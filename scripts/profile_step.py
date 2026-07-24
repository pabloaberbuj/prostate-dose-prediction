"""
Diagnóstico de cuello de botella por componente.
Mide: I/O disco, GPU transfer, forward, backward, loss, métricas.
Usa solo 5 pacientes — no OOM, termina en <2 minutos.

Uso:
    python scripts/profile_step.py --config configs/exp001_unet2d_baseline.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datamodules.dose_datamodule import DosePatientDataset
from src.models.lightning_module import DosePredictionModule


def timer(label, fn, warmup=False):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000
    tag = " (warmup)" if warmup else ""
    print(f"  {label:<35} {elapsed:7.1f} ms{tag}")
    return result, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n-patients", type=int, default=5)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*55}")
    print(f"PROFILE: {args.n_patients} pacientes, device={device}")
    print(f"{'='*55}")

    # ── 1. Obtener paths de pacientes ────────────────────────────────
    import json
    splits = json.loads(Path(cfg.data.splits_file).read_text())
    processed = Path(cfg.data.processed_dir)
    paths = [processed / f"{a}.npz" for a in splits['train'][:args.n_patients]
             if (processed / f"{a}.npz").exists()]
    print(f"\n[1] Paths: {len(paths)} pacientes de train")

    # ── 2. Calcular n_target ─────────────────────────────────────────
    z_vals = []
    for p in paths:
        d = np.load(str(p), allow_pickle=True)
        z_vals.append(d['ct'].shape[0])
        d.close()
    n_target = ((max(z_vals) + 7) // 8) * 8
    print(f"   Z range: {min(z_vals)}-{max(z_vals)}, n_target={n_target}")

    # ── 3. Medir I/O por paciente ────────────────────────────────────
    print(f"\n[2] I/O por paciente (carga NPZ + pad):")
    ds_nocache = DosePatientDataset(paths, augment=False,
                                    fixed_n_slices=n_target, cache_in_ram=False)
    io_times = []
    for i in range(len(paths)):
        _, ms = timer(f"  paciente {i}", lambda idx=i: ds_nocache[idx])
        io_times.append(ms)
    print(f"  Promedio I/O: {sum(io_times)/len(io_times):.0f} ms  |  "
          f"min={min(io_times):.0f}  max={max(io_times):.0f} ms")

    # ── 4. Medir GPU transfer ────────────────────────────────────────
    print(f"\n[3] GPU transfer (CPU -> {device}):")
    sample = ds_nocache[0]
    tensors = {k: v for k, v in sample.items() if torch.is_tensor(v)}
    _, ms_transfer = timer("  batch CPU->GPU",
                           lambda: {k: v.to(device) for k, v in tensors.items()})
    batch_gpu = {k: v.unsqueeze(0).to(device) for k, v in tensors.items()}
    batch_gpu['anonid'] = sample['anonid']
    batch_gpu['factor_norm'] = sample['factor_norm']

    # ── 5. Construir modelo ──────────────────────────────────────────
    print(f"\n[4] Modelo:")
    model = DosePredictionModule(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parámetros: {n_params:,}")
    if torch.cuda.is_available():
        vram_model = torch.cuda.memory_allocated() / 1e6
        print(f"  VRAM modelo: {vram_model:.0f} MB")

    # ── 6. Warmup ────────────────────────────────────────────────────
    print(f"\n[5] Forward + backward (1 warmup + 3 mediciones):")
    scaler = torch.cuda.amp.GradScaler()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def full_step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16,
                            enabled=torch.cuda.is_available()):
            x = model._build_input(batch_gpu)
            pred = model(x)
            target = batch_gpu['dose']
            body   = batch_gpu['body_mask']
            structs = {
                'ptv':     batch_gpu['ptv_mask'],
                'rectum':  batch_gpu['rectum_mask'],
                'bladder': batch_gpu['bladder_mask'],
            }
            losses = model.loss_fn(pred, target, body, structs)
        scaler.scale(losses['total']).backward()
        scaler.step(opt)
        scaler.update()
        return pred, losses

    # Warmup
    timer("warmup", full_step, warmup=True)

    # Mediciones separadas
    fw_times, bw_times, step_times = [], [], []
    for i in range(3):
        opt.zero_grad(set_to_none=True)

        # Forward solo
        def fwd_only():
            with torch.autocast(device_type='cuda', dtype=torch.float16,
                                enabled=torch.cuda.is_available()):
                x = model._build_input(batch_gpu)
                return model(x)

        pred, ms_fw = timer(f"  forward #{i+1}", fwd_only)
        fw_times.append(ms_fw)

        # Backward solo (sobre la última predicción)
        with torch.autocast(device_type='cuda', dtype=torch.float16,
                            enabled=torch.cuda.is_available()):
            x = model._build_input(batch_gpu)
            pred = model(x)
            losses = model.loss_fn(pred, batch_gpu['dose'],
                                   batch_gpu['body_mask'],
                                   {'ptv': batch_gpu['ptv_mask'],
                                    'rectum': batch_gpu['rectum_mask'],
                                    'bladder': batch_gpu['bladder_mask']})
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        scaler.scale(losses['total']).backward()
        scaler.step(opt)
        scaler.update()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ms_bw = (time.perf_counter() - t0) * 1000
        print(f"  {'backward+optim #'+str(i+1):<35} {ms_bw:7.1f} ms")
        bw_times.append(ms_bw)

    # Step completo (forward + backward + optim)
    print()
    for i in range(3):
        _, ms = timer(f"  step completo #{i+1}", full_step)
        step_times.append(ms)

    # ── 7. Resumen ───────────────────────────────────────────────────
    if torch.cuda.is_available():
        vram_peak = torch.cuda.max_memory_allocated() / 1e6
        vram_cur  = torch.cuda.memory_allocated() / 1e6
    else:
        vram_peak = vram_cur = 0

    print(f"\n{'='*55}")
    print(f"RESUMEN (promedio de 3 runs)")
    print(f"{'='*55}")
    print(f"  I/O por paciente (NPZ + pad):  {sum(io_times)/len(io_times):6.0f} ms")
    print(f"  GPU transfer (CPU->GPU):        {ms_transfer:6.0f} ms")
    print(f"  Forward (GPU):                 {sum(fw_times)/3:6.0f} ms")
    print(f"  Backward + optim (GPU):        {sum(bw_times)/3:6.0f} ms")
    print(f"  Step completo:                 {sum(step_times)/3:6.0f} ms")
    print(f"  VRAM pico:                     {vram_peak:6.0f} MB")
    print(f"  VRAM actual:                   {vram_cur:6.0f} MB")
    print()
    total_step = sum(io_times)/len(io_times) + sum(step_times)/3
    epoch_min = 265 * total_step / 1000 / 60
    print(f"  Epoch estimada (265 pac):     {epoch_min:5.1f} min")
    print(f"  Bottleneck: ", end="")
    io_avg   = sum(io_times)/len(io_times)
    step_avg = sum(step_times)/3
    if io_avg > step_avg * 0.3:
        print(f"I/O ({io_avg:.0f}ms) — activar cache_train si hay RAM")
    elif step_avg > 5000:
        print(f"COMPUTE ({step_avg:.0f}ms) — reducir base_features o n_slices")
    else:
        print(f"equilibrado")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
