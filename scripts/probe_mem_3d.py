"""
Sondeo de memoria — U-Net 3D sobre volúmenes de normo (BLOQUE 1, exp_normo_3dunet).

Instancia UNet3D, corre un forward+backward+optimizer.step() con batch=1
sobre un volumen del tamaño real de normo (worst-case, no típico) y mide
torch.cuda.max_memory_allocated()/reserved(). Barre:
  - Volumen: 'full' (256x256x80, worst-case Z del train set) vs.
             'crop'  (128x240x80, bbox real PTV+Rectum+Bladder+2cm margen,
                      worst-case sobre 265 pacientes de train, ver
                      scratchpad/compute_bbox_stats.py).
  - base_features: 16, 24, 32.
  - AMP (autocast fp16 + GradScaler): on/off.
  - Gradient checkpointing: on/off.

Cada combinación corre en un subproceso aislado (Python fresco) para que la
medición de pico de memoria de una config no quede contaminada por el
allocator cacheado de la config anterior.

Uso:
    python scripts/probe_mem_3d.py                     # barrido completo
    python scripts/probe_mem_3d.py --single --volume full \
        --base-features 16 --amp --grad-checkpoint     # una sola config (debug)
"""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IN_CHANNELS = 5  # CT + BODY + PSDM_ptv + PSDM_rectum + PSDM_bladder (igual que exp002, en 3D)
OUT_CHANNELS = 1
DEPTH = 4  # misma profundidad que la serie 2D

# (D, H, W) = (Z, Y, X). Z=80 en ambos: ya es el worst-case tras recorte axial
# ROI+margen en preprocess.py (max 77 cortes en train, redondeado a multiplo
# de 8 como hace DoseDataModule._calcular_n_slices_target).
VOLUMES = {
    "full": (80, 256, 256),
    "crop": (80, 240, 128),  # bbox real PTV|Rectum|Bladder + 2cm, worst-case train, /16
}

RESULT_MARKER = "RESULT_JSON:"
GPU_TOTAL_GB = 12.0
HEADROOM_GB = 1.5

# Techo DURO de memoria CUDA por proceso, como fraccion de la VRAM real del
# device (no de GPU_TOTAL_GB fijo). CRITICO en Windows/WDDM: sin este cap, un
# alloc que excede la VRAM fisica NO tira un OutOfMemoryError limpio — el
# driver empieza a paginar silenciosamente a RAM del sistema ("shared GPU
# memory"), lo que trabó la PC entera 17+ minutos la primera vez que se corrió
# este sondeo (ver incidente documentado en la sesion). Con el cap, el
# allocator de PyTorch respeta el limite y tira OOM limpio en vez de pagear.
HARD_MEM_FRACTION = 0.88  # deja ~12%  (~1.4GB en una A2000 12GB) para driver/SO


def run_single(volume: str, base_features: int, amp: bool, grad_checkpoint: bool,
              n_timed_steps: int = 5) -> dict:
    import time
    import torch
    from src.models.unet3d import UNet3D

    torch.cuda.set_per_process_memory_fraction(HARD_MEM_FRACTION, device=0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    D, H, W = VOLUMES[volume]
    device = "cuda"

    model = UNet3D(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS,
        base_features=base_features, depth=DEPTH, dropout=0.0,
        grad_checkpointing=grad_checkpoint,
    ).to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    x = torch.randn(1, IN_CHANNELS, D, H, W, device=device)
    y = torch.randn(1, OUT_CHANNELS, D, H, W, device=device)

    def _step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    # Paso 0 (warmup, no cronometrado): incluye compilación de kernels/cudnn
    # autotune — se descarta del promedio de tiempo pero cuenta para el pico
    # de memoria (peor caso real, el pico no baja en pasos siguientes).
    _step()
    torch.cuda.synchronize()

    step_times_s = []
    for _ in range(n_timed_steps):
        t0 = time.perf_counter()
        _step()
        torch.cuda.synchronize()
        step_times_s.append(time.perf_counter() - t0)

    peak_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
    peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9

    return {
        "volume": volume, "shape_dhw": [D, H, W],
        "base_features": base_features, "amp": amp,
        "grad_checkpoint": grad_checkpoint,
        "n_params_millions": round(n_params / 1e6, 2),
        "peak_alloc_gb": round(peak_alloc_gb, 3),
        "peak_reserved_gb": round(peak_reserved_gb, 3),
        "step_time_s_mean": round(sum(step_times_s) / len(step_times_s), 3),
        "step_time_s_min": round(min(step_times_s), 3),
        "fits_12gb_with_headroom": (peak_reserved_gb + HEADROOM_GB) <= GPU_TOTAL_GB,
        "status": "ok",
    }


def main_single(args):
    try:
        result = run_single(args.volume, args.base_features, args.amp, args.grad_checkpoint)
    except Exception as e:
        import torch
        is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
        result = {
            "volume": args.volume, "shape_dhw": list(VOLUMES[args.volume]),
            "base_features": args.base_features, "amp": args.amp,
            "grad_checkpoint": args.grad_checkpoint,
            "status": "OOM" if is_oom else f"ERROR: {e}",
            "fits_12gb_with_headroom": False,
        }
    print(RESULT_MARKER + json.dumps(result))


def main_sweep():
    repo_root = Path(__file__).resolve().parent.parent
    combos = list(itertools.product(
        VOLUMES.keys(), [16, 24, 32], [True, False], [True, False]
    ))
    results = []
    for i, (volume, bf, amp, gckpt) in enumerate(combos):
        cmd = [
            sys.executable, str(Path(__file__).resolve()), "--single",
            "--volume", volume, "--base-features", str(bf),
        ]
        if amp:
            cmd.append("--amp")
        if gckpt:
            cmd.append("--grad-checkpoint")
        print(f"[{i+1}/{len(combos)}] volume={volume} base_features={bf} "
              f"amp={amp} grad_ckpt={gckpt} ...", flush=True)
        # Timeout corto y estricto: una config sana (6 pasos fwd+bwd, batch=1)
        # termina en segundos. Cualquier cosa que tarde >90s ya no es una
        # medicion valida — probablemente es el mismo cuadro de oversubscription
        # WDDM que congelo la PC la primera vez (ver HARD_MEM_FRACTION arriba).
        # Con timeout=90 detectamos y matamos ese caso rapido en vez de dejarlo
        # degradar el sistema por minutos.
        try:
            proc = subprocess.Popen(cmd, cwd=str(repo_root), stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = proc.communicate(timeout=90)
            except subprocess.TimeoutExpired:
                # taskkill /T mata el arbol completo por las dudas (defensa en
                # profundidad ante el problema ya visto de procesos huerfanos
                # que subprocess.Popen.kill() no siempre alcanza a limpiar).
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True)
                proc.wait(timeout=15)
                results.append({
                    "volume": volume, "shape_dhw": list(VOLUMES[volume]),
                    "base_features": bf, "amp": amp, "grad_checkpoint": gckpt,
                    "status": "TIMEOUT (posible oversubscription WDDM, proceso matado)",
                    "fits_12gb_with_headroom": False,
                })
                print("  -> TIMEOUT (matado via taskkill /T)")
                continue
            proc = argparse.Namespace(stdout=stdout, stderr=stderr)
        except Exception as e:
            results.append({
                "volume": volume, "shape_dhw": list(VOLUMES[volume]),
                "base_features": bf, "amp": amp, "grad_checkpoint": gckpt,
                "status": f"LAUNCH_ERROR: {e}", "fits_12gb_with_headroom": False,
            })
            print(f"  -> LAUNCH_ERROR: {e}")
            continue

        line = None
        for out_line in proc.stdout.splitlines():
            if out_line.startswith(RESULT_MARKER):
                line = out_line[len(RESULT_MARKER):]
                break
        if line is None:
            results.append({
                "volume": volume, "shape_dhw": list(VOLUMES[volume]),
                "base_features": bf, "amp": amp, "grad_checkpoint": gckpt,
                "status": f"NO_RESULT stderr_tail={proc.stderr[-500:]!r}",
                "fits_12gb_with_headroom": False,
            })
            print(f"  -> sin resultado (revisar stderr)")
            continue
        result = json.loads(line)
        results.append(result)
        if result["status"] == "ok":
            print(f"  -> alloc={result['peak_alloc_gb']}GB reserved={result['peak_reserved_gb']}GB "
                  f"fits={result['fits_12gb_with_headroom']}")
        else:
            print(f"  -> {result['status']}")

    out_path = repo_root / "results" / "probe_mem_3d.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "in_channels": IN_CHANNELS, "depth": DEPTH, "volumes": VOLUMES,
            "gpu_total_gb": GPU_TOTAL_GB, "headroom_gb": HEADROOM_GB,
            "results": results,
        }, f, indent=2)
    print(f"\nGuardado en {out_path}")

    print("\n| Volumen | shape(D,H,W) | base_feat | AMP | grad_ckpt | alloc(GB) | reserved(GB) | s/paso | entra en 12GB |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        shape = "x".join(str(v) for v in r["shape_dhw"])
        alloc = r.get("peak_alloc_gb", "-")
        reserved = r.get("peak_reserved_gb", "-")
        step_t = r.get("step_time_s_mean", "-")
        fits = "SI" if r.get("fits_12gb_with_headroom") else "no"
        status = r["status"]
        estado = status if status != "ok" else fits
        print(f"| {r['volume']} | {shape} | {r['base_features']} | {r['amp']} | "
              f"{r['grad_checkpoint']} | {alloc} | {reserved} | {step_t} | {estado} |")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--single", action="store_true")
    p.add_argument("--volume", choices=list(VOLUMES.keys()))
    p.add_argument("--base-features", type=int)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad-checkpoint", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.single:
        main_single(args)
    else:
        main_sweep()
