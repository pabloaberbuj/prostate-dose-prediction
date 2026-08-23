"""Diagnostico rapido (2026-08-16), dos partes independientes, pensado para
costar minutos y no horas de GPU:

PARTE A (punto 2 de Pablo -- cuello de botella de tiempo): mide el costo real
de forward+backward de `engine.compute_dose` con distintos `beam_chunk_size`
sobre UN arco real (mismo numero de control points que cualquier corrida),
sin optimizar nada -- solo timing.

PARTE B (punto 1 -- hipotesis del freeze): confirma si `dvh_Dp_loss` (D98,
pydosert/objectives/losses.py) da un gradiente ONE-HOT (un solo voxel del
PTV recibe gradiente no-cero) y si la IDENTIDAD de ese voxel es mas o menos
estable entre dos estados de parametros cercanos, comparando
leaf_margin_mm=5.0 (v6, convergio) vs leaf_margin_mm=2.0 (v7, se congelo).
No corre LBFGS ni ninguna optimizacion real -- solo dos forward/backward
por margen (uno en el init, uno despues de un unico paso manual pequeno de
Adam) para no gastar tiempo de mas.
"""
import sys
import time
from pathlib import Path

import torch

REPO = Path(r"c:\Pablo\ProstateDoseProject\repo")
sys.path.insert(0, str(REPO))

from src.planning.build_beams import STRUCT_NAMES_DEFAULT, load_patient_and_arcs
from src.planning.engine_setup import build_dose_engine, load_machine_config
from src.bridge.unet_to_target import unet_dose_to_pdrt_grid
from src.planning.mimicking import (
    OptimizableArc, compute_dvh_targets, dvh_loss, STRUCT_WEIGHTS_DEFAULT, build_mse_weights,
    D98_WINDOW_FRAC,
)

PATIENT_DIR = r"c:\Pablo\ProstateDoseProject\dicom_pilot\PT_003fe2bb84986507"
PRED_NPZ = r"c:\Pablo\ProstateDoseProject\repo\results\pilot_rd\pred_PT_003fe2bb84986507.npz"
RX_GY_TOTAL = 78.0
N_FRACCIONES = 39
RUN_PART_A = False  # ya corrida con exito (promedios: 1=32.8s 8=16.5s 16=15.5s 32=15.7s) -- desactivada para no repetir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
print(f"[diag] device={device}")

t0 = time.perf_counter()
patient, arcs = load_patient_and_arcs(PATIENT_DIR, struct_names=STRUCT_NAMES_DEFAULT, device=device, dtype=dtype)
machine_config = load_machine_config()
engine = build_dose_engine(patient, beam_template=arcs[0], device=device, dtype=dtype)
engine.train()
body_mask = patient.structures["BODY"].to(device)
density_image = patient.density_image.to(device=device, dtype=dtype).unsqueeze(0)
print(f"[diag] setup patient+engine: {time.perf_counter() - t0:.1f}s")

# ----------------------------------------------------------------------
# PARTE A: timing de beam_chunk_size sobre un arco real (sin optimizar)
# ----------------------------------------------------------------------
if RUN_PART_A:
    print("\n=== PARTE A: timing beam_chunk_size (forward+backward, 1 arco real, 2 repeticiones c/u) ===")
    print("(178, sin chunking, YA CONFIRMADO OOM la corrida anterior -- no se reintenta)")
    arco_real = arcs[0]
    leaf = arco_real.leaf_positions.clone().requires_grad_(True)
    mu = arco_real.mus.clone().requires_grad_(True)

    from pydosert.data import BeamSequence

    resultados = {}
    for chunk in (1, 8, 16, 32):
        tiempos = []
        for rep in range(2):
            bs = BeamSequence(
                mus=mu, leaf_positions=leaf, jaw_positions=arco_real.jaw_positions,
                gantry_angles=arco_real.gantry_angles, collimator_angles=arco_real.collimator_angles,
                field_size=arco_real.field_size, iso_center=arco_real.iso_center, sid=arco_real.sid,
            )
            try:
                t0 = time.perf_counter()
                d = engine.compute_dose(bs, density_image=density_image, beam_chunk_size=chunk, overwrite=True)
                d.sum().backward()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                tiempos.append(dt)
                print(f"[diag] beam_chunk_size={chunk:>3d}  rep={rep+1}  forward+backward = {dt:.1f}s")
            except (torch.cuda.OutOfMemoryError, torch.AcceleratorError) as e:
                print(f"[diag] beam_chunk_size={chunk:>3d}  rep={rep+1}  OOM: {e}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            leaf.grad = None
            mu.grad = None
        if tiempos:
            resultados[chunk] = sum(tiempos) / len(tiempos)

    print("\n[diag] promedios:")
    for chunk, prom in resultados.items():
        print(f"    beam_chunk_size={chunk:>3d}  promedio={prom:.1f}s")

# ----------------------------------------------------------------------
# PARTE B: one-hot / estabilidad del voxel D98 bajo margin=5 vs margin=2
# ----------------------------------------------------------------------
print("\n=== PARTE B: gradiente de dvh_Dp_loss (D98) -- one-hot? estable? ===")
dose_target_np, _ = unet_dose_to_pdrt_grid(PATIENT_DIR, PRED_NPZ, RX_GY_TOTAL, N_FRACCIONES)
dose_target = torch.from_numpy(dose_target_np).to(device=device, dtype=dtype)
rx_frac_gy = RX_GY_TOTAL / N_FRACCIONES
dvh_targets = compute_dvh_targets(dose_target, patient, rx_frac_gy)
print(f"[diag] D98 target (crudo) = {dvh_targets['ptv_d98_gy']:.3f} Gy/fx")

weights_vol, _ = build_mse_weights(patient, STRUCT_WEIGHTS_DEFAULT, device, dtype)
body_bool = body_mask > 0
w_body = weights_vol[body_bool]
w_sum = w_body.sum()
tgt_body = dose_target[body_bool]
ptv_mask = patient.structures["PTV_High"].to(device) > 0

leaf_widths = machine_config.leaf_widths

for margin in (5.0, 2.0):
    print(f"\n--- leaf_margin_mm={margin} ---")
    optim_arcs = [
        OptimizableArc(
            a, ptv_conform_start=True, ptv_mask=patient.structures["PTV_High"],
            resolution=patient.resolution, leaf_widths=leaf_widths,
            leaf_margin_mm=margin, jaw_margin_mm=2.0, leaf_max_step_mm=None,
            machine_config=machine_config, optimize_jaws=False,
        )
        for a in arcs
    ]
    params = [p for oa in optim_arcs for p in oa.parameters()]
    opt = torch.optim.Adam(params, lr=0.01)

    def forward_and_d98():
        dose_pred = None
        for oa in optim_arcs:
            bs = oa.to_beam_sequence()
            d = engine.compute_dose(bs, density_image=density_image, beam_chunk_size=8)
            dose_pred = d if dose_pred is None else dose_pred + d
        dose_pred = dose_pred[0]
        return dose_pred

    def reportar_voxel_ganador(dose_pred, etiqueta):
        vals = dose_pred[ptv_mask]
        n = vals.numel()
        k = int(0.02 * (n - 1))
        vals_sorted, idx_sorted = torch.sort(vals)
        orig_idx = int(idx_sorted[k])
        Dp = vals_sorted[k]
        opt.zero_grad(set_to_none=True)
        Dp.backward(retain_graph=True)
        nz = 0
        for oa in optim_arcs:
            if oa.raw_leaf.grad is not None:
                nz += int((oa.raw_leaf.grad != 0).sum())
            if oa.raw_mu.grad is not None:
                nz += int((oa.raw_mu.grad != 0).sum())
        print(f"    [{etiqueta}] Dp_1voxel(D98 crudo)={float(Dp.detach()):.4f}Gy  "
              f"voxel_idx(dentro del PTV)={orig_idx}  params_no_cero_por_Dp={nz}")
        opt.zero_grad(set_to_none=True)

        # Ventana usada por smooth_d98_loss (mismo k, mismo ancho) -- el
        # conjunto de indices ORIGINALES (no el rango ordenado) que promedia.
        half = max(1, int(D98_WINDOW_FRAC * n) // 2)
        lo, hi = max(0, k - half), min(n, k + half + 1)
        ventana = set(int(i) for i in idx_sorted[lo:hi])
        return orig_idx, ventana

    t0 = time.perf_counter()
    dose_pred_1 = forward_and_d98()
    idx_1, ventana_1 = reportar_voxel_ganador(dose_pred_1, "estado inicial (init conformado)")

    # Un solo paso manual de Adam con la loss COMPLETA (mse + dvh), para
    # simular "un paso de optimizacion" barato (sin closure/line-search de LBFGS).
    mse = (w_body * (dose_pred_1[body_bool] - tgt_body) ** 2).sum() / w_sum
    dh = dvh_loss(dose_pred_1, patient, dvh_targets)
    loss = mse + 1.0 * dh["total"]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    print(f"    [paso manual Adam] mse={float(mse):.6f}  dvh_raw={float(dh['total']):.6f}  "
          f"(tiempo total hasta aca: {time.perf_counter() - t0:.1f}s)")

    dose_pred_2 = forward_and_d98()
    idx_2, ventana_2 = reportar_voxel_ganador(dose_pred_2, "tras 1 paso manual de Adam (ya con smooth_d98_loss activo)")

    overlap = len(ventana_1 & ventana_2)
    print(f"    => voxel unico D98 {'SE MANTUVO' if idx_1 == idx_2 else f'CAMBIO ({idx_1} -> {idx_2})'}  |  "
          f"ventana smooth_d98_loss: {overlap}/{len(ventana_1)} vóxeles en común entre los dos estados "
          f"(margin={margin})")
