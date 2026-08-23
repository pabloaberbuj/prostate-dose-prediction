"""Chequeo de sandbox (COMISIONAMIENTO_PDRT.md): recalcula un plan VMAT real
de próstata (2 arcos) con el motor de PyDoseRT usando el machine_config
comisionado, y compara contra la dosis AAA real (RD de Eclipse) con índice
gamma 3D.

No busca precisión clínica (fuera de alcance del proyecto) - sólo confirma
que el sandbox no está roto en heterogeneidad real antes de invertir en el
mimicking. Ver COMISIONAMIENTO_PDRT.md sección "Chequeo de sandbox".

Uso: python commissioning/recompute_check.py <carpeta_paciente_dicom>
La carpeta debe tener los CT.*.dcm, un RS.*.dcm, un RP.*.dcm y uno o más
RD.*.dcm (todos del mismo paciente/plan, exportados de Eclipse).
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.planning.build_beams import STRUCT_NAMES_DEFAULT, load_patient_and_arcs
from src.planning.engine_setup import DEFAULT_MACHINE_CONFIG, build_dose_engine

RESULTS_DIR = Path(__file__).resolve().parent / "sandbox_results"
STRUCT_NAMES = STRUCT_NAMES_DEFAULT


def gamma_3d(dose_ref, dose_eval, spacing_mm, dose_percent_threshold, distance_mm_threshold,
             lower_cutoff_frac=0.20, mask=None, search_factor=1.5):
    """Gamma 3D global (normalizado al máximo de dose_ref), mismo grid para
    ambas dosis (no hace falta interpolar entre grillas distintas). Búsqueda
    local por ventana de voxels en vez de la interpolación fina de pymedphys
    (cuyo backend `interpolation`/econforge no corre en Python 3.13 - usa el
    módulo `cgi`, eliminado en 3.11+/removido en 3.13). Suficiente para un
    chequeo de sandbox, no para una auditoría clínica de precisión.
    """
    sz, sy, sx = spacing_mm
    dta = distance_mm_threshold
    global_ref = float(dose_ref.max())
    dose_crit = dose_percent_threshold / 100.0 * global_ref

    rz = int(np.ceil(dta * search_factor / sz))
    ry = int(np.ceil(dta * search_factor / sy))
    rx = int(np.ceil(dta * search_factor / sx))
    padded = np.pad(
        dose_eval.astype(np.float32),
        ((rz, rz), (ry, ry), (rx, rx)),
        mode="constant",
        constant_values=-1e6,
    )
    shape = dose_ref.shape
    best_g2 = np.full(shape, np.inf, dtype=np.float32)
    ref = dose_ref.astype(np.float32)
    max_dist2 = (dta * search_factor) ** 2
    for dz in range(-rz, rz + 1):
        for dy in range(-ry, ry + 1):
            for dx in range(-rx, rx + 1):
                dist2 = (dz * sz) ** 2 + (dy * sy) ** 2 + (dx * sx) ** 2
                if dist2 > max_dist2:
                    continue
                shifted = padded[rz + dz:rz + dz + shape[0], ry + dy:ry + dy + shape[1], rx + dx:rx + dx + shape[2]]
                dd = (shifted - ref) / dose_crit
                g2 = dd * dd + dist2 / (dta * dta)
                np.minimum(best_g2, g2, out=best_g2)
    gamma = np.sqrt(best_g2)
    valid = dose_ref >= lower_cutoff_frac * global_ref
    if mask is not None:
        valid &= mask
    return gamma, valid


def main(patient_dir_str):
    patient_dir = Path(patient_dir_str)
    out_dir = RESULTS_DIR / patient_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    patient, beam_sequences = load_patient_and_arcs(
        patient_dir, struct_names=STRUCT_NAMES, device=device, tmp_dir=out_dir,
    )
    print(f"Arcos cargados: {len(beam_sequences)}")
    for i, bs in enumerate(beam_sequences):
        print(f"  arco {i+1}: {bs.mus.shape[0]} control points, MU total={float(bs.mus.sum()):.1f}")

    dose_engine = build_dose_engine(patient, device=device)

    density_image = patient.density_image.to(device).to(torch.float32)
    dose_pred = None
    for i, bs in enumerate(beam_sequences):
        d = dose_engine.compute_dose(
            bs.to(device).to(torch.float32),
            density_image=density_image,
            beam_chunk_size=1,
        )
        d = d.squeeze(0).detach().cpu()
        dose_pred = d if dose_pred is None else dose_pred + d
        print(f"  arco {i+1} calculado, dosis max={float(d.max()):.4f} Gy")

    dose_ref = patient.dose.cpu().numpy()
    dose_pred_np = dose_pred.numpy()

    body = patient.structures.get("BODY")
    body_mask = body.cpu().numpy() > 0 if body is not None else np.ones_like(dose_ref, dtype=bool)

    # Reescalado a un percentil alto de dosis, igual que el ejemplo oficial
    # rtplan.ipynb (auto_calibrate ya calibra la escala aproximada; esto
    # corrige el residuo fino antes de comparar formas de distribución).
    scale = float(np.quantile(dose_ref[body_mask], 0.999) / np.quantile(dose_pred_np[body_mask], 0.999))
    dose_pred_scaled = dose_pred_np * scale
    print(f"Dosis ref max={dose_ref.max():.3f} Gy | dosis PDRT max (antes de reescalar)={dose_pred_np.max():.3f} Gy | factor reescalado={scale:.4f}")

    np.save(out_dir / "dose_ref.npy", dose_ref)
    np.save(out_dir / "dose_pdrt.npy", dose_pred_scaled)

    spacing_mm = patient.resolution  # mm, orden (z,y,x) según docstring de load_dicom

    results = {
        "patient": patient_dir.name,
        "n_arcs": len(beam_sequences),
        "scale_factor": scale,
        "dose_ref_max": float(dose_ref.max()),
        "dose_pdrt_max_prescale": float(dose_pred_np.max()),
    }

    for crit_pct, crit_mm in [(5.0, 5.0), (3.0, 3.0)]:
        gamma, valid = gamma_3d(
            dose_ref, dose_pred_scaled, spacing_mm,
            dose_percent_threshold=crit_pct, distance_mm_threshold=crit_mm,
            lower_cutoff_frac=0.20, mask=body_mask,
        )
        pass_rate = float(np.mean(gamma[valid] <= 1.0)) * 100.0
        print(f"Gamma {crit_pct:.0f}%/{crit_mm:.0f}mm (global, corte 20%, dentro de BODY): pass rate = {pass_rate:.1f}% (n={int(valid.sum())})")
        results[f"gamma_{crit_pct:.0f}pct_{crit_mm:.0f}mm"] = pass_rate

    for name in STRUCT_NAMES:
        mask = patient.structures.get(name)
        if mask is None:
            print(f"  [{name}: no encontrada en el RS]")
            results[f"diff_{name}_pct"] = None
            continue
        m = mask.cpu().numpy() > 0
        if not m.any():
            print(f"  [{name}: máscara vacía]")
            results[f"diff_{name}_pct"] = None
            continue
        d_ref_roi = dose_ref[m]
        d_pred_roi = dose_pred_scaled[m]
        diff_pct = 100.0 * (d_pred_roi.mean() - d_ref_roi.mean()) / d_ref_roi.mean()
        print(f"  {name}: dosis media ref={d_ref_roi.mean():.3f} Gy, PDRT={d_pred_roi.mean():.3f} Gy, diff={diff_pct:+.1f}%")
        results[f"diff_{name}_pct"] = diff_pct

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
