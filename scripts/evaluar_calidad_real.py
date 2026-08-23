"""
Piloto substructuras -- calidad REAL (desde RD de Eclipse, no prediccion) de
PTV_High / Rectum / Rectum_hot / Rectum_cold / Bladder / Rectum!PTV (overlap),
para comparar planes ya optimizados (ej. los "push10" control/test).

Reusa `masks.npz` (Rectum_hot/Rectum_cold/Rectum, generados en Bloque 1) --
misma geometria de paciente, no depende del plan que se este evaluando.
PTV_High/Bladder/Rectum!PTV se leen frescos del RS via rt-utils.

Uso:
    .venv/Scripts/python.exe scripts/evaluar_calidad_real.py \
        --dicom-dir "//10.100.0.252/.../PT_44a81316c45f64f5" \
        --masks-npz results/substruct/PT_44a81316c45f64f5/masks.npz \
        --rd "//10.100.0.252/.../xml substructure/Control/RD.....dcm" \
        --rx-gy 78 --nombre "Control_push10"
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.bridge.unet_to_target import leer_geometria_ct, rd_real_a_grid_nativo  # noqa: E402
from src.bridge.rtstruct_io import cargar_rtstruct_ct_only  # noqa: E402


def stats_estructura(dose_pct: np.ndarray, mask: np.ndarray) -> dict:
    v = dose_pct[mask]
    return {
        "vol_vox": int(mask.sum()),
        "mean%": float(v.mean()),
        "median%": float(np.median(v)),
        "min%": float(v.min()),
        "max%": float(v.max()),
        "D95%": float(np.percentile(v, 5)),
        "D98%": float(np.percentile(v, 2)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", required=True)
    parser.add_argument("--masks-npz", required=True)
    parser.add_argument("--rd", required=True)
    parser.add_argument("--rx-gy", type=float, required=True)
    parser.add_argument("--nombre", required=True, help="Etiqueta del plan para el reporte")
    parser.add_argument("--bladder-roi-name", default="Bladder")
    parser.add_argument("--ptv-id", default="PTV_High")
    parser.add_argument("--overlap-roi-name", default="Rectum!PTV")
    args = parser.parse_args()

    dicom_dir = Path(args.dicom_dir)
    ct_geom = leer_geometria_ct(dicom_dir)

    print(f"Resampleando RD real ({args.rd}) al grid nativo ...")
    dose_gy_native = rd_real_a_grid_nativo(args.rd, ct_geom)
    dose_pct_rt = (dose_gy_native / args.rx_gy * 100.0).transpose(1, 2, 0)  # -> convencion rt-utils

    d = np.load(args.masks_npz, allow_pickle=True)
    roi_name = str(d["roi_name"])

    rtstruct = cargar_rtstruct_ct_only(dicom_dir)
    disponibles = rtstruct.get_roi_names()

    estructuras = {
        args.ptv_id: rtstruct.get_roi_mask_by_name(args.ptv_id) if args.ptv_id in disponibles else None,
        "Rectum": d["roi_mask"],
        f"{roi_name}_hot": d["hot"],
        f"{roi_name}_cold": d["cold"],
        args.bladder_roi_name: (rtstruct.get_roi_mask_by_name(args.bladder_roi_name)
                                if args.bladder_roi_name in disponibles else None),
        args.overlap_roi_name: (rtstruct.get_roi_mask_by_name(args.overlap_roi_name)
                                if args.overlap_roi_name in disponibles else None),
    }

    print(f"\n=== {args.nombre} ===")
    header = f"{'estructura':>15} {'vox':>7} {'mean%':>7} {'median%':>8} {'min%':>7} {'max%':>7} {'D95%':>7} {'D98%':>7}"
    print(header)
    filas = {}
    for nombre, mask in estructuras.items():
        if mask is None:
            print(f"{nombre:>15}  [no encontrada en el RS: {disponibles}]")
            continue
        if mask.shape != dose_pct_rt.shape:
            print(f"{nombre:>15}  [shape no coincide: {mask.shape} vs {dose_pct_rt.shape}]")
            continue
        s = stats_estructura(dose_pct_rt, mask)
        filas[nombre] = s
        print(f"{nombre:>15} {s['vol_vox']:7d} {s['mean%']:7.1f} {s['median%']:8.1f} "
              f"{s['min%']:7.1f} {s['max%']:7.1f} {s['D95%']:7.1f} {s['D98%']:7.1f}")

    return filas


if __name__ == "__main__":
    main()
