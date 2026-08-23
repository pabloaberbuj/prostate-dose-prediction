"""
Piloto substructuras -- calidad REAL (desde RD de Eclipse) de las subestructuras
del modo 3 vias (overlap Rectum&PTV separado, ver build_substruct_objtemplates.py
--overlap-from-ptv-intersection). Reusa masks.npz (Rectum/hot/cold) + PTV_High/
Bladder frescos via rt-utils, y calcula overlap/hot_no_overlap/cold_no_overlap
igual que en la generacion del XML.

Uso:
    .venv/Scripts/python.exe scripts/evaluar_calidad_real_3vias.py \
        --dicom-dir "//10.100.0.252/.../PT_44a81316c45f64f5" \
        --masks-npz results/substruct/PT_44a81316c45f64f5/masks.npz \
        --rd "//.../xml substructure/Control/RD....dcm" \
        --rx-gy 78 --nombre "Control_push15_overlap"
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
    parser.add_argument("--nombre", required=True)
    parser.add_argument("--bladder-roi-name", default="Bladder")
    parser.add_argument("--ptv-id", default="PTV_High")
    args = parser.parse_args()

    dicom_dir = Path(args.dicom_dir)
    ct_geom = leer_geometria_ct(dicom_dir)

    print(f"Resampleando RD real ({Path(args.rd).name}) al grid nativo ...")
    dose_gy_native = rd_real_a_grid_nativo(args.rd, ct_geom)
    dose_pct_rt = (dose_gy_native / args.rx_gy * 100.0).transpose(1, 2, 0)  # convencion rt-utils

    d = np.load(args.masks_npz, allow_pickle=True)
    roi_mask, hot, cold = d["roi_mask"], d["hot"], d["cold"]

    rtstruct = cargar_rtstruct_ct_only(dicom_dir)
    ptv_mask = rtstruct.get_roi_mask_by_name(args.ptv_id)
    bladder_mask = rtstruct.get_roi_mask_by_name(args.bladder_roi_name)

    overlap = roi_mask & ptv_mask
    hot_no_overlap = hot & ~overlap
    cold_no_overlap = cold & ~overlap
    assert np.array_equal(overlap | hot_no_overlap | cold_no_overlap, roi_mask)

    estructuras = {
        args.ptv_id: ptv_mask,
        "Rectum": roi_mask,
        "Rectum_overlap_PTV": overlap,
        "hot_no_overlap": hot_no_overlap,
        "cold_no_overlap": cold_no_overlap,
        args.bladder_roi_name: bladder_mask,
    }

    print(f"\n=== {args.nombre} ===")
    print(f"{'estructura':>18} {'vox':>7} {'mean%':>7} {'median%':>8} {'min%':>7} {'max%':>7} {'D95%':>7} {'D98%':>7}")
    filas = {}
    for nombre, mask in estructuras.items():
        s = stats_estructura(dose_pct_rt, mask)
        filas[nombre] = s
        print(f"{nombre:>18} {s['vol_vox']:7d} {s['mean%']:7.1f} {s['median%']:8.1f} "
              f"{s['min%']:7.1f} {s['max%']:7.1f} {s['D95%']:7.1f} {s['D98%']:7.1f}")
    return filas


if __name__ == "__main__":
    main()
