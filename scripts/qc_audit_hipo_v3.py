"""QC rapido (Paso 2) sobre una muestra de NPZ del dataset hipo nuevo.

Verifica por paciente: rango de dosis (~[0,110]% prescripcion), shape consistente
entre canales, PSDM en [-1,1], mascaras binarias sin valores raros.

Uso:
    python scripts/qc_audit_hipo_v3.py PT_xxx PT_yyy ...
"""

import json
import sys
from pathlib import Path

import numpy as np

PROCESSED_DIR = Path(r"C:\Pablo\ProstateDoseProject\processed_hipo")


def audit_one(anonid: str):
    npz = np.load(PROCESSED_DIR / f"{anonid}.npz", allow_pickle=True)
    meta = json.loads(npz["meta"][0])

    print(f"\n=== {anonid} (Status={meta.get('Status')}, PTV={meta.get('PTV_name_usado')}) ===")

    shapes = {k: npz[k].shape for k in ["ct", "dose", "ptv_mask", "body_mask", "rectum_mask",
                                          "bladder_mask", "psdm_ptv", "psdm_rectum", "psdm_bladder"]}
    shapes_set = set(shapes.values())
    ok_shape = len(shapes_set) == 1
    print(f"  shapes: {shapes_set} {'OK' if ok_shape else 'MISMATCH ' + str(shapes)}")

    dose = npz["dose"]
    dmin, dmax = float(dose.min()), float(dose.max())
    ok_dose = -1.0 <= dmin and dmax <= 115.0
    print(f"  dose range: [{dmin:.2f}, {dmax:.2f}] % Rx  {'OK' if ok_dose else 'FUERA DE RANGO'}")

    for psdm_name in ["psdm_ptv", "psdm_rectum", "psdm_bladder"]:
        p = npz[psdm_name]
        pmin, pmax = float(p.min()), float(p.max())
        ok_psdm = -1.05 <= pmin and pmax <= 1.05
        print(f"  {psdm_name} range: [{pmin:.3f}, {pmax:.3f}]  {'OK' if ok_psdm else 'FUERA DE [-1,1]'}")

    for mask_name in ["ptv_mask", "body_mask", "rectum_mask", "bladder_mask"]:
        m = npz[mask_name]
        vals = set(np.unique(m).tolist())
        ok_mask = vals <= {0, 1}
        n_on = int(m.sum())
        print(f"  {mask_name}: valores={vals} n_voxels_on={n_on}  {'OK' if ok_mask else 'VALORES NO BINARIOS'}")

    d95_check = float(np.percentile(dose[npz["ptv_mask"] > 0], 5))
    print(f"  D95(PTV) recalculado sobre grid downsampleado: {d95_check:.2f}% (deberia ser ~100, "
          f"s_D95 meta={meta.get('s_D95')})")

    print(f"  n_slices={meta.get('n_slices')}  vol_ptv_cc={meta.get('vol_ptv_cc')}  "
          f"D95_delivered_Gy={meta.get('D95_delivered_Gy')}")


def main():
    ids = sys.argv[1:]
    if not ids:
        print("Uso: python scripts/qc_audit_hipo_v3.py PT_xxx PT_yyy ...")
        sys.exit(1)
    for anonid in ids:
        audit_one(anonid)


if __name__ == "__main__":
    main()
