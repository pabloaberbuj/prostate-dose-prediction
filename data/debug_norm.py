"""
Debug puntual: verifica el factor_norm guardado en el NPZ
y recalcula manualmente para encontrar el bug.

Uso:
    python data/debug_norm.py --npz C:/ruta/processed/PT_0a3eac085fccff90.npz
"""
import argparse
import json
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    meta = json.loads(str(data['meta'][0]))

    dose    = data['dose']
    ptv     = data['ptv_mask']
    dosis_ptv = dose[ptv > 0]

    print(f"=== Meta guardado en NPZ ===")
    print(f"  factor_norm:  {meta.get('factor_norm')}")
    print(f"  vol_ptv_cc:   {meta.get('vol_ptv_cc')}")
    print(f"  n_slices:     {meta.get('n_slices')}")

    print(f"\n=== Dosis en NPZ (ya normalizada) ===")
    print(f"  dose.min():   {dose.min():.4f}")
    print(f"  dose.max():   {dose.max():.4f}")
    print(f"  dose.mean():  {dose.mean():.4f}")
    print(f"  PTV voxels:   {ptv.sum()}")

    if ptv.sum() > 0:
        print(f"\n=== DVH PTV desde NPZ ===")
        print(f"  D95 (percentil 5):  {np.percentile(dosis_ptv, 5):.4f} %")
        print(f"  D98 (percentil 2):  {np.percentile(dosis_ptv, 2):.4f} %")
        print(f"  Dmean:              {dosis_ptv.mean():.4f} %")
        print(f"  Dmax:               {dosis_ptv.max():.4f} %")

    print(f"\n=== Diagnóstico ===")
    factor = meta.get('factor_norm', None)
    if factor is not None:
        print(f"  factor_norm guardado = {factor:.6f}")
        if factor > 10:
            print(f"  ⚠ factor > 10 → probable bug: se guardó prescripcion_cgy/d95_cgy*100")
            print(f"  Factor corregido sería: {factor/100:.6f}")
        elif 0.9 < factor < 1.1:
            print(f"  ✓ factor en rango correcto (~1.0)")
        else:
            print(f"  ⚠ factor fuera de rango esperado")

    # Recalcular manualmente como debería hacerlo preprocess.py
    if ptv.sum() > 0:
        print(f"\n=== Recálculo manual ===")
        # Si la dosis ya está en % y D95 debería ser 100%
        d95_pct = float(np.percentile(dosis_ptv, 5))
        print(f"  D95 de la dosis guardada: {d95_pct:.4f}%")
        print(f"  ¿Es ~100%? {'SI' if 98 < d95_pct < 102 else 'NO → la normalización no funcionó correctamente'}")

        # Si el factor guardado es ~99, la dosis en el NPZ en realidad está en cGy/100 no en %
        if factor and factor > 10:
            print(f"\n  Hipótesis: la dosis se guardó sin multiplicar por 100")
            print(f"  dose.max() actual = {dose.max():.2f}")
            print(f"  Si estuviera en % debería ser ~105-115")
            print(f"  Si estuviera en cGy/prescripción debería ser ~1.05-1.15")


if __name__ == "__main__":
    main()
