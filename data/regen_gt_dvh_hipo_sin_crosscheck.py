"""
Regenera data/gt_dvh_hipo_256.csv SIN el cross-check contra el CSV de ESAPI
(metricas_planes_hipofx_D95norm.csv, cuya carpeta fuente "dicoms hipofx" ya no
existe en disco — limpieza de espacio, 2026-08, ver CLAUDE_CODE_CONTEXT.md).

Reusa calcular_dvh_paciente() de compute_gt_dvh_hipo.py (mismas columnas
*_gt256, incluidas D99/D01cc/D30Gy/D15Gy/D10Gy/V50Gy/V40Gy agregadas el
2026-08-18) para todo el universo train+val+test-excluidos de splits_hipo_v1.json.
Usar esto en vez de compute_gt_dvh_hipo.py cuando no se necesite (o no se pueda)
correr la Parte C (cross-check de flags).

Uso:
    python data/regen_gt_dvh_hipo_sin_crosscheck.py \
        --processed-dir "c:/Pablo/ProstateDoseProject/processed_hipo" \
        --splits        data/splits/splits_hipo_v1.json \
        --out-dvh       data/gt_dvh_hipo_256.csv
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from compute_gt_dvh_hipo import calcular_dvh_paciente

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out-dvh", required=True)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    with open(args.splits) as f:
        splits = json.load(f)
    excluidos = set(splits.get("excluidos", []))
    anonids = [a for a in splits["train"] + splits["val"] + splits["test"] if a not in excluidos]

    filas = []
    faltantes = []
    for anonid in anonids:
        npz_path = processed_dir / f"{anonid}.npz"
        if not npz_path.exists():
            faltantes.append(anonid)
            continue
        filas.append(calcular_dvh_paciente(npz_path))

    if faltantes:
        log.warning(f"{len(faltantes)} pacientes sin NPZ, se omiten: {faltantes}")

    df_gt = pd.DataFrame(filas)
    out_path = Path(args.out_dvh)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_gt.to_csv(out_path, index=False)
    log.info(f"GT DVH regenerado en {out_path} ({len(df_gt)} pacientes, {len(df_gt.columns)} columnas)")


if __name__ == "__main__":
    main()
