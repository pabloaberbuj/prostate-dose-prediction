"""error_analysis.json para exp_hipo_003_finetune_v3 (Paso 6 del handoff).

Adaptacion de analyze_errors.py (normo) a las columnas realmente disponibles en
per_patient_metrics.csv del pipeline hipo (no hay VolPTV_cc/VolBladder_cc ahi --
se cruzan desde metricas_planes_hipofx_D95norm.csv por AnonID).

Uso:
    python scripts/error_analysis_hipo_v3.py \
        --results-dir results/exp_hipo_003_finetune_v3_test_hipo_v3 \
        --metricas-csv data/metricas_planes_hipofx_D95norm.csv \
        --splits data/splits/splits_hipo_v3_unet.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def analisis_subgrupo(df, variable_grupo, metricas):
    out = {}
    for g, sub in df.groupby(variable_grupo, observed=True):
        out[str(g)] = {
            "n": int(len(sub)),
            "metrics": {
                m: {"mean": float(sub[m].mean()), "std": float(sub[m].std()), "median": float(sub[m].median())}
                for m in metricas if m in sub.columns
            },
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--metricas-csv", required=True)
    p.add_argument("--splits", required=True)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    df = pd.read_csv(results_dir / "per_patient_metrics.csv")

    meta = json.load(open(args.splits))["metadata"]
    t1, t2 = meta["terciles_overlap_cc"]

    csv = pd.read_csv(args.metricas_csv, sep=";", encoding="utf-8-sig")
    csv = csv.drop_duplicates(subset="AnonID", keep="first")[["AnonID", "Solap_PTV_Rectum_cc", "VolPTV_cc"]]
    df = df.merge(csv, on="AnonID", how="left")

    def overlap_bin(v):
        if pd.isna(v):
            return "sin_dato"
        if v <= t1:
            return "overlap_bajo"
        if v <= t2:
            return "overlap_medio"
        return "overlap_alto"

    df["tercil_overlap"] = df["Solap_PTV_Rectum_cc"].apply(overlap_bin)

    metricas_imagen = ["mae_body", "mae_ptv", "mae_rectum", "mae_bladder", "dose_score_openkbp"]

    resumen = {
        "n_test": int(len(df)),
        "overlap_fuente": "Solap_PTV_Rectum_cc (confirmado por Pablo como overlap real, bug C# ya corregido)",
        "terciles_overlap_cc": [t1, t2],
        "global": {m: {"mean": float(df[m].mean()), "std": float(df[m].std())} for m in metricas_imagen},
        "by_overlap_tercil": analisis_subgrupo(df, "tercil_overlap", metricas_imagen),
        "by_status": analisis_subgrupo(df, "Status", metricas_imagen),
        "by_any_op_fail_real": analisis_subgrupo(df, "any_op_fail_real", metricas_imagen),
    }

    with open(results_dir / "error_analysis.json", "w") as f:
        json.dump(resumen, f, indent=2)
    print(f"Guardado: {results_dir / 'error_analysis.json'}")

    df_ordenado = df.sort_values("mae_body", ascending=False)
    cols = ["AnonID", "Status", "mae_body", "mae_ptv", "mae_rectum", "mae_bladder",
            "dose_score_openkbp", "any_op_fail_real", "Solap_PTV_Rectum_cc", "VolPTV_cc"]
    df_ordenado[cols].head(10).to_csv(results_dir / "worst_cases.csv", index=False, sep=";")
    df_ordenado[cols].tail(10).to_csv(results_dir / "best_cases.csv", index=False, sep=";")
    print(f"Guardado: {results_dir / 'worst_cases.csv'} / best_cases.csv")

    print("\n=== TOP 5 PEORES (mayor mae_body) ===")
    print(df_ordenado[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
