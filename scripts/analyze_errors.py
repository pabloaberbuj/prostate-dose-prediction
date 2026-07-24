"""
Análisis de errores estratificado.

Cruza las métricas de test_metrics.csv con metricas_planes.csv (ESAPI)
y analiza cómo varía el error según:
  - Volumen del PTV (terciles)
  - Solapamiento PTV-Recto (terciles)
  - Volumen de vejiga (terciles, predictor más variable según EDA)
  - Casos subóptimos (FactorNorm > 1.05)

Genera:
  - figures/errors_vs_<variable>.png — boxplot del error por tercil
  - figures/dvh_score_by_subgroup.png — comparación de DVH score por subgrupo
  - error_analysis.json — tabla resumen
  - worst_cases.csv — los 10 peores casos (para inspección visual)
  - best_cases.csv  — los 10 mejores casos

Uso:
    python scripts/analyze_errors.py \
        --test-metrics  results/exp001_test/test_metrics.csv \
        --plan-metrics  data/metricas_planes.csv \
        --output-dir    results/exp001_test/analysis
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cargar_y_cruzar(test_csv: Path, plan_csv: Path, overlap_real_csv: Path = None) -> pd.DataFrame:
    df_test = pd.read_csv(test_csv, sep=";")
    df_plan = pd.read_csv(plan_csv, sep=";")
    # Normalizar clave de merge a 'anonid' en ambos DataFrames
    if "AnonID" in df_test.columns:
        df_test = df_test.rename(columns={"AnonID": "anonid"})
    if "AnonID" in df_plan.columns:
        df_plan = df_plan.rename(columns={"AnonID": "anonid"})
    df = df_test.merge(df_plan, on="anonid", how="left", suffixes=("", "_plan"))
    # Si un paciente tiene más de un plan en metricas_planes, conservar el primero
    df = df.drop_duplicates(subset="anonid", keep="first")
    df = df.rename(columns={"anonid": "AnonID"})

    # Derivadas
    # pct_solap_rectum bugueado (Math.Min en extract_dicom_csharp, ver "Correccion overlap.md")
    df["pct_solap_rectum"] = df["Solap_PTV_Rectum_cc"] / df["VolPTV_cc"] * 100
    df["es_suboptimo"]     = df["FactorNorm_D95"] > 1.05

    if overlap_real_csv is not None:
        df_overlap = pd.read_csv(overlap_real_csv, sep=";")
        df_overlap = df_overlap[["AnonID", "pct_solap_real", "tercil_nuevo"]]
        df = df.merge(df_overlap, on="AnonID", how="left")
        faltantes = df[df["pct_solap_real"].isna()]["AnonID"].tolist()
        if faltantes:
            print(f"⚠️  {len(faltantes)} pacientes de test sin overlap real (fuera de los 385 auditados): {faltantes}")
        # Overlap real reemplaza al bugueado para el análisis estratificado
        df["pct_solap_rectum"] = df["pct_solap_real"]
        df["tercil_solap_rect_precalculado"] = df["tercil_nuevo"]

    return df


def boxplot_por_grupo(df: pd.DataFrame, variable: str, metricas: list,
                      output_path: Path, etiqueta_grupo: str):
    """Genera boxplots de las métricas para cada tercil/categoría de la variable."""
    fig, axes = plt.subplots(1, len(metricas), figsize=(5 * len(metricas), 4),
                              squeeze=False)
    for i, metrica in enumerate(metricas):
        ax = axes[0, i]
        grupos = df.groupby(variable, observed=True)[metrica]
        labels  = []
        data    = []
        for g, vals in grupos:
            vals = vals.dropna()
            if len(vals) > 0:
                labels.append(str(g))
                data.append(vals.values)
        if data:
            bp = ax.boxplot(data, patch_artist=True, widths=0.6)
            ax.set_xticklabels(labels)
            for patch, c in zip(bp["boxes"], ["#9FE1CB", "#B5D4F4", "#F4C0D1"]):
                patch.set_facecolor(c)
                patch.set_edgecolor("#444")
        ax.set_title(metrica, fontsize=10)
        ax.set_xlabel(etiqueta_grupo, fontsize=9)
        ax.set_ylabel(metrica, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def analisis_subgrupo(df: pd.DataFrame, variable_grupo: str, metricas: list) -> dict:
    """Calcula mean/std/n por grupo."""
    resultado = {}
    for g, sub in df.groupby(variable_grupo, observed=True):
        resultado[str(g)] = {
            "n": int(len(sub)),
            "metrics": {
                m: {
                    "mean":   float(sub[m].mean()),
                    "std":    float(sub[m].std()),
                    "median": float(sub[m].median()),
                }
                for m in metricas if m in sub.columns
            },
        }
    return resultado


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-metrics",  required=True)
    parser.add_argument("--plan-metrics",  required=True)
    parser.add_argument("--output-dir",    required=True)
    parser.add_argument("--overlap-real-csv", default=None,
                        help="CSV de scripts/compute_overlap_real.py (overlap por intersección real "
                             "de máscaras, con columnas AnonID, pct_solap_real, tercil_nuevo). "
                             "Si se pasa, reemplaza el tercil de solapamiento PTV-Recto bugueado "
                             "(Math.Min) por el real. Ver 'Correccion overlap.md'.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    overlap_real_csv = Path(args.overlap_real_csv) if args.overlap_real_csv else None
    df = cargar_y_cruzar(Path(args.test_metrics), Path(args.plan_metrics), overlap_real_csv)
    print(f"Pacientes cruzados: {len(df)}")

    # ── Estratificar por terciles
    df["tercil_vol_ptv"]    = pd.qcut(df["VolPTV_cc"],          3, labels=["bajo", "medio", "alto"])
    df["tercil_vol_blad"]   = pd.qcut(df["VolBladder_cc"],      3, labels=["bajo", "medio", "alto"])
    if overlap_real_csv is not None:
        # Terciles precalculados sobre los 385 pacientes normo (no solo el test set),
        # para ser consistentes con la estratificación real de "Correccion overlap.md".
        df["tercil_solap_rect"] = df["tercil_solap_rect_precalculado"]
    else:
        df["tercil_solap_rect"] = pd.qcut(df["pct_solap_rectum"], 3, labels=["bajo", "medio", "alto"])

    metricas_imagen = ["mae_body", "mae_ptv", "mae_rectum", "mae_bladder",
                       "dose_score_openkbp", "dvh_score_openkbp"]

    # ── Boxplots por subgrupo
    print("Generando figuras...")
    boxplot_por_grupo(df, "tercil_vol_ptv",    metricas_imagen,
                      figures_dir / "errors_vs_vol_ptv.png",    "Tercil VolPTV")
    boxplot_por_grupo(df, "tercil_vol_blad",   metricas_imagen,
                      figures_dir / "errors_vs_vol_bladder.png", "Tercil VolBladder")
    boxplot_por_grupo(df, "tercil_solap_rect", metricas_imagen,
                      figures_dir / "errors_vs_solap_rectum.png", "Tercil solapamiento PTV-Recto")

    # Subóptimos
    if df["es_suboptimo"].sum() > 0:
        boxplot_por_grupo(df, "es_suboptimo", metricas_imagen,
                          figures_dir / "errors_suboptimal.png",
                          "Plan subóptimo (FactorNorm>1.05)")

    # ── Resumen JSON
    resumen = {
        "n_test":             int(len(df)),
        "overlap_solap_rectum_fuente": (
            "interseccion_real_mascaras_npz" if overlap_real_csv is not None
            else "bug_math_min_extract_dicom_csharp"
        ),
        "global": {
            m: {"mean": float(df[m].mean()), "std": float(df[m].std())}
            for m in metricas_imagen
        },
        "by_vol_ptv":    analisis_subgrupo(df, "tercil_vol_ptv",    metricas_imagen),
        "by_vol_bladder": analisis_subgrupo(df, "tercil_vol_blad",   metricas_imagen),
        "by_solap_rectum": analisis_subgrupo(df, "tercil_solap_rect", metricas_imagen),
        "by_suboptimal":   analisis_subgrupo(df, "es_suboptimo",     metricas_imagen),
    }

    with open(output_dir / "error_analysis.json", "w") as f:
        json.dump(resumen, f, indent=2)
    print(f"Resumen guardado en {output_dir / 'error_analysis.json'}")

    # ── Peores y mejores casos
    df_ordenado = df.sort_values("dvh_score_openkbp", ascending=False)
    cols_clave = ["AnonID", "dvh_score_openkbp", "dose_score_openkbp",
                  "mae_body", "mae_rectum", "mae_bladder",
                  "VolPTV_cc", "VolBladder_cc", "pct_solap_rectum"]
    df_ordenado[cols_clave].head(10).to_csv(output_dir / "worst_cases.csv",
                                             index=False, sep=";")
    df_ordenado[cols_clave].tail(10).to_csv(output_dir / "best_cases.csv",
                                             index=False, sep=";")

    print("\n=== TOP 5 PEORES CASOS (mayor DVH score) ===")
    print(df_ordenado[cols_clave].head(5).to_string(index=False))
    print("\n=== TOP 5 MEJORES CASOS (menor DVH score) ===")
    print(df_ordenado[cols_clave].tail(5).to_string(index=False))

    # ── Análisis rápido en consola
    print("\n=== ERROR POR TERCIL DE VolBladder ===")
    for grupo, sub in df.groupby("tercil_vol_blad", observed=True):
        print(f"  {grupo} (n={len(sub):2d}): "
              f"mae_body={sub['mae_body'].mean():.2f} ± {sub['mae_body'].std():.2f}, "
              f"dvh_score={sub['dvh_score_openkbp'].mean():.2f}")

    print("\n=== ERROR POR TERCIL DE solapamiento PTV-Recto ===")
    for grupo, sub in df.groupby("tercil_solap_rect", observed=True):
        print(f"  {grupo} (n={len(sub):2d}): "
              f"mae_rectum={sub['mae_rectum'].mean():.2f} ± {sub['mae_rectum'].std():.2f}")

    if df["es_suboptimo"].sum() > 0:
        print(f"\n=== Subóptimos (n={df['es_suboptimo'].sum()}) ===")
        sub = df[df["es_suboptimo"]]
        nor = df[~df["es_suboptimo"]]
        print(f"  mae_body subóptimos: {sub['mae_body'].mean():.2f} | "
              f"normales: {nor['mae_body'].mean():.2f}")


if __name__ == "__main__":
    main()
