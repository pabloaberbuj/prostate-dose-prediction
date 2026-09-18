"""Exporta los coeficientes de la regresion logistica (modelo DESPLEGADO de
clasificacion, Tarea 2) por constraint, desde los joblib serializados en
models/proyecto1/clf_<tag>.joblib (ver manifest.json). Los coeficientes ya estan en
escala estandarizada (el LogReg se fitea sobre features pasadas por StandardScaler) —
son directamente comparables entre si dentro de un mismo constraint.

Imprime tabla y guarda results/proyecto1_v3_cv/feature_importance_logreg.json.

Uso:
    python scripts/export_feature_importance_p1.py
"""

import json
from pathlib import Path

import joblib

_REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = _REPO_ROOT / "models" / "proyecto1"
OUT_PATH = _REPO_ROOT / "results" / "proyecto1_v3_cv" / "feature_importance_logreg.json"

FEATURE_COLS = [
    "VolRectum_cc", "VolBladder_cc", "VolPTV_cc",
    "Solap_PTV_Rectum_cc", "Solap_PTV_Bladder_cc",
    "overlap_rel_recto", "overlap_rel_vejiga",
]
CONSTRAINTS = ["RV65", "RV55", "BV65"]


def extraer_importancia(tag: str) -> dict:
    bundle = joblib.load(MODELS_DIR / f"clf_{tag}.joblib")
    assert bundle["feature_cols"] == FEATURE_COLS, \
        f"clf_{tag}.joblib: orden de features distinto al esperado ({bundle['feature_cols']})"

    logreg = bundle["logreg"]
    coefs = logreg.coef_[0]
    filas = sorted(
        (
            {"feature": feat, "coef_estandarizado": float(c), "abs_coef": float(abs(c))}
            for feat, c in zip(FEATURE_COLS, coefs)
        ),
        key=lambda r: -r["abs_coef"],
    )
    for rank, fila in enumerate(filas, start=1):
        fila["rank_importancia"] = rank

    return {
        "constraint": tag,
        "intercepto": float(logreg.intercept_[0]),
        "coeficientes": filas,
        "orden_importancia_absoluta": [f["feature"] for f in filas],
    }


def imprimir_tabla(resultados: dict):
    for tag, r in resultados.items():
        print(f"\n=== {tag} (intercepto={r['intercepto']:+.3f}) ===")
        print(f"{'rank':>4s}  {'feature':24s} {'coef (std)':>12s}  {'|coef|':>8s}")
        for f in r["coeficientes"]:
            print(f"{f['rank_importancia']:>4d}  {f['feature']:24s} {f['coef_estandarizado']:>+12.4f}  {f['abs_coef']:>8.4f}")


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    resultados = {tag: extraer_importancia(tag) for tag in CONSTRAINTS}
    imprimir_tabla(resultados)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "fuente": "models/proyecto1/clf_<tag>.joblib (logreg desplegado, Tarea 2)",
            "nota": "coeficientes en escala estandarizada (StandardScaler fiteado en train) -> comparables entre features dentro de un mismo constraint",
            "constraints": resultados,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado: {OUT_PATH}")


if __name__ == "__main__":
    main()
