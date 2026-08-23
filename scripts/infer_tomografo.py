"""Tarea 7 (Proyecto 1) — wrapper de inferencia end-to-end: RS DICOM del autocontour
(en el tomografo) -> {prob, zona verde/naranja/rojo, V_pred} por constraint.

Encadena scripts/extract_features_live.py (features desde DICOM) con los modelos
serializados en models/proyecto1/ (Tareas 2-4/6) usando EXACTAMENTE la misma logica de
combinacion por OAR que scripts/calibrate_p1.py / scripts/eval_p1.py:
  - Recto:  score  = max(P_fail_RV65, P_fail_RV55)
            margen = max(V_pred_RV65 - 15, V_pred_RV55 - 25)
  - Vejiga: score  = P_fail_BV65
            margen = V_pred_BV65 - 15
  - zona = "verde" si score < umbral_verde_OAR; si no, "rojo" si margen >= Delta_OAR,
    si no "naranja".

Uso:
    python scripts/infer_tomografo.py --carpeta <ruta con CT+RS DICOM del paciente>
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from extract_features_live import extraer_features, FEATURE_COLS_FINAL, cargar_config  # noqa: E402

MODELS_DIR = _REPO_ROOT / "models" / "proyecto1"


def cargar_manifest():
    with open(MODELS_DIR / "manifest.json", encoding="utf-8") as f:
        return json.load(f)


def cargar_thresholds():
    with open(MODELS_DIR / "thresholds.json", encoding="utf-8") as f:
        return json.load(f)


def predecir_paciente(feats: dict) -> dict:
    """feats: dict con las 7 features (ver extraer_features). Devuelve dict con
    resultado por constraint (RV65/RV55/BV65) y por OAR (recto/vejiga)."""
    thresholds = cargar_thresholds()
    x = np.array([[feats[c] for c in FEATURE_COLS_FINAL]])

    prob_fail, val_pred = {}, {}
    for tag in ("RV65", "RV55", "BV65"):
        clf = joblib.load(MODELS_DIR / f"clf_{tag}.joblib")
        reg = joblib.load(MODELS_DIR / f"reg_{tag}.joblib")
        x_clf = clf["scaler"].transform(x)
        prob_fail[tag] = float(clf["logreg"].predict_proba(x_clf)[0, 1])
        x_reg = reg["scaler"].transform(x) if reg["usa_scaler"] else x
        val_pred[tag] = float(reg["modelo"].predict(x_reg)[0])

    umbral_clinico = thresholds["umbrales_clinicos"]
    margen = {tag: val_pred[tag] - umbral_clinico[tag] for tag in val_pred}

    def zona_oar(score, margen_oar, cfg_oar):
        if score < cfg_oar["umbral_verde_proba"]:
            return "verde"
        return "rojo" if margen_oar >= cfg_oar["delta_severidad_pp"] else "naranja"

    score_recto = max(prob_fail["RV65"], prob_fail["RV55"])
    margen_recto = max(margen["RV65"], margen["RV55"])
    score_vejiga = prob_fail["BV65"]
    margen_vejiga = margen["BV65"]

    return {
        "features": feats,
        "por_constraint": {
            tag: {"prob_falla": prob_fail[tag], "V_pred": val_pred[tag],
                  "umbral_clinico": umbral_clinico[tag], "margen": margen[tag]}
            for tag in ("RV65", "RV55", "BV65")
        },
        "por_oar": {
            "recto": {"score": score_recto, "margen": margen_recto,
                      "zona": zona_oar(score_recto, margen_recto, thresholds["recto"])},
            "vejiga": {"score": score_vejiga, "margen": margen_vejiga,
                       "zona": zona_oar(score_vejiga, margen_vejiga, thresholds["vejiga"])},
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", type=Path, required=True, help="Carpeta con CT+RS DICOM del paciente")
    args = ap.parse_args()

    config = cargar_config()
    feats = extraer_features(args.carpeta, config)
    resultado = predecir_paciente(feats)

    print("Features:")
    for k, v in feats.items():
        print(f"  {k:24s} {v:.4f}")
    print("\nPor constraint:")
    for tag, r in resultado["por_constraint"].items():
        print(f"  {tag}: prob_falla={r['prob_falla']:.3f}  V_pred={r['V_pred']:.2f}  "
              f"umbral_clinico={r['umbral_clinico']}  margen={r['margen']:+.2f}pp")
    print("\nSemaforo:")
    for oar, r in resultado["por_oar"].items():
        print(f"  {oar.upper()}: zona={r['zona'].upper()}  score={r['score']:.3f}  margen={r['margen']:+.2f}pp")


if __name__ == "__main__":
    main()
