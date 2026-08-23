"""Tarea 4 (Proyecto 1): calibracion del semaforo de 3 zonas (verde/naranja/rojo),
separado por Recto y por Vejiga, congelado en VAL.

Diseno hibrido:
  - Frontera VERDE / no-verde (de la CLASIFICACION, critica de seguridad -> evita FN):
    umbral en val donde la sensibilidad del "no-verde" alcanza el objetivo.
      Recto:  sens ~= 0.85 (score_recto  = max(P_fail_RV65, P_fail_RV55), combinacion
              pessimista — si CUALQUIERA de los 2 constraints de recto predice falla,
              el paciente es no-verde; label_recto_fail = fail_RV65 OR fail_RV55, el
              mismo criterio ya usado para el estrato 2D del split).
      Vejiga: sens ~= 0.90 (score_vejiga = P_fail_BV65, unico constraint de vejiga).
  - Split NARANJA / ROJO (solo dentro de no-verde, de la REGRESION de severidad):
    margen = valor_predicho - umbral_clinico (recto: max(margen_RV65, margen_RV55) del
    mismo paciente, mismo criterio pesimista que arriba; vejiga: margen_BV65 directo).
    Delta calibrado en val como la MEDIANA del margen entre los pacientes no-verde de
    val (parte a la mitad la poblacion de riesgo real observada en val — "ajustar Delta
    mirando la distribucion", tal como pide la tarea). margen >= Delta -> ROJO;
    margen < Delta -> NARANJA.
  - Todo se CONGELA (umbral verde + Delta) y se guarda en
    models/proyecto1/thresholds.json. NO se recalibra contra test (Tarea 5).

Uso:
    python scripts/calibrate_p1.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from evaluate_hipo import punto_operacion_sens_objetivo, confusion_binaria, MIN_VAL_POSITIVOS  # noqa: E402

DATASET_CSV = _REPO_ROOT / "data" / "dataset_p1.csv"
MODELS_DIR = _REPO_ROOT / "models" / "proyecto1"
OUT_THRESHOLDS = MODELS_DIR / "thresholds.json"

SENS_OBJETIVO = {"recto": 0.85, "vejiga": 0.90}
FALLBACK_THRESHOLD_PROBA = 0.5


def cargar_modelo(kind, tag):
    return joblib.load(MODELS_DIR / f"{kind}_{tag}.joblib")


def predecir_prob_fail(clf_bundle, X_raw):
    X = clf_bundle["scaler"].transform(X_raw)
    return clf_bundle["logreg"].predict_proba(X)[:, 1]


def predecir_valor(reg_bundle, X_raw):
    X = reg_bundle["scaler"].transform(X_raw) if reg_bundle["usa_scaler"] else X_raw
    return reg_bundle["modelo"].predict(X)


def calibrar_frontera_verde(y_fail_val, score_val, sens_objetivo, oar_label):
    n_pos = int(y_fail_val.sum())
    es_fallback = n_pos < MIN_VAL_POSITIVOS
    if es_fallback:
        print(f"  [{oar_label}] WARNING: solo {n_pos} positivos en val (< {MIN_VAL_POSITIVOS}) "
              f"-> umbral fallback nominal ({FALLBACK_THRESHOLD_PROBA})")
        umbral = FALLBACK_THRESHOLD_PROBA
    else:
        umbral = punto_operacion_sens_objetivo(y_fail_val, score_val, sens_objetivo)
    y_pred = (score_val >= umbral).astype(int)
    conf = confusion_binaria(y_fail_val, y_pred)
    print(f"  [{oar_label}] umbral_verde(proba)={umbral:.4f}  "
          f"sens={conf['sensibilidad']:.3f} esp={conf['especificidad']:.3f}  "
          f"({conf['TP']}TP {conf['FP']}FP {conf['TN']}TN {conf['FN']}FN)  "
          f"n_pos_val={n_pos}{' [FALLBACK]' if es_fallback else ''}")
    return umbral, es_fallback, conf


def calibrar_delta_severidad(margen_no_verde, oar_label):
    if len(margen_no_verde) == 0:
        print(f"  [{oar_label}] sin pacientes no-verde en val -> Delta = 0.0 (sin datos para calibrar)")
        return 0.0, {}
    stats = {
        "n": int(len(margen_no_verde)),
        "min": float(np.min(margen_no_verde)),
        "median": float(np.median(margen_no_verde)),
        "mean": float(np.mean(margen_no_verde)),
        "std": float(np.std(margen_no_verde)),
        "max": float(np.max(margen_no_verde)),
    }
    delta = stats["median"]
    print(f"  [{oar_label}] margen no-verde en val: n={stats['n']} "
          f"min={stats['min']:.2f} mediana={stats['median']:.2f} media={stats['mean']:.2f} "
          f"std={stats['std']:.2f} max={stats['max']:.2f}  -> Delta={delta:.2f}pp")
    return delta, stats


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_CSV).set_index("AnonID")
    ids_val = df.index[df["split"] == "val"]

    feature_cols_clf = cargar_modelo("clf", "RV65")["feature_cols"]
    X_val = df.loc[ids_val, feature_cols_clf].to_numpy()

    clf = {tag: cargar_modelo("clf", tag) for tag in ("RV65", "RV55", "BV65")}
    reg = {tag: cargar_modelo("reg", tag) for tag in ("RV65", "RV55", "BV65")}

    prob_fail_val = {tag: predecir_prob_fail(clf[tag], X_val) for tag in clf}
    val_pred = {tag: predecir_valor(reg[tag], X_val) for tag in reg}
    umbral_clinico = {tag: reg[tag]["threshold"] for tag in reg}

    fail_recto_val = ((df.loc[ids_val, "fail_RV65"] == 1) | (df.loc[ids_val, "fail_RV55"] == 1)).to_numpy().astype(int)
    fail_vejiga_val = df.loc[ids_val, "fail_BV65"].to_numpy().astype(int)

    score_recto_val = np.maximum(prob_fail_val["RV65"], prob_fail_val["RV55"])
    score_vejiga_val = prob_fail_val["BV65"]

    margen_RV65_val = val_pred["RV65"] - umbral_clinico["RV65"]
    margen_RV55_val = val_pred["RV55"] - umbral_clinico["RV55"]
    margen_recto_val = np.maximum(margen_RV65_val, margen_RV55_val)
    margen_vejiga_val = val_pred["BV65"] - umbral_clinico["BV65"]

    print("=" * 78)
    print("PASO 1 — Frontera VERDE / no-verde (val, congelado)")
    print("=" * 78)
    umbral_verde_recto, fb_recto, conf_recto = calibrar_frontera_verde(
        fail_recto_val, score_recto_val, SENS_OBJETIVO["recto"], "RECTO")
    umbral_verde_vejiga, fb_vejiga, conf_vejiga = calibrar_frontera_verde(
        fail_vejiga_val, score_vejiga_val, SENS_OBJETIVO["vejiga"], "VEJIGA")

    print("\n" + "=" * 78)
    print("PASO 2 — Delta NARANJA/ROJO dentro de no-verde (val, congelado)")
    print("=" * 78)
    no_verde_recto = score_recto_val >= umbral_verde_recto
    no_verde_vejiga = score_vejiga_val >= umbral_verde_vejiga
    delta_recto, stats_recto = calibrar_delta_severidad(margen_recto_val[no_verde_recto], "RECTO")
    delta_vejiga, stats_vejiga = calibrar_delta_severidad(margen_vejiga_val[no_verde_vejiga], "VEJIGA")

    def zonas(score, umbral_verde, margen, delta):
        zona = np.where(score < umbral_verde, "verde", np.where(margen >= delta, "rojo", "naranja"))
        return zona

    zona_recto_val = zonas(score_recto_val, umbral_verde_recto, margen_recto_val, delta_recto)
    zona_vejiga_val = zonas(score_vejiga_val, umbral_verde_vejiga, margen_vejiga_val, delta_vejiga)

    print("\n" + "=" * 78)
    print("PASO 3 — Distribucion de zonas en VAL (n=%d)" % len(ids_val))
    print("=" * 78)
    for label, zona in (("RECTO", zona_recto_val), ("VEJIGA", zona_vejiga_val)):
        vals, counts = np.unique(zona, return_counts=True)
        dist = dict(zip(vals.tolist(), counts.tolist()))
        print(f"  {label}: {dist}")

    thresholds = {
        "recto": {
            "umbral_verde_proba": float(umbral_verde_recto),
            "umbral_verde_es_fallback": fb_recto,
            "sens_objetivo": SENS_OBJETIVO["recto"],
            "sens_esp_val": {"sensibilidad": conf_recto["sensibilidad"], "especificidad": conf_recto["especificidad"]},
            "delta_severidad_pp": float(delta_recto),
            "distribucion_margen_no_verde_val": stats_recto,
            "combinacion": "max(P_fail_RV65, P_fail_RV55) para score; max(margen_RV65, margen_RV55) para severidad",
        },
        "vejiga": {
            "umbral_verde_proba": float(umbral_verde_vejiga),
            "umbral_verde_es_fallback": fb_vejiga,
            "sens_objetivo": SENS_OBJETIVO["vejiga"],
            "sens_esp_val": {"sensibilidad": conf_vejiga["sensibilidad"], "especificidad": conf_vejiga["especificidad"]},
            "delta_severidad_pp": float(delta_vejiga),
            "distribucion_margen_no_verde_val": stats_vejiga,
            "combinacion": "P_fail_BV65 directo (unico constraint de vejiga)",
        },
        "umbrales_clinicos": umbral_clinico,
        "n_val": len(ids_val),
    }
    with open(OUT_THRESHOLDS, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado: {OUT_THRESHOLDS}")


if __name__ == "__main__":
    main()
