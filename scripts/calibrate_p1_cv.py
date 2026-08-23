"""SPEC_calibracion_cv_p1.md — calibracion del punto de operacion por CV estratificado
+ IC bootstrap del umbral, con dos variantes de la frontera verde (A=probabilidad del
clasificador, B=margen de la regresion de severidad). Deja `calibrate_p1.py` (v3) y
`models/proyecto1/thresholds.json` intactos como referencia — este script escribe
`models/proyecto1/thresholds_cv.json` aparte.

Motivacion (ver SPEC_calibracion_cv_p1.md): en v3, el umbral verde de Recto se calibro
sobre val (n_pos=9) y la sensibilidad cayo de 0.89 (val) a 0.50 (test) — no un modelo
malo (AUC test=0.921), sino un umbral con varianza enorme por N chico. Esta version:

1. Pool de calibracion = poblacion NATURAL (Approved+Rejected) de train+val agrupados.
   Los UnApproved se EXCLUYEN del calculo del umbral (su prevalencia de falla 60-70% lo
   sesgaria) — pero siguen en train para fitear los modelos DESPLEGADOS (sin cambios
   respecto a las Tareas 2-3; ese fit no se toca aca).
2. Umbral = MEDIANA de 5 folds StratifiedKFold (no un val fijo) — refitea scaler+modelo
   en cada fold SOLO para estimar el umbral, nunca para el modelo final desplegado.
3. Dos variantes de la frontera verde, ambas calibradas y reportadas (no se elige
   ganadora aca — la decision de cual usar queda para Pablo/Claude.ai con el IC en
   mano):
     A = umbral sobre P_fail del clasificador (igual que v3).
     B = umbral sobre el margen de la regresion (valor_predicho - umbral_clinico).
4. Objetivo de sensibilidad de RECTO sube a 0.95 (antes 0.85 en v3) — eleccion
   deliberada por costo asimetrico de FN de recto (Cambio 4 de la spec), NO un ajuste
   contra test. Vejiga se mantiene en 0.90.
5. Delta (split naranja/rojo dentro de no-verde) se recalibra sobre el mismo pool
   ampliado (antes solo val), usando el margen del modelo DESPLEGADO — mas estable que
   v3 por tener mas casos, aunque la spec no lo pedia explicitamente; es consistente
   con el espiritu de reducir varianza de calibracion por N chico.

Uso:
    python scripts/calibrate_p1_cv.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from evaluate_hipo import punto_operacion_sens_objetivo  # noqa: E402

DATASET_CSV = _REPO_ROOT / "data" / "dataset_p1.csv"
MODELS_DIR = _REPO_ROOT / "models" / "proyecto1"
OUT_THRESHOLDS = MODELS_DIR / "thresholds_cv.json"

FEATURE_COLS = [
    "VolRectum_cc", "VolBladder_cc", "VolPTV_cc",
    "Solap_PTV_Rectum_cc", "Solap_PTV_Bladder_cc",
    "overlap_rel_recto", "overlap_rel_vejiga",
]
UMBRAL_CLINICO = {"RV65": 15.0, "RV55": 25.0, "BV65": 15.0}
OAR_TAGS = {"recto": ["RV65", "RV55"], "vejiga": ["BV65"]}
SENS_OBJETIVO = {"recto": 0.95, "vejiga": 0.90}  # Cambio 4: recto 0.85 -> 0.95
N_FOLDS = 5
N_BOOTSTRAP = 1000
SEED = 42
VARIANTE_ACTIVA = "A"  # status quo (igual que v3); B queda reportada para comparar, no elegida por mirar test


def pool_calibracion(df):
    natural = df[df["Status"] != "UnApproved"]
    return natural[natural["split"].isin(["train", "val"])].copy()


def fit_logreg(X_tr, y_tr):
    scaler = StandardScaler().fit(X_tr)
    modelo = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    modelo.fit(scaler.transform(X_tr), y_tr)
    return scaler, modelo


def fit_ridge(X_tr, y_tr):
    scaler = StandardScaler().fit(X_tr)
    modelo = Ridge(alpha=1.0, random_state=SEED)
    modelo.fit(scaler.transform(X_tr), y_tr)
    return scaler, modelo


def cv_calibrar_oar(pool, tags, sens_objetivo):
    """5-fold StratifiedKFold: refitea LogReg (variante A) y Ridge (variante B) SOLO
    para estimar el umbral. Devuelve listas de umbrales por fold para A y B."""
    X = pool[FEATURE_COLS].to_numpy()
    label = np.max([pool[f"fail_{t}"].to_numpy() for t in tags], axis=0)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    umbrales_A, umbrales_B = [], []
    folds_omitidos = 0
    for train_idx, val_idx in skf.split(X, label):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_val = label[val_idx]
        if y_val.sum() == 0 or y_val.sum() == len(y_val):
            folds_omitidos += 1
            continue

        scores_A, scores_B = [], []
        for t in tags:
            y_fail_tr = pool[f"fail_{t}"].to_numpy()[train_idx]
            scaler_c, clf = fit_logreg(X_tr, y_fail_tr)
            scores_A.append(clf.predict_proba(scaler_c.transform(X_val))[:, 1])

            y_val_tr = pool[f"value_{t}"].to_numpy()[train_idx]
            scaler_r, reg = fit_ridge(X_tr, y_val_tr)
            pred_val = reg.predict(scaler_r.transform(X_val))
            scores_B.append(pred_val - UMBRAL_CLINICO[t])

        score_A = np.max(scores_A, axis=0)
        score_B = np.max(scores_B, axis=0)
        umbrales_A.append(punto_operacion_sens_objetivo(y_val, score_A, sens_objetivo))
        umbrales_B.append(punto_operacion_sens_objetivo(y_val, score_B, sens_objetivo))

    return umbrales_A, umbrales_B, folds_omitidos


def resumen_folds(umbrales):
    arr = np.array(umbrales)
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "umbrales_por_fold": arr.tolist(),
        "mediana": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "iqr": [float(q1), float(q3)],
    }


def deployed_scores(pool, tags, kind):
    """Scores del modelo DESPLEGADO (fit unico en train completo, Tareas 2-3) sobre el
    pool — usados para el bootstrap del umbral y para calibrar Delta. kind='A' ->
    probabilidad; kind='B' -> margen de regresion."""
    X = pool[FEATURE_COLS].to_numpy()
    out = []
    for t in tags:
        if kind == "A":
            clf = joblib.load(MODELS_DIR / f"clf_{t}.joblib")
            X_s = clf["scaler"].transform(X)
            out.append(clf["logreg"].predict_proba(X_s)[:, 1])
        else:
            reg = joblib.load(MODELS_DIR / f"reg_{t}.joblib")
            X_s = reg["scaler"].transform(X) if reg["usa_scaler"] else X
            pred = reg["modelo"].predict(X_s)
            out.append(pred - UMBRAL_CLINICO[t])
    return np.max(out, axis=0)


def bootstrap_umbral_ci(label, score, sens_objetivo, rng):
    n = len(label)
    vals = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        y_b, s_b = label[idx], score[idx]
        if y_b.sum() == 0 or y_b.sum() == n:
            continue
        vals.append(punto_operacion_sens_objetivo(y_b, s_b, sens_objetivo))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)], len(vals)


def calibrar_delta(margen_pool, no_verde_mask, oar_label, variante):
    margen_no_verde = margen_pool[no_verde_mask]
    if len(margen_no_verde) == 0:
        print(f"    [{oar_label}/{variante}] sin pacientes no-verde en el pool -> Delta=0.0")
        return 0.0, {"n": 0}
    delta = float(np.median(margen_no_verde))
    stats = {
        "n": int(len(margen_no_verde)),
        "min": float(np.min(margen_no_verde)), "median": delta,
        "mean": float(np.mean(margen_no_verde)), "std": float(np.std(margen_no_verde)),
        "max": float(np.max(margen_no_verde)),
    }
    print(f"    [{oar_label}/{variante}] Delta (mediana margen no-verde, n={stats['n']}) = {delta:.2f}pp")
    return delta, stats


def calibrar_oar_completo(pool, oar_label, tags, sens_objetivo, rng):
    print(f"\n=== {oar_label.upper()} (sens_objetivo={sens_objetivo}) ===")
    umbrales_A, umbrales_B, folds_omitidos = cv_calibrar_oar(pool, tags, sens_objetivo)
    if folds_omitidos:
        print(f"  [WARN] {folds_omitidos} fold(s) omitidos por ser degenerados (val de fold sin positivos o sin negativos)")

    resumen_A = resumen_folds(umbrales_A)
    resumen_B = resumen_folds(umbrales_B)
    print(f"  Variante A (prob)  : mediana={resumen_A['mediana']:.4f}  IQR={resumen_A['iqr']}  "
          f"min/max={resumen_A['min']:.4f}/{resumen_A['max']:.4f}")
    print(f"  Variante B (margen): mediana={resumen_B['mediana']:.4f}  IQR={resumen_B['iqr']}  "
          f"min/max={resumen_B['min']:.4f}/{resumen_B['max']:.4f}")

    label_pool = np.max([pool[f"fail_{t}"].to_numpy() for t in tags], axis=0)
    score_A_deployed = deployed_scores(pool, tags, "A")
    score_B_deployed = deployed_scores(pool, tags, "B")  # esto YA es el margen (variante B)

    ci_A, n_efectivo_A = bootstrap_umbral_ci(label_pool, score_A_deployed, sens_objetivo, rng)
    ci_B, n_efectivo_B = bootstrap_umbral_ci(label_pool, score_B_deployed, sens_objetivo, rng)
    print(f"  Bootstrap IC95 umbral A: {ci_A}  (n_efectivo={n_efectivo_A}/{N_BOOTSTRAP})")
    print(f"  Bootstrap IC95 umbral B: {ci_B}  (n_efectivo={n_efectivo_B}/{N_BOOTSTRAP})")

    no_verde_A = score_A_deployed >= resumen_A["mediana"]
    no_verde_B = score_B_deployed >= resumen_B["mediana"]
    delta_A, stats_delta_A = calibrar_delta(score_B_deployed, no_verde_A, oar_label, "A")
    delta_B, stats_delta_B = calibrar_delta(score_B_deployed, no_verde_B, oar_label, "B")

    return {
        "sens_objetivo": sens_objetivo,
        "variant_A_prob": {
            **resumen_A,
            "umbral_bootstrap_ci95": ci_A,
            "delta_severidad_pp": delta_A,
            "distribucion_margen_no_verde_pool": stats_delta_A,
            "n_no_verde_pool": int(no_verde_A.sum()),
        },
        "variant_B_margen": {
            **resumen_B,
            "umbral_bootstrap_ci95": ci_B,
            "delta_severidad_pp": delta_B,
            "distribucion_margen_no_verde_pool": stats_delta_B,
            "n_no_verde_pool": int(no_verde_B.sum()),
        },
        "variante_activa": VARIANTE_ACTIVA,
        "nota_variante_activa": "A = status quo (igual que v3). B queda calibrada y reportada para "
                                  "comparar (spec: 'no elegir la ganadora mirando test') — decision "
                                  "de cambiar a B pendiente de Pablo/Claude.ai con el IC en mano.",
    }


def main():
    df = pd.read_csv(DATASET_CSV).set_index("AnonID")
    pool = pool_calibracion(df.reset_index())
    print(f"Pool de calibracion (natural train+val, UnApproved excluido): n={len(pool)}")
    print(f"  fail_RV65={pool['fail_RV65'].sum()}  fail_RV55={pool['fail_RV55'].sum()}  fail_BV65={pool['fail_BV65'].sum()}")

    rng = np.random.default_rng(SEED)
    resultado_recto = calibrar_oar_completo(pool, "recto", OAR_TAGS["recto"], SENS_OBJETIVO["recto"], rng)
    resultado_vejiga = calibrar_oar_completo(pool, "vejiga", OAR_TAGS["vejiga"], SENS_OBJETIVO["vejiga"], rng)

    thresholds = {
        "recto": resultado_recto,
        "vejiga": resultado_vejiga,
        "umbrales_clinicos": UMBRAL_CLINICO,
        "pool_calibracion": {
            "n": len(pool), "criterio": "natural (Approved+Rejected) de train+val agrupados, UnApproved excluido",
            "n_folds": N_FOLDS, "seed": SEED, "n_bootstrap": N_BOOTSTRAP,
        },
    }
    with open(OUT_THRESHOLDS, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado: {OUT_THRESHOLDS}")


if __name__ == "__main__":
    main()
