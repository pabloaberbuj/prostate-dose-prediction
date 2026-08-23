"""Tarea 5 (Proyecto 1): evaluacion en TEST (prevalencia natural) con umbrales
congelados (Tarea 4) y modelos ya entrenados (Tareas 2 y 3). No se recalibra nada
contra test.

Reporta, por constraint (RV65/RV55/BV65 individuales) y por OAR (Recto/Vejiga, el
semaforo realmente desplegado):
  - AUC [IC95 bootstrap 1000, seed=42] logreg vs GradientBoosting (control).
  - Sens/esp de la frontera verde en el punto de operacion congelado.
  - Matriz de las 3 zonas (verde/naranja/rojo) vs label real (cumple/falla), por OAR.

Guarda results/proyecto1_v3/metrics_summary.json.

Uso:
    python scripts/eval_p1.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from evaluate_hipo import confusion_binaria  # noqa: E402

DATASET_CSV = _REPO_ROOT / "data" / "dataset_p1.csv"
MODELS_DIR = _REPO_ROOT / "models" / "proyecto1"
OUT_DIR = _REPO_ROOT / "results" / "proyecto1_v3"

N_BOOTSTRAP = 1000
SEED = 42


def cargar_modelo(kind, tag):
    return joblib.load(MODELS_DIR / f"{kind}_{tag}.joblib")


def predecir_prob_fail(clf_bundle, X_raw, modelo_key="logreg"):
    if modelo_key == "logreg":
        X = clf_bundle["scaler"].transform(X_raw)
        return clf_bundle["logreg"].predict_proba(X)[:, 1]
    return clf_bundle["gb"].predict_proba(X_raw)[:, 1]


def predecir_valor(reg_bundle, X_raw):
    X = reg_bundle["scaler"].transform(X_raw) if reg_bundle["usa_scaler"] else X_raw
    return reg_bundle["modelo"].predict(X)


def bootstrap_auc_ci(y_true, y_score, n_boot, rng):
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            vals.append(roc_auc_score(y_true[idx], y_score[idx]))
        except ValueError:
            vals.append(float("nan"))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap_sens_esp_ci(y_true, y_pred, n_boot, rng):
    sens_list, esp_list = [], []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        c = confusion_binaria(y_true[idx], y_pred[idx])
        sens_list.append(c["sensibilidad"])
        esp_list.append(c["especificidad"])

    def ci(vals):
        lo, hi = np.nanpercentile(np.array(vals), [2.5, 97.5])
        return [float(lo), float(hi)]
    return {"sensibilidad_ci95": ci(sens_list), "especificidad_ci95": ci(esp_list)}


def evaluar_constraint_individual(df, ids_test, tag, rng):
    clf = cargar_modelo("clf", tag)
    X_test = df.loc[ids_test, clf["feature_cols"]].to_numpy()
    y_test = df.loc[ids_test, f"fail_{tag}"].to_numpy()

    score_lr = predecir_prob_fail(clf, X_test, "logreg")
    score_gb = predecir_prob_fail(clf, X_test, "gb")
    auc_lr = float(roc_auc_score(y_test, score_lr))
    auc_gb = float(roc_auc_score(y_test, score_gb))
    ci_lr = bootstrap_auc_ci(y_test, score_lr, N_BOOTSTRAP, rng)
    ci_gb = bootstrap_auc_ci(y_test, score_gb, N_BOOTSTRAP, rng)

    dentro_ic = ci_lr[0] <= auc_gb <= ci_lr[1] or ci_gb[0] <= auc_lr <= ci_gb[1]
    print(f"  {tag}: LogReg AUC={auc_lr:.3f} {ci_lr}  |  GB(ceiling) AUC={auc_gb:.3f} {ci_gb}  "
          f"{'[dentro de rango esperado]' if dentro_ic else '[FUERA del rango esperado]'}")

    return {
        "n_test": len(ids_test), "n_positivos_test": int(y_test.sum()),
        "logreg": {"auc": auc_lr, "auc_ci95": ci_lr},
        "gradient_boosting": {"auc": auc_gb, "auc_ci95": ci_gb},
        "logreg_vs_gb_dentro_ic_esperado": bool(dentro_ic),
    }


def evaluar_oar(df, ids_test, oar_label, tags, thresholds_oar, umbral_clinico, rng):
    feature_cols = cargar_modelo("clf", tags[0])["feature_cols"]
    X_test = df.loc[ids_test, feature_cols].to_numpy()

    clf = {t: cargar_modelo("clf", t) for t in tags}
    reg = {t: cargar_modelo("reg", t) for t in tags}
    prob_fail = {t: predecir_prob_fail(clf[t], X_test, "logreg") for t in tags}
    val_pred = {t: predecir_valor(reg[t], X_test) for t in tags}

    score = np.max([prob_fail[t] for t in tags], axis=0)
    margen = np.max([val_pred[t] - umbral_clinico[t] for t in tags], axis=0)
    label_fail = np.max([df.loc[ids_test, f"fail_{t}"].to_numpy() for t in tags], axis=0)

    umbral_verde = thresholds_oar["umbral_verde_proba"]
    delta = thresholds_oar["delta_severidad_pp"]

    y_pred_no_verde = (score >= umbral_verde).astype(int)
    conf = confusion_binaria(label_fail, y_pred_no_verde)
    ci_sens_esp = bootstrap_sens_esp_ci(label_fail, y_pred_no_verde, N_BOOTSTRAP, rng)

    auc = float(roc_auc_score(label_fail, score))
    ci_auc = bootstrap_auc_ci(label_fail, score, N_BOOTSTRAP, rng)

    zona = np.where(score < umbral_verde, "verde", np.where(margen >= delta, "rojo", "naranja"))
    zona_vs_real = pd.crosstab(
        pd.Series(zona, name="zona_predicha"),
        pd.Series(np.where(label_fail == 1, "falla", "cumple"), name="real"),
    )

    print(f"\n  [{oar_label}] AUC={auc:.3f} {ci_auc}  n_test={len(ids_test)} n_falla_real={int(label_fail.sum())}")
    print(f"    Frontera verde: sens={conf['sensibilidad']:.3f} {ci_sens_esp['sensibilidad_ci95']}  "
          f"esp={conf['especificidad']:.3f} {ci_sens_esp['especificidad_ci95']}  "
          f"({conf['TP']}TP {conf['FP']}FP {conf['TN']}TN {conf['FN']}FN)")
    print(f"    Matriz de zonas vs real:\n{zona_vs_real.to_string()}")

    return {
        "auc": auc, "auc_ci95": ci_auc,
        "n_test": len(ids_test), "n_falla_real": int(label_fail.sum()),
        "frontera_verde": {**conf, **ci_sens_esp,
                            "umbral_verde_proba": umbral_verde, "sens_objetivo": thresholds_oar["sens_objetivo"]},
        "delta_severidad_pp": delta,
        "matriz_zonas_vs_real": zona_vs_real.to_dict(),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_CSV).set_index("AnonID")
    ids_test = df.index[df["split"] == "test"]
    print(f"n_test={len(ids_test)}  (prevalencia natural — TreatmentApproved + Rejected)")

    with open(MODELS_DIR / "thresholds.json", encoding="utf-8") as f:
        thresholds = json.load(f)

    rng = np.random.default_rng(SEED)

    print("\n" + "=" * 78)
    print("AUC POR CONSTRAINT INDIVIDUAL (LogReg vs GradBoost control)")
    print("=" * 78)
    resultados_constraints = {}
    for tag in ("RV65", "RV55", "BV65"):
        resultados_constraints[tag] = evaluar_constraint_individual(df, ids_test, tag, rng)

    print("\n" + "=" * 78)
    print("SEMAFORO POR OAR (el que realmente se despliega)")
    print("=" * 78)
    umbral_clinico = thresholds["umbrales_clinicos"]
    resultado_recto = evaluar_oar(df, ids_test, "RECTO", ["RV65", "RV55"], thresholds["recto"], umbral_clinico, rng)
    resultado_vejiga = evaluar_oar(df, ids_test, "VEJIGA", ["BV65"], thresholds["vejiga"], umbral_clinico, rng)

    summary = {
        "n_test": len(ids_test),
        "splits_file": str(_REPO_ROOT / "data" / "splits" / "splits_hipo_v3.json"),
        "n_bootstrap": N_BOOTSTRAP, "bootstrap_seed": SEED,
        "por_constraint": resultados_constraints,
        "por_oar": {"recto": resultado_recto, "vejiga": resultado_vejiga},
    }
    with open(OUT_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nGuardado: {OUT_DIR / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
