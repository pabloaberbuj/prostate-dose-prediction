"""SPEC_calibracion_cv_p1.md, Cambio 3 — evaluacion en TEST con los umbrales
CV-calibrados (`thresholds_cv.json`, congelados, NUNCA recalculados contra test), con
IC bootstrap del punto de operacion (sens/esp) ademas del AUC. Reporta las dos
variantes de la frontera verde (A=prob, B=margen) lado a lado, sin elegir ganadora.

Deja `eval_p1.py` (v3) y `results/proyecto1_v3/` intactos como referencia — este script
escribe `results/proyecto1_v3_cv/metrics_summary.json` aparte.

Uso:
    python scripts/eval_p1_cv.py
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
OUT_DIR = _REPO_ROOT / "results" / "proyecto1_v3_cv"

FEATURE_COLS = [
    "VolRectum_cc", "VolBladder_cc", "VolPTV_cc",
    "Solap_PTV_Rectum_cc", "Solap_PTV_Bladder_cc",
    "overlap_rel_recto", "overlap_rel_vejiga",
]
UMBRAL_CLINICO = {"RV65": 15.0, "RV55": 25.0, "BV65": 15.0}
OAR_TAGS = {"recto": ["RV65", "RV55"], "vejiga": ["BV65"]}
N_BOOTSTRAP = 1000
SEED = 42


def cargar_modelo(kind, tag):
    return joblib.load(MODELS_DIR / f"{kind}_{tag}.joblib")


def score_variante(df, ids, tags, kind):
    X = df.loc[ids, FEATURE_COLS].to_numpy()
    out = []
    for t in tags:
        if kind == "A":
            clf = cargar_modelo("clf", t)
            out.append(clf["logreg"].predict_proba(clf["scaler"].transform(X))[:, 1])
        else:
            reg = cargar_modelo("reg", t)
            X_s = reg["scaler"].transform(X) if reg["usa_scaler"] else X
            out.append(reg["modelo"].predict(X_s) - UMBRAL_CLINICO[t])
    return np.max(out, axis=0)


def bootstrap_auc_ci(y_true, y_score, rng):
    vals = []
    n = len(y_true)
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        try:
            vals.append(roc_auc_score(y_true[idx], y_score[idx]))
        except ValueError:
            vals.append(float("nan"))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap_sens_esp_ci(y_true, y_pred, rng):
    sens_list, esp_list = [], []
    n = len(y_true)
    for _ in range(N_BOOTSTRAP):
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
    X_test = df.loc[ids_test, FEATURE_COLS].to_numpy()
    y_test = df.loc[ids_test, f"fail_{tag}"].to_numpy()
    score = clf["logreg"].predict_proba(clf["scaler"].transform(X_test))[:, 1]
    auc = float(roc_auc_score(y_test, score))
    ci = bootstrap_auc_ci(y_test, score, rng)
    print(f"  {tag}: AUC={auc:.3f} {ci}")
    return {"auc": auc, "auc_ci95": ci, "n_test": len(ids_test), "n_positivos_test": int(y_test.sum())}


def evaluar_variante(df, ids_test, oar_label, tags, variante_cfg, kind, rng):
    label_test = np.max([df.loc[ids_test, f"fail_{t}"].to_numpy() for t in tags], axis=0)
    score_test = score_variante(df, ids_test, tags, kind)
    umbral = variante_cfg["mediana"]
    delta = variante_cfg["delta_severidad_pp"]

    y_pred_no_verde = (score_test >= umbral).astype(int)
    conf = confusion_binaria(label_test, y_pred_no_verde)
    ci_sens_esp = bootstrap_sens_esp_ci(label_test, y_pred_no_verde, rng)
    auc = float(roc_auc_score(label_test, score_test))
    ci_auc = bootstrap_auc_ci(label_test, score_test, rng)

    # Naranja/rojo siempre sobre el margen de la regresion (variante B), independiente
    # de que variante define el corte verde/no-verde — ver docstring de calibrate_p1_cv.py.
    margen_test = score_variante(df, ids_test, tags, "B") if kind == "A" else score_test
    zona = np.where(score_test < umbral, "verde", np.where(margen_test >= delta, "rojo", "naranja"))
    zona_counts = {z: int((zona == z).sum()) for z in ("verde", "naranja", "rojo")}
    zona_vs_real = pd.crosstab(
        pd.Series(zona, name="zona_predicha"),
        pd.Series(np.where(label_test == 1, "falla", "cumple"), name="real"),
    )

    print(f"    Variante {kind}: umbral(mediana CV)={umbral:.4f}  AUC={auc:.3f} {ci_auc}")
    print(f"      sens={conf['sensibilidad']:.3f} {ci_sens_esp['sensibilidad_ci95']}  "
          f"esp={conf['especificidad']:.3f} {ci_sens_esp['especificidad_ci95']}  "
          f"({conf['TP']}TP {conf['FP']}FP {conf['TN']}TN {conf['FN']}FN)")
    print(f"      zonas: {zona_counts}")

    return {
        "umbral_mediana_cv": umbral, "umbral_iqr_cv": variante_cfg["iqr"],
        "umbral_bootstrap_ci95": variante_cfg["umbral_bootstrap_ci95"],
        "auc": auc, "auc_ci95": ci_auc,
        "sens_esp_test": {**conf, **ci_sens_esp},
        "delta_severidad_pp": delta,
        "zona_counts": zona_counts,
        "zona_vs_real": zona_vs_real.to_dict(),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_CSV).set_index("AnonID")
    ids_test = df.index[df["split"] == "test"]
    print(f"n_test={len(ids_test)}  (prevalencia natural, umbrales CONGELADOS del CV — no se tocan)")

    with open(MODELS_DIR / "thresholds_cv.json", encoding="utf-8") as f:
        thresholds = json.load(f)

    rng = np.random.default_rng(SEED)

    print("\n" + "=" * 78)
    print("AUC POR CONSTRAINT INDIVIDUAL (sin cambios respecto a v3)")
    print("=" * 78)
    resultados_constraints = {tag: evaluar_constraint_individual(df, ids_test, tag, rng)
                               for tag in ("RV65", "RV55", "BV65")}

    print("\n" + "=" * 78)
    print("SEMAFORO POR OAR — variantes A (prob) vs B (margen), umbral CV congelado")
    print("=" * 78)
    resultado_oar = {}
    for oar_label, tags in OAR_TAGS.items():
        print(f"\n[{oar_label.upper()}]")
        cfg_oar = thresholds[oar_label]
        res_A = evaluar_variante(df, ids_test, oar_label, tags, cfg_oar["variant_A_prob"], "A", rng)
        res_B = evaluar_variante(df, ids_test, oar_label, tags, cfg_oar["variant_B_margen"], "B", rng)
        resultado_oar[oar_label] = {
            "sens_objetivo": cfg_oar["sens_objetivo"],
            "variante_activa": cfg_oar["variante_activa"],
            "variant_A_prob": res_A,
            "variant_B_margen": res_B,
        }

    summary = {
        "n_test": len(ids_test),
        "n_bootstrap": N_BOOTSTRAP, "bootstrap_seed": SEED,
        "thresholds_file": "thresholds_cv.json",
        "por_constraint": resultados_constraints,
        "por_oar": resultado_oar,
    }
    with open(OUT_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nGuardado: {OUT_DIR / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
