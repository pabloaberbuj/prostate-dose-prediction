"""Baseline de ML clasico sobre features geometricas escalares (control metodologico).

Pregunta: ¿hace falta la U-Net, o unos pocos numeros geometricos conocibles en el
tomografo (antes de existir el plan) resuelven la clasificacion de constraints
operativos igual de bien?

7 features (TODAS pre-plan, no dependen de como se calculo el tratamiento):
  VolRectum_cc, VolBladder_cc, VolPTV_cc,
  Solap_PTV_Rectum_cc, Solap_PTV_Bladder_cc,
  overlap_rel_recto  = Solap_PTV_Rectum_cc  / VolRectum_cc,
  overlap_rel_vejiga = Solap_PTV_Bladder_cc / VolBladder_cc

Labels: flags GT operativos de gt_dvh_hipo_256.csv (los mismos que evaluate_hipo.py),
un clasificador binario (fail=1) por constraint: RV65, RV55, BV65.

Split: EXACTAMENTE data/splits/splits_hipo_v2_clean_balanced.json (mismo train/val/test
que la red, sin re-split).

Protocolo de evaluacion IDENTICO a evaluate_hipo.py (mismas funciones importadas, no
reimplementadas, para garantizar comparabilidad total):
  - AUC en test + IC95 bootstrap (1000 resamples, seed=42).
  - Umbral operativo calibrado en VAL a sensibilidad>=0.90 (congelado), aplicado a test.
    Fallback al umbral 0.5 (nominal en espacio de probabilidad) si val tiene <5 positivos.
  - Matriz de confusion en test, sens/esp/PPV/NPV/prevalencia + IC bootstrap de sens/esp.

Modelos (CPU puro, sin tocar la red/GPU):
  1. Regresion logistica (StandardScaler + class_weight='balanced').
  2. HistGradientBoostingClassifier (sklearn, hiperparametros default salvo
     random_state/class_weight — conservador, n_train~98 no da margen para tunear).

Uso:
    python scripts/baseline_ml_clasico.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from evaluate_hipo import (  # noqa: E402
    punto_operacion_sens_objetivo,
    confusion_binaria,
    bootstrap_ci_clasificacion,
    MIN_VAL_POSITIVOS,
)

CSV_FEATURES = Path(r"c:\Pablo\ProstateDoseProject\dicoms hipofx\metricas_planes_hipofx_D95norm_clean.csv")
GT_DVH_PATH = _REPO_ROOT / "data" / "gt_dvh_hipo_256.csv"
SPLITS_PATH = _REPO_ROOT / "data" / "splits" / "splits_hipo_v2_clean_balanced.json"
UNET_FINETUNE_SUMMARY = _REPO_ROOT / "results" / "exp_hipo_002b_finetune_clean_test_hipo_v2_balanced" / "exp002_metrics_summary.json"

OUT_DIR = _REPO_ROOT / "results" / "baseline_clasico"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
FALLBACK_THRESHOLD_PROBA = 0.5  # equivalente al "umbral clinico nominal" de evaluate_hipo.py, en espacio de probabilidad

FEATURE_COLS_RAW = [
    "VolRectum_cc", "VolBladder_cc", "VolPTV_cc",
    "Solap_PTV_Rectum_cc", "Solap_PTV_Bladder_cc",
]
FEATURE_COLS_FINAL = FEATURE_COLS_RAW + ["overlap_rel_recto", "overlap_rel_vejiga"]

CONSTRAINTS = {
    "RV65": "Flag_Rectum_V65Gy_lt15_gt256",
    "RV55": "Flag_Rectum_V55Gy_lt25_gt256",
    "BV65": "Flag_Bladder_V65Gy_lt15_gt256",
}


def cargar_features() -> pd.DataFrame:
    df = pd.read_csv(CSV_FEATURES, sep=";", encoding="utf-8-sig").set_index("AnonID")
    df = df[FEATURE_COLS_RAW].astype(float)
    df["overlap_rel_recto"] = df["Solap_PTV_Rectum_cc"] / df["VolRectum_cc"]
    df["overlap_rel_vejiga"] = df["Solap_PTV_Bladder_cc"] / df["VolBladder_cc"]
    return df[FEATURE_COLS_FINAL]


def cargar_labels() -> pd.DataFrame:
    gt = pd.read_csv(GT_DVH_PATH).set_index("AnonID")
    out = pd.DataFrame(index=gt.index)
    for tag, col in CONSTRAINTS.items():
        out[tag] = (gt[col] == 0).astype(int)  # 1 = falla el constraint
    return out


def analizar_constraint_clasico(y_val, score_val, y_test, score_test, tag: str) -> dict:
    """Replica analizar_constraint() de evaluate_hipo.py pero sobre scores de un
    clasificador clasico (probabilidad de clase 'falla') en vez de un VxGy predicho."""
    n_pos_val = int(y_val.sum())
    es_fallback = n_pos_val < MIN_VAL_POSITIVOS
    if es_fallback:
        print(f"  [{tag}] WARNING: solo {n_pos_val} positivos en val (< {MIN_VAL_POSITIVOS}) — "
              f"calibracion inestable, uso umbral nominal en proba ({FALLBACK_THRESHOLD_PROBA}) como fallback")
        umbral_operativo = FALLBACK_THRESHOLD_PROBA
    else:
        umbral_operativo = punto_operacion_sens_objetivo(y_val, score_val, 0.90)

    y_pred_fail_val = (score_val >= umbral_operativo).astype(int)
    conf_val = confusion_binaria(y_val, y_pred_fail_val)

    try:
        auc = float(roc_auc_score(y_test, score_test))
    except ValueError:
        auc = float("nan")

    y_pred_fail_test = (score_test >= umbral_operativo).astype(int)
    conf = confusion_binaria(y_test, y_pred_fail_test)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ci_cls = bootstrap_ci_clasificacion(y_test, score_test, y_pred_fail_test, N_BOOTSTRAP, rng)

    return {
        "auc": auc,
        "auc_ci95": ci_cls["auc_ci95"],
        "umbral_operativo_val": float(umbral_operativo),
        "umbral_operativo_es_fallback": es_fallback,
        "val_calibracion": {"n_val": len(y_val), "n_positivos_val": n_pos_val, **conf_val},
        **conf,
        "sensibilidad_ci95": ci_cls["sensibilidad_ci95"],
        "especificidad_ci95": ci_cls["especificidad_ci95"],
    }


def main():
    feats = cargar_features()
    labels = cargar_labels()
    splits = json.load(open(SPLITS_PATH))

    ids_train = [i for i in splits["train"] if i in feats.index]
    ids_val = [i for i in splits["val"] if i in feats.index]
    ids_test = [i for i in splits["test"] if i in feats.index]
    print(f"n_train={len(ids_train)}  n_val={len(ids_val)}  n_test={len(ids_test)}")

    X_train_raw = feats.loc[ids_train].to_numpy()
    X_val_raw = feats.loc[ids_val].to_numpy()
    X_test_raw = feats.loc[ids_test].to_numpy()

    resultados = {"logistic_regression": {}, "gradient_boosting": {}}
    feature_importance = {"logistic_regression": {}, "gradient_boosting": {}}

    for tag in CONSTRAINTS:
        print(f"\n=== Constraint {tag} ===")
        y_train = labels.loc[ids_train, tag].to_numpy()
        y_val = labels.loc[ids_val, tag].to_numpy()
        y_test = labels.loc[ids_test, tag].to_numpy()
        print(f"  positivos: train={y_train.sum()}/{len(y_train)}  val={y_val.sum()}/{len(y_val)}  test={y_test.sum()}/{len(y_test)}")

        # --- Regresion logistica (con estandarizacion) ---
        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)

        logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        logreg.fit(X_train, y_train)
        score_val_lr = logreg.predict_proba(X_val)[:, 1]
        score_test_lr = logreg.predict_proba(X_test)[:, 1]
        r_lr = analizar_constraint_clasico(y_val, score_val_lr, y_test, score_test_lr, f"{tag}/logreg")
        resultados["logistic_regression"][tag] = r_lr
        feature_importance["logistic_regression"][tag] = dict(zip(FEATURE_COLS_FINAL, logreg.coef_[0].tolist()))

        # --- Gradient boosting (sobre features sin estandarizar, no lo necesita) ---
        gb = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
        gb.fit(X_train_raw, y_train)
        score_val_gb = gb.predict_proba(X_val_raw)[:, 1]
        score_test_gb = gb.predict_proba(X_test_raw)[:, 1]
        r_gb = analizar_constraint_clasico(y_val, score_val_gb, y_test, score_test_gb, f"{tag}/gb")
        resultados["gradient_boosting"][tag] = r_gb

        try:
            from sklearn.inspection import permutation_importance
            imp = permutation_importance(gb, X_test_raw, y_test, n_repeats=30, random_state=42, scoring="roc_auc")
            feature_importance["gradient_boosting"][tag] = dict(zip(FEATURE_COLS_FINAL, imp.importances_mean.tolist()))
        except Exception as e:
            feature_importance["gradient_boosting"][tag] = {"error": str(e)}

        print(f"  LogReg     : AUC={r_lr['auc']:.3f} (IC95 {r_lr['auc_ci95']})  "
              f"sens={r_lr['sensibilidad']:.3f} esp={r_lr['especificidad']:.3f}  "
              f"umbral_op={r_lr['umbral_operativo_val']:.3f}{' [FALLBACK]' if r_lr['umbral_operativo_es_fallback'] else ''}")
        print(f"  GradBoost  : AUC={r_gb['auc']:.3f} (IC95 {r_gb['auc_ci95']})  "
              f"sens={r_gb['sensibilidad']:.3f} esp={r_gb['especificidad']:.3f}  "
              f"umbral_op={r_gb['umbral_operativo_val']:.3f}{' [FALLBACK]' if r_gb['umbral_operativo_es_fallback'] else ''}")

    # --- Traer AUC de la U-Net finetune (referencia) ---
    unet_auc = {}
    if UNET_FINETUNE_SUMMARY.exists():
        unet_summary = json.load(open(UNET_FINETUNE_SUMMARY))
        for tag in CONSTRAINTS:
            r = unet_summary["capa3_constraints_operativos"][tag]
            unet_auc[tag] = {"auc": r["auc"], "auc_ci95": r["auc_ci95"],
                              "sensibilidad": r["sensibilidad"], "especificidad": r["especificidad"]}
    else:
        print(f"[WARN] No se encontro {UNET_FINETUNE_SUMMARY} — tabla comparativa sin U-Net")

    print("\n" + "=" * 78)
    print("TABLA COMPARATIVA — AUC [IC95] por constraint")
    print("=" * 78)
    for tag in CONSTRAINTS:
        lr = resultados["logistic_regression"][tag]
        gb = resultados["gradient_boosting"][tag]
        u = unet_auc.get(tag, {"auc": float("nan"), "auc_ci95": [float("nan")] * 2})
        print(f"{tag}: LogReg={lr['auc']:.3f} {lr['auc_ci95']}  |  "
              f"GradBoost={gb['auc']:.3f} {gb['auc_ci95']}  |  "
              f"U-Net finetune={u['auc']:.3f} {u['auc_ci95']}")

    print("\n" + "=" * 78)
    print("IMPORTANCIA DE FEATURES")
    print("=" * 78)
    for tag in CONSTRAINTS:
        print(f"\n{tag} — coeficientes LogReg (estandarizados):")
        for feat, coef in sorted(feature_importance["logistic_regression"][tag].items(), key=lambda kv: -abs(kv[1])):
            print(f"    {feat:24s} {coef:+.3f}")
        print(f"{tag} — permutation importance GradBoost (delta AUC):")
        gb_imp = feature_importance["gradient_boosting"][tag]
        if "error" not in gb_imp:
            for feat, imp in sorted(gb_imp.items(), key=lambda kv: -abs(kv[1])):
                print(f"    {feat:24s} {imp:+.4f}")

    summary = {
        "features": FEATURE_COLS_FINAL,
        "n_train": len(ids_train), "n_val": len(ids_val), "n_test": len(ids_test),
        "splits_file": str(SPLITS_PATH),
        "n_bootstrap": N_BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED,
        "resultados": {
            modelo: {tag: {k: v for k, v in r.items()} for tag, r in tags.items()}
            for modelo, tags in resultados.items()
        },
        "feature_importance": feature_importance,
        "unet_finetune_referencia": unet_auc,
    }
    with open(OUT_DIR / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nResumen guardado en {OUT_DIR / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
