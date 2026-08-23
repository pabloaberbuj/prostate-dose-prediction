"""Tarea 2 (Proyecto 1): modelos de clasificacion binaria (falla=1) por constraint.

Para cada uno de los 3 constraints operativos (RV65, RV55, BV65):
  - Regresion logistica (features estandarizadas, scaler fiteado SOLO en train,
    class_weight='balanced') -> MODELO A DESPLEGAR (interpretable, estable con n chico).
  - HistGradientBoostingClassifier (class_weight='balanced', sobre features crudas,
    no necesita escalado) -> techo de performance / control, NO se despliega.

Guarda un joblib por constraint en models/proyecto1/clf_<tag>.joblib con
{"scaler", "logreg", "gb", "feature_cols"}. Imprime AUC val/test de sanity check (la
evaluacion formal con bootstrap + umbrales congelados es la Tarea 5).

Uso:
    python scripts/train_p1_clf.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET_CSV = _REPO_ROOT / "data" / "dataset_p1.csv"
MODELS_DIR = _REPO_ROOT / "models" / "proyecto1"

FEATURE_COLS = [
    "VolRectum_cc", "VolBladder_cc", "VolPTV_cc",
    "Solap_PTV_Rectum_cc", "Solap_PTV_Bladder_cc",
    "overlap_rel_recto", "overlap_rel_vejiga",
]
CONSTRAINTS = ["RV65", "RV55", "BV65"]
SEED = 42


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_CSV).set_index("AnonID")

    ids_train = df.index[df["split"] == "train"]
    ids_val = df.index[df["split"] == "val"]
    ids_test = df.index[df["split"] == "test"]
    print(f"n_train={len(ids_train)}  n_val={len(ids_val)}  n_test={len(ids_test)}")

    X_train_raw = df.loc[ids_train, FEATURE_COLS].to_numpy()
    X_val_raw = df.loc[ids_val, FEATURE_COLS].to_numpy()
    X_test_raw = df.loc[ids_test, FEATURE_COLS].to_numpy()

    print(f"\n{'constraint':10s} {'modelo':12s} {'AUC val':>8s} {'AUC test':>9s}")
    for tag in CONSTRAINTS:
        y_train = df.loc[ids_train, f"fail_{tag}"].to_numpy()
        y_val = df.loc[ids_val, f"fail_{tag}"].to_numpy()
        y_test = df.loc[ids_test, f"fail_{tag}"].to_numpy()

        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)

        logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
        logreg.fit(X_train, y_train)
        auc_val_lr = roc_auc_score(y_val, logreg.predict_proba(X_val)[:, 1])
        auc_test_lr = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])
        print(f"{tag:10s} {'logreg':12s} {auc_val_lr:8.3f} {auc_test_lr:9.3f}")

        gb = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)
        gb.fit(X_train_raw, y_train)
        auc_val_gb = roc_auc_score(y_val, gb.predict_proba(X_val_raw)[:, 1])
        auc_test_gb = roc_auc_score(y_test, gb.predict_proba(X_test_raw)[:, 1])
        print(f"{tag:10s} {'gb (ceiling)':12s} {auc_val_gb:8.3f} {auc_test_gb:9.3f}")

        out_path = MODELS_DIR / f"clf_{tag}.joblib"
        joblib.dump({
            "scaler": scaler,
            "logreg": logreg,
            "gb": gb,
            "feature_cols": FEATURE_COLS,
            "constraint": tag,
        }, out_path)
        print(f"    -> guardado {out_path}")


if __name__ == "__main__":
    main()
