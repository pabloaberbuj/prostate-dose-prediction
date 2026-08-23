"""Tarea 3 (Proyecto 1): regresion de severidad — eje "rescatable vs imposible".

Para cada constraint, regresion sobre el VALOR CONTINUO del DVH (no el flag binario):
value_RV65 = %V65Gy Rectum (D95norm), value_RV55 = %V55Gy Rectum, value_BV65 = %V65Gy
Bladder. Mismas 7 features, mismo scaler (fiteado en train), mismo split que la Tarea 2.

El objetivo NO es predecir el valor con precision — es RANKEAR "apenas pasado el
constraint" vs "muy pasado" dentro del grupo que falla, para la Tarea 4 (split
naranja/rojo). Por eso, ademas del MAE/correlacion globales, se reporta la correlacion
en la banda CERCA DEL UMBRAL (+/-10pp alrededor del threshold del constraint) — la banda
que realmente importa para el semaforo.

Empieza con Ridge (lineal, estable con n chico). Si la correlacion en la banda cercana al
umbral en VAL es pobre (<0.3), usa HistGradientBoostingRegressor en su lugar para ese
constraint. Guarda un joblib por constraint en models/proyecto1/reg_<tag>.joblib con
{"scaler", "modelo", "modelo_tipo", "feature_cols", "threshold"}.

Uso:
    python scripts/train_p1_reg.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET_CSV = _REPO_ROOT / "data" / "dataset_p1.csv"
MODELS_DIR = _REPO_ROOT / "models" / "proyecto1"

FEATURE_COLS = [
    "VolRectum_cc", "VolBladder_cc", "VolPTV_cc",
    "Solap_PTV_Rectum_cc", "Solap_PTV_Bladder_cc",
    "overlap_rel_recto", "overlap_rel_vejiga",
]
CONSTRAINTS = {"RV65": 15.0, "RV55": 25.0, "BV65": 15.0}  # tag -> umbral (%)
BANDA_UMBRAL_PP = 10.0  # +/- pp alrededor del umbral
CORR_MINIMA_RIDGE = 0.3
SEED = 42


def corr_seguro(y_true, y_pred):
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(pearsonr(y_true, y_pred)[0])


def evaluar(y_true, y_pred, threshold, banda_pp):
    mae = float(mean_absolute_error(y_true, y_pred))
    corr = corr_seguro(y_true, y_pred)
    cerca = np.abs(y_true - threshold) <= banda_pp
    n_cerca = int(cerca.sum())
    mae_cerca = float(mean_absolute_error(y_true[cerca], y_pred[cerca])) if n_cerca >= 2 else float("nan")
    corr_cerca = corr_seguro(y_true[cerca], y_pred[cerca]) if n_cerca >= 3 else float("nan")
    return {"mae": mae, "corr": corr, "n_cerca_umbral": n_cerca, "mae_cerca_umbral": mae_cerca, "corr_cerca_umbral": corr_cerca}


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_CSV).set_index("AnonID")

    ids_train = df.index[df["split"] == "train"]
    ids_val = df.index[df["split"] == "val"]
    ids_test = df.index[df["split"] == "test"]

    X_train_raw = df.loc[ids_train, FEATURE_COLS].to_numpy()
    X_val_raw = df.loc[ids_val, FEATURE_COLS].to_numpy()
    X_test_raw = df.loc[ids_test, FEATURE_COLS].to_numpy()

    for tag, threshold in CONSTRAINTS.items():
        print(f"\n=== {tag} (umbral={threshold}%) ===")
        y_train = df.loc[ids_train, f"value_{tag}"].to_numpy()
        y_val = df.loc[ids_val, f"value_{tag}"].to_numpy()
        y_test = df.loc[ids_test, f"value_{tag}"].to_numpy()

        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)

        ridge = Ridge(alpha=1.0, random_state=SEED)
        ridge.fit(X_train, y_train)
        pred_val_ridge = ridge.predict(X_val)
        pred_test_ridge = ridge.predict(X_test)
        m_val_ridge = evaluar(y_val, pred_val_ridge, threshold, BANDA_UMBRAL_PP)
        m_test_ridge = evaluar(y_test, pred_test_ridge, threshold, BANDA_UMBRAL_PP)
        print(f"  Ridge : val  MAE={m_val_ridge['mae']:.2f} corr={m_val_ridge['corr']:.3f}  "
              f"cerca-umbral(n={m_val_ridge['n_cerca_umbral']}) MAE={m_val_ridge['mae_cerca_umbral']:.2f} corr={m_val_ridge['corr_cerca_umbral']:.3f}")
        print(f"          test MAE={m_test_ridge['mae']:.2f} corr={m_test_ridge['corr']:.3f}  "
              f"cerca-umbral(n={m_test_ridge['n_cerca_umbral']}) MAE={m_test_ridge['mae_cerca_umbral']:.2f} corr={m_test_ridge['corr_cerca_umbral']:.3f}")

        hgb = HistGradientBoostingRegressor(random_state=SEED)
        hgb.fit(X_train_raw, y_train)
        pred_val_hgb = hgb.predict(X_val_raw)
        pred_test_hgb = hgb.predict(X_test_raw)
        m_val_hgb = evaluar(y_val, pred_val_hgb, threshold, BANDA_UMBRAL_PP)
        m_test_hgb = evaluar(y_test, pred_test_hgb, threshold, BANDA_UMBRAL_PP)
        print(f"  HGB   : val  MAE={m_val_hgb['mae']:.2f} corr={m_val_hgb['corr']:.3f}  "
              f"cerca-umbral(n={m_val_hgb['n_cerca_umbral']}) MAE={m_val_hgb['mae_cerca_umbral']:.2f} corr={m_val_hgb['corr_cerca_umbral']:.3f}")
        print(f"          test MAE={m_test_hgb['mae']:.2f} corr={m_test_hgb['corr']:.3f}  "
              f"cerca-umbral(n={m_test_hgb['n_cerca_umbral']}) MAE={m_test_hgb['mae_cerca_umbral']:.2f} corr={m_test_hgb['corr_cerca_umbral']:.3f}")

        corr_ridge_cerca = m_val_ridge["corr_cerca_umbral"]
        usar_hgb = np.isnan(corr_ridge_cerca) or corr_ridge_cerca < CORR_MINIMA_RIDGE
        modelo_final = hgb if usar_hgb else ridge
        modelo_tipo = "hgb" if usar_hgb else "ridge"
        print(f"  -> modelo elegido: {modelo_tipo} "
              f"({'corr Ridge cerca-umbral en val = ' + f'{corr_ridge_cerca:.3f} < {CORR_MINIMA_RIDGE}' if usar_hgb else 'Ridge OK'})")

        out_path = MODELS_DIR / f"reg_{tag}.joblib"
        joblib.dump({
            "scaler": scaler,
            "modelo": modelo_final,
            "modelo_tipo": modelo_tipo,
            "feature_cols": FEATURE_COLS,
            "constraint": tag,
            "threshold": threshold,
            "usa_scaler": modelo_tipo == "ridge",
        }, out_path)
        print(f"    -> guardado {out_path}")


if __name__ == "__main__":
    main()
