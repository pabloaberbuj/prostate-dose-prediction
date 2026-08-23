"""
Evaluación del modelo sobre el dataset HIPOFRACCIONADO (28 fx x 2.5 Gy = 70 Gy).

Módulo paralelo a evaluate.py (que sigue siendo exclusivo del dataset normofraccionado,
sin cambios). Se invoca vía:

    python scripts/evaluate.py --dataset hipo --exp <exp_name> --checkpoint <path>

Corre sobre CUALQUIER checkpoint, incluido un modelo entrenado en el dataset normo
(zero-shot) — la arquitectura se define por --config (o por defecto configs/<exp>.yaml),
los datos de evaluación siempre son el test set hipofraccionado.

Ground truth: data/gt_dvh_hipo_256.csv (DVH calculado en Python sobre la grilla 256x256,
NO los flags del C#/ESAPI — ver CLAUDE_CODE_CONTEXT.md / HIPOFX_KICKOFF.md, decisiones
cerradas hasta D5).

Genera en results/<exp>_test_hipo/:
  - per_patient_metrics.csv
  - metrics_summary.json
  - plots/ (DVH+slice comparativo para mediana/mejores/peores por MAE-body,
    scatter de puntos DVH, scatter de constraints operativos con banda gris)

Capas de evaluación:
  1. Regresión de dosis: MAE voxel-wise por estructura, dose score OpenKBP.
  2. Acuerdo en puntos DVH (PTV D95/D98/V70Gy; Rectum/Bladder V65/V55/V45): bias + MAE.
  3. Clasificación clínica sobre los 3 constraints operativos con señal suficiente
     (Rectum V65<15, Rectum V55<25, Bladder V65<15): AUC, punto de operación a
     sensibilidad>=0.90, PPV/NPV, estratificación por zona gris/clara, misses
     clínicamente significativos, matriz de confusión "algún constraint falla".
  Bootstrap (1000 resamples, seed fijo) para IC 95% de capa 1 y capa 3.

Bloques adicionales de puntos DVH (bias/MAE, agregados en metrics_summary.json,
puntos crudos en Gy/% ya en per_patient_metrics.csv — ver BLOQUE{1,2,3}_* arriba):
  Bloque 1 — PTV: D95/D98/D99/D01cc (proxy de Dmax), bias/MAE en % de Rx (70 Gy).
  Bloque 2 — OARs: Rectum Dmean/D30/D15/D10, Bladder Dmean, bias/MAE en % de Rx.
  Bloque 3 — OARs: Rectum/Bladder V65/V55/V50/V45/V40Gy, bias/MAE en pp de volumen.

Formato Lempart 2021 / Kajikawa 2019 (mismos puntos que evaluate.py, ver ahí el
detalle — acá los valores nativos están en Gy y se convierten a % de Rx=70Gy
antes de agregar, salvo los puntos V65Gy que ya son % de volumen):
  lempart_format:  PTV D95/D98/D99/D2, Rectum Dmean/D30/D15/D10, Bladder Dmean,
                   Body D0.1% — mean_pct_error/std_pct_error/mae_pct.
  kajikawa_format: PTV D2/D98, Rectum/Bladder V65Gy — MAE ± 1SD.
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "data"))

from src.datamodules.dose_datamodule import DoseDataModule  # noqa: E402
from src.models.lightning_module import DosePredictionModule  # noqa: E402
from compute_gt_dvh_hipo import volumen_pct_sobre_umbral, dosis_percentil, dosis_d01cc  # noqa: E402

from evaluate import (cargar_modelo, figura_paciente, dose_score_openkbp, cargar_vol_ptv_cc,  # noqa: E402
                       bias_mae, diff_stats)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


PRESCRIPCION_GY = 70.0

# Mínimo de positivos (falla) en VAL para calibrar el umbral operativo de forma
# estable. Por debajo de esto, se cae al umbral clínico nominal como fallback
# (ver analizar_constraint).
MIN_VAL_POSITIVOS = 5

# Constraints operativos — los únicos con señal suficiente (ver D5). El resto
# (V45 ambos OAR, Bladder V55, Dmean) se reporta (capa 1/2) pero no se clasifica.
OPERATIONAL_CONSTRAINTS = {
    "RV65": {"struct": "Rectum",  "punto": "V65Gy", "umbral_clinico": 15.0,
             "gt_flag_col": "Flag_Rectum_V65Gy_lt15_gt256"},
    "RV55": {"struct": "Rectum",  "punto": "V55Gy", "umbral_clinico": 25.0,
             "gt_flag_col": "Flag_Rectum_V55Gy_lt25_gt256"},
    "BV65": {"struct": "Bladder", "punto": "V65Gy", "umbral_clinico": 15.0,
             "gt_flag_col": "Flag_Bladder_V65Gy_lt15_gt256"},
}

# Puntos DVH de capa 2 (acuerdo interpretable)
DVH_POINTS_CAPA2 = [
    ("PTV_D95Gy",     "ptv"),
    ("PTV_D98Gy",     "ptv"),
    ("PTV_V70Gy",     "ptv"),
    ("Rectum_V65Gy",  "rectum"),
    ("Rectum_V55Gy",  "rectum"),
    ("Rectum_V45Gy",  "rectum"),
    ("Bladder_V65Gy", "bladder"),
    ("Bladder_V55Gy", "bladder"),
    ("Bladder_V45Gy", "bladder"),
]

# Columnas GT (sufijo _gt256) para cada punto de capa 2
GT_COL_MAP = {
    "PTV_D95Gy":     "PTV_D95_Gy_gt256",
    "PTV_D98Gy":     "PTV_D98_Gy_gt256",
    "PTV_D99Gy":     "PTV_D99_Gy_gt256",
    "PTV_D2Gy":      "PTV_D2_Gy_gt256",
    "PTV_D01cc":     "PTV_D01cc_Gy_gt256",
    "PTV_V70Gy":     "PTV_V70Gy_pct_gt256",
    "Rectum_V65Gy":  "Rectum_V65Gy_gt256",
    "Rectum_V55Gy":  "Rectum_V55Gy_gt256",
    "Rectum_V50Gy":  "Rectum_V50Gy_gt256",
    "Rectum_V45Gy":  "Rectum_V45Gy_gt256",
    "Rectum_V40Gy":  "Rectum_V40Gy_gt256",
    "Rectum_Dmean":  "Rectum_Dmean_gt256",
    "Rectum_D30Gy":  "Rectum_D30Gy_gt256",
    "Rectum_D15Gy":  "Rectum_D15Gy_gt256",
    "Rectum_D10Gy":  "Rectum_D10Gy_gt256",
    "Bladder_V65Gy": "Bladder_V65Gy_gt256",
    "Bladder_V55Gy": "Bladder_V55Gy_gt256",
    "Bladder_V50Gy": "Bladder_V50Gy_gt256",
    "Bladder_V45Gy": "Bladder_V45Gy_gt256",
    "Bladder_V40Gy": "Bladder_V40Gy_gt256",
    "Bladder_Dmean": "Bladder_Dmean_gt256",
    "Body_D0_1pctGy": "Body_D0_1pct_Gy_gt256",
}

# Puntos de solo-reporte (no entran en la lista formal de capa 2, pero se guardan
# en el CSV per-patient y se agregan bias/MAE igual, es info "gratis" del GT)
REPORT_ONLY_POINTS = [
    "Rectum_Dmean", "Bladder_Dmean",
    "PTV_D99Gy", "PTV_D2Gy", "PTV_D01cc",
    "Rectum_D30Gy", "Rectum_D15Gy", "Rectum_D10Gy",
    "Rectum_V50Gy", "Rectum_V40Gy",
    "Bladder_V50Gy", "Bladder_V40Gy",
    "Body_D0_1pctGy",
]

# Los 3 bloques adicionales pedidos (bias/MAE, ver calcular_bloques_dvh_adicionales_hipo).
# Bloque 1/2 se reportan en % de prescripción (70 Gy) aunque los puntos se
# almacenan en Gy en el CSV; bloque 3 ya está en puntos porcentuales de volumen.
BLOQUE1_PTV_DVH_GY     = ["PTV_D95Gy", "PTV_D98Gy", "PTV_D99Gy", "PTV_D01cc"]
BLOQUE2_OAR_DVH_GY     = ["Rectum_Dmean", "Rectum_D30Gy", "Rectum_D15Gy", "Rectum_D10Gy",
                          "Bladder_Dmean"]

# Formato Lempart 2021 / Kajikawa 2019 — ver evaluate.py para la lista análoga en
# normo. Acá los puntos base ya existen en Gy (GT_COL_MAP); se convierten a %Rx
# (70Gy) al agregar, salvo V65Gy que ya es % de volumen.
LEMPART_POINTS_GY_HIPO = {
    "ptv":     ["PTV_D95Gy", "PTV_D98Gy", "PTV_D99Gy", "PTV_D2Gy"],
    "rectum":  ["Rectum_Dmean", "Rectum_D30Gy", "Rectum_D15Gy", "Rectum_D10Gy"],
    "bladder": ["Bladder_Dmean"],
    "body":    ["Body_D0_1pctGy"],
}
KAJIKAWA_POINTS_HIPO_GY = ["PTV_D2Gy", "PTV_D98Gy"]           # convertir a %Rx
KAJIKAWA_POINTS_HIPO_PCT = ["Rectum_V65Gy", "Bladder_V65Gy"]  # ya en % volumen
BLOQUE3_OAR_VOLUMEN    = ["Rectum_V65Gy", "Rectum_V55Gy", "Rectum_V50Gy", "Rectum_V45Gy", "Rectum_V40Gy",
                          "Bladder_V65Gy", "Bladder_V55Gy", "Bladder_V50Gy", "Bladder_V45Gy", "Bladder_V40Gy"]


# ──────────────────────────────────────────────────────────────────────────────
# Cálculo de puntos DVH predichos (misma formula que compute_gt_dvh_hipo.py)
# ──────────────────────────────────────────────────────────────────────────────

def calcular_puntos_dvh_pred(dose_pred_gy: np.ndarray, ptv: np.ndarray,
                              rectum: np.ndarray, bladder: np.ndarray,
                              vol_ptv_cc: float = None, body: np.ndarray = None) -> dict:
    out = {}
    out["PTV_D95Gy"] = dosis_percentil(dose_pred_gy, ptv, 95)
    out["PTV_D98Gy"] = dosis_percentil(dose_pred_gy, ptv, 98)
    out["PTV_D99Gy"] = dosis_percentil(dose_pred_gy, ptv, 99)
    out["PTV_D2Gy"] = dosis_percentil(dose_pred_gy, ptv, 2)
    out["PTV_D01cc"] = dosis_d01cc(dose_pred_gy, ptv, vol_ptv_cc)
    out["PTV_V70Gy"] = volumen_pct_sobre_umbral(dose_pred_gy, ptv, 70.0)
    out["PTV_V95pct"] = volumen_pct_sobre_umbral(dose_pred_gy, ptv, 0.95 * PRESCRIPCION_GY)
    out["PTV_V98pct"] = volumen_pct_sobre_umbral(dose_pred_gy, ptv, 0.98 * PRESCRIPCION_GY)
    for nombre, mask in (("Rectum", rectum), ("Bladder", bladder)):
        out[f"{nombre}_V65Gy"] = volumen_pct_sobre_umbral(dose_pred_gy, mask, 65.0)
        out[f"{nombre}_V55Gy"] = volumen_pct_sobre_umbral(dose_pred_gy, mask, 55.0)
        out[f"{nombre}_V50Gy"] = volumen_pct_sobre_umbral(dose_pred_gy, mask, 50.0)
        out[f"{nombre}_V45Gy"] = volumen_pct_sobre_umbral(dose_pred_gy, mask, 45.0)
        out[f"{nombre}_V40Gy"] = volumen_pct_sobre_umbral(dose_pred_gy, mask, 40.0)
        vals = dose_pred_gy[mask > 0]
        out[f"{nombre}_Dmean"] = float(vals.mean()) if len(vals) else float("nan")
    out["Rectum_D30Gy"] = dosis_percentil(dose_pred_gy, rectum, 30)
    out["Rectum_D15Gy"] = dosis_percentil(dose_pred_gy, rectum, 15)
    out["Rectum_D10Gy"] = dosis_percentil(dose_pred_gy, rectum, 10)
    if body is not None:
        out["Body_D0_1pctGy"] = dosis_percentil(dose_pred_gy, body, 0.1)
    else:
        out["Body_D0_1pctGy"] = float("nan")
    return out


def mae_en_mascara(dose_pred_pct: np.ndarray, dose_real_pct: np.ndarray, mask: np.ndarray) -> float:
    return float((np.abs(dose_pred_pct - dose_real_pct) * mask).sum() / max(mask.sum(), 1))


def bias_mae_pct_rx(df: pd.DataFrame, nombre: str, prescripcion_gy: float) -> dict:
    """bias/MAE de un punto DVH almacenado en Gy (columnas {nombre}_real/_pred),
    convertido a % de prescripción para reportar en las mismas unidades que el
    resto del pipeline."""
    real_gy = df[f"{nombre}_real"].to_numpy(dtype=float)
    pred_gy = df[f"{nombre}_pred"].to_numpy(dtype=float)
    valid = ~np.isnan(real_gy) & ~np.isnan(pred_gy)
    if valid.sum() == 0:
        return {"bias": float("nan"), "mae": float("nan"), "n": 0}
    diff_pct = (pred_gy[valid] - real_gy[valid]) / prescripcion_gy * 100.0
    return {"bias": float(np.mean(diff_pct)), "mae": float(np.mean(np.abs(diff_pct))), "n": int(valid.sum())}


def diff_stats_pct_rx(df: pd.DataFrame, nombre: str, prescripcion_gy: float) -> dict:
    """Igual que evaluate.diff_stats pero convirtiendo Gy -> % de prescripción
    primero (los puntos base de este script están en Gy, no en %Rx)."""
    real_gy = df[f"{nombre}_real"].to_numpy(dtype=float)
    pred_gy = df[f"{nombre}_pred"].to_numpy(dtype=float)
    valid = ~np.isnan(real_gy) & ~np.isnan(pred_gy)
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "mean_signed": float("nan"), "std_signed": float("nan"),
                "mae": float("nan"), "std_abs": float("nan")}
    diff_pct = (pred_gy[valid] - real_gy[valid]) / prescripcion_gy * 100.0
    abs_diff = np.abs(diff_pct)
    return {
        "n": n,
        "mean_signed": float(np.mean(diff_pct)),
        "std_signed":  float(np.std(diff_pct, ddof=1)) if n > 1 else float("nan"),
        "mae":         float(np.mean(abs_diff)),
        "std_abs":     float(np.std(abs_diff, ddof=1)) if n > 1 else float("nan"),
    }


def calcular_formato_lempart_kajikawa_hipo(df: pd.DataFrame, prescripcion_gy: float) -> dict:
    """Igual criterio que evaluate.calcular_formato_lempart_kajikawa: los puntos
    base están en Gy acá (no en %Rx como en evaluate.py), así que se convierten
    antes de agregar. Las columnas {nombre}_real/_pred ya existen en el df
    (evaluar_dataset ya las pobló vía GT_COL_MAP)."""
    lempart = {}
    for struct, nombres in LEMPART_POINTS_GY_HIPO.items():
        for nombre in nombres:
            s = diff_stats_pct_rx(df, nombre, prescripcion_gy)
            lempart[f"{struct}_{nombre}"] = {
                "mean_pct_error": s["mean_signed"],
                "std_pct_error":  s["std_signed"],
                "mae_pct":        s["mae"],
                "n":              s["n"],
            }

    kajikawa = {}
    for nombre in KAJIKAWA_POINTS_HIPO_GY:
        s = diff_stats_pct_rx(df, nombre, prescripcion_gy)
        kajikawa[nombre] = {"mae": s["mae"], "sd": s["std_abs"], "n": s["n"]}
    for nombre in KAJIKAWA_POINTS_HIPO_PCT:
        s = diff_stats(df, f"{nombre}_real", f"{nombre}_pred")
        kajikawa[nombre] = {"mae": s["mae"], "sd": s["std_abs"], "n": s["n"]}

    return {"lempart_format": lempart, "kajikawa_format": kajikawa}


def agregar_columnas_lempart_kajikawa_hipo(df: pd.DataFrame, prescripcion_gy: float) -> pd.DataFrame:
    """Columnas lf_/kf_ per-patient — error con signo pred-real, ya convertido a
    % de prescripción para los puntos de dosis (los puntos V65Gy quedan en pp
    de volumen, sin conversión)."""
    for struct, nombres in LEMPART_POINTS_GY_HIPO.items():
        for nombre in nombres:
            df[f"lf_{struct}_{nombre}"] = (
                (df[f"{nombre}_pred"] - df[f"{nombre}_real"]) / prescripcion_gy * 100.0
            )
    for nombre in KAJIKAWA_POINTS_HIPO_GY:
        df[f"kf_{nombre}"] = (df[f"{nombre}_pred"] - df[f"{nombre}_real"]) / prescripcion_gy * 100.0
    for nombre in KAJIKAWA_POINTS_HIPO_PCT:
        df[f"kf_{nombre}"] = df[f"{nombre}_pred"] - df[f"{nombre}_real"]
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Capa 3 — clasificación clínica
# ──────────────────────────────────────────────────────────────────────────────

def punto_operacion_sens_objetivo(y_true_fail: np.ndarray, y_score: np.ndarray,
                                   sens_objetivo: float = 0.90) -> float:
    """Umbral (sobre y_score, el VxGy predicho) que logra sensibilidad >= objetivo
    maximizando especificidad (el mayor umbral entre los que alcanzan el objetivo)."""
    fpr, tpr, thresholds = roc_curve(y_true_fail, y_score)
    candidatos = [(t, tp) for t, tp in zip(thresholds, tpr) if tp >= sens_objetivo]
    if not candidatos:
        idx = int(np.argmax(tpr))
        return float(thresholds[idx])
    return float(max(candidatos, key=lambda c: c[0])[0])


def confusion_binaria(y_true_fail: np.ndarray, y_pred_fail: np.ndarray) -> dict:
    tp = int(np.sum((y_true_fail == 1) & (y_pred_fail == 1)))
    fp = int(np.sum((y_true_fail == 0) & (y_pred_fail == 1)))
    tn = int(np.sum((y_true_fail == 0) & (y_pred_fail == 0)))
    fn = int(np.sum((y_true_fail == 1) & (y_pred_fail == 0)))
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    esp  = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv  = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "sensibilidad": sens, "especificidad": esp, "PPV": ppv, "NPV": npv,
            "prevalencia": (tp + fn) / max(tp + fp + tn + fn, 1)}


def analizar_constraint(df_val: pd.DataFrame, df_test: pd.DataFrame, tag: str, cfg_c: dict) -> dict:
    """Calibra el umbral operativo (sensibilidad>=0.90) en VAL, lo congela, y lo
    aplica a TEST para reportar sens/esp/PPV/NPV/confusión — evita el problema
    circular de calibrar y evaluar sobre el mismo test set."""
    struct, punto, umbral = cfg_c["struct"], cfg_c["punto"], cfg_c["umbral_clinico"]
    col_pred = f"{struct}_{punto}_pred"
    col_real = f"{struct}_{punto}_real"

    # ── Calibración en VAL (congelada)
    y_real_val = df_val[col_real].to_numpy()
    y_pred_val = df_val[col_pred].to_numpy()
    y_true_fail_val = (y_real_val >= umbral).astype(int)   # GT: 1 = falla el constraint
    n_pos_val = int(y_true_fail_val.sum())

    es_fallback = n_pos_val < MIN_VAL_POSITIVOS
    if es_fallback:
        log.warning(f"{tag}: solo {n_pos_val} positivos en val (< {MIN_VAL_POSITIVOS}) — "
                    f"calibración inestable, uso el umbral clínico nominal ({umbral}%) como fallback")
        umbral_operativo = float(umbral)
    else:
        umbral_operativo = punto_operacion_sens_objetivo(y_true_fail_val, y_pred_val, 0.90)

    y_pred_fail_val = (y_pred_val >= umbral_operativo).astype(int)
    conf_val = confusion_binaria(y_true_fail_val, y_pred_fail_val)

    # ── Aplicación a TEST (umbral ya congelado, sin circularidad)
    y_real = df_test[col_real].to_numpy()
    y_pred = df_test[col_pred].to_numpy()
    y_true_fail = (y_real >= umbral).astype(int)

    # AUC (barrido completo de umbral sobre el score predicho) — threshold-free,
    # se sigue calculando sobre TEST.
    try:
        auc = float(roc_auc_score(y_true_fail, y_pred))
    except ValueError:
        auc = float("nan")  # una sola clase presente

    y_pred_fail_operativo = (y_pred >= umbral_operativo).astype(int)
    conf = confusion_binaria(y_true_fail, y_pred_fail_operativo)

    # Zona gris (|V_real - umbral_clinico| < 2pp) vs. zona clara sobre TEST, sens/esp
    # del clasificador calibrado en val (umbral_operativo) estratificado por esa distancia.
    dist = np.abs(y_real - umbral)
    zonas = {}
    for nombre_zona, mask_zona in [("zona_gris", dist < 2.0), ("zona_clara", dist >= 2.0)]:
        if mask_zona.sum() == 0:
            zonas[nombre_zona] = {"n": 0, "sensibilidad": float("nan"), "especificidad": float("nan")}
            continue
        c = confusion_binaria(y_true_fail[mask_zona], y_pred_fail_operativo[mask_zona])
        zonas[nombre_zona] = {"n": int(mask_zona.sum()),
                               "sensibilidad": c["sensibilidad"], "especificidad": c["especificidad"]}

    # Miss clínicamente significativo: predicción cruza el umbral CLÍNICO (no el
    # operativo) por >3pp en el lado equivocado del valor real. Sobre TEST.
    pred_fail_clinico = (y_pred >= umbral)
    real_fail_clinico = (y_real >= umbral)
    cruza_mal = (pred_fail_clinico != real_fail_clinico) & (np.abs(y_pred - umbral) > 3.0)
    pacientes_miss = df_test.loc[cruza_mal, "AnonID"].tolist()

    return {
        "auc": auc,
        "umbral_clinico_pct": umbral,
        "umbral_operativo_val": umbral_operativo,
        "umbral_operativo_es_fallback_clinico": es_fallback,
        "val_calibracion": {"n_val": len(df_val), "n_positivos_val": n_pos_val, **conf_val},
        **conf,
        "zona_gris": zonas["zona_gris"],
        "zona_clara": zonas["zona_clara"],
        "misses_clinicamente_significativos": {"n": int(cruza_mal.sum()), "pacientes": pacientes_miss},
        # Guardado para bootstrap (sobre TEST, con el umbral ya congelado)
        "_y_true_fail": y_true_fail,
        "_y_score": y_pred,
        "_y_pred_fail_operativo": y_pred_fail_operativo,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(valores: np.ndarray, n_boot: int, rng: np.random.Generator) -> list:
    n = len(valores)
    medias = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        medias[i] = np.nanmean(valores[idx])
    lo, hi = np.nanpercentile(medias, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap_ci_clasificacion(y_true_fail: np.ndarray, y_score: np.ndarray,
                                y_pred_fail_operativo: np.ndarray,
                                n_boot: int, rng: np.random.Generator) -> dict:
    n = len(y_true_fail)
    sens_list, esp_list, auc_list = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp_op, ys = y_true_fail[idx], y_pred_fail_operativo[idx], y_score[idx]
        c = confusion_binaria(yt, yp_op)
        sens_list.append(c["sensibilidad"])
        esp_list.append(c["especificidad"])
        try:
            auc_list.append(roc_auc_score(yt, ys))
        except ValueError:
            auc_list.append(float("nan"))

    def ci(vals):
        lo, hi = np.nanpercentile(np.array(vals), [2.5, 97.5])
        return [float(lo), float(hi)]

    return {"sensibilidad_ci95": ci(sens_list), "especificidad_ci95": ci(esp_list), "auc_ci95": ci(auc_list)}


# ──────────────────────────────────────────────────────────────────────────────
# Desglose por geometria de arcos (NArcos) — sección ADICIONAL, no reemplaza el
# reporte agregado. Racional: sobre el agregado de 36 los casos "arc-limited"
# (ver CLAUDE_CODE_CONTEXT.md) tapan cualquier diferencia entre modelos — hay
# que comparar zero-shot vs. baseline vs. finetuning sobre lo tratable.
# ──────────────────────────────────────────────────────────────────────────────

MIN_SUBGRUPO_POSITIVOS = 5
MAE_BODY_ARC_LIMITED_THRESHOLD = 8.0


def cargar_narcos(metricas_csv_path: Path) -> dict:
    """dict {AnonID: NArcos} desde metricas_planes_hipofx_D95norm.csv (separador ';')."""
    df = pd.read_csv(metricas_csv_path, sep=";", encoding="utf-8-sig")
    return dict(zip(df["AnonID"], df["NArcos"].astype(int)))


def capa1_mae_subgrupo(df_sub: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> dict:
    out = {}
    for struct in ["body", "ptv", "rectum", "bladder"]:
        vals = df_sub[f"mae_{struct}"].to_numpy()
        out[f"mae_{struct}"] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "ci95": bootstrap_ci(vals, n_boot, rng) if len(vals) > 1 else [float("nan"), float("nan")],
        }
    return out


def capa3_auc_subgrupo(df_sub: pd.DataFrame, cfg_c: dict, n_boot: int, rng: np.random.Generator) -> dict:
    struct, punto, umbral = cfg_c["struct"], cfg_c["punto"], cfg_c["umbral_clinico"]
    y_real = df_sub[f"{struct}_{punto}_real"].to_numpy()
    y_pred = df_sub[f"{struct}_{punto}_pred"].to_numpy()
    y_true_fail = (y_real >= umbral).astype(int)
    n_pos = int(y_true_fail.sum())

    try:
        auc = float(roc_auc_score(y_true_fail, y_pred))
    except ValueError:
        auc = float("nan")

    resultado = {"n": len(df_sub), "n_positivos": n_pos, "auc": auc}
    if n_pos < MIN_SUBGRUPO_POSITIVOS or (len(df_sub) - n_pos) < MIN_SUBGRUPO_POSITIVOS:
        resultado["warning"] = f"n_positivos={n_pos} insuficiente (<{MIN_SUBGRUPO_POSITIVOS}) para IC estable — solo punto estimado"
        log.warning(f"Subgrupo NArcos: {resultado['warning']} (n={len(df_sub)})")
        return resultado

    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true_fail), size=len(y_true_fail))
        try:
            aucs.append(roc_auc_score(y_true_fail[idx], y_pred[idx]))
        except ValueError:
            aucs.append(float("nan"))
    lo, hi = np.nanpercentile(aucs, [2.5, 97.5])
    resultado["auc_ci95"] = [float(lo), float(hi)]
    return resultado


def desglosar_por_narcos(df: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> dict:
    desglose = {}
    for n_arcos in sorted(df["NArcos"].dropna().unique()):
        df_sub = df[df["NArcos"] == n_arcos]
        entry = {
            "n": int(len(df_sub)),
            "capa1_mae": capa1_mae_subgrupo(df_sub, n_boot, rng) if len(df_sub) > 1 else None,
            "capa3_auc": {tag: capa3_auc_subgrupo(df_sub, cfg_c, n_boot, rng)
                          for tag, cfg_c in OPERATIONAL_CONSTRAINTS.items()},
        }
        desglose[f"{int(n_arcos)}_arcos"] = entry
    return desglose


def casos_arc_limited(df: pd.DataFrame, umbral_mae_body: float = MAE_BODY_ARC_LIMITED_THRESHOLD) -> list:
    """Lista de pacientes con mae_body > umbral, con tipos nativos de Python
    (numpy/pandas scalars no son serializables por json.dump directamente)."""
    sub = df[df["mae_body"] > umbral_mae_body]
    registros = []
    for _, row in sub.iterrows():
        registros.append({
            "AnonID": row["AnonID"],
            "NArcos": int(row["NArcos"]) if pd.notna(row["NArcos"]) else None,
            "mae_body": float(row["mae_body"]),
            "mae_ptv": float(row["mae_ptv"]),
            "mae_rectum": float(row["mae_rectum"]),
            "mae_bladder": float(row["mae_bladder"]),
        })
    return registros


# ──────────────────────────────────────────────────────────────────────────────
# Inferencia + métricas escalares por paciente (compartido entre val y test)
# ──────────────────────────────────────────────────────────────────────────────

def evaluar_dataset(model, device, loader, gt_dvh: pd.DataFrame, desc: str,
                     processed_dir) -> pd.DataFrame:
    """Corre inferencia sobre un dataloader (val o test) y devuelve un DataFrame
    con una fila por paciente: MAE por estructura, dose score, puntos DVH
    predichos/reales, y flags de los constraints operativos. No retiene los
    volúmenes completos (livianos de RAM, aptos para val + test en la misma corrida)."""
    filas = []
    for batch in tqdm(loader, desc=desc):
        anonid = batch["anonid"][0] if isinstance(batch["anonid"], list) else batch["anonid"]
        if isinstance(anonid, (list, tuple)):
            anonid = anonid[0]

        batch_gpu = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            x = model._build_input(batch_gpu)
            pred = model(x)

        dose_pred_pct = pred[0].cpu().numpy()
        dose_real_pct = batch["dose"][0].numpy()
        body    = batch["body_mask"][0].numpy()
        ptv     = batch["ptv_mask"][0].numpy()
        rectum  = batch["rectum_mask"][0].numpy()
        bladder = batch["bladder_mask"][0].numpy()

        vol_ptv_cc = cargar_vol_ptv_cc(processed_dir, anonid)
        dose_pred_gy = dose_pred_pct * PRESCRIPCION_GY / 100.0
        pred_pts = calcular_puntos_dvh_pred(dose_pred_gy, ptv, rectum, bladder,
                                             vol_ptv_cc=vol_ptv_cc, body=body)
        gt_row = gt_dvh.loc[anonid]

        fila = {
            "AnonID": anonid,
            "Status": gt_row["Status"],
            "mae_body":    mae_en_mascara(dose_pred_pct, dose_real_pct, body),
            "mae_ptv":     mae_en_mascara(dose_pred_pct, dose_real_pct, ptv),
            "mae_rectum":  mae_en_mascara(dose_pred_pct, dose_real_pct, rectum),
            "mae_bladder": mae_en_mascara(dose_pred_pct, dose_real_pct, bladder),
            "dose_score_openkbp": dose_score_openkbp(dose_real_pct, dose_pred_pct, body),
        }
        for nombre, gt_col in GT_COL_MAP.items():
            fila[f"{nombre}_real"] = float(gt_row[gt_col])
        for nombre in GT_COL_MAP:
            fila[f"{nombre}_pred"] = pred_pts[nombre]

        # Flags (umbral clínico, 1 = cumple) — real ya viene del CSV GT
        for tag, cfg_c in OPERATIONAL_CONSTRAINTS.items():
            struct, punto, umbral = cfg_c["struct"], cfg_c["punto"], cfg_c["umbral_clinico"]
            fila[f"Flag_{tag}_real"] = int(gt_row[cfg_c["gt_flag_col"]])
            fila[f"Flag_{tag}_pred"] = int(fila[f"{struct}_{punto}_pred"] < umbral)

        fila["any_op_fail_real"] = int(any(fila[f"Flag_{t}_real"] == 0 for t in OPERATIONAL_CONSTRAINTS))
        fila["any_op_fail_pred_clinico"] = int(any(fila[f"Flag_{t}_pred"] == 0 for t in OPERATIONAL_CONSTRAINTS))

        filas.append(fila)

    return pd.DataFrame(filas)


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def scatter_dvh_points(df: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for ax, (nombre, _struct) in zip(axes.flat, DVH_POINTS_CAPA2):
        real = df[f"{nombre}_real"].to_numpy()
        pred = df[f"{nombre}_pred"].to_numpy()
        lim = [min(real.min(), pred.min()) - 1, max(real.max(), pred.max()) + 1]
        ax.scatter(real, pred, s=18, alpha=0.7, color="steelblue")
        ax.plot(lim, lim, color="gray", linestyle="--", linewidth=1)
        bias = float(np.mean(pred - real))
        mae = float(np.mean(np.abs(pred - real)))
        ax.set_title(f"{nombre}\nbias={bias:+.2f}  MAE={mae:.2f}", fontsize=9)
        ax.set_xlabel("Real")
        ax.set_ylabel("Predicho")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.grid(alpha=0.3)
    fig.suptitle("Capa 2 — puntos DVH: predicho vs. real (grilla 256)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=110, bbox_inches="tight")
    plt.close(fig)


def scatter_constraints_operativos(df: pd.DataFrame, resultados_capa3: dict, output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (tag, cfg_c) in zip(axes, OPERATIONAL_CONSTRAINTS.items()):
        struct, punto, umbral = cfg_c["struct"], cfg_c["punto"], cfg_c["umbral_clinico"]
        real = df[f"{struct}_{punto}_real"].to_numpy()
        pred = df[f"{struct}_{punto}_pred"].to_numpy()
        lim = [0, max(real.max(), pred.max()) + 2]

        ax.axvspan(umbral - 2.0, umbral + 2.0, color="gray", alpha=0.2, label="zona gris (|real-umbral|<2pp)")
        ax.axvline(umbral, color="black", linestyle=":", linewidth=1)
        ax.axhline(umbral, color="black", linestyle=":", linewidth=1)
        ax.plot(lim, lim, color="gray", linestyle="--", linewidth=1)
        ax.scatter(real, pred, s=20, alpha=0.75, color="firebrick")

        r = resultados_capa3[tag]
        ax.set_title(f"{tag} ({struct} {punto} < {umbral}%)\n"
                     f"AUC={r['auc']:.3f}  sens={r['sensibilidad']:.2f}  esp={r['especificidad']:.2f}",
                     fontsize=9)
        ax.set_xlabel(f"{punto} real (%)")
        ax.set_ylabel(f"{punto} predicho (%)")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Capa 3 — constraints operativos: predicho vs. real", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_hipo_evaluation(args):
    exp_name = args.exp
    config_path = args.config or (f"configs/{exp_name}.yaml" if exp_name else None)
    if config_path is None:
        raise ValueError("--dataset hipo requiere --config o --exp (para derivar configs/<exp>.yaml)")
    output_dir = Path(args.output_dir or f"results/{exp_name or Path(config_path).stem}_test_hipo")
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    cfg = OmegaConf.load(config_path)

    # Overrides: los datos de evaluación SIEMPRE son el dataset hipo, sin importar
    # en qué dataset se entrenó el checkpoint (permite zero-shot normo -> hipo).
    cfg.data.processed_dir = args.processed_dir_hipo
    cfg.data.splits_file   = args.splits_hipo
    cfg.data.cache_train   = False
    cfg.data.cache_val     = False

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.experiment.seed)

    model, device = cargar_modelo(args.checkpoint, cfg)

    dm = DoseDataModule(cfg)
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    val_loader  = dm.val_dataloader()

    # ── Ground truth DVH (grilla 256, calculado en Python — NO flags del C#)
    gt_dvh = pd.read_csv(args.gt_dvh_csv_hipo).set_index("AnonID")

    anonids_test = [Path(p).stem for p in dm.test_ds.npz_paths]
    anonids_val  = [Path(p).stem for p in dm.val_ds.npz_paths]
    faltantes_gt = sorted((set(anonids_test) | set(anonids_val)) - set(gt_dvh.index))
    if faltantes_gt:
        raise ValueError(f"{len(faltantes_gt)} pacientes de val/test hipo no están en "
                          f"{args.gt_dvh_csv_hipo}: {faltantes_gt}")

    # ── Val: solo para calibrar (congelar) el umbral operativo de capa 3 — evita
    # calibrar y evaluar sobre el mismo test set (circularidad).
    print(f"\nInfiriendo {len(anonids_val)} pacientes de val (calibración umbral, hipofraccionado)...")
    df_val = evaluar_dataset(model, device, val_loader, gt_dvh, desc="Val hipo (calibracion)",
                              processed_dir=args.processed_dir_hipo)

    print(f"\nEvaluando {len(anonids_test)} pacientes de test (hipofraccionado)...")
    df = evaluar_dataset(model, device, test_loader, gt_dvh, desc="Test hipo",
                          processed_dir=args.processed_dir_hipo)

    # NArcos por paciente (join con metricas_planes_hipofx_D95norm.csv) — para el
    # desglose por geometria de arcos (sección adicional, no afecta el agregado).
    # ⚠️ La carpeta "dicoms hipofx" (fuente de este CSV) ya no existe en disco
    # (limpieza de espacio, 2026-08 — ver CLAUDE_CODE_CONTEXT.md). Tolerar su
    # ausencia: el desglose por NArcos y casos_arc_limited quedan vacios/omitidos,
    # pero las capas 1/2/3 (lo que SI se puede calcular sin ese CSV) no se pierden.
    if Path(args.metricas_csv_hipo).exists():
        narcos_map = cargar_narcos(Path(args.metricas_csv_hipo))
        df["NArcos"] = df["AnonID"].map(narcos_map)
        faltantes_narcos = df[df["NArcos"].isna()]["AnonID"].tolist()
        if faltantes_narcos:
            log.warning(f"{len(faltantes_narcos)} pacientes de test sin NArcos en "
                        f"{args.metricas_csv_hipo}: {faltantes_narcos}")
    else:
        log.warning(f"{args.metricas_csv_hipo} no existe (dato perdido, ver CLAUDE_CODE_CONTEXT.md) "
                     f"— se omite el desglose por NArcos y casos_arc_limited.")
        df["NArcos"] = float("nan")

    df = agregar_columnas_lempart_kajikawa_hipo(df, PRESCRIPCION_GY)
    df.to_csv(output_dir / "per_patient_metrics.csv", index=False)
    print(f"Métricas por paciente guardadas en {output_dir / 'per_patient_metrics.csv'}")

    # ── Capa 3 — clasificación clínica (umbral calibrado en val, congelado, aplicado a test)
    resultados_capa3 = {tag: analizar_constraint(df_val, df, tag, cfg_c)
                         for tag, cfg_c in OPERATIONAL_CONSTRAINTS.items()}

    # any-op-fail con el clasificador CALIBRADO (umbral_operativo por constraint), bonus
    pred_fail_operativo_any = np.zeros(len(df), dtype=int)
    for tag in OPERATIONAL_CONSTRAINTS:
        pred_fail_operativo_any |= resultados_capa3[tag]["_y_pred_fail_operativo"]
    any_fail_real = df["any_op_fail_real"].to_numpy()
    conf_any = confusion_binaria(any_fail_real, pred_fail_operativo_any)

    # ── Bootstrap (capa 1 + capa 3)
    rng = np.random.default_rng(args.bootstrap_seed)
    capa1_ci = {
        struct: bootstrap_ci(df[f"mae_{struct}"].to_numpy(), args.n_bootstrap, rng)
        for struct in ["body", "ptv", "rectum", "bladder"]
    }
    capa1_ci["dose_score_openkbp"] = bootstrap_ci(df["dose_score_openkbp"].to_numpy(), args.n_bootstrap, rng)

    for tag in OPERATIONAL_CONSTRAINTS:
        r = resultados_capa3[tag]
        ci = bootstrap_ci_clasificacion(r["_y_true_fail"], r["_y_score"], r["_y_pred_fail_operativo"],
                                         args.n_bootstrap, rng)
        r.update(ci)

    # ── Capa 2 — bias/MAE por punto DVH (+ puntos de solo-reporte)
    capa2 = {}
    for nombre, _struct in DVH_POINTS_CAPA2:
        real = df[f"{nombre}_real"].to_numpy()
        pred = df[f"{nombre}_pred"].to_numpy()
        capa2[nombre] = {"bias": float(np.mean(pred - real)), "mae": float(np.mean(np.abs(pred - real)))}
    puntos_reporte_adicionales = {}
    for nombre in REPORT_ONLY_POINTS:
        real = df[f"{nombre}_real"].to_numpy()
        pred = df[f"{nombre}_pred"].to_numpy()
        puntos_reporte_adicionales[nombre] = {"bias": float(np.mean(pred - real)), "mae": float(np.mean(np.abs(pred - real)))}

    # ── Bloques 1/2/3 pedidos: mismos puntos de arriba (algunos ya en capa2,
    # otros solo-reporte), pero bias/MAE convertido a % de prescripción (70 Gy)
    # para bloques 1 y 2 — bloque 3 ya está en puntos porcentuales de volumen.
    bloque1_ptv = {nombre: bias_mae_pct_rx(df, nombre, PRESCRIPCION_GY) for nombre in BLOQUE1_PTV_DVH_GY}
    bloque2_oar = {nombre: bias_mae_pct_rx(df, nombre, PRESCRIPCION_GY) for nombre in BLOQUE2_OAR_DVH_GY}
    bloque3_oar = {nombre: bias_mae(df, f"{nombre}_real", f"{nombre}_pred") for nombre in BLOQUE3_OAR_VOLUMEN}

    # ── Formato Lempart 2021 / Kajikawa 2019
    formato_lk = calcular_formato_lempart_kajikawa_hipo(df, PRESCRIPCION_GY)

    # ── Plots — capa 2 y capa 3 (no requieren volúmenes, solo el dataframe)
    scatter_dvh_points(df, plots_dir / "dvh_points_scatter.png")
    scatter_constraints_operativos(df, resultados_capa3, plots_dir / "constraints_operativos_scatter.png")

    # ── Plots — DVH+slice comparativo (mediana, 2 mejores, 2 peores por MAE-body)
    # Segunda pasada liviana: solo re-infiere para los 5 pacientes seleccionados
    # (evita retener los 36 volumenes completos en RAM durante la pasada 1).
    orden = df.sort_values("mae_body").reset_index(drop=True)
    idx_mejores = orden.index[:2].tolist()
    idx_peores  = orden.index[-2:].tolist()
    idx_mediana = [len(orden) // 2]
    seleccion = {}
    for etiqueta, idxs in [("mejor", idx_mejores), ("mediana", idx_mediana), ("peor", idx_peores)]:
        for rank, i in enumerate(idxs, start=1):
            anonid = orden.loc[i, "AnonID"]
            seleccion[anonid] = f"{etiqueta}{rank if len(idxs) > 1 else ''}_mae{orden.loc[i, 'mae_body']:.2f}"

    for batch in tqdm(test_loader, desc="Plots DVH+slice"):
        anonid = batch["anonid"][0] if isinstance(batch["anonid"], list) else batch["anonid"]
        if isinstance(anonid, (list, tuple)):
            anonid = anonid[0]
        if anonid not in seleccion:
            continue

        batch_gpu = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            x = model._build_input(batch_gpu)
            pred = model(x)

        dose_pred_pct = pred[0].cpu().numpy()
        dose_real_pct = batch["dose"][0].numpy()
        ct      = batch["ct"][0].numpy()
        body    = batch["body_mask"][0].numpy()
        ptv     = batch["ptv_mask"][0].numpy()
        rectum  = batch["rectum_mask"][0].numpy()
        bladder = batch["bladder_mask"][0].numpy()

        figura_paciente(
            anonid, ct, dose_real_pct, dose_pred_pct, body, ptv, rectum, bladder,
            plots_dir / f"dvh_slice_{seleccion[anonid]}_{anonid}.png",
        )

    # ── Desglose por geometria de arcos (sección adicional, ver racional arriba)
    desglose_narcos = desglosar_por_narcos(df, args.n_bootstrap, rng) if df["NArcos"].notna().any() else {}
    lista_arc_limited = casos_arc_limited(df)

    # ── metrics_summary.json
    summary = {
        "dataset": "hipo",
        "exp": exp_name,
        "checkpoint": str(args.checkpoint),
        "config": str(config_path),
        "n_test": len(df),
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "capa1_regresion_dosis": {
            "mae_body":    {"mean": float(df["mae_body"].mean()),    "std": float(df["mae_body"].std()),    "ci95": capa1_ci["body"]},
            "mae_ptv":     {"mean": float(df["mae_ptv"].mean()),     "std": float(df["mae_ptv"].std()),     "ci95": capa1_ci["ptv"]},
            "mae_rectum":  {"mean": float(df["mae_rectum"].mean()),  "std": float(df["mae_rectum"].std()),  "ci95": capa1_ci["rectum"]},
            "mae_bladder": {"mean": float(df["mae_bladder"].mean()), "std": float(df["mae_bladder"].std()), "ci95": capa1_ci["bladder"]},
            "dose_score_openkbp": {"mean": float(df["dose_score_openkbp"].mean()),
                                    "std": float(df["dose_score_openkbp"].std()),
                                    "ci95": capa1_ci["dose_score_openkbp"]},
        },
        "capa2_puntos_dvh": capa2,
        "capa2_puntos_reporte_adicionales": puntos_reporte_adicionales,
        "bloque1_ptv_puntos_dvh_pct_rx": bloque1_ptv,
        "bloque2_oar_puntos_dvh_pct_rx": bloque2_oar,
        "bloque3_oar_puntos_v_pp_volumen": bloque3_oar,
        "lempart_format": formato_lk["lempart_format"],
        "kajikawa_format": formato_lk["kajikawa_format"],
        "capa3_constraints_operativos": {
            tag: {k: v for k, v in r.items() if not k.startswith("_")}
            for tag, r in resultados_capa3.items()
        },
        "any_op_fail_confusion_calibrado": conf_any,
        "desglose_narcos": desglose_narcos,
        "casos_arc_limited": {
            "criterio": f"mae_body > {MAE_BODY_ARC_LIMITED_THRESHOLD}",
            "n": len(lista_arc_limited),
            "pacientes": lista_arc_limited,
        },
    }
    with open(output_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Resumen guardado en {output_dir / 'metrics_summary.json'}")

    # ── Consola
    print("\n=== RESUMEN — dataset hipofraccionado ===")
    print(f"Pacientes evaluados: {len(df)}")
    print(f"MAE body:    {summary['capa1_regresion_dosis']['mae_body']['mean']:.3f} "
          f"± {summary['capa1_regresion_dosis']['mae_body']['std']:.3f} % "
          f"(IC95 {capa1_ci['body']})")
    print(f"MAE Rectum:  {summary['capa1_regresion_dosis']['mae_rectum']['mean']:.3f} "
          f"± {summary['capa1_regresion_dosis']['mae_rectum']['std']:.3f} %")
    print(f"MAE Bladder: {summary['capa1_regresion_dosis']['mae_bladder']['mean']:.3f} "
          f"± {summary['capa1_regresion_dosis']['mae_bladder']['std']:.3f} %")

    print("\n=== Bloque 1 — PTV, puntos DVH (bias / MAE, % Rx) ===")
    for k, v in bloque1_ptv.items():
        print(f"  {k}: bias={v['bias']:+.3f}  MAE={v['mae']:.3f}  (n={v['n']})")

    print("\n=== Bloque 2 — OARs, puntos DVH (bias / MAE, % Rx) ===")
    for k, v in bloque2_oar.items():
        print(f"  {k}: bias={v['bias']:+.3f}  MAE={v['mae']:.3f}  (n={v['n']})")

    print("\n=== Bloque 3 — OARs, puntos V (bias / MAE, pp de volumen) ===")
    for k, v in bloque3_oar.items():
        print(f"  {k}: bias={v['bias']:+.3f}  MAE={v['mae']:.3f}  (n={v['n']})")

    print("\n=== Formato Lempart 2021 (mean_pct_error ± std_pct_error, mae_pct) ===")
    for k, v in formato_lk["lempart_format"].items():
        print(f"  {k}: {v['mean_pct_error']:+.3f} ± {v['std_pct_error']:.3f}  "
              f"MAE={v['mae_pct']:.3f}  (n={v['n']})")

    print("\n=== Formato Kajikawa 2019 (MAE ± 1SD) ===")
    for k, v in formato_lk["kajikawa_format"].items():
        print(f"  {k}: {v['mae']:.3f} ± {v['sd']:.3f}  (n={v['n']})")

    print("\n=== CAPA 3 — constraints operativos (umbral calibrado en VAL, congelado) ===")
    for tag, r in resultados_capa3.items():
        fallback_tag = " [FALLBACK clinico — val con pocos positivos]" if r["umbral_operativo_es_fallback_clinico"] else ""
        print(f"\n{tag} (umbral clinico {r['umbral_clinico_pct']}%, "
              f"umbral operativo calibrado en val {r['umbral_operativo_val']:.2f}%{fallback_tag}):")
        vc = r["val_calibracion"]
        print(f"  Val calibracion: n={vc['n_val']} positivos={vc['n_positivos_val']} "
              f"sens_val={vc['sensibilidad']:.3f} esp_val={vc['especificidad']:.3f}")
        print(f"  AUC (test): {r['auc']:.3f} (IC95 {r['auc_ci95']})")
        print(f"  Sens: {r['sensibilidad']:.3f} (IC95 {r['sensibilidad_ci95']})  "
              f"Esp: {r['especificidad']:.3f} (IC95 {r['especificidad_ci95']})")
        print(f"  TP={r['TP']} FP={r['FP']} TN={r['TN']} FN={r['FN']}  "
              f"PPV={r['PPV']:.3f} NPV={r['NPV']:.3f}  prevalencia={r['prevalencia']:.3f}")
        print(f"  Zona gris (n={r['zona_gris']['n']}): sens={r['zona_gris']['sensibilidad']:.3f} "
              f"esp={r['zona_gris']['especificidad']:.3f}")
        print(f"  Zona clara (n={r['zona_clara']['n']}): sens={r['zona_clara']['sensibilidad']:.3f} "
              f"esp={r['zona_clara']['especificidad']:.3f}")
        miss = r['misses_clinicamente_significativos']
        print(f"  Misses clinicamente significativos: {miss['n']} {miss['pacientes']}")

    print(f"\nAny-op-fail (calibrado, cualquier constraint operativo falla):")
    print(f"  TP={conf_any['TP']} FP={conf_any['FP']} TN={conf_any['TN']} FN={conf_any['FN']}  "
          f"sens={conf_any['sensibilidad']:.3f} esp={conf_any['especificidad']:.3f}")

    print("\n=== DESGLOSE POR NARCOS ===")
    for grupo, entry in desglose_narcos.items():
        print(f"\n{grupo} (n={entry['n']}):")
        if entry["capa1_mae"] is not None:
            m = entry["capa1_mae"]
            print(f"  MAE body:    {m['mae_body']['mean']:.3f} (IC95 {m['mae_body']['ci95']})")
            print(f"  MAE rectum:  {m['mae_rectum']['mean']:.3f} (IC95 {m['mae_rectum']['ci95']})")
            print(f"  MAE bladder: {m['mae_bladder']['mean']:.3f} (IC95 {m['mae_bladder']['ci95']})")
        else:
            print("  (n<=1, sin estadisticas de capa1)")
        for tag, r in entry["capa3_auc"].items():
            if "warning" in r:
                print(f"  {tag}: AUC={r['auc']:.3f} n={r['n']} n_pos={r['n_positivos']} — {r['warning']}")
            else:
                print(f"  {tag}: AUC={r['auc']:.3f} (IC95 {r['auc_ci95']}) n={r['n']} n_pos={r['n_positivos']}")

    print(f"\nCasos arc-limited (mae_body > {MAE_BODY_ARC_LIMITED_THRESHOLD}): {len(lista_arc_limited)}")
    for caso in lista_arc_limited:
        print(f"  {caso['AnonID']} (NArcos={caso['NArcos']}): "
              f"mae_body={caso['mae_body']:.2f} mae_ptv={caso['mae_ptv']:.2f} "
              f"mae_rectum={caso['mae_rectum']:.2f} mae_bladder={caso['mae_bladder']:.2f}")
