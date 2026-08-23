"""
Métrica de fidelidad de DVH-curva-completa — vara KBP, línea base sobre el modelo
de referencia (exp_hipo_002b_finetune_clean), ANTES de exp007 (loss DVH).

Tarea de EVALUACION pura: NO entrena nada, NO toca configs ni splits. Solo
inferencia (reusa la carga de modelo/NPZ de scripts/analisis_angular.py, ya
validada) + curvas DVH densas real vs. predicha.

Dataset: SOLO test (n=31) de splits_hipo_v2_clean_balanced.json — no usa val
(a diferencia de analisis_angular.py, que usaba val+test para tener mas n en
un diagnostico exploratorio; esto es una metrica de evaluacion, mismo cohorte
que se reporta como "test" en el resto del proyecto).

GT: mismo NPZ (processed_hipo/, dosis ya normalizada a D95(PTV_real)=100% desde
preprocess_hipo.py) — misma fuente que data/compute_gt_dvh_hipo.py, para que
sea idéntico al resto del proyecto. Reusa las mismas funciones de esa fuente
(volumen_pct_sobre_umbral, dosis_percentil) para los D95/D98 de chequeo.

Renormalizacion (critico, ver docstring de la tarea): la dosis PREDICHA se
renormaliza a D95(PTV_pred)=100% ANTES de calcular su DVH — mide fidelidad de
FORMA sin arrastrar el sesgo de escala global (que ya se sabe existe, ver
CLAUDE_CODE_CONTEXT: bias V70Gy ~-0.5pp en el modelo finetune). El modelo no
tiene activacion final -> puede predecir ruido negativo cerca de 0% (ver hallazgo
del piloto RD DICOM, results/pilot_rd) -> se clipea a 0 antes de renormalizar
(mismo fix que scripts/write_rd_dicom.py aplico al exportar a DICOM).

Uso:
    .venv/Scripts/python.exe scripts/dvh_curva_completa.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "data"))

from analisis_angular import (  # noqa: E402
    cargar_npz, inferir_dosis, calcular_n_slices_target,
    CHECKPOINT, CONFIG_YAML, PROCESSED_DIR,
)
from evaluate import cargar_modelo, dose_score_openkbp  # noqa: E402
from compute_gt_dvh_hipo import dosis_percentil, volumen_pct_sobre_umbral  # noqa: E402

SPLITS_PATH = _REPO_ROOT / "data/splits/splits_hipo_v2_clean_balanced.json"

OUT_DIR = _REPO_ROOT / "results/dvh_curva_completa"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PRESCRIPCION_GY = 70.0

D_BINS = np.arange(0, 111, 1)  # 0..110 %Rx, paso 1% -> 111 bins
IDX_BAJA = slice(0, 41)    # D in [0,40]   -> 41 bins
IDX_MEDIA = slice(41, 81)  # D in (40,80]  -> 40 bins
IDX_ALTA = slice(81, 111)  # D in (80,110] -> 30 bins
assert (IDX_BAJA.stop - IDX_BAJA.start) + (IDX_MEDIA.stop - IDX_MEDIA.start) + (IDX_ALTA.stop - IDX_ALTA.start) == len(D_BINS)

D95_TOL_PP = 0.5  # tolerancia para el chequeo D95(PTV_pred) post-renorm ~ 100%

STRUCTURES = ["PTV", "Rectum", "Bladder", "BODY"]
MASK_KEY = {"PTV": "ptv_mask", "Rectum": "rectum_mask", "Bladder": "bladder_mask", "BODY": "body_mask"}


# ──────────────────────────────────────────────────────────────────────────────
# DVH densa
# ──────────────────────────────────────────────────────────────────────────────

def curva_v_d(dose_pct: np.ndarray, mask: np.ndarray, d_bins: np.ndarray = D_BINS) -> np.ndarray:
    """V(D) = %volumen de la estructura con dosis >= D, para cada D en d_bins."""
    vals = dose_pct[mask > 0]
    if len(vals) == 0:
        return np.full(len(d_bins), np.nan)
    # vectorizado: (n_bins, n_vals) seria caro en memoria para BODY; usar searchsorted
    vals_sorted = np.sort(vals)
    n = len(vals_sorted)
    # posicion donde vals_sorted >= d  ->  n - idx = cuenta de vals >= d
    idx = np.searchsorted(vals_sorted, d_bins, side="left")
    return 100.0 * (n - idx) / n


def d95_ptv(dose_pct: np.ndarray, ptv_mask: np.ndarray) -> float:
    vals = dose_pct[ptv_mask > 0]
    if len(vals) == 0:
        return float("nan")
    return float(np.percentile(vals, 5))  # D95 = percentil (100-95)=5


# ──────────────────────────────────────────────────────────────────────────────
# Por paciente
# ──────────────────────────────────────────────────────────────────────────────

def analizar_paciente(anonid: str, arrays: dict, dose_pred_raw: np.ndarray) -> tuple:
    dose_real = arrays["dose"]
    ptv = arrays["ptv_mask"]

    # Renormalizacion de la predicha a D95(PTV_pred)=100%. Clip a 0 primero
    # (el modelo no tiene activacion final, puede dar ruido negativo cerca de
    # 0% — ver docstring del script; sin esto V(0) de la predicha no daria 100%).
    dose_pred_raw = np.clip(dose_pred_raw, 0.0, None)
    d95_pred_raw = d95_ptv(dose_pred_raw, ptv)
    factor_renorm = 100.0 / d95_pred_raw if d95_pred_raw > 0 else float("nan")
    dose_pred = dose_pred_raw * factor_renorm

    d95_real_check = d95_ptv(dose_real, ptv)
    d95_pred_check = d95_ptv(dose_pred, ptv)

    filas = []
    curvas = {}
    for struct in STRUCTURES:
        mask = arrays[MASK_KEY[struct]]
        v_real = curva_v_d(dose_real, mask)
        v_pred = curva_v_d(dose_pred, mask)
        delta = v_pred - v_real

        # Sanity checks
        v0_real_ok = bool(np.isclose(v_real[0], 100.0, atol=1e-6)) if mask.sum() > 0 else None
        v0_pred_ok = bool(np.isclose(v_pred[0], 100.0, atol=1e-6)) if mask.sum() > 0 else None
        mono_real_ok = bool(np.all(np.diff(v_real) <= 1e-9)) if mask.sum() > 0 else None
        mono_pred_ok = bool(np.all(np.diff(v_pred) <= 1e-9)) if mask.sum() > 0 else None

        mean_abs_delta = float(np.mean(np.abs(delta)))
        mean_abs_delta_baja = float(np.mean(np.abs(delta[IDX_BAJA])))
        mean_abs_delta_media = float(np.mean(np.abs(delta[IDX_MEDIA])))
        mean_abs_delta_alta = float(np.mean(np.abs(delta[IDX_ALTA])))

        filas.append({
            "AnonID": anonid, "estructura": struct,
            "mean_abs_delta_V_global": mean_abs_delta,
            "mean_abs_delta_V_baja_0_40": mean_abs_delta_baja,
            "mean_abs_delta_V_media_40_80": mean_abs_delta_media,
            "mean_abs_delta_V_alta_80_110": mean_abs_delta_alta,
            "n_voxeles": int(mask.sum()),
            "factor_renorm_pred": factor_renorm,
            "D95_PTV_real_pct": d95_real_check,
            "D95_PTV_pred_post_renorm_pct": d95_pred_check,
            "D95_PTV_pred_check_ok": bool(abs(d95_pred_check - 100.0) <= D95_TOL_PP),
            "V0_real_100_ok": v0_real_ok,
            "V0_pred_100_ok": v0_pred_ok,
            "monotona_real_ok": mono_real_ok,
            "monotona_pred_ok": mono_pred_ok,
        })
        curvas[struct] = {"v_real": v_real, "v_pred": v_pred, "delta": delta}

    return filas, curvas, factor_renorm, d95_real_check, d95_pred_check


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_dvh_medio_por_estructura(todas_curvas: dict, plots_dir: Path = PLOTS_DIR):
    for struct in STRUCTURES:
        v_real_all = np.array([c[struct]["v_real"] for c in todas_curvas.values()])
        v_pred_all = np.array([c[struct]["v_pred"] for c in todas_curvas.values()])
        delta_all = np.array([c[struct]["delta"] for c in todas_curvas.values()])

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        real_mean, real_std = v_real_all.mean(axis=0), v_real_all.std(axis=0)
        pred_mean, pred_std = v_pred_all.mean(axis=0), v_pred_all.std(axis=0)
        ax.plot(D_BINS, real_mean, color="black", label="Real (media)")
        ax.fill_between(D_BINS, real_mean - real_std, real_mean + real_std, color="black", alpha=0.15)
        ax.plot(D_BINS, pred_mean, color="firebrick", linestyle="--", label="Predicha renorm. (media)")
        ax.fill_between(D_BINS, pred_mean - pred_std, pred_mean + pred_std, color="firebrick", alpha=0.15)
        ax.axvspan(0, 40, color="gray", alpha=0.08)
        ax.axvspan(40, 80, color="steelblue", alpha=0.08)
        ax.axvspan(80, 110, color="darkorange", alpha=0.08)
        ax.set_xlabel("Dosis (% Rx)")
        ax.set_ylabel("Volumen (%)")
        ax.set_title(f"DVH medio ± std (n={len(todas_curvas)}) — {struct}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax2 = axes[1]
        delta_mean = delta_all.mean(axis=0)
        delta_std = delta_all.std(axis=0)
        ax2.plot(D_BINS, delta_mean, color="darkorange")
        ax2.fill_between(D_BINS, delta_mean - delta_std, delta_mean + delta_std, color="darkorange", alpha=0.2)
        ax2.axhline(0, color="gray", linewidth=0.8)
        ax2.axvspan(0, 40, color="gray", alpha=0.08, label="banda baja")
        ax2.axvspan(40, 80, color="steelblue", alpha=0.08, label="banda media")
        ax2.axvspan(80, 110, color="darkorange", alpha=0.08, label="banda alta")
        ax2.set_xlabel("Dosis (% Rx)")
        ax2.set_ylabel("ΔV = V_pred − V_real (pp)")
        ax2.set_title(f"ΔV(D) medio ± std — {struct}")
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(str(plots_dir / f"dvh_medio_{struct.lower()}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)


def plot_overlay_3_pacientes(df: pd.DataFrame, todas_curvas: dict, plots_dir: Path = PLOTS_DIR):
    rectum_rank = df[df["estructura"] == "Rectum"].sort_values("mean_abs_delta_V_global")
    elegidos = {
        "bajo": rectum_rank.iloc[0]["AnonID"],
        "medio": rectum_rank.iloc[len(rectum_rank) // 2]["AnonID"],
        "alto": rectum_rank.iloc[-1]["AnonID"],
    }
    for struct in ["Rectum", "Bladder"]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (etiqueta, anonid) in zip(axes, elegidos.items()):
            c = todas_curvas[anonid][struct]
            ax.plot(D_BINS, c["v_real"], color="black", label="Real")
            ax.plot(D_BINS, c["v_pred"], color="firebrick", linestyle="--", label="Predicha (renorm.)")
            mad = df[(df["AnonID"] == anonid) & (df["estructura"] == "Rectum")]["mean_abs_delta_V_global"].values[0]
            ax.set_title(f"{etiqueta} mean|ΔV|(recto)={mad:.2f}pp\n{anonid}", fontsize=9)
            ax.set_xlabel("Dosis (% Rx)")
            ax.set_ylabel("Volumen (%)")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        fig.suptitle(f"DVH real vs. predicha — {struct} (3 pacientes, ranking por mean|ΔV| de Rectum)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(str(plots_dir / f"overlay_3_pacientes_{struct.lower()}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# DVH score OpenKBP (vara vieja, para comparar al lado de la nueva)
# ──────────────────────────────────────────────────────────────────────────────

def dvh_score_openkbp_desde_curvas(dose_real: np.ndarray, dose_pred: np.ndarray, arrays: dict) -> float:
    errores = []
    for struct in ["PTV", "Rectum", "Bladder"]:
        mask = arrays[MASK_KEY[struct]]
        roi_real = dose_real[mask > 0]
        roi_pred = dose_pred[mask > 0]
        if len(roi_real) == 0:
            continue
        if struct == "PTV":
            for q in (98, 5, 1):  # D2, D95, D99
                errores.append(abs(np.percentile(roi_real, q) - np.percentile(roi_pred, q)))
        else:
            errores.append(abs(roi_real.mean() - roi_pred.mean()))
            errores.append(abs(roi_real.max() - roi_pred.max()))
    return float(np.mean(errores)) if errores else float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(checkpoint=None, config_yaml=None, out_dir=None, splits_path=None):
    """Parametrizable (checkpoint/config/out_dir/splits_path) para poder reusar
    este script sobre otros experimentos (ej. exp_hipo_003_dvhloss) sin duplicar
    codigo — defaults = el checkpoint/config/split de referencia (002b)."""
    checkpoint = Path(checkpoint) if checkpoint else CHECKPOINT
    config_yaml = Path(config_yaml) if config_yaml else CONFIG_YAML
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    splits_path = Path(splits_path) if splits_path else SPLITS_PATH
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_path) as f:
        splits = json.load(f)
    ids_test = splits["test"]
    print(f"Pacientes a evaluar (SOLO test): {len(ids_test)}")

    cfg = OmegaConf.load(str(config_yaml))
    cfg.data.processed_dir = str(PROCESSED_DIR)
    torch.set_float32_matmul_precision("high")
    model, device = cargar_modelo(str(checkpoint), cfg)

    conv1 = model.model.inc.block[0]
    print(f"Conv1 in_channels real del checkpoint: {conv1.in_channels} (cfg.model.in_channels={cfg.model.in_channels})")
    if conv1.in_channels != cfg.model.in_channels:
        raise RuntimeError(f"Mismatch de canales: checkpoint espera {conv1.in_channels}, cfg dice {cfg.model.in_channels}")

    pad_z_to = None
    if cfg.model.arch == "unet3d":
        # CRITICO: el 3D fue entrenado con Z SIEMPRE padeado a un tamano fijo
        # (ver docstring de inferir_dosis) -- sin este pad, el MAE se dispara
        # (BODY-MAE 7.44 vs 1.60 verificado en un paciente, exp_normo_3dunet).
        pad_z_to = calcular_n_slices_target(PROCESSED_DIR, splits_path)
        print(f"Modelo 3D detectado -- pad_z_to={pad_z_to} (replica el padding de entrenamiento)")

    todas_filas = []
    todas_curvas = {}
    dose_scores_openkbp = []
    dvh_scores_openkbp = []
    fallas_sanity = []

    for anonid in ids_test:
        arrays, meta = cargar_npz(PROCESSED_DIR / f"{anonid}.npz")
        dose_pred_raw = inferir_dosis(model, device, arrays, pad_z_to=pad_z_to)

        filas, curvas, factor_renorm, d95_real, d95_pred = analizar_paciente(anonid, arrays, dose_pred_raw)
        todas_filas.extend(filas)
        todas_curvas[anonid] = curvas

        # Vara vieja al lado (OpenKBP), sobre la dosis renormalizada (misma base
        # de comparacion que la metrica nueva, no la dosis cruda del modelo)
        d95_ptv_local = d95_ptv(arrays["dose"], arrays["ptv_mask"])
        factor = 100.0 / d95_ptv(np.clip(dose_pred_raw, 0.0, None), arrays["ptv_mask"])
        dose_pred_renorm_full = np.clip(dose_pred_raw, 0.0, None) * factor
        dose_scores_openkbp.append(dose_score_openkbp(arrays["dose"], dose_pred_renorm_full, arrays["body_mask"]))
        dvh_scores_openkbp.append(dvh_score_openkbp_desde_curvas(arrays["dose"], dose_pred_renorm_full, arrays))

        for f in filas:
            problemas = [k for k in ["D95_PTV_pred_check_ok", "V0_real_100_ok", "V0_pred_100_ok",
                                      "monotona_real_ok", "monotona_pred_ok"]
                         if f[k] is False]
            if problemas:
                fallas_sanity.append({"AnonID": anonid, "estructura": f["estructura"], "fallas": problemas})

        print(f"  {anonid}: factor_renorm={factor_renorm:.4f}  D95_real={d95_real:.2f}  "
              f"D95_pred_post_renorm={d95_pred:.2f}")

    if fallas_sanity:
        print(f"\n⚠️ SANITY CHECKS FALLIDOS ({len(fallas_sanity)}):")
        for f in fallas_sanity:
            print(f"  {f}")
    else:
        print("\nSanity checks: todos OK (V(0)=100%, DVH monotona, D95(PTV_pred) post-renorm ~100%).")

    df = pd.DataFrame(todas_filas)
    df.to_csv(out_dir / "dvh_fidelity_por_paciente.csv", index=False)
    print(f"\nGuardado: {out_dir / 'dvh_fidelity_por_paciente.csv'} ({len(df)} filas, "
          f"{df['AnonID'].nunique()} pacientes x {df['estructura'].nunique()} estructuras)")

    print("Generando plots...")
    plot_dvh_medio_por_estructura(todas_curvas, plots_dir)
    plot_overlay_3_pacientes(df, todas_curvas, plots_dir)

    resumen_por_estructura = {}
    for struct in STRUCTURES:
        sub = df[df["estructura"] == struct]
        resumen_por_estructura[struct] = {
            "mean_abs_delta_V_global":     {"mean": float(sub["mean_abs_delta_V_global"].mean()),
                                             "std": float(sub["mean_abs_delta_V_global"].std())},
            "mean_abs_delta_V_baja_0_40":  {"mean": float(sub["mean_abs_delta_V_baja_0_40"].mean()),
                                             "std": float(sub["mean_abs_delta_V_baja_0_40"].std())},
            "mean_abs_delta_V_media_40_80":{"mean": float(sub["mean_abs_delta_V_media_40_80"].mean()),
                                             "std": float(sub["mean_abs_delta_V_media_40_80"].std())},
            "mean_abs_delta_V_alta_80_110":{"mean": float(sub["mean_abs_delta_V_alta_80_110"].mean()),
                                             "std": float(sub["mean_abs_delta_V_alta_80_110"].std())},
        }
        # banda dominante (mayor mean|DeltaV|)
        bandas = {"baja_0_40": resumen_por_estructura[struct]["mean_abs_delta_V_baja_0_40"]["mean"],
                  "media_40_80": resumen_por_estructura[struct]["mean_abs_delta_V_media_40_80"]["mean"],
                  "alta_80_110": resumen_por_estructura[struct]["mean_abs_delta_V_alta_80_110"]["mean"]}
        resumen_por_estructura[struct]["banda_dominante"] = max(bandas, key=bandas.get)

    summary = {
        "n_test": len(ids_test),
        "split_usado": f"test ({splits_path.name}) — NO val",
        "checkpoint": str(checkpoint),
        "config": str(config_yaml),
        "d_bins_pct_rx": {"min": 0, "max": 110, "paso": 1, "n_bins": len(D_BINS)},
        "bandas_dosis_pct_rx": {"baja": [0, 40], "media": [40, 80], "alta": [80, 110]},
        "renormalizacion": "dosis predicha renormalizada a D95(PTV_pred)=100% antes de "
                            "calcular su DVH (previo clip a 0 del ruido negativo del modelo "
                            "sin activacion final)",
        "d95_tol_pp_chequeo": D95_TOL_PP,
        "sanity_checks_fallidos": fallas_sanity,
        "factor_renorm": {"mean": float(df.drop_duplicates("AnonID")["factor_renorm_pred"].mean()),
                           "std": float(df.drop_duplicates("AnonID")["factor_renorm_pred"].std()),
                           "min": float(df.drop_duplicates("AnonID")["factor_renorm_pred"].min()),
                           "max": float(df.drop_duplicates("AnonID")["factor_renorm_pred"].max())},
        "mean_abs_delta_V_por_estructura": resumen_por_estructura,
        "vara_vieja_openkbp_referencia": {
            "dose_score_openkbp": {"mean": float(np.mean(dose_scores_openkbp)), "std": float(np.std(dose_scores_openkbp))},
            "dvh_score_openkbp": {"mean": float(np.mean(dvh_scores_openkbp)), "std": float(np.std(dvh_scores_openkbp))},
            "nota": "calculado sobre la MISMA dosis predicha renormalizada que la metrica nueva "
                    "(no la dosis cruda del modelo), para comparar varas sobre la misma base.",
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Guardado: {out_dir / 'summary.json'}")

    print("\n=== RESUMEN mean|deltaV| por estructura (pp de volumen) ===")
    for struct in STRUCTURES:
        r = resumen_por_estructura[struct]
        print(f"{struct:8s}: global={r['mean_abs_delta_V_global']['mean']:.2f} std={r['mean_abs_delta_V_global']['std']:.2f}  "
              f"baja={r['mean_abs_delta_V_baja_0_40']['mean']:.2f}  "
              f"media={r['mean_abs_delta_V_media_40_80']['mean']:.2f}  "
              f"alta={r['mean_abs_delta_V_alta_80_110']['mean']:.2f}  "
              f"(banda dominante: {r['banda_dominante']})")

    return df, summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help=f"default: {CHECKPOINT}")
    parser.add_argument("--config", default=None, help=f"default: {CONFIG_YAML}")
    parser.add_argument("--out-dir", default=None, help=f"default: {OUT_DIR}")
    parser.add_argument("--splits", default=None, help=f"default: {SPLITS_PATH}")
    args = parser.parse_args()
    main(checkpoint=args.checkpoint, config_yaml=args.config, out_dir=args.out_dir, splits_path=args.splits)
