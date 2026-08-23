"""
Extension CHICA de dvh_curva_completa.py: agrega el SIGNO del error de DVH que
mean|deltaV| colapsaba. Evaluacion pura — NO entrena, NO toca configs/splits.
MISMO checkpoint (exp_hipo_002b_finetune_clean), MISMO test=31, MISMA renorm
D95(PTV_pred)=100%, MISMOS bins (0-110%Rx paso 1%) que dvh_curva_completa.py —
reusa esas funciones directamente (curva_v_d, analizar_paciente, D_BINS, IDX_*),
no las reimplementa.

Convencion de signo (fijada, ver curva_v_d/analizar_paciente en dvh_curva_completa.py):
    deltaV(D) = V_pred(D) - V_real(D)
    deltaV > 0 -> el modelo predice MAS volumen a esa dosis = OAR mas caliente
                  que la realidad = PESIMISTA.
    deltaV < 0 -> predice menos dosis que la real = OPTIMISTA (el caso peligroso
                  para un target de PDRT: se "comeria" violaciones reales).

Discriminador principal (banda media 40-80%Rx, por estructura):
    mean|deltaV|_media       -> el numero viejo (blur + bias, sin separar)
    |mean_signed deltaV|_media -> solo la componente de BIAS direccional
    ratio_bias = |mean_signed|_media / mean|deltaV|_media
        ~0 -> domina blur simetrico (forma S, sube y baja) -> loss DVH sigmoide sola alcanza
        ~1 -> domina bias direccional (mono-signo) -> hace falta un termino de
              gradiente/edge ademas de la loss DVH simetrica

Uso:
    .venv/Scripts/python.exe scripts/dvh_signed.py
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

from analisis_angular import cargar_npz, inferir_dosis  # noqa: E402
from dvh_curva_completa import (  # noqa: E402
    CHECKPOINT, CONFIG_YAML, PROCESSED_DIR, SPLITS_PATH, D_BINS, IDX_BAJA, IDX_MEDIA, IDX_ALTA,
    analizar_paciente,
)
from evaluate import cargar_modelo  # noqa: E402

OUT_DIR = _REPO_ROOT / "results/dvh_curva_completa"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

GT_DVH_CSV = _REPO_ROOT / "data/gt_dvh_hipo_256.csv"
FLAG_RV65_COL = "Flag_Rectum_V65Gy_lt15_gt256"  # 1 = cumple, 0 = no cumple

STRUCTURES = ["PTV", "Rectum", "Bladder"]  # BODY no pedido en esta extension

# Sub-bandas de la banda media, para el chequeo de forma S (blur de hombro)
IDX_MEDIA_BAJA = slice(41, 61)   # D en (40,60] -> 20 bins
IDX_MEDIA_ALTA = slice(61, 81)   # D en (60,80] -> 20 bins
assert (IDX_MEDIA_BAJA.stop - IDX_MEDIA_BAJA.start) + (IDX_MEDIA_ALTA.stop - IDX_MEDIA_ALTA.start) == (IDX_MEDIA.stop - IDX_MEDIA.start)

CRUCE_MIN_AMPLITUD_PP = 0.1  # umbral para filtrar cruces por cero espurios (ruido de piso plano, ej. PTV)


# ──────────────────────────────────────────────────────────────────────────────
# Metricas con signo, por paciente
# ──────────────────────────────────────────────────────────────────────────────

def medias_con_signo(delta: np.ndarray) -> dict:
    return {
        "signed_mean_global": float(np.mean(delta)),
        "signed_mean_baja_0_40": float(np.mean(delta[IDX_BAJA])),
        "signed_mean_media_40_80": float(np.mean(delta[IDX_MEDIA])),
        "signed_mean_alta_80_110": float(np.mean(delta[IDX_ALTA])),
        "signed_mean_media_baja_40_60": float(np.mean(delta[IDX_MEDIA_BAJA])),
        "signed_mean_media_alta_60_80": float(np.mean(delta[IDX_MEDIA_ALTA])),
    }


def cruces_por_cero(curva: np.ndarray, d_bins: np.ndarray = D_BINS, min_amplitud: float = 0.0) -> list:
    """Dosis (interpolada linealmente) donde la curva cambia de signo.

    `min_amplitud`: ignora cruces donde ambos vecinos tienen |valor| por debajo
    de este umbral (pp) — sin esto, tramos de la curva pegados a 0 por ruido de
    punto flotante (ej. PTV en la banda baja/media, donde real y pred saturan a
    ~100% de volumen y casi no hay señal) generan decenas de "cruces" espurios."""
    cruces = []
    signos = np.sign(curva)
    for i in range(len(curva) - 1):
        if signos[i] != signos[i + 1] or signos[i] == 0:
            if max(abs(curva[i]), abs(curva[i + 1])) < min_amplitud:
                continue
            d0, d1 = d_bins[i], d_bins[i + 1]
            v0, v1 = curva[i], curva[i + 1]
            if v1 == v0:
                cruces.append(float(d0))
            else:
                d_cruce = d0 + (0 - v0) * (d1 - d0) / (v1 - v0)
                cruces.append(float(d_cruce))
    return cruces


def interpretar_ratio_bias(ratio: float) -> str:
    if np.isnan(ratio):
        return "sin datos"
    if ratio < 0.35:
        return "domina BLUR SIMETRICO (forma S) — una loss de DVH sigmoide sola deberia alcanzar"
    if ratio > 0.65:
        return "domina BIAS DIRECCIONAL (mono-signo) — la loss de DVH simetrica sola NO lo corrige, hace falta termino de gradiente/edge"
    return "MIXTO — ni blur puro ni bias puro, revisar la curva completa"


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_delta_con_signo(curvas_por_estructura: dict, plots_dir: Path = PLOTS_DIR):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, struct in zip(axes, STRUCTURES):
        arr = curvas_por_estructura[struct]  # (n_pac, 111)
        mean_curve = np.mean(arr, axis=0)
        q25, q75 = np.percentile(arr, [25, 75], axis=0)
        ax.plot(D_BINS, mean_curve, color="darkorange", linewidth=1.8, label="mean ΔV(D)")
        ax.fill_between(D_BINS, q25, q75, color="darkorange", alpha=0.2, label="IQR (25-75)")
        ax.axhline(0, color="gray", linewidth=1.0)
        ax.axvspan(0, 40, color="gray", alpha=0.06)
        ax.axvspan(40, 80, color="steelblue", alpha=0.08)
        ax.axvspan(80, 110, color="darkorange", alpha=0.06)
        cruces = [c for c in cruces_por_cero(mean_curve, min_amplitud=CRUCE_MIN_AMPLITUD_PP) if 0 < c < 110]
        for c in cruces:
            ax.axvline(c, color="black", linestyle=":", linewidth=1)
        ax.set_xlabel("Dosis (% Rx)")
        ax.set_ylabel("ΔV = V_pred − V_real (pp)  [+pesimista / -optimista]")
        n_cruces_media = sum(1 for c in cruces if 40 < c < 80)
        ax.set_title(f"{struct}  (n cruces significativos={len(cruces)}, "
                     f"dentro de banda media={n_cruces_media})", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("ΔV(D) CON SIGNO — media poblacional (n=31, test)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(plots_dir / "delta_v_con_signo_por_estructura.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_rectum_cumple_vs_no(delta_rectum: np.ndarray, flags: np.ndarray, plots_dir: Path = PLOTS_DIR):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for etiqueta, cond, color in [("Cumple RV65 (real)", flags == 1, "seagreen"),
                                    ("NO cumple RV65 (real)", flags == 0, "firebrick")]:
        sub = delta_rectum[cond]
        if len(sub) == 0:
            continue
        mean_curve = sub.mean(axis=0)
        q25, q75 = np.percentile(sub, [25, 75], axis=0)
        ax.plot(D_BINS, mean_curve, color=color, linewidth=1.8, label=f"{etiqueta} (n={len(sub)})")
        ax.fill_between(D_BINS, q25, q75, color=color, alpha=0.15)
    ax.axhline(0, color="gray", linewidth=1.0)
    ax.axvspan(40, 80, color="steelblue", alpha=0.08, label="banda media")
    ax.set_xlabel("Dosis (% Rx)")
    ax.set_ylabel("ΔV = V_pred − V_real (pp)  [+pesimista / -optimista]")
    ax.set_title("Rectum — ΔV(D) con signo, cumple vs. no-cumple RV65 (real)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(plots_dir / "delta_v_rectum_cumple_vs_no.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(checkpoint=None, config_yaml=None, out_dir=None, splits_path=None):
    """Parametrizable (checkpoint/config/out_dir/splits_path) para reusar este
    script sobre otros experimentos (ej. exp_hipo_003_dvhloss) — defaults =
    checkpoint/config/split de referencia (002b)."""
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

    gt_dvh = pd.read_csv(GT_DVH_CSV).set_index("AnonID")

    cfg = OmegaConf.load(str(config_yaml))
    cfg.data.processed_dir = str(PROCESSED_DIR)
    torch.set_float32_matmul_precision("high")
    model, device = cargar_modelo(str(checkpoint), cfg)

    filas_signed = []
    curvas_por_estructura = {s: [] for s in STRUCTURES}
    anonids_orden = []

    for anonid in ids_test:
        arrays, meta = cargar_npz(PROCESSED_DIR / f"{anonid}.npz")
        dose_pred_raw = inferir_dosis(model, device, arrays)
        filas_abs, curvas, factor_renorm, d95_real, d95_pred = analizar_paciente(anonid, arrays, dose_pred_raw)

        fila_abs_por_estructura = {f["estructura"]: f for f in filas_abs}
        rv65_real_cumple = int(gt_dvh.loc[anonid, FLAG_RV65_COL]) if anonid in gt_dvh.index else None

        anonids_orden.append(anonid)
        for struct in STRUCTURES:
            delta = curvas[struct]["delta"]
            curvas_por_estructura[struct].append(delta)
            sm = medias_con_signo(delta)

            mean_abs_media = fila_abs_por_estructura[struct]["mean_abs_delta_V_media_40_80"]
            ratio_bias_media = (abs(sm["signed_mean_media_40_80"]) / mean_abs_media
                                 if mean_abs_media > 0 else float("nan"))

            filas_signed.append({
                "AnonID": anonid, "estructura": struct,
                "Flag_RV65_real_cumple": rv65_real_cumple if struct == "Rectum" else None,
                "mean_abs_delta_V_media_40_80": mean_abs_media,
                **sm,
                "ratio_bias_media_pp": ratio_bias_media,
            })

    df = pd.DataFrame(filas_signed)
    df.to_csv(out_dir / "dvh_fidelity_por_paciente_signed.csv", index=False)
    print(f"Guardado: {out_dir / 'dvh_fidelity_por_paciente_signed.csv'} ({len(df)} filas)")

    for struct in STRUCTURES:
        curvas_por_estructura[struct] = np.array(curvas_por_estructura[struct])  # (n_pac, 111)

    print("Generando plots...")
    plot_delta_con_signo(curvas_por_estructura, plots_dir)

    flags_rectum = df[df["estructura"] == "Rectum"]["Flag_RV65_real_cumple"].to_numpy()
    delta_rectum_arr = curvas_por_estructura["Rectum"]
    plot_rectum_cumple_vs_no(delta_rectum_arr, flags_rectum, plots_dir)

    # ── summary_signed.json ────────────────────────────────────────────────
    resumen = {}
    for struct in STRUCTURES:
        sub = df[df["estructura"] == struct]
        arr = curvas_por_estructura[struct]
        mean_curve = arr.mean(axis=0)
        q25, q75 = np.percentile(arr, [25, 75], axis=0)

        mean_abs_media_pobl = float(sub["mean_abs_delta_V_media_40_80"].mean())
        signed_mean_media_por_paciente = sub["signed_mean_media_40_80"].to_numpy()
        signed_mean_media_pobl = float(signed_mean_media_por_paciente.mean())
        abs_signed_mean_media_pobl = abs(signed_mean_media_pobl)
        ratio_bias_pobl = abs_signed_mean_media_pobl / mean_abs_media_pobl if mean_abs_media_pobl > 0 else float("nan")

        signo_pobl = np.sign(signed_mean_media_pobl)
        frac_mismo_signo = float(np.mean(np.sign(signed_mean_media_por_paciente) == signo_pobl)) if signo_pobl != 0 else float("nan")

        cruces_raw = cruces_por_cero(mean_curve)
        cruces_sig = cruces_por_cero(mean_curve, min_amplitud=CRUCE_MIN_AMPLITUD_PP)
        cruces_en_banda_media = [c for c in cruces_sig if 40 < c < 80]

        s_40_60 = float(sub["signed_mean_media_baja_40_60"].mean())
        s_60_80 = float(sub["signed_mean_media_alta_60_80"].mean())
        if np.sign(s_40_60) == np.sign(s_60_80) and s_40_60 != 0:
            patron_observado = ("HUMP/JOROBA, no S — mismo signo en 40-60 y 60-80, sin cruce dentro "
                                 "de la banda media (el sesgo no se auto-cancela ahi adentro)")
        else:
            patron_observado = "S-shape — signos opuestos en 40-60 vs 60-80, consistente con blur de hombro centrado ~60%"

        resumen[struct] = {
            "curva_poblacional_mean_delta_v": mean_curve.tolist(),
            "curva_poblacional_iqr25": q25.tolist(),
            "curva_poblacional_iqr75": q75.tolist(),
            "d_bins": D_BINS.tolist(),
            "banda_media_40_80": {
                "mean_abs_delta_V_media_poblacional": mean_abs_media_pobl,
                "signed_mean_media_poblacional": signed_mean_media_pobl,
                "abs_signed_mean_media_poblacional": abs_signed_mean_media_pobl,
                "ratio_bias_poblacional": ratio_bias_pobl,
                "interpretacion_poblacional": interpretar_ratio_bias(ratio_bias_pobl),
                "signo_bias_poblacional": ("PESIMISTA (+)" if signo_pobl > 0
                                            else "OPTIMISTA (-)" if signo_pobl < 0 else "neutro"),
                "distribucion_per_paciente_signed_mean_media": {
                    "mean": signed_mean_media_pobl,
                    "std": float(signed_mean_media_por_paciente.std()),
                    "median": float(np.median(signed_mean_media_por_paciente)),
                    "min": float(signed_mean_media_por_paciente.min()),
                    "max": float(signed_mean_media_por_paciente.max()),
                    "frac_pacientes_mismo_signo_que_poblacional": frac_mismo_signo,
                    "ratio_bias_per_paciente_mean": float(sub["ratio_bias_media_pp"].mean()),
                    "ratio_bias_per_paciente_std": float(sub["ratio_bias_media_pp"].std()),
                },
                "nota_poblacional_vs_per_paciente": (
                    "El ratio_bias POBLACIONAL puede ser bajo (blur-looking) solo porque distintos "
                    "pacientes tienen sesgos direccionales OPUESTOS que se cancelan al promediar — "
                    "eso NO implica que cada paciente individualmente tenga un error simetrico. "
                    "Mirar ratio_bias_per_paciente_mean (arriba) y frac_pacientes_mismo_signo: si "
                    "ese ratio per-paciente es alto (~0.7-1.0) y la fraccion de mismo signo es ~0.5, "
                    "la lectura correcta es 'cada paciente tiene un sesgo direccional fuerte, pero la "
                    "direccion varia de paciente a paciente' — no 'sesgo simetrico dentro de cada paciente'."
                ),
            },
            "forma_S_sub_bandas": {
                "signed_mean_40_60": s_40_60,
                "signed_mean_60_80": s_60_80,
                "patron_esperado_blur_hombro": "positivo en 40-60, negativo en 60-80 (o viceversa), "
                                                "con cruce por cero cerca del punto medio (~60%)",
                "patron_observado": patron_observado,
            },
            "cruces_por_cero_pct_rx_raw": cruces_raw,
            "cruces_por_cero_pct_rx_significativos": cruces_sig,
            "cruce_dentro_banda_media": cruces_en_banda_media,
        }

    # Desglose Rectum cumple/no-cumple RV65
    sub_r = df[df["estructura"] == "Rectum"]
    desglose_rv65 = {}
    for etiqueta, flag_val in [("cumple", 1), ("no_cumple", 0)]:
        grupo = sub_r[sub_r["Flag_RV65_real_cumple"] == flag_val]
        if len(grupo) == 0:
            desglose_rv65[etiqueta] = {"n": 0}
            continue
        signed_media = grupo["signed_mean_media_40_80"].to_numpy()
        mean_abs_media_g = float(grupo["mean_abs_delta_V_media_40_80"].mean())
        signed_mean_g = float(signed_media.mean())
        ratio_g = abs(signed_mean_g) / mean_abs_media_g if mean_abs_media_g > 0 else float("nan")
        desglose_rv65[etiqueta] = {
            "n": int(len(grupo)),
            "mean_abs_delta_V_media": mean_abs_media_g,
            "signed_mean_media": signed_mean_g,
            "ratio_bias_media": ratio_g,
            "signo": ("PESIMISTA (+)" if signed_mean_g > 0 else "OPTIMISTA (-)" if signed_mean_g < 0 else "neutro"),
            "frac_pacientes_optimistas_media": float(np.mean(signed_media < 0)),
        }
    alerta_optimismo_no_cumple = (desglose_rv65.get("no_cumple", {}).get("n", 0) > 0
                                   and desglose_rv65["no_cumple"]["signed_mean_media"] < 0)
    resumen["Rectum"]["desglose_RV65_cumple_vs_no_cumple"] = {
        **desglose_rv65,
        "ALERTA_optimista_en_no_cumple": alerta_optimismo_no_cumple,
        "nota": "Si ALERTA=true: el modelo es OPTIMISTA (subestima dosis) especificamente en "
                "los pacientes que en la realidad NO cumplen RV65 -> riesgo de que un target "
                "de PDRT basado en esta prediccion 'se coma' violaciones reales de constraint.",
    }

    with open(out_dir / "summary_signed.json", "w") as f:
        json.dump(resumen, f, indent=2)
    print(f"Guardado: {out_dir / 'summary_signed.json'}")

    print("\n=== RESUMEN banda media [40-80%] — blur simetrico vs bias direccional ===")
    for struct in STRUCTURES:
        r = resumen[struct]["banda_media_40_80"]
        dist = r["distribucion_per_paciente_signed_mean_media"]
        print(f"{struct:8s}: mean|dV|_media={r['mean_abs_delta_V_media_poblacional']:.2f}pp  "
              f"|signed_mean|_media={r['abs_signed_mean_media_poblacional']:.2f}pp  "
              f"ratio_bias_POBLACIONAL={r['ratio_bias_poblacional']:.2f}  signo={r['signo_bias_poblacional']}")
        print(f"          poblacional -> {r['interpretacion_poblacional']}")
        print(f"          PER-PACIENTE: ratio_bias_mean={dist['ratio_bias_per_paciente_mean']:.2f} "
              f"(std={dist['ratio_bias_per_paciente_std']:.2f})  "
              f"frac_mismo_signo_q_poblacional={dist['frac_pacientes_mismo_signo_que_poblacional']:.2f}  "
              f"signed_mean std entre pacientes={dist['std']:.2f}pp")
        if dist['ratio_bias_per_paciente_mean'] > 0.6 and 0.3 < dist['frac_pacientes_mismo_signo_que_poblacional'] < 0.7:
            print("          *** el bajo ratio poblacional es un artefacto de CANCELACION entre pacientes "
                  "con sesgo direccional fuerte pero de signo distinto — NO es blur simetrico real ***")

    print("\n=== Rectum — cumple vs no-cumple RV65 (real) ===")
    for etiqueta in ["cumple", "no_cumple"]:
        g = desglose_rv65.get(etiqueta, {})
        if g.get("n", 0) == 0:
            print(f"  {etiqueta}: sin datos")
            continue
        print(f"  {etiqueta} (n={g['n']}): signed_mean_media={g['signed_mean_media']:+.2f}pp "
              f"({g['signo']})  ratio_bias={g['ratio_bias_media']:.2f}  "
              f"frac_optimistas={g['frac_pacientes_optimistas_media']:.2f}")
    if alerta_optimismo_no_cumple:
        print("  *** ALERTA: el modelo es OPTIMISTA en los no-cumple (RV65) ***")

    return df, resumen


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--splits", default=None)
    args = parser.parse_args()
    main(checkpoint=args.checkpoint, config_yaml=args.config, out_dir=args.out_dir, splits_path=args.splits)
