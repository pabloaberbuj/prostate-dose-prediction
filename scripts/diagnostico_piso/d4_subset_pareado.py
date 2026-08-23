"""
D4 — Subset PAREADO (mismo paciente, dos generadores): RapidPlan-normo vs.
manual-hipo, sobre la MISMA anatomia real (misma persona, AnonID identico —
verificado que el hash de HC es el mismo en processed/ y processed_hipo/).

Pregunta: para el MISMO paciente, ¿el generador manual (hipo) produce un hombro
rectal (banda media, %Rx) mas disperso ENTRE PACIENTES que el generador
automatico RapidPlan (normo)? Si RapidPlan es consistente, su varianza
inter-paciente deberia ser chica incluso con anatomias distintas (15 personas
distintas); si el manual es idiosincratico, su varianza deberia ser mayor.

⚠️ CONFUSOR DE FRACCIONAMIENTO (70Gy/28fx hipo vs 78Gy/39fx normo): NO se
comparan las MEDIAS (el fraccionamiento las corre en un sentido u otro vía
constraints/practica clinica distinta) — se compara la DISPERSION
inter-paciente (std), que es robusta a un shift sistematico de la media.
Normalizado a %Rx (ya lo esta el GT, D95(PTV)=100% en ambos preprocesados) +
secundariamente en EQD2 (alpha/beta=3, recto) para robustez adicional.

Puramente GT, no usa el modelo. Reusa cargar_npz/overlap_y_volumenes
(analisis_angular.py) y curva_v_d/D_BINS/IDX_MEDIA (dvh_curva_completa.py).

Uso:
    .venv/Scripts/python.exe scripts/diagnostico_piso/d4_subset_pareado.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from analisis_angular import cargar_npz  # noqa: E402
from dvh_curva_completa import curva_v_d, D_BINS, IDX_MEDIA  # noqa: E402

PROCESSED_NORMO = Path(r"C:\Pablo\ProstateDoseProject\processed")
PROCESSED_HIPO = Path(r"C:\Pablo\ProstateDoseProject\processed_hipo")
SPLITS_HIPO_CLEAN = _REPO_ROOT / "data/splits/splits_hipo_v2_clean_balanced.json"
SPLITS_NORMO = _REPO_ROOT / "data/splits/splits_v1.json"

OUT_DIR = _REPO_ROOT / "results/diagnostico_piso"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RX_GY = {"normo": 78.0, "hipo": 70.0}
N_FX = {"normo": 39, "hipo": 28}
ALPHA_BETA_RECTUM_GY = 3.0
D_MEDIA_POINTS = [40, 50, 60, 70, 80]  # puntos individuales dentro de la banda, para robustez


def ids_overlap() -> list:
    with open(SPLITS_NORMO) as f:
        s_normo = json.load(f)
    with open(SPLITS_HIPO_CLEAN) as f:
        s_hipo = json.load(f)
    ids_normo = set(s_normo["train"]) | set(s_normo["val"]) | set(s_normo["test"])
    ids_hipo = set(s_hipo["train"]) | set(s_hipo["val"]) | set(s_hipo["test"])
    return sorted(ids_normo & ids_hipo)


def eqd2_dmean_rectum_gy(dose_pct: np.ndarray, rectum_mask: np.ndarray, rx_gy: float, n_fx: int) -> float:
    """EQD2 (alpha/beta=3) de la dosis MEDIA de recto. Conversion por-voxel
    (dosis total del voxel -> dosis/fraccion asumiendo fraccionamiento uniforme
    -> EQD2), promediada despues sobre el volumen de recto."""
    roi_pct = dose_pct[rectum_mask > 0]
    if len(roi_pct) == 0:
        return float("nan")
    dose_gy = roi_pct / 100.0 * rx_gy
    d_per_fx = dose_gy / n_fx
    eqd2 = dose_gy * (d_per_fx + ALPHA_BETA_RECTUM_GY) / (2.0 + ALPHA_BETA_RECTUM_GY)
    return float(eqd2.mean())


def cargar_fila(anonid: str, dataset: str) -> dict:
    processed_dir = PROCESSED_NORMO if dataset == "normo" else PROCESSED_HIPO
    arrays, meta = cargar_npz(processed_dir / f"{anonid}.npz")
    v_real = curva_v_d(arrays["dose"], arrays["rectum_mask"])
    v_media = v_real[IDX_MEDIA]
    fila = {
        "AnonID": anonid, "dataset": dataset,
        "shoulder_media_mean_pct": float(np.mean(v_media)),
        "vol_rectum_voxels": int(arrays["rectum_mask"].sum()),
        "eqd2_dmean_rectum_gy": eqd2_dmean_rectum_gy(
            arrays["dose"], arrays["rectum_mask"], RX_GY[dataset], N_FX[dataset]),
    }
    for d in D_MEDIA_POINTS:
        idx_bin = np.where(D_BINS == d)[0][0]
        fila[f"V{d}_pct"] = float(v_real[idx_bin])
    return fila


def levene_y_ratio(a: pd.Series, b: pd.Series, nombre: str) -> dict:
    """a=hipo, b=normo. Compara DISPERSION (std/var), no medias."""
    lev = stats.levene(a, b)
    return {
        "metrica": nombre,
        "std_hipo": float(a.std()), "std_normo": float(b.std()),
        "var_hipo": float(a.var()), "var_normo": float(b.var()),
        "ratio_std_hipo_vs_normo": float(a.std() / b.std()) if b.std() > 0 else float("nan"),
        "mean_hipo_NO_COMPARAR_confundido_por_fx": float(a.mean()),
        "mean_normo_NO_COMPARAR_confundido_por_fx": float(b.mean()),
        "levene_p": float(lev.pvalue),
    }


def plot_boxplots(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    datos = [df[df["dataset"] == "hipo"]["shoulder_media_mean_pct"],
             df[df["dataset"] == "normo"]["shoulder_media_mean_pct"]]
    bp = ax.boxplot(datos, tick_labels=["hipo\n(manual)", "normo\n(RapidPlan)"], widths=0.5, patch_artist=True)
    for patch, color in zip(bp["boxes"], ["firebrick", "steelblue"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    for i, d in enumerate(datos):
        x = np.full(len(d), i + 1) + np.random.default_rng(0).uniform(-0.08, 0.08, len(d))
        ax.scatter(x, d, s=20, color="black", alpha=0.6, zorder=3)
    ax.set_ylabel("shoulder banda media Rectum (%V, mean 40-80%Rx)")
    ax.set_title(f"D4 — dispersion inter-paciente pareada (n={len(datos[0])})\n"
                 "(comparar ANCHO de la caja, NO la mediana — confundida por fx)", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    ax2 = axes[1]
    datos2 = [df[df["dataset"] == "hipo"]["eqd2_dmean_rectum_gy"],
              df[df["dataset"] == "normo"]["eqd2_dmean_rectum_gy"]]
    bp2 = ax2.boxplot(datos2, tick_labels=["hipo\n(manual)", "normo\n(RapidPlan)"], widths=0.5, patch_artist=True)
    for patch, color in zip(bp2["boxes"], ["firebrick", "steelblue"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    for i, d in enumerate(datos2):
        x = np.full(len(d), i + 1) + np.random.default_rng(1).uniform(-0.08, 0.08, len(d))
        ax2.scatter(x, d, s=20, color="black", alpha=0.6, zorder=3)
    ax2.set_ylabel("EQD2 Dmean Rectum (Gy, α/β=3)")
    ax2.set_title("Secundario — dispersion en EQD2 (Gy)", fontsize=10)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("D4 — Subset pareado (mismo paciente, RapidPlan-normo vs. manual-hipo)",
                  fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / "d4_boxplot_varianza_pareada.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    overlap = ids_overlap()
    print(f"Pacientes en AMBOS datasets (normo + hipo clean): {len(overlap)}")

    filas = []
    for anonid in overlap:
        filas.append(cargar_fila(anonid, "hipo"))
        filas.append(cargar_fila(anonid, "normo"))
    df = pd.DataFrame(filas)
    df.to_csv(OUT_DIR / "d4_subset_pareado_por_paciente.csv", index=False)

    plot_boxplots(df)

    hipo = df[df["dataset"] == "hipo"].set_index("AnonID")
    normo = df[df["dataset"] == "normo"].set_index("AnonID")

    resultado_shoulder = levene_y_ratio(hipo["shoulder_media_mean_pct"], normo["shoulder_media_mean_pct"],
                                         "shoulder_media_mean_pct (%V, 40-80%Rx)")
    resultado_eqd2 = levene_y_ratio(hipo["eqd2_dmean_rectum_gy"], normo["eqd2_dmean_rectum_gy"],
                                      "eqd2_dmean_rectum_gy")

    resultados_por_punto = {}
    for d in D_MEDIA_POINTS:
        resultados_por_punto[f"V{d}_pct"] = levene_y_ratio(hipo[f"V{d}_pct"], normo[f"V{d}_pct"], f"V{d}_pct")

    def veredicto_de(r):
        if r["levene_p"] < 0.05 and r["ratio_std_hipo_vs_normo"] > 1.0:
            return "CONFIRMADA — varianza inter-paciente hipo (manual) > normo (RapidPlan), misma anatomia"
        elif r["levene_p"] < 0.05:
            return "INVERTIDA — varianza normo > hipo (inesperado, revisar)"
        else:
            return "NO SIGNIFICATIVA — con n=15 el test de Levene no distingue las varianzas"

    veredicto_principal = veredicto_de(resultado_shoulder)
    veredicto_eqd2 = veredicto_de(resultado_eqd2)

    summary = {
        "n_pareado": len(overlap),
        "ids_overlap": overlap,
        "nota_n": (
            "La tarea original estimaba '~21' pacientes en ambos datasets; el conteo real "
            f"verificado (interseccion de AnonID entre splits_v1.json completo y "
            f"splits_hipo_v2_clean_balanced.json completo, hash de HC identico entre datasets) "
            f"es {len(overlap)}. Diferencia probable: la estimacion original no descontaba los "
            "28 pacientes con contaminacion nodal excluidos del hipo 'clean', ni la posibilidad "
            "de que algunos no tengan NPZ en ambos preprocesados."
        ),
        "confusor_fraccionamiento": "70Gy/28fx (hipo) vs 78Gy/39fx (normo) — las MEDIAS no se "
                                     "interpretan (confundidas), solo la DISPERSION (std/var, test de Levene).",
        "metrica_principal_shoulder_pct_rx": resultado_shoulder,
        "veredicto_principal": veredicto_principal,
        "metrica_secundaria_eqd2_gy": resultado_eqd2,
        "veredicto_eqd2": veredicto_eqd2,
        "puntos_individuales_banda_media": resultados_por_punto,
    }
    with open(OUT_DIR / "d4_subset_pareado.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== D4 — RESUMEN ===")
    print(f"n pareado = {len(overlap)}")
    r = resultado_shoulder
    print(f"shoulder banda media: std_hipo={r['std_hipo']:.2f}  std_normo={r['std_normo']:.2f}  "
          f"ratio={r['ratio_std_hipo_vs_normo']:.2f}  Levene p={r['levene_p']:.4f}")
    print(f"  -> {veredicto_principal}")
    r2 = resultado_eqd2
    print(f"EQD2 Dmean: std_hipo={r2['std_hipo']:.2f}Gy  std_normo={r2['std_normo']:.2f}Gy  "
          f"ratio={r2['ratio_std_hipo_vs_normo']:.2f}  Levene p={r2['levene_p']:.4f}")
    print(f"  -> {veredicto_eqd2}")
    print(f"Guardado: {OUT_DIR / 'd4_subset_pareado.json'}")
    return summary


if __name__ == "__main__":
    main()
