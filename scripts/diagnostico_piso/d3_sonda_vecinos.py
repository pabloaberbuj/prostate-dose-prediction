"""
D3 — Sonda uno-a-muchos POR DATASET: ¿el piso del hombro es de GENERACION
(consistencia del planificador) o de ANATOMIA?

Puramente sobre GT (no usa el modelo, no hace inferencia). Para normo (RapidPlan,
planificador automatico, homogeneo) y para hipo (manual, planificador humano) POR
SEPARADO:
  1. 7 features geometricas escalares (las mismas de "Baseline ML clasico" en
     CLAUDE_CODE_CONTEXT: VolPTV/Rectum/Bladder_cc, Solap_PTV_Rectum/Bladder_cc,
     overlap_rel_recto/vejiga), recalculadas DESDE LAS MASCARAS del NPZ (via
     overlap_y_volumenes de analisis_angular.py — mismo patron de calibracion de
     voxel que compute_overlap_real.py, evita el bug Math.Min conocido).
  2. Para cada paciente, sus k=3 vecinos mas cercanos en ese espacio de 7
     features (estandarizado con StandardScaler, AJUSTADO POR DATASET —
     nunca mezclando normo con hipo en el mismo espacio).
  3. Grupo = paciente + sus 3 vecinos (n=4). Spread = std del %V(D) REAL de
     Rectum dentro del grupo, en banda media [40,80]%Rx, promediado sobre los
     bins de esa banda -> un escalar de "cuanto varia el DVH real entre
     anatomias casi identicas" por paciente.
  4. Comparar la distribucion de ese spread, normo vs. hipo.

Hipotesis: normo (generador consistente, RapidPlan) -> spread CHICO. hipo
(manual) -> spread GRANDE -> el piso del hombro es de CONSISTENCIA DE
GENERACION, no de anatomia (si la anatomia fuera el limite, el spread seria
similar en ambos datasets para anatomias igual de parecidas).

⚠️ Solapamiento de pacientes entre datasets (mismo AnonID = mismo hash de HC,
verificado: el hash es identico en processed/ y processed_hipo/ para el mismo
paciente): la sonda es SIEMPRE intra-dataset — el espacio de 7 features y el
KNN se calculan SEPARADO para normo y para hipo, así que un paciente presente en
ambos datasets nunca busca vecinos cruzando de un dataset al otro (los dos
universos de pacientes ni siquiera se concatenan en ningun paso de este script).
Tampoco se cuenta dos veces DENTRO de un mismo dataset (cada AnonID aparece una
sola vez en processed/ y una sola vez en processed_hipo/). El paciente SI
contribuye una vez a la cuenta de normo (con su anatomia/dosis normo) y una vez
a la de hipo (con su anatomia/dosis hipo, potencialmente distinta anatomia real
por año/preparacion) — son dos observaciones legitimas y distintas, no un
duplicado. El conteo real de la interseccion se reporta en summary.json.

Uso:
    .venv/Scripts/python.exe scripts/diagnostico_piso/d3_sonda_vecinos.py
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
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from analisis_angular import cargar_npz, overlap_y_volumenes  # noqa: E402
from dvh_curva_completa import curva_v_d, D_BINS, IDX_MEDIA  # noqa: E402

PROCESSED_NORMO = Path(r"C:\Pablo\ProstateDoseProject\processed")
PROCESSED_HIPO = Path(r"C:\Pablo\ProstateDoseProject\processed_hipo")
SPLITS_HIPO_CLEAN = _REPO_ROOT / "data/splits/splits_hipo_v2_clean_balanced.json"
SPLITS_NORMO = _REPO_ROOT / "data/splits/splits_v1.json"

OUT_DIR = _REPO_ROOT / "results/diagnostico_piso"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["VolPTV_cc", "VolRectum_cc", "VolBladder_cc",
            "Solap_PTV_Rectum_cc", "Solap_PTV_Bladder_cc",
            "overlap_rel_recto", "overlap_rel_vejiga"]
K_VECINOS = 3


def listar_ids(dataset: str) -> list:
    if dataset == "normo":
        with open(SPLITS_NORMO) as f:
            s = json.load(f)
        ids = sorted(set(s["train"]) | set(s["val"]) | set(s["test"]))
        return ids, PROCESSED_NORMO
    else:
        with open(SPLITS_HIPO_CLEAN) as f:
            s = json.load(f)
        ids = sorted(set(s["train"]) | set(s["val"]) | set(s["test"]))
        return ids, PROCESSED_HIPO


def cargar_dataset(dataset: str) -> pd.DataFrame:
    ids, processed_dir = listar_ids(dataset)
    filas = []
    for anonid in tqdm(ids, desc=f"D3 — cargando {dataset}"):
        npz_path = processed_dir / f"{anonid}.npz"
        if not npz_path.exists():
            continue
        arrays, meta = cargar_npz(npz_path)
        feats = overlap_y_volumenes(meta, arrays["ptv_mask"], arrays["rectum_mask"], arrays["bladder_mask"])
        v_real = curva_v_d(arrays["dose"], arrays["rectum_mask"])
        fila = {"AnonID": anonid, "dataset": dataset}
        fila.update(feats)
        fila["v_rectum_media_bins"] = v_real[IDX_MEDIA]
        filas.append(fila)
    return pd.DataFrame(filas)


def sonda_vecinos(df: pd.DataFrame, k: int = K_VECINOS) -> pd.DataFrame:
    X = df[FEATURES].to_numpy(dtype=float)
    valid = ~np.isnan(X).any(axis=1)
    df = df[valid].reset_index(drop=True)
    X = X[valid]
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xz)  # +1 porque incluye el propio punto
    dist, idx = nn.kneighbors(Xz)

    v_media = np.stack(df["v_rectum_media_bins"].to_numpy())  # (n_pacientes, n_bins_media)
    spreads = []
    dist_vecino_mas_cercano = []
    for i in range(len(df)):
        grupo_idx = idx[i]  # incluye a si mismo en la posicion 0
        curvas_grupo = v_media[grupo_idx]  # (k+1, n_bins)
        std_por_bin = curvas_grupo.std(axis=0)
        spreads.append(float(std_por_bin.mean()))
        dist_vecino_mas_cercano.append(float(dist[i, 1]))  # excluye self (columna 0)

    df = df.copy()
    df["spread_vecinos_banda_media_pp"] = spreads
    df["dist_al_vecino_mas_cercano_z"] = dist_vecino_mas_cercano
    return df


def plot_spread_por_dataset(df_normo: pd.DataFrame, df_hipo: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    rng = np.random.default_rng(0)
    for i, (nombre, df) in enumerate([("normo\n(RapidPlan)", df_normo), ("hipo\n(manual)", df_hipo)]):
        y = df["spread_vecinos_banda_media_pp"].to_numpy()
        x = i + rng.uniform(-0.15, 0.15, size=len(y))
        ax.scatter(x, y, s=14, alpha=0.5, color="steelblue" if i == 0 else "firebrick")
        ax.boxplot([y], positions=[i], widths=0.35, showfliers=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["normo\n(RapidPlan)", "hipo\n(manual)"])
    ax.set_ylabel("spread vecinos, banda media Rectum (pp de volumen)")
    ax.set_title("D3 — Spread uno-a-muchos (GT), por dataset", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    ax2 = axes[1]
    ax2.scatter(df_normo["dist_al_vecino_mas_cercano_z"], df_normo["spread_vecinos_banda_media_pp"],
                s=14, alpha=0.5, color="steelblue", label="normo")
    ax2.scatter(df_hipo["dist_al_vecino_mas_cercano_z"], df_hipo["spread_vecinos_banda_media_pp"],
                s=14, alpha=0.5, color="firebrick", label="hipo")
    ax2.set_xlabel("distancia (z-score) al vecino mas cercano")
    ax2.set_ylabel("spread vecinos, banda media Rectum (pp)")
    ax2.set_title("Spread vs. calidad del match anatomico", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / "d3_scatter_spread_vecinos.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    df_normo_raw = cargar_dataset("normo")
    df_hipo_raw = cargar_dataset("hipo")

    ids_normo = set(df_normo_raw["AnonID"])
    ids_hipo = set(df_hipo_raw["AnonID"])
    interseccion = sorted(ids_normo & ids_hipo)

    df_normo = sonda_vecinos(df_normo_raw)
    df_hipo = sonda_vecinos(df_hipo_raw)

    df_normo.drop(columns=["v_rectum_media_bins"]).to_csv(OUT_DIR / "d3_normo_por_paciente.csv", index=False)
    df_hipo.drop(columns=["v_rectum_media_bins"]).to_csv(OUT_DIR / "d3_hipo_por_paciente.csv", index=False)

    plot_spread_por_dataset(df_normo, df_hipo)

    s_normo = df_normo["spread_vecinos_banda_media_pp"]
    s_hipo = df_hipo["spread_vecinos_banda_media_pp"]
    u = stats.mannwhitneyu(s_hipo, s_normo, alternative="greater")
    ratio = float(s_hipo.mean() / s_normo.mean()) if s_normo.mean() > 0 else float("nan")

    if u.pvalue < 0.01 and ratio > 1.3:
        veredicto = ("CONFIRMADA — el spread hipo (manual) es sustancialmente mayor que el "
                     "spread normo (RapidPlan) entre vecinos anatomicamente casi identicos: "
                     "el piso del hombro es de CONSISTENCIA DE GENERACION del GT, no de anatomia.")
    elif u.pvalue < 0.05:
        veredicto = "CONFIRMADA (debil) — diferencia significativa pero de magnitud moderada."
    else:
        veredicto = ("NO CONFIRMADA — el spread no difiere significativamente entre datasets; "
                     "no hay evidencia de que el piso sea de consistencia de generacion via esta sonda.")

    summary = {
        "k_vecinos": K_VECINOS,
        "features_geometricas": FEATURES,
        "banda_media_pct_rx": [40, 80],
        "n_normo": int(len(df_normo)), "n_hipo": int(len(df_hipo)),
        "n_interseccion_normo_hipo": len(interseccion),
        "nota_solapamiento": (
            f"{len(interseccion)} AnonID presentes en AMBOS datasets (mismo paciente fisico, "
            "hash de HC identico verificado). La sonda es intra-dataset: el KNN de normo se "
            "calcula solo con pacientes normo, el de hipo solo con pacientes hipo — nunca se "
            "concatenan los dos espacios de features, así que un paciente en ambos NUNCA es "
            "vecino de si mismo cruzando dataset. Contribuye una observacion (anatomia+dosis "
            "propias de ese dataset) a cada KNN por separado — no es un duplicado dentro de "
            "ningun calculo individual."
        ),
        "spread_normo": {"mean": float(s_normo.mean()), "std": float(s_normo.std()),
                          "median": float(s_normo.median()), "n": int(len(s_normo))},
        "spread_hipo": {"mean": float(s_hipo.mean()), "std": float(s_hipo.std()),
                         "median": float(s_hipo.median()), "n": int(len(s_hipo))},
        "ratio_hipo_vs_normo": ratio,
        "mannwhitney_p_hipo_mayor": float(u.pvalue),
        "veredicto": veredicto,
    }
    with open(OUT_DIR / "d3_sonda_vecinos.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== D3 — RESUMEN ===")
    print(f"normo: spread={s_normo.mean():.2f}±{s_normo.std():.2f}pp (n={len(s_normo)})")
    print(f"hipo:  spread={s_hipo.mean():.2f}±{s_hipo.std():.2f}pp (n={len(s_hipo)})")
    print(f"ratio hipo/normo={ratio:.2f}  Mann-Whitney p={u.pvalue:.4g}")
    print(f"Interseccion de pacientes entre datasets: {len(interseccion)}")
    print(f"-> {veredicto}")
    print(f"Guardado: {OUT_DIR / 'd3_sonda_vecinos.json'}")
    return summary


if __name__ == "__main__":
    main()
