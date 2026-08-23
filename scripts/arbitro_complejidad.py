"""
Parte B — ARBITRO de complejidad: usa las 3 metricas VMAT extraidas del RP real
(scripts/extraer_complejidad_rp.py) para decidir si el piso del hombro medio-bajo
de OAR (results/diagnostico_piso/summary.json, hallazgo D3) es de GENERACION
(agresividad de plan) o de ANATOMIA no capturada por las 7 features geometricas.

⚠️ ALCANCE REDUCIDO A NORMO (decision de Pablo, 2026-08-19): los RP de hipofx no
estan disponibles localmente (dicom_pilot/ solo tiene 7-8 casos de piloto, muy
lejos de los n=151 que D3 uso para hipo) y el arbitraje por vecinos necesita
cientos de pacientes para tener sentido estadistico. Esto tiene una consecuencia
IMPORTANTE para la interpretacion: D3 encontro que el spread de DVH entre vecinos
es CHICO en normo (RapidPlan, generador consistente) y GRANDE en hipo (manual,
generador idiosincratico) -- la pregunta de fondo (¿el piso de hipo es de
generacion o anatomia?) NO se puede responder directamente sin RP de hipo. Lo que
este script SI puede hacer es una prueba mas debil pero honesta: dentro de normo
(donde ya sabemos que el spread de DVH es chico), ¿la variacion RESIDUAL de
complejidad de plan explica la poca variacion de DVH que SI queda entre vecinos
anatomicamente casi identicos? Un resultado positivo aca es evidencia indirecta
a favor del mecanismo (planificador mas/menos agresivo -> mas/menos DVH), pero
NO reemplaza la comparacion directa manual-hipo vs. RapidPlan-normo que pedia la
tarea original (Validacion 1 y 2, bloqueadas por falta de RP de hipo -- ver
summary.json, campos "validacion_1_bloqueada"/"validacion_2_bloqueada").

Metodologia (complejidad RESIDUAL, no cruda):
  1. Por cada una de las 3 metricas de complejidad (MU_factor, MCSv, SAS10):
     regresion lineal metrica ~ 7 features geometricas de D3 (VolPTV/Rectum/
     Bladder_cc, Solap_PTV_Rectum/Bladder_cc, overlap_rel_recto/vejiga) -> el
     residuo es la parte de complejidad NO explicada por la anatomia ya
     matcheada en D3.
  2. Reusa el KNN de D3 (k=3, mismo espacio de 7 features estandarizado, mismo
     codigo de scripts/diagnostico_piso/d3_sonda_vecinos.py) para tener los
     MISMOS grupos de vecinos anatomicos que D3 uso -- no se recalculan vecinos
     en un espacio distinto.
  3. Para cada par (paciente, vecino) donde AMBOS tienen RP parseado: Spearman
     entre |Δcomplejidad-residual| y |ΔDVH-banda-media(40-80%Rx) real de Rectum|
     (V media de la banda, mismo indicador escalar que D3/D4 usan). Se reporta
     tambien la correlacion con complejidad CRUDA al lado (mismo par, mismo
     Δmetrica pero sin restar la anatomia) para mostrar cuanto del efecto ya
     esta en D3/D4 como "confundido por anatomia" vs. lo que aporta esto de mas.

Uso:
    .venv/Scripts/python.exe scripts/arbitro_complejidad.py
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
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "diagnostico_piso"))

from d3_sonda_vecinos import cargar_dataset, FEATURES, K_VECINOS  # noqa: E402

COMPLEJIDAD_CSV = _REPO_ROOT / "results/complejidad_arbitro/data/complejidad_rp.csv"

OUT_DIR = _REPO_ROOT / "results/complejidad_arbitro"
PLOTS_DIR = OUT_DIR / "plots"
DATA_DIR = OUT_DIR / "data"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

METRICAS = ["MU_factor", "MCSv", "SAS10"]


def knn_grupos(df_anatomia: pd.DataFrame, k: int = K_VECINOS) -> np.ndarray:
    """Mismo codigo que d3_sonda_vecinos.sonda_vecinos, pero devuelve los INDICES
    de vecinos (no solo el spread agregado) para poder armar pares (i,j)."""
    X = df_anatomia[FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xz)
    _, idx = nn.kneighbors(Xz)
    return idx  # idx[i] = [i, vecino1, vecino2, vecino3]


def residualizar(df: pd.DataFrame, metrica: str) -> np.ndarray:
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[metrica].to_numpy(dtype=float)
    modelo = LinearRegression().fit(X, y)
    pred = modelo.predict(X)
    return y - pred, float(modelo.score(X, y))


def pares_unicos_con_complejidad(idx: np.ndarray, tiene_complejidad: np.ndarray) -> list:
    """De los grupos de vecinos de D3 (paciente + 3 vecinos), arma los pares
    (i, vecino) UNICOS (sin duplicar (i,j) y (j,i)) restringidos a los pacientes
    que efectivamente tienen RP parseado (tiene_complejidad[i] y [j] True)."""
    pares = set()
    for i in range(len(idx)):
        if not tiene_complejidad[i]:
            continue
        for v in idx[i][1:]:
            if tiene_complejidad[v]:
                pares.add(frozenset((i, v)))
    return [tuple(p) for p in pares if len(p) == 2]


def plot_scatter_residual_vs_dvh(resultados: dict):
    fig, axes = plt.subplots(1, len(METRICAS), figsize=(5 * len(METRICAS), 4.5))
    for ax, metrica in zip(axes, METRICAS):
        r = resultados[metrica]
        ax.scatter(r["_delta_residual"], r["_delta_dvh"], s=16, alpha=0.5, color="steelblue")
        ax.set_xlabel(f"|Δ{metrica} residual|")
        ax.set_ylabel("|ΔDVH banda media Rectum| (pp)")
        ax.set_title(f"{metrica}\nspearman={r['spearman_residual']['rho']:.2f} "
                     f"(p={r['spearman_residual']['p']:.3g})", fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Árbitro — Δcomplejidad RESIDUAL vs ΔDVH entre vecinos anatómicos (normo)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / "arbitro_scatter_residual_vs_dvh.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    df_complejidad = pd.read_csv(COMPLEJIDAD_CSV)
    df_complejidad = df_complejidad[df_complejidad["dataset"] == "normo"]

    df_anatomia = cargar_dataset("normo")  # AnonID, dataset, 7 features, v_rectum_media_bins
    X = df_anatomia[FEATURES].to_numpy(dtype=float)
    valid = ~np.isnan(X).any(axis=1)
    df_anatomia = df_anatomia[valid].reset_index(drop=True)

    idx_vecinos = knn_grupos(df_anatomia, k=K_VECINOS)

    df = df_anatomia.merge(df_complejidad[["AnonID"] + METRICAS + ["NArcos", "MU_total"]],
                            on="AnonID", how="left")
    tiene_complejidad = df[METRICAS].notna().all(axis=1).to_numpy()

    v_media = np.stack(df["v_rectum_media_bins"].to_numpy())
    v_media_mean = v_media.mean(axis=1)  # escalar por paciente, igual que D3

    n_total_anatomia = len(df)
    n_con_complejidad = int(tiene_complejidad.sum())
    print(f"Pacientes con anatomia (cohorte D3 normo): {n_total_anatomia}")
    print(f"De esos, con RP parseado OK: {n_con_complejidad}")

    pares = pares_unicos_con_complejidad(idx_vecinos, tiene_complejidad)
    print(f"Pares (paciente, vecino-D3) unicos con complejidad en AMBOS: {len(pares)}")

    resultados = {}
    for metrica in METRICAS:
        residual, r2_anatomia = residualizar(df.loc[tiene_complejidad], metrica)
        # mapear el residuo de vuelta al indice completo de df (NaN donde no hay complejidad)
        residual_full = np.full(len(df), np.nan)
        residual_full[np.where(tiene_complejidad)[0]] = residual
        cruda_full = df[metrica].to_numpy(dtype=float)

        d_residual = np.array([abs(residual_full[i] - residual_full[j]) for i, j in pares])
        d_cruda = np.array([abs(cruda_full[i] - cruda_full[j]) for i, j in pares])
        d_dvh = np.array([abs(v_media_mean[i] - v_media_mean[j]) for i, j in pares])

        rho_res, p_res = stats.spearmanr(d_residual, d_dvh)
        rho_cru, p_cru = stats.spearmanr(d_cruda, d_dvh)

        resultados[metrica] = {
            "r2_metrica_explicado_por_anatomia": r2_anatomia,
            "n_pares": len(pares),
            "spearman_residual": {"rho": float(rho_res), "p": float(p_res)},
            "spearman_cruda": {"rho": float(rho_cru), "p": float(p_cru)},
            "_delta_residual": d_residual, "_delta_cruda": d_cruda, "_delta_dvh": d_dvh,
        }
        print(f"\n{metrica}: R2(anatomia)={r2_anatomia:.3f}  "
              f"spearman(residual,dDVH)={rho_res:.3f} (p={p_res:.3g})  "
              f"spearman(cruda,dDVH)={rho_cru:.3f} (p={p_cru:.3g})")

    plot_scatter_residual_vs_dvh(resultados)

    def veredicto_metrica(r):
        if r["spearman_residual"]["p"] < 0.05 and r["spearman_residual"]["rho"] > 0:
            return ("ARBITRO A FAVOR DE GENERACION — la complejidad residual (no explicada "
                    "por anatomia) SI correlaciona con el spread de DVH entre vecinos.")
        elif r["spearman_residual"]["p"] < 0.05 and r["spearman_residual"]["rho"] < 0:
            return "CORRELACION NEGATIVA SIGNIFICATIVA — inesperado, revisar."
        else:
            return ("NO SIGNIFICATIVO — con este n, la complejidad residual de plan no explica "
                    "el spread de DVH entre vecinos anatomicos (dentro de normo).")

    summary = {
        "alcance": "SOLO normofx (ver docstring) -- Validacion 1 y 2 de la tarea original "
                   "(pareado manual-hipo vs RapidPlan-normo, terciles de complejidad en hipo) "
                   "requieren RP de hipofx, NO DISPONIBLES localmente. Bloqueadas.",
        "validacion_1_bloqueada": "requiere RP de hipofx (no disponibles) -- no ejecutada",
        "validacion_2_bloqueada": "requiere RP de hipofx (no disponibles) -- no ejecutada",
        "n_pacientes_cohorte_d3_normo": n_total_anatomia,
        "n_pacientes_con_rp_parseado": n_con_complejidad,
        "n_pares_vecinos_evaluables": len(pares),
        "resultados_por_metrica": {
            m: {k: v for k, v in r.items() if not k.startswith("_")}
            for m, r in resultados.items()
        },
        "veredicto_por_metrica": {m: veredicto_metrica(r) for m, r in resultados.items()},
    }
    with open(DATA_DIR.parent / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nGuardado: {DATA_DIR.parent / 'summary.json'}")
    return summary


if __name__ == "__main__":
    main()
