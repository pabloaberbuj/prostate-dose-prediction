"""
Figura para charla — PISO DE CONSISTENCIA DE GENERACION (D3, sonda uno-a-muchos, hipo).

Puramente de VISUALIZACION: no entrena nada, no corre analisis nuevo. Reusa
directamente el resultado ya publicado de D3 (results/diagnostico_piso/d3_hipo_por_paciente.csv,
d3_sonda_vecinos.json) para elegir un grupo (indice + 3 vecinos KNN de las mismas 7
features geometricas) que ilustre "misma anatomia, dosis real distinta" en la banda
media [40,80]%Rx del recto.

⚠️ Nota de datos: `processed_hipo/` (usado por d3_sonda_vecinos.py) ya no existe en disco
(fue reemplazado durante el fix de CT, ver memoria project_ct_channel_corrupted_wrong_series /
project_ctfix_fov_resolution_change). Los 7 features + el spread por paciente en el CSV de D3
son el resultado YA PUBLICADO (no se recalculan aca). Para las imagenes (CT/mascaras/dosis) se
usa `processed_hipo_ctfix/` (149/151 pacientes de D3 disponibles ahi) — el fix de CT solo
corrigio el canal CT; dosis y formas de mascara son "fisicamente consistentes" con la version
vieja (ver memoria), asi que el DVH real recalculado aca desde ctfix es una reproduccion fiel del
mismo dato, no un analisis distinto. La identidad de vecinos (KNN) se reproduce desde las 7
features YA GUARDADAS en el CSV de D3 (no se re-extraen de mascaras), preservando exactamente el
mismo grafo de vecinos que uso D3.

Alineacion espacial entre pacientes (para poder computar Dice y elegir 1 corte comun):
cada NPZ de processed_hipo_ctfix ya esta recortado a una ventana XY de 500x500mm CENTRADA en el
centroide del PTV de ESE paciente (`crop_lado_mm`, `centroide_ptv_xy_mm`, confirmado constante
en 149/149 pacientes: centroide PTV cae siempre en (~127.8,~127.8) de 256x256). Es decir, el
plano XY YA es un sistema de coordenadas comun (offset en mm respecto del centroide del PTV
propio) entre pacientes — no hace falta registrar/resamplear en XY. Solo Z se alinea
manualmente aqui (centroide Z del PTV propio como origen, spacing_mm[2]=3.0mm en el 100% de la
muestra).

Uso:
    .venv/Scripts/python.exe scripts/diagnostico_piso/figura_piso_generacion.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from dvh_curva_completa import curva_v_d, D_BINS, IDX_MEDIA  # noqa: E402

PROCESSED_HIPO_CTFIX = Path(r"C:\Pablo\ProstateDoseProject\processed_hipo_ctfix")
D3_CSV = _REPO_ROOT / "results/diagnostico_piso/d3_hipo_por_paciente.csv"
COMPLEJIDAD_CSV = _REPO_ROOT / "results/complejidad_arbitro/data/complejidad_rp.csv"

OUT_DIR = _REPO_ROOT / "results/figura_piso_generacion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["VolPTV_cc", "VolRectum_cc", "VolBladder_cc",
            "Solap_PTV_Rectum_cc", "Solap_PTV_Bladder_cc",
            "overlap_rel_recto", "overlap_rel_vejiga"]
K_VECINOS = 3

DICE_PTV_MIN = 0.70
# El umbral sugerido por la tarea (0.60) resulto INALCANZABLE escaneando los 146
# candidatos disponibles (max Dice_recto par-a-par en TODO el dataset = 0.582,
# ver results/figura_piso_generacion/diagnostico_todos_los_candidatos.json) —
# el recto es un tubo largo cuya forma/posicion varia mas entre "vecinos"
# escalares de lo que un blob como el PTV varia. Se releja a 0.55 (tope real
# ~0.58), documentado explicitamente en la salida — no se fuerza el umbral
# original de forma artificial.
DICE_RECTO_MIN = 0.55
Z_OFFSET_RANGE = range(-40, 41)   # indices de slice relativos al centroide Z del PTV propio (~3mm/paso => ±120mm)


# ──────────────────────────────────────────────────────────────────────────────
# 1) Reproducir el grafo de vecinos EXACTO de D3 desde el CSV ya publicado
# ──────────────────────────────────────────────────────────────────────────────

def cargar_df_d3():
    df = pd.read_csv(D3_CSV).sort_values("AnonID").reset_index(drop=True)
    ctfix_ids = set(p.stem for p in PROCESSED_HIPO_CTFIX.glob("*.npz"))
    df["disponible_ctfix"] = df["AnonID"].isin(ctfix_ids)
    return df, ctfix_ids


def knn_grupos(df: pd.DataFrame, k: int = K_VECINOS):
    X = df[FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xz)
    dist, idx = nn.kneighbors(Xz)
    return idx, dist  # idx[i,0]==i (self), idx[i,1:] = 3 vecinos


# ──────────────────────────────────────────────────────────────────────────────
# 2) Datos por paciente (masks/dose/ct) + alineacion Z relativa al centroide PTV
# ──────────────────────────────────────────────────────────────────────────────

_CACHE_PACIENTES = {}


def get_array(pac: dict, key: str) -> np.ndarray:
    """Carga perezosa por-key (evita traer ct/dose/psdm a memoria para los
    ~40 candidatos que solo necesitan las 3 mascaras para el filtro Dice)."""
    if key not in pac["_arr_cache"]:
        dt = np.uint8 if "mask" in key else np.float32
        pac["_arr_cache"][key] = np.array(pac["_data"][key], dtype=dt)
    return pac["_arr_cache"][key]


def cargar_paciente(anonid: str):
    if anonid in _CACHE_PACIENTES:
        return _CACHE_PACIENTES[anonid]
    data = np.load(str(PROCESSED_HIPO_CTFIX / f"{anonid}.npz"), allow_pickle=True)
    meta = json.loads(str(data["meta"][0]))
    pac = {"anonid": anonid, "_data": data, "_arr_cache": {}, "meta": meta}
    ptv = get_array(pac, "ptv_mask")
    zs = np.nonzero(ptv.sum(axis=(1, 2)))[0]
    pac["z_centroid"] = float(zs.mean()) if len(zs) else ptv.shape[0] / 2.0
    pac["n_slices"] = ptv.shape[0]
    _CACHE_PACIENTES[anonid] = pac
    return pac


def slice_en_offset(pac: dict, key: str, offset: int) -> np.ndarray:
    """Devuelve el slice 2D (256x256) de arrays[key] en el indice
    round(z_centroid_ptv + offset); fuera de rango -> ceros (estructura no
    capturada en el crop Z de ese paciente en esa posicion relativa)."""
    idx = int(round(pac["z_centroid"] + offset))
    arr = get_array(pac, key)
    if 0 <= idx < arr.shape[0]:
        return arr[idx]
    return np.zeros(arr.shape[1:], dtype=arr.dtype)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    s = a.sum() + b.sum()
    if s == 0:
        return float("nan")
    return float(2.0 * (a & b).sum() / s)


def volumen_offset(pac: dict, key: str) -> np.ndarray:
    return np.stack([slice_en_offset(pac, key, off) for off in Z_OFFSET_RANGE], axis=0)


def dice_pairwise_grupo(pacientes: list, key: str) -> dict:
    vols = [volumen_offset(p, key) for p in pacientes]
    n = len(pacientes)
    pares = []
    for i in range(n):
        for j in range(i + 1, n):
            pares.append(dice(vols[i], vols[j]))
    pares = [d for d in pares if not np.isnan(d)]
    return {
        "min": float(min(pares)) if pares else float("nan"),
        "mean": float(np.mean(pares)) if pares else float("nan"),
        "pares": pares,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3) Seleccion de candidatos: ranking por spread D3 ya publicado + filtro Dice
# ──────────────────────────────────────────────────────────────────────────────

def evaluar_candidato(df, idx_matriz, i, ctfix_ids):
    grupo_idx = idx_matriz[i]  # [self, v1, v2, v3]
    grupo_ids = df.loc[grupo_idx, "AnonID"].tolist()
    if not all(a in ctfix_ids for a in grupo_ids):
        return None

    pacientes = [cargar_paciente(a) for a in grupo_ids]

    # Filtro Dice primero (solo mascaras) — evita cargar dose/ct para candidatos
    # que ya van a descartarse, manteniendo la memoria acotada.
    dice_ptv = dice_pairwise_grupo(pacientes, "ptv_mask")
    dice_recto = dice_pairwise_grupo(pacientes, "rectum_mask")
    pasa_filtro = (dice_ptv["min"] >= DICE_PTV_MIN) and (dice_recto["min"] >= DICE_RECTO_MIN)

    if not pasa_filtro:
        return {
            "anonid_indice": grupo_ids[0], "grupo_ids": grupo_ids, "pacientes": pacientes,
            "spread_csv_original": float(df.loc[i, "spread_vecinos_banda_media_pp"]),
            "dice_ptv": dice_ptv, "dice_recto": dice_recto,
            "dice_vejiga": {"min": float("nan"), "mean": float("nan"), "pares": []},
            "pasa_filtro_dice": False, "spread_recalculado_ctfix": float("nan"),
        }

    dice_vejiga = dice_pairwise_grupo(pacientes, "bladder_mask")

    # DVH real completo (todas las bandas) del recto, recalculado desde ctfix
    curvas = np.stack([curva_v_d(get_array(p, "dose"), get_array(p, "rectum_mask")) for p in pacientes])
    std_por_bin_media = curvas[:, IDX_MEDIA].std(axis=0)
    spread_recalculado = float(std_por_bin_media.mean())
    media_por_paciente = curvas[:, IDX_MEDIA].mean(axis=1)  # (4,) valor "banda media" propio de c/u

    return {
        "anonid_indice": grupo_ids[0],
        "grupo_ids": grupo_ids,
        "pacientes": pacientes,
        "curvas_dvh_recto": curvas,
        "spread_csv_original": float(df.loc[i, "spread_vecinos_banda_media_pp"]),
        "spread_recalculado_ctfix": spread_recalculado,
        "media_banda_por_paciente": media_por_paciente.tolist(),
        "dice_ptv": dice_ptv, "dice_recto": dice_recto, "dice_vejiga": dice_vejiga,
        "pasa_filtro_dice": bool(pasa_filtro),
    }


def elegir_grupos(df, idx_matriz, ctfix_ids, n_candidatos_a_revisar=200):
    orden = df["spread_vecinos_banda_media_pp"].to_numpy().argsort()[::-1]
    resultados = []
    for rank, i in enumerate(orden):
        if rank >= n_candidatos_a_revisar:
            break
        r = evaluar_candidato(df, idx_matriz, i, ctfix_ids)
        if r is None:
            continue
        r["rank_por_spread_csv"] = int(rank)
        resultados.append(r)

    pasan = [r for r in resultados if r["pasa_filtro_dice"]]
    pasan.sort(key=lambda r: r["spread_recalculado_ctfix"], reverse=True)
    return pasan, resultados


# ──────────────────────────────────────────────────────────────────────────────
# 4) Eleccion del corte Z comun
# ──────────────────────────────────────────────────────────────────────────────

def elegir_z_comun(pacientes: list):
    filas = []
    for off in Z_OFFSET_RANGE:
        ptv_slices = [slice_en_offset(p, "ptv_mask", off) for p in pacientes]
        recto_slices = [slice_en_offset(p, "rectum_mask", off) for p in pacientes]
        dose_slices = [slice_en_offset(p, "dose", off) for p in pacientes]

        # el align_score (Dice de PTV|Recto) puede pasar el umbral aun si a UN
        # paciente el PTV le da vacio en este offset (el recto solo ya alcanza
        # para "explicar" el Dice combinado) — eso se vio en la figura (un
        # paciente sin contorno PTV visible). Se exige ademas que TODOS tengan
        # PTV no-vacio en el offset elegido.
        ptv_presente_en_todos = all(ps.sum() > 0 for ps in ptv_slices)

        combo = [((ps > 0) | (rs > 0)).astype(np.uint8) for ps, rs in zip(ptv_slices, recto_slices)]
        n = len(combo)
        pares_align = []
        for a in range(n):
            for b in range(a + 1, n):
                pares_align.append(dice(combo[a], combo[b]))
        pares_align = [d for d in pares_align if not np.isnan(d)]
        align_score = float(np.mean(pares_align)) if pares_align else 0.0

        dosis_media_recto = []
        for rs, ds in zip(recto_slices, dose_slices):
            if rs.sum() > 0:
                dosis_media_recto.append(float(ds[rs > 0].mean()))
        if len(dosis_media_recto) == n:
            diff_score = float(np.std(dosis_media_recto))
        else:
            diff_score = float("nan")

        filas.append({"offset": off, "align_score": align_score, "diff_score": diff_score,
                       "ptv_presente_en_todos": ptv_presente_en_todos,
                       "dosis_media_recto_por_paciente": dosis_media_recto if len(dosis_media_recto) == n else None})

    validos = [f for f in filas if not np.isnan(f["diff_score"]) and f["align_score"] >= 0.5 and f["ptv_presente_en_todos"]]
    if not validos:
        validos = [f for f in filas if not np.isnan(f["diff_score"]) and f["ptv_presente_en_todos"]]
    if not validos:
        validos = [f for f in filas if not np.isnan(f["diff_score"])]
    elegido = max(validos, key=lambda f: f["diff_score"])
    return elegido, filas


# ──────────────────────────────────────────────────────────────────────────────
# 5) Figuras (grilla CT+dosis, y DVH aparte)
# ──────────────────────────────────────────────────────────────────────────────

COLOR_INDICE = "black"
PALETA_VECINOS = ["steelblue", "darkorange", "firebrick", "seagreen", "purple"]


def etiquetas_y_colores(n_pacientes: int):
    etiquetas = [f"paciente {i}" for i in range(1, n_pacientes + 1)]
    colores = [COLOR_INDICE] + PALETA_VECINOS[: n_pacientes - 1]
    return etiquetas, colores


def quitar_filas(candidato: dict, posiciones: list) -> dict:
    """Devuelve una copia del candidato sin las filas en `posiciones` (0=indice,
    nunca se debe quitar), con Dice/spread/DVH RECALCULADOS para el subgrupo
    resultante (no se reusan los valores del grupo de 4)."""
    keep = [i for i in range(len(candidato["grupo_ids"])) if i not in posiciones]
    grupo_ids = [candidato["grupo_ids"][i] for i in keep]
    pacientes = [candidato["pacientes"][i] for i in keep]
    curvas = candidato["curvas_dvh_recto"][keep, :]

    dice_ptv = dice_pairwise_grupo(pacientes, "ptv_mask")
    dice_recto = dice_pairwise_grupo(pacientes, "rectum_mask")
    dice_vejiga = dice_pairwise_grupo(pacientes, "bladder_mask")
    std_por_bin_media = curvas[:, IDX_MEDIA].std(axis=0)
    spread_recalculado = float(std_por_bin_media.mean())
    media_por_paciente = curvas[:, IDX_MEDIA].mean(axis=1)

    return {
        **candidato,
        "grupo_ids": grupo_ids, "pacientes": pacientes, "curvas_dvh_recto": curvas,
        "dice_ptv": dice_ptv, "dice_recto": dice_recto, "dice_vejiga": dice_vejiga,
        "spread_recalculado_ctfix": spread_recalculado,
        "media_banda_por_paciente": media_por_paciente.tolist(),
        "filas_originales_removidas": posiciones,
    }


def construir_figura(candidato: dict, z_info: dict, out_path: Path, complejidad_df: dict):
    """Grilla CT+contornos / Dosis real — SIN el panel de DVH (va en figura aparte,
    ver construir_dvh_figura), para que cada imagen no quede desproporcionada."""
    pacientes = candidato["pacientes"]
    grupo_ids = candidato["grupo_ids"]
    offset = z_info["offset"]
    media_banda = candidato["media_banda_por_paciente"]
    n = len(pacientes)
    etiquetas, colores = etiquetas_y_colores(n)

    dose_slices = [slice_en_offset(p, "dose", offset) for p in pacientes]
    ct_slices = [slice_en_offset(p, "ct", offset) for p in pacientes]
    ptv_slices = [slice_en_offset(p, "ptv_mask", offset) for p in pacientes]
    recto_slices = [slice_en_offset(p, "rectum_mask", offset) for p in pacientes]
    vejiga_slices = [slice_en_offset(p, "bladder_mask", offset) for p in pacientes]

    vmax_dose = max(float(d.max()) for d in dose_slices)
    vmax_dose = max(vmax_dose, 100.0)

    fig = plt.figure(figsize=(9.5, 4.3 * n))
    gs = GridSpec(n, 2, wspace=0.06, hspace=0.16, figure=fig)

    im_dose_ref = None
    for i, (pid, ct, dose, ptv, recto, vejiga) in enumerate(
            zip(grupo_ids, ct_slices, dose_slices, ptv_slices, recto_slices, vejiga_slices)):
        ax_ct = fig.add_subplot(gs[i, 0])
        ax_ct.imshow(ct, cmap="gray", vmin=-1, vmax=1)
        if ptv.sum() > 0:
            ax_ct.contour(ptv, levels=[0.5], colors="gold", linewidths=1.6)
        if recto.sum() > 0:
            ax_ct.contour(recto, levels=[0.5], colors="chocolate", linewidths=1.6)
        if vejiga.sum() > 0:
            ax_ct.contour(vejiga, levels=[0.5], colors="deepskyblue", linewidths=1.6)
        ax_ct.set_xticks([]); ax_ct.set_yticks([])
        etiqueta_mcsv = complejidad_df.get(pid, "N/D (sin RP hipofx)")
        ax_ct.set_ylabel(f"{etiquetas[i]}\n{pid}", fontsize=8, color=colores[i], fontweight="bold")
        if i == 0:
            ax_ct.set_title("CT + contornos\n(PTV oro / Recto marron / Vejiga celeste)", fontsize=9)

        ax_d = fig.add_subplot(gs[i, 1])
        im = ax_d.imshow(dose, cmap="jet", vmin=0, vmax=vmax_dose)
        cs = ax_d.contour(dose, levels=[40, 60, 80], colors=["white", "white", "white"],
                           linewidths=[1.0, 1.3, 1.6], linestyles=["dotted", "dashed", "solid"])
        ax_d.clabel(cs, inline=True, fontsize=6, fmt="%d%%Rx")
        if recto.sum() > 0:
            ax_d.contour(recto, levels=[0.5], colors="chocolate", linewidths=1.3)
        if ptv.sum() > 0:
            ax_d.contour(ptv, levels=[0.5], colors="gold", linewidths=1.0)
        ax_d.set_xticks([]); ax_d.set_yticks([])
        if i == 0:
            ax_d.set_title("Dosis REAL (%Rx)\nisodosis 40/60/80%Rx", fontsize=9)
        im_dose_ref = im

        txt = f"DVH banda media Recto [40-80%Rx]: {media_banda[i]:.1f} %vol\nMCSv (RP): {etiqueta_mcsv}"
        ax_d.text(1.05, 0.5, txt, transform=ax_d.transAxes, fontsize=7.5, va="center",
                   bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))

    fig.subplots_adjust(bottom=0.09)
    cbar_ax = fig.add_axes([0.15, 0.035, 0.4, 0.012])
    fig.colorbar(im_dose_ref, cax=cbar_ax, orientation="horizontal", label=f"Dosis (%Rx) — escala comun a los {n}")

    fig.suptitle(
        f"Piso de consistencia de generacion (D3, hipo) — grupo indice {grupo_ids[0]}\n"
        f"spread banda media recalculado={candidato['spread_recalculado_ctfix']:.2f} pp | "
        f"Dice_PTV(min/mean)={candidato['dice_ptv']['min']:.2f}/{candidato['dice_ptv']['mean']:.2f} | "
        f"Dice_Recto(min/mean)={candidato['dice_recto']['min']:.2f}/{candidato['dice_recto']['mean']:.2f} | "
        f"corte z_offset={offset} (align={z_info['align_score']:.2f}, diff_dosis_recto_std={z_info['diff_score']:.2f}pp)",
        fontsize=10, y=0.995)

    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def construir_dvh_figura(candidato: dict, out_path: Path):
    """DVH del recto de los N pacientes del grupo, en figura propia (no
    embebida en la grilla) para que se lea bien en la charla."""
    grupo_ids = candidato["grupo_ids"]
    curvas = candidato["curvas_dvh_recto"]
    n = len(grupo_ids)
    etiquetas, colores = etiquetas_y_colores(n)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.axvspan(40, 80, color="steelblue", alpha=0.10, label="banda media [40,80]%Rx")
    for i, aid in enumerate(grupo_ids):
        ax.plot(D_BINS, curvas[i], color=colores[i], lw=2.0, label=f"{etiquetas[i]} — {aid}")
    ax.set_xlabel("Dosis (% Rx)")
    ax.set_ylabel("Volumen Rectum (%)")
    ax.set_title(f"DVH real del Recto — grupo indice {grupo_ids[0]}\n"
                  f"(anatomia casi identica, banda media sombreada)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 102)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def cargar_complejidad_hipo(grupo_ids):
    if not COMPLEJIDAD_CSV.exists():
        return {a: "N/D (sin RP hipofx)" for a in grupo_ids}
    df = pd.read_csv(COMPLEJIDAD_CSV)
    out = {}
    for a in grupo_ids:
        fila = df[df["AnonID"] == a]
        if len(fila) == 0:
            out[a] = "N/D (sin RP hipofx)"
        else:
            out[a] = f"{fila.iloc[0]['MCSv']:.3f}"
    return out


def resumen_candidato_para_json(candidato: dict, z_info: dict, df_features: pd.DataFrame):
    grupo_ids = candidato["grupo_ids"]
    etiquetas, _ = etiquetas_y_colores(len(grupo_ids))
    features_tabla = df_features.set_index("AnonID").loc[grupo_ids, FEATURES].round(3).to_dict(orient="index")
    return {
        "anonid_indice": candidato["anonid_indice"],
        "grupo_ids": grupo_ids,
        "etiquetas": etiquetas,
        "filas_originales_removidas_del_grupo_knn_de_4": candidato.get("filas_originales_removidas", []),
        "features_geometricas": features_tabla,
        "dvh_banda_media_recto_pct_vol": {aid: round(v, 2) for aid, v in zip(grupo_ids, candidato["media_banda_por_paciente"])},
        "spread_banda_media_grupo": {
            "valor_original_d3_csv_pp": round(candidato["spread_csv_original"], 3),
            "valor_recalculado_desde_ctfix_pp": round(candidato["spread_recalculado_ctfix"], 3),
            "nota": "el original viene del CSV ya publicado de D3 (grid pre-fix-CT, hoy borrado de disco); "
                    "el recalculado usa processed_hipo_ctfix (misma dosis/mascaras, FOV/resolucion "
                    "distinta pero 'fisicamente consistente' segun memoria project_ctfix_fov_resolution_change) "
                    "— se reportan ambos, deberian ser cercanos.",
        },
        "dice_interpaciente": {
            "PTV": candidato["dice_ptv"], "Rectum": candidato["dice_recto"], "Bladder": candidato["dice_vejiga"],
            "umbral_aplicado": {"Dice_PTV_min": DICE_PTV_MIN, "Dice_Recto_min": DICE_RECTO_MIN},
            "pasa_filtro": candidato["pasa_filtro_dice"],
        },
        "corte_z_elegido": {
            "offset_relativo_a_centroide_ptv_propio_slices": z_info["offset"],
            "criterio": (
                "para cada offset entero en [-40,40] (relativo al centroide Z del PTV de CADA "
                "paciente, spacing_z=3.0mm => rango ±120mm), se calculo (a) align_score = Dice "
                "promedio par-a-par de (PTV|Recto) entre los 4 pacientes en ese offset [mide "
                "cuan bien se superponen las mascaras en ese corte], y (b) diff_score = std "
                "entre los 4 de la dosis media dentro del recto en ese corte [mide cuanto "
                "difiere la dosis real ahi]. Se filtro a offsets con align_score>=0.5 y se "
                "eligio el que MAXIMIZA diff_score (mas contraste de dosis real, sujeto a "
                "anatomia bien alineada)."
            ),
            "align_score": round(z_info["align_score"], 3),
            "diff_score_dosis_recto_pp": round(z_info["diff_score"], 3),
            "dosis_media_recto_por_paciente_pctRx": {
                aid: round(v, 2) for aid, v in zip(grupo_ids, z_info["dosis_media_recto_por_paciente"])
            } if z_info["dosis_media_recto_por_paciente"] else None,
        },
        "complejidad_plan_MCSv": {
            "valores": cargar_complejidad_hipo(grupo_ids),
            "nota": "complejidad_rp.csv (arbitro de complejidad) NO tiene filas de dataset='hipo' — "
                    "solo se pudo extraer MU_factor/MCSv/SAS10 para normo (RP no disponibles para "
                    "hipofx, ver memoria project_p1_ml_clasico_status / seccion arbitro de "
                    "CLAUDE_CODE_CONTEXT.md). No se insinua agresividad de plan para el grupo hipo "
                    "por esta via — limitacion de datos, no un resultado nulo.",
        },
    }


def main():
    df, ctfix_ids = cargar_df_d3()
    idx_matriz, _ = knn_grupos(df, k=K_VECINOS)

    pasan, revisados = elegir_grupos(df, idx_matriz, ctfix_ids)

    diag_full = [{"anonid_indice": r["anonid_indice"], "grupo_ids": r["grupo_ids"],
                  "rank_por_spread_csv": r["rank_por_spread_csv"],
                  "spread_csv_original": r["spread_csv_original"],
                  "dice_ptv_min": r["dice_ptv"]["min"], "dice_ptv_mean": r["dice_ptv"]["mean"],
                  "dice_recto_min": r["dice_recto"]["min"], "dice_recto_mean": r["dice_recto"]["mean"]}
                 for r in revisados]
    with open(OUT_DIR / "diagnostico_todos_los_candidatos.json", "w") as f:
        json.dump(diag_full, f, indent=2)

    print(f"Candidatos revisados: {len(revisados)}  |  que pasan filtro Dice: {len(pasan)}")
    for r in pasan[:6]:
        print(f"  indice={r['anonid_indice']}  spread_recalc={r['spread_recalculado_ctfix']:.2f}pp  "
              f"Dice_PTV(min)={r['dice_ptv']['min']:.2f}  Dice_Recto(min)={r['dice_recto']['min']:.2f}  "
              f"rank_spread_csv={r['rank_por_spread_csv']}")

    if not pasan:
        print("AVISO: ningun candidato paso el filtro de Dice -- relajar umbrales o revisar 'revisados'.")
        # Igual guardamos diagnostico de lo revisado para inspeccion manual
        diag = [{"anonid_indice": r["anonid_indice"], "grupo_ids": r["grupo_ids"],
                 "spread_recalculado_ctfix": r["spread_recalculado_ctfix"],
                 "dice_ptv_min": r["dice_ptv"]["min"], "dice_recto_min": r["dice_recto"]["min"],
                 "dice_vejiga_min": r["dice_vejiga"]["min"]} for r in revisados]
        with open(OUT_DIR / "diagnostico_candidatos_sin_pasar_filtro.json", "w") as f:
            json.dump(diag, f, indent=2)
        return

    salida = {"criterio_seleccion": f"grupo (indice+3 vecinos KNN de 7 features D3) que maximiza "
                                     f"spread real de DVH banda media del recto, sujeto a Dice_PTV>="
                                     f"{DICE_PTV_MIN} y Dice_Recto>={DICE_RECTO_MIN} par-a-par entre los "
                                     f"4 (alineados por centroide PTV propio en XY [ya comun por "
                                     f"construccion del preprocesado] y Z [centroide PTV propio como "
                                     f"origen]).",
              "nota_umbral_dice_recto": (
                  "la tarea sugeria Dice_Recto>=0.60; se escaneraron los 146 candidatos disponibles "
                  "(processed_hipo_ctfix) y el MAXIMO Dice_recto par-a-par alcanzable en TODO el "
                  "dataset fue 0.582 (ningun grupo llega a 0.60) — ver "
                  "diagnostico_todos_los_candidatos.json. Se uso 0.55 en su lugar, documentado aca en "
                  "vez de forzar el umbral original. Interpretacion: el recto (tubo largo, forma/"
                  "posicion variable dia a dia) tiene techo de similitud de forma mas bajo que un "
                  "organo tipo blob (PTV) para 'vecinos' definidos solo por 7 features escalares — la "
                  "similitud de forma real del recto entre 'vecinos anatomicos' de D3 es mas floja de "
                  "lo que sugieren volumen/overlap solos, un matiz a mencionar en la charla."
              ),
              "k_vecinos": K_VECINOS, "features_usadas": FEATURES,
              "n_candidatos_revisados": len(revisados), "n_candidatos_que_pasan_filtro": len(pasan)}

    # Revision visual (Pablo, 2026-08-24): en el grupo principal y en la alternativa 1,
    # el "vecino 1" (1er vecino KNN) mostraba un corte axial claramente distinto del
    # resto del grupo en esa misma posicion z (anatomia no comparable en la imagen,
    # aun con align_score de grupo pasando el umbral -- un promedio de 6 pares puede
    # esconder 1 par malo). Se saca esa fila y se RECALCULA Dice/spread/z-corte sobre
    # el subgrupo de 3 restante (no se reusan los numeros del grupo de 4). La ex-
    # alternativa 2 tenia ese problema en 2 de sus 3 vecinos (vecino 1 Y vecino 3) --
    # sacar ambos deja solo 2 pacientes, muy poco para ilustrar "vecinos", asi que se
    # DESCARTA esa alternativa en vez de forzarla.
    candidatos_finales = [
        ("figura_axial_vecinos.png", "figura_dvh_principal.png", quitar_filas(pasan[0], [1])),
        ("figura_axial_vecinos_alternativa1.png", "figura_dvh_alternativa1.png", quitar_filas(pasan[1], [1])),
    ]
    print("AVISO: ex-alternativa 2 (indice=%s) descartada -- vecino 1 Y vecino 3 mostraban "
          "cortes no comparables, quitar ambos dejaba solo 2 pacientes." % pasan[2]["anonid_indice"])

    grupos_json = []
    for nombre_grilla, nombre_dvh, candidato in candidatos_finales:
        z_info, _ = elegir_z_comun(candidato["pacientes"])
        complejidad = cargar_complejidad_hipo(candidato["grupo_ids"])
        construir_figura(candidato, z_info, OUT_DIR / nombre_grilla, complejidad)
        construir_dvh_figura(candidato, OUT_DIR / nombre_dvh)
        resumen = resumen_candidato_para_json(candidato, z_info, df)
        resumen["archivo_figura_grilla"] = nombre_grilla
        resumen["archivo_figura_dvh"] = nombre_dvh
        grupos_json.append(resumen)
        print(f"Figuras guardadas: {OUT_DIR / nombre_grilla}  |  {OUT_DIR / nombre_dvh}")

    salida["nota_filas_removidas"] = (
        "el grupo principal y la alternativa 1 originalmente tenian 4 pacientes "
        "(indice+3 vecinos KNN); se saco el 1er vecino KNN de ambos porque su corte axial "
        "en la posicion z comun no era anatomicamente comparable al resto (visible en la "
        "figura, no solo en el Dice promedio). Dice/spread/z quedaron RECALCULADOS sobre "
        "el subgrupo de 3 (indice+2 vecinos), no son los del grupo original de 4. La "
        "ex-alternativa 2 se descarto por completo (2 de sus 3 vecinos con el mismo problema)."
    )
    salida["grupo_elegido"] = grupos_json[0]
    salida["alternativas"] = grupos_json[1:]

    with open(OUT_DIR / "grupo_elegido.json", "w") as f:
        json.dump(salida, f, indent=2, default=str)
    print(f"\nGuardado: {OUT_DIR / 'grupo_elegido.json'}")


if __name__ == "__main__":
    main()
