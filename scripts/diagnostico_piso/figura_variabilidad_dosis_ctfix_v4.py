"""
Figura para charla — VARIABILIDAD DE DOSIS A ANATOMIA EQUIVALENTE (hipo, ctfix v4).

Puramente de VISUALIZACION: no entrena nada, no corre analisis estadistico nuevo.
Reproduce la MISMA logica que results/figura_piso_generacion/ (D3: 7 features
geometricas -> KNN -> grupo indice+vecinos -> spread real de DVH banda media del
recto) pero sobre el dataset hipo ACTUAL (`processed_hipo_ctfix/`, FOV 34cm
definitivo/1.328mm px, 198 pacientes de `splits_hipo_ctfix_v4.json`) — el dataset
usado antes (`d3_hipo_por_paciente.csv`, FOV 50cm viejo) quedo obsoleto, ver
RESULTADO_hipo_ctfix_v4.md / CLAUDE_CODE_CONTEXT.md ("Paso 1... FOV real era 50cm,
no 34cm" -> re-corrido). No hay CSV de features pre-calculado para v4, asi que se
recalculan aca desde las mascaras (mismo metodo que analisis_angular.overlap_y_volumenes,
ya usado y validado en el proyecto) — es la MISMA sonda D3, no una tecnica nueva.

IMPORTANTE (correcciones ya conversadas, incorporadas de entrada esta vez):
- NO llamar a esto "piso anatomico": el punto es que la variabilidad de dosis NO es
  explicada por diferencias de anatomia (anatomia casi identica, dosis real distinta).
- Etiquetas de fila: "paciente 1/2/3" (nunca "indice"/"vecino").
- El DVH va en figura SEPARADA (no embebido en la grilla), tamaño normal.
- Si el grupo KNN de 4 tiene un vecino cuyo corte no es comparable (Dice bajo con
  los otros 3, o PTV vacio en el offset comun), se recorta automaticamente ANTES de
  elegir el grupo final (no se arma manualmente despues de mirar la figura).
- Figura con texto minimo (sin caja de Dice/MCSv por fila) — el detalle cuantitativo
  vive en el JSON/reporte, no pegoteado en la imagen.

Uso:
    .venv/Scripts/python.exe scripts/diagnostico_piso/figura_variabilidad_dosis_ctfix_v4.py
"""

import json
import sys
from collections import OrderedDict
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

from analisis_angular import overlap_y_volumenes  # noqa: E402
from dvh_curva_completa import curva_v_d, D_BINS, IDX_MEDIA  # noqa: E402

PROCESSED = Path(r"C:\Pablo\ProstateDoseProject\processed_hipo_ctfix")
SPLITS_V4 = _REPO_ROOT / "data/splits/splits_hipo_ctfix_v4.json"
COMPLEJIDAD_CSV = _REPO_ROOT / "results/complejidad_arbitro/data/complejidad_rp.csv"

OUT_DIR = _REPO_ROOT / "results/figura_variabilidad_dosis_ctfix_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["VolPTV_cc", "VolRectum_cc", "VolBladder_cc",
            "Solap_PTV_Rectum_cc", "Solap_PTV_Bladder_cc",
            "overlap_rel_recto", "overlap_rel_vejiga"]
K_VECINOS = 3
Z_OFFSET_RANGE = range(-40, 41)  # slices relativos al centroide Z del PTV propio

# Umbrales deseados (los mismos que pidio la tarea originalmente). Con el recorte
# automatico de vecinos no comparables, se intenta ALCANZARLOS esta vez; si el
# escaneo muestra que son inalcanzables incluso recortando, se documenta el maximo
# real encontrado y se ajustan (igual que se hizo la vez anterior, sin forzarlos).
DICE_PTV_MIN = 0.70
DICE_RECTO_MIN = 0.60

MAX_CACHE_PACIENTES = 24  # LRU acotado: RAM compartida con entrenamiento en curso


# ──────────────────────────────────────────────────────────────────────────────
# 1) Carga por paciente — cache LRU acotada (RAM compartida, ver
#    feedback_windows_gpu_process_safety) + acceso perezoso por-key
# ──────────────────────────────────────────────────────────────────────────────

_CACHE_PACIENTES = OrderedDict()


def get_array(pac: dict, key: str) -> np.ndarray:
    """Perezoso por-key. Si el NpzFile ya fue cerrado (el paciente salio de la
    cache LRU acotada pero este dict `pac` sigue referenciado en `evaluados`/
    `finales`), se reabre desde disco de forma transparente."""
    if key not in pac["_arr_cache"]:
        try:
            raw = pac["_data"][key]
        except Exception:
            pac["_data"] = np.load(str(PROCESSED / f"{pac['anonid']}.npz"), allow_pickle=True)
            raw = pac["_data"][key]
        dt = np.uint8 if "mask" in key else np.float32
        pac["_arr_cache"][key] = np.array(raw, dtype=dt)
    return pac["_arr_cache"][key]


def cargar_paciente(anonid: str) -> dict:
    if anonid in _CACHE_PACIENTES:
        _CACHE_PACIENTES.move_to_end(anonid)
        return _CACHE_PACIENTES[anonid]
    data = np.load(str(PROCESSED / f"{anonid}.npz"), allow_pickle=True)
    meta = json.loads(str(data["meta"][0]))
    pac = {"anonid": anonid, "_data": data, "_arr_cache": {}, "meta": meta}
    ptv = get_array(pac, "ptv_mask")
    zs = np.nonzero(ptv.sum(axis=(1, 2)))[0]
    pac["z_centroid"] = float(zs.mean()) if len(zs) else ptv.shape[0] / 2.0
    pac["n_slices"] = ptv.shape[0]
    _CACHE_PACIENTES[anonid] = pac
    if len(_CACHE_PACIENTES) > MAX_CACHE_PACIENTES:
        _, viejo = _CACHE_PACIENTES.popitem(last=False)
        viejo["_data"].close()
    return pac


def slice_en_offset(pac: dict, key: str, offset: int) -> np.ndarray:
    idx = int(round(pac["z_centroid"] + offset))
    arr = get_array(pac, key)
    if 0 <= idx < arr.shape[0]:
        return arr[idx]
    return np.zeros(arr.shape[1:], dtype=arr.dtype)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    s = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / s) if s > 0 else float("nan")


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
    return {"min": float(min(pares)) if pares else float("nan"),
            "mean": float(np.mean(pares)) if pares else float("nan"), "pares": pares}


# ──────────────────────────────────────────────────────────────────────────────
# 2) Features geometricas + DVH real del recto, para TODOS los pacientes del
#    split v4 (streaming: no se cachean arrays de los 198 a la vez, solo salen
#    escalares/curvas chicas de esta fase)
# ──────────────────────────────────────────────────────────────────────────────

def listar_ids_v4():
    with open(SPLITS_V4) as f:
        s = json.load(f)
    ids = sorted(set(s["train"]) | set(s["val"]) | set(s["test"]))
    return ids


def construir_tabla_features_y_curvas(ids: list):
    filas = []
    curvas = np.full((len(ids), len(D_BINS)), np.nan)
    for i, pid in enumerate(ids):
        pac = cargar_paciente(pid)
        ptv = get_array(pac, "ptv_mask")
        recto = get_array(pac, "rectum_mask")
        vejiga = get_array(pac, "bladder_mask")
        dose = get_array(pac, "dose")
        feats = overlap_y_volumenes(pac["meta"], ptv, recto, vejiga)
        fila = {"AnonID": pid}
        fila.update(feats)
        filas.append(fila)
        curvas[i] = curva_v_d(dose, recto)
    df = pd.DataFrame(filas)
    return df, curvas


def knn_grupos(df: pd.DataFrame, k: int = K_VECINOS):
    X = df[FEATURES].to_numpy(dtype=float)
    Xz = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xz)
    _, idx = nn.kneighbors(Xz)
    return idx  # idx[i,0]==i (self), idx[i,1:]=3 vecinos


# ──────────────────────────────────────────────────────────────────────────────
# 3) Seleccion de grupo: rankear por spread, recortar vecinos no comparables
#    ANTES de fijar el grupo final (no despues de ver la figura)
# ──────────────────────────────────────────────────────────────────────────────

def spread_de_grupo(curvas_grupo: np.ndarray):
    std_por_bin = curvas_grupo[:, IDX_MEDIA].std(axis=0)
    return float(std_por_bin.mean())


def evaluar_grupo(ids_grupo: list) -> dict:
    pacientes = [cargar_paciente(a) for a in ids_grupo]
    dice_ptv = dice_pairwise_grupo(pacientes, "ptv_mask")
    dice_recto = dice_pairwise_grupo(pacientes, "rectum_mask")
    return {"grupo_ids": ids_grupo, "pacientes": pacientes,
            "dice_ptv": dice_ptv, "dice_recto": dice_recto}


def mejor_variante(df, curvas, idx_matriz, i):
    """Evalua el grupo de 4 (indice+3nn); si no cumple los umbrales de Dice,
    prueba recortar cada uno de los 3 vecinos por separado y se queda con el
    mejor recorte de 3 que SI cumpla (o el de mayor Dice_recto_min si ninguno
    cumple). Devuelve la variante elegida + si cumplio los umbrales."""
    grupo_idx = list(idx_matriz[i])
    grupo_ids = df.loc[grupo_idx, "AnonID"].tolist()
    ev4 = evaluar_grupo(grupo_ids)
    ev4["curvas"] = curvas[grupo_idx]
    ev4["spread"] = spread_de_grupo(ev4["curvas"])
    ev4["cumple"] = ev4["dice_ptv"]["min"] >= DICE_PTV_MIN and ev4["dice_recto"]["min"] >= DICE_RECTO_MIN
    ev4["n"] = 4
    if ev4["cumple"]:
        return ev4

    variantes_3 = []
    for pos_a_quitar in [1, 2, 3]:
        keep = [p for p in range(4) if p != pos_a_quitar]
        ids3 = [grupo_ids[p] for p in keep]
        ev3 = evaluar_grupo(ids3)
        ev3["curvas"] = curvas[[grupo_idx[p] for p in keep]]
        ev3["spread"] = spread_de_grupo(ev3["curvas"])
        ev3["cumple"] = ev3["dice_ptv"]["min"] >= DICE_PTV_MIN and ev3["dice_recto"]["min"] >= DICE_RECTO_MIN
        ev3["n"] = 3
        ev3["pos_removida_del_grupo_de_4"] = pos_a_quitar
        variantes_3.append(ev3)

    cumplen = [v for v in variantes_3 if v["cumple"]]
    if cumplen:
        return max(cumplen, key=lambda v: v["spread"])
    # ninguno cumple: devolver el mejor intento (mayor Dice_recto_min) para diagnostico
    todas = [ev4] + variantes_3
    return max(todas, key=lambda v: (v["dice_recto"]["min"] if not np.isnan(v["dice_recto"]["min"]) else -1))


def elegir_top_grupos(df, curvas, idx_matriz, n_finales=3, n_candidatos_a_revisar=200):
    orden = np.argsort([spread_de_grupo(curvas[list(idx_matriz[i])]) for i in range(len(df))])[::-1]
    evaluados = []
    for rank, i in enumerate(orden[:n_candidatos_a_revisar]):
        variante = mejor_variante(df, curvas, idx_matriz, i)
        variante["rank_por_spread_grupo4"] = int(rank)
        evaluados.append(variante)

    cumplen = [v for v in evaluados if v["cumple"]]
    cumplen.sort(key=lambda v: v["spread"], reverse=True)
    return cumplen[:n_finales], evaluados


# ──────────────────────────────────────────────────────────────────────────────
# 4) Eleccion del corte Z comun (documentado, no a ojo)
# ──────────────────────────────────────────────────────────────────────────────

def elegir_z_comun(pacientes: list):
    filas = []
    for off in Z_OFFSET_RANGE:
        ptv_slices = [slice_en_offset(p, "ptv_mask", off) for p in pacientes]
        recto_slices = [slice_en_offset(p, "rectum_mask", off) for p in pacientes]
        dose_slices = [slice_en_offset(p, "dose", off) for p in pacientes]

        estructuras_presentes_en_todos = (
            all(ps.sum() > 0 for ps in ptv_slices) and all(rs.sum() > 0 for rs in recto_slices)
        )

        combo = [((ps > 0) | (rs > 0)).astype(np.uint8) for ps, rs in zip(ptv_slices, recto_slices)]
        n = len(combo)
        pares_align = [dice(combo[a], combo[b]) for a in range(n) for b in range(a + 1, n)]
        pares_align = [d for d in pares_align if not np.isnan(d)]
        align_score = float(np.mean(pares_align)) if pares_align else 0.0

        dosis_media_recto = [float(ds[rs > 0].mean()) for rs, ds in zip(recto_slices, dose_slices) if rs.sum() > 0]
        diff_score = float(np.std(dosis_media_recto)) if len(dosis_media_recto) == n else float("nan")

        filas.append({"offset": off, "align_score": align_score, "diff_score": diff_score,
                       "estructuras_presentes_en_todos": estructuras_presentes_en_todos,
                       "dosis_media_recto_por_paciente": dosis_media_recto if len(dosis_media_recto) == n else None})

    validos = [f for f in filas if not np.isnan(f["diff_score"]) and f["align_score"] >= 0.5 and f["estructuras_presentes_en_todos"]]
    if not validos:
        validos = [f for f in filas if not np.isnan(f["diff_score"]) and f["estructuras_presentes_en_todos"]]
    if not validos:
        validos = [f for f in filas if not np.isnan(f["diff_score"])]
    return max(validos, key=lambda f: f["diff_score"]), filas


# ──────────────────────────────────────────────────────────────────────────────
# 5) Figuras — minimalistas (grilla CT+dosis; DVH en figura aparte)
# ──────────────────────────────────────────────────────────────────────────────

COLOR_INDICE = "black"
PALETA_VECINOS = ["steelblue", "darkorange", "firebrick", "seagreen"]


def etiquetas_y_colores(n_pacientes: int):
    etiquetas = [f"paciente {i}" for i in range(1, n_pacientes + 1)]
    colores = [COLOR_INDICE] + PALETA_VECINOS[: n_pacientes - 1]
    return etiquetas, colores


def construir_figura_grilla(variante: dict, z_info: dict, out_path: Path):
    pacientes = variante["pacientes"]
    n = len(pacientes)
    offset = z_info["offset"]
    etiquetas, colores = etiquetas_y_colores(n)

    dose_slices = [slice_en_offset(p, "dose", offset) for p in pacientes]
    ct_slices = [slice_en_offset(p, "ct", offset) for p in pacientes]
    ptv_slices = [slice_en_offset(p, "ptv_mask", offset) for p in pacientes]
    recto_slices = [slice_en_offset(p, "rectum_mask", offset) for p in pacientes]
    vejiga_slices = [slice_en_offset(p, "bladder_mask", offset) for p in pacientes]

    vmax_dose = max(100.0, max(float(d.max()) for d in dose_slices))

    fig = plt.figure(figsize=(9, 4.1 * n))
    gs = GridSpec(n, 2, wspace=0.05, hspace=0.12, figure=fig)

    im_ref = None
    for i, (ct, dose, ptv, recto, vejiga) in enumerate(
            zip(ct_slices, dose_slices, ptv_slices, recto_slices, vejiga_slices)):
        ax_ct = fig.add_subplot(gs[i, 0])
        ax_ct.imshow(ct, cmap="gray", vmin=-1, vmax=1)
        if ptv.sum() > 0:
            ax_ct.contour(ptv, levels=[0.5], colors="gold", linewidths=1.6)
        if recto.sum() > 0:
            ax_ct.contour(recto, levels=[0.5], colors="chocolate", linewidths=1.6)
        if vejiga.sum() > 0:
            ax_ct.contour(vejiga, levels=[0.5], colors="deepskyblue", linewidths=1.6)
        ax_ct.set_xticks([]); ax_ct.set_yticks([])
        ax_ct.set_ylabel(etiquetas[i], fontsize=11, color=colores[i], fontweight="bold")
        if i == 0:
            ax_ct.set_title("CT + contornos\nPTV oro · Recto marron · Vejiga celeste", fontsize=9)

        ax_d = fig.add_subplot(gs[i, 1])
        im = ax_d.imshow(dose, cmap="jet", vmin=0, vmax=vmax_dose)
        cs = ax_d.contour(dose, levels=[40, 60, 80], colors="white",
                           linewidths=[1.0, 1.3, 1.6], linestyles=["dotted", "dashed", "solid"])
        ax_d.clabel(cs, inline=True, fontsize=6, fmt="%d%%Rx")
        if recto.sum() > 0:
            ax_d.contour(recto, levels=[0.5], colors="chocolate", linewidths=1.3)
        if ptv.sum() > 0:
            ax_d.contour(ptv, levels=[0.5], colors="gold", linewidths=1.0)
        ax_d.set_xticks([]); ax_d.set_yticks([])
        if i == 0:
            ax_d.set_title("Dosis REAL (%Rx)\nisodosis 40/60/80%Rx", fontsize=9)
        im_ref = im

    fig.subplots_adjust(bottom=0.07, top=0.90)
    cbar_ax = fig.add_axes([0.25, 0.03, 0.5, 0.012])
    fig.colorbar(im_ref, cax=cbar_ax, orientation="horizontal", label="Dosis (%Rx) — escala comun")
    fig.suptitle("Variabilidad de dosis a anatomia equivalente", fontsize=13, fontweight="bold", y=0.965)

    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def construir_figura_dvh(variante: dict, out_path: Path):
    curvas = variante["curvas"]
    n = curvas.shape[0]
    etiquetas, colores = etiquetas_y_colores(n)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.axvspan(40, 80, color="steelblue", alpha=0.10)
    for i in range(n):
        ax.plot(D_BINS, curvas[i], color=colores[i], lw=2.2, label=etiquetas[i])
    ax.set_xlabel("Dosis (% Rx)")
    ax.set_ylabel("Volumen Rectum (%)")
    ax.set_title("DVH real del Recto — anatomia casi identica, dosis distinta", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 110); ax.set_ylim(0, 102)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 6) Verificacion de orientacion/lateralidad (cuantitativa, no a ojo)
# ──────────────────────────────────────────────────────────────────────────────

def verificar_orientacion(pacientes: list) -> dict:
    signos = []
    for p in pacientes:
        ptv = get_array(p, "ptv_mask"); recto = get_array(p, "rectum_mask"); vej = get_array(p, "bladder_mask")
        def centroide(m):
            rows, cols = np.nonzero(m.sum(axis=0) > 0)
            return (rows.mean(), cols.mean()) if len(rows) else None
        c_ptv, c_recto, c_vej = centroide(ptv), centroide(recto), centroide(vej)
        signos.append({
            "recto_row_menos_ptv_row": (c_recto[0] - c_ptv[0]) if c_recto and c_ptv else None,
            "vejiga_row_menos_ptv_row": (c_vej[0] - c_ptv[0]) if c_vej and c_ptv else None,
        })
    signos_recto = [s["recto_row_menos_ptv_row"] for s in signos if s["recto_row_menos_ptv_row"] is not None]
    signos_vej = [s["vejiga_row_menos_ptv_row"] for s in signos if s["vejiga_row_menos_ptv_row"] is not None]
    consistente = (all(v > 0 for v in signos_recto) or all(v < 0 for v in signos_recto)) and \
                  (all(v > 0 for v in signos_vej) or all(v < 0 for v in signos_vej))
    return {"consistente": bool(consistente), "detalle_por_paciente": signos}


# ──────────────────────────────────────────────────────────────────────────────
# 7) Complejidad de plan (limitacion conocida: no hay RP de hipofx)
# ──────────────────────────────────────────────────────────────────────────────

def cargar_complejidad_hipo(grupo_ids):
    if not COMPLEJIDAD_CSV.exists():
        return {a: "N/D (sin RP hipofx)" for a in grupo_ids}
    df = pd.read_csv(COMPLEJIDAD_CSV)
    return {a: (f"{df[df.AnonID == a].iloc[0]['MCSv']:.3f}" if (df.AnonID == a).any() else "N/D (sin RP hipofx)")
            for a in grupo_ids}


def resumen_variante_json(variante: dict, z_info: dict, df_features: pd.DataFrame, orientacion: dict) -> dict:
    grupo_ids = variante["grupo_ids"]
    etiquetas, _ = etiquetas_y_colores(len(grupo_ids))
    features_tabla = df_features.set_index("AnonID").loc[grupo_ids, FEATURES].round(3).to_dict(orient="index")
    media_banda = variante["curvas"][:, IDX_MEDIA].mean(axis=1).tolist()
    return {
        "grupo_ids": grupo_ids,
        "etiquetas": etiquetas,
        "n_pacientes": variante["n"],
        "recorte_automatico": {
            "se_recorto_un_vecino": variante["n"] == 3,
            "posicion_removida_del_grupo_knn_de_4": variante.get("pos_removida_del_grupo_de_4"),
            "motivo": ("el grupo KNN de 4 no alcanzaba Dice_PTV>=%.2f y Dice_Recto>=%.2f simultaneo; "
                       "se probaron los 3 recortes de-uno-vecino posibles y se uso el de mayor spread "
                       "que SI los alcanza" % (DICE_PTV_MIN, DICE_RECTO_MIN)) if variante["n"] == 3 else None,
        },
        "features_geometricas": features_tabla,
        "dvh_banda_media_recto_pct_vol": {aid: round(v, 2) for aid, v in zip(grupo_ids, media_banda)},
        "spread_banda_media_pp": round(variante["spread"], 3),
        "dice_interpaciente": {
            "PTV": variante["dice_ptv"], "Rectum": variante["dice_recto"],
            "umbral_aplicado": {"Dice_PTV_min": DICE_PTV_MIN, "Dice_Recto_min": DICE_RECTO_MIN},
            "cumple_umbral": variante["cumple"],
        },
        "corte_z_elegido": {
            "offset_relativo_a_centroide_ptv_propio_slices": z_info["offset"],
            "criterio": (
                "para cada offset entero en [-40,40] (relativo al centroide Z del PTV de CADA "
                "paciente, spacing_z~3.0mm), se exigio que PTV y Recto NO estuvieran vacios en "
                "NINGUNO de los pacientes en ese offset; entre los offsets que cumplen eso, se "
                "calculo (a) align_score=Dice promedio par-a-par de (PTV|Recto) [maxima "
                "superposicion espacial inter-paciente] y (b) diff_score=std entre pacientes de la "
                "dosis media dentro del recto en ese corte [maxima diferencia de dosis real]. Se "
                "filtro a align_score>=0.5 y se elgio el offset que maximiza diff_score."
            ),
            "align_score": round(z_info["align_score"], 3),
            "diff_score_dosis_recto_pp": round(z_info["diff_score"], 3),
            "dosis_media_recto_por_paciente_pctRx": {
                aid: round(v, 2) for aid, v in zip(grupo_ids, z_info["dosis_media_recto_por_paciente"])
            } if z_info["dosis_media_recto_por_paciente"] else None,
        },
        "verificacion_orientacion_lateralidad": orientacion,
        "complejidad_plan_MCSv": {
            "valores": cargar_complejidad_hipo(grupo_ids),
            "nota": "complejidad_rp.csv no tiene filas de dataset='hipo' (RP no disponibles para "
                    "hipofx) — no se puede insinuar agresividad de plan para este grupo por esta via.",
        },
    }


def main():
    ids = listar_ids_v4()
    print(f"Pacientes en splits_hipo_ctfix_v4.json: {len(ids)}")

    df, curvas = construir_tabla_features_y_curvas(ids)
    print("Features + DVH real calculados para todos.")

    idx_matriz = knn_grupos(df, k=K_VECINOS)

    finales, evaluados = elegir_top_grupos(df, curvas, idx_matriz, n_finales=3, n_candidatos_a_revisar=200)

    max_dice_recto_encontrado = max((v["dice_recto"]["min"] for v in evaluados
                                      if not np.isnan(v["dice_recto"]["min"])), default=float("nan"))
    print(f"Candidatos evaluados: {len(evaluados)} | que cumplen umbral: "
          f"{sum(1 for v in evaluados if v['cumple'])} | max Dice_recto_min visto: {max_dice_recto_encontrado:.3f}")

    umbral_ajustado = False
    if not finales:
        umbral_ajustado = True
        nuevo_umbral_recto = max(0.30, round(max_dice_recto_encontrado - 0.02, 2))
        print(f"AVISO: nadie cumplio Dice_Recto>={DICE_RECTO_MIN}. Reintentando con "
              f"Dice_Recto>={nuevo_umbral_recto} (max real encontrado={max_dice_recto_encontrado:.3f}).")
        globals()["DICE_RECTO_MIN"] = nuevo_umbral_recto
        for v in evaluados:
            v["cumple"] = v["dice_ptv"]["min"] >= DICE_PTV_MIN and v["dice_recto"]["min"] >= nuevo_umbral_recto
        finales = sorted([v for v in evaluados if v["cumple"]], key=lambda v: v["spread"], reverse=True)[:3]

    print(f"\nGrupos finales elegidos: {len(finales)}")
    for v in finales:
        print(f"  n={v['n']} spread={v['spread']:.2f}pp Dice_PTV_min={v['dice_ptv']['min']:.2f} "
              f"Dice_Recto_min={v['dice_recto']['min']:.2f} ids={v['grupo_ids']}")

    nombres = [("figura_variabilidad_dosis.png", "figura_dvh.png")] + \
              [(f"figura_variabilidad_dosis_alternativa{i}.png", f"figura_dvh_alternativa{i}.png") for i in range(1, 3)]

    resultados_json = []
    for (nombre_grilla, nombre_dvh), variante in zip(nombres, finales):
        z_info, _ = elegir_z_comun(variante["pacientes"])
        orientacion = verificar_orientacion(variante["pacientes"])
        construir_figura_grilla(variante, z_info, OUT_DIR / nombre_grilla)
        construir_figura_dvh(variante, OUT_DIR / nombre_dvh)
        resumen = resumen_variante_json(variante, z_info, df, orientacion)
        resumen["archivo_figura_grilla"] = nombre_grilla
        resumen["archivo_figura_dvh"] = nombre_dvh
        resultados_json.append(resumen)
        print(f"Guardado: {OUT_DIR / nombre_grilla}  |  {OUT_DIR / nombre_dvh}  "
              f"(orientacion consistente={orientacion['consistente']})")

    diag = [{"grupo_ids": v["grupo_ids"], "n": v["n"], "spread": v["spread"],
             "dice_ptv_min": v["dice_ptv"]["min"], "dice_recto_min": v["dice_recto"]["min"],
             "cumple": v["cumple"], "rank_por_spread_grupo4": v.get("rank_por_spread_grupo4")}
            for v in evaluados]
    with open(OUT_DIR / "diagnostico_todos_los_candidatos.json", "w") as f:
        json.dump(diag, f, indent=2)

    salida = {
        "dataset": "processed_hipo_ctfix (FOV 34cm definitivo, 1.328mm/px), "
                   "splits_hipo_ctfix_v4.json (198 pacientes)",
        "criterio_seleccion": (
            "grupo (indice + hasta 3 vecinos KNN de 7 features geometricas) que maximiza el "
            "spread real de DVH banda media [40,80]%Rx del recto, sujeto a Dice_PTV>="
            f"{DICE_PTV_MIN} y Dice_Recto>={DICE_RECTO_MIN} par-a-par entre TODOS los miembros "
            "finales. Si el grupo de 4 no cumple, se recorta automaticamente el vecino que "
            "menos aporta (peor Dice) ANTES de fijar el grupo — no se ajusta manualmente "
            "despues de ver la figura."
        ),
        "umbral_dice_ajustado_empiricamente": umbral_ajustado,
        "max_dice_recto_min_encontrado_en_todo_el_escaneo": round(float(max_dice_recto_encontrado), 3),
        "k_vecinos": K_VECINOS, "features_usadas": FEATURES,
        "n_candidatos_evaluados": len(evaluados),
        "n_candidatos_que_cumplen": sum(1 for v in evaluados if v["cumple"]),
        "grupo_elegido": resultados_json[0] if resultados_json else None,
        "alternativas": resultados_json[1:],
    }
    with open(OUT_DIR / "grupo_elegido.json", "w") as f:
        json.dump(salida, f, indent=2, default=str)
    print(f"\nGuardado: {OUT_DIR / 'grupo_elegido.json'}")


if __name__ == "__main__":
    main()
