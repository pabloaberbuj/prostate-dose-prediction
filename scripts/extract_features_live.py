"""Tarea 7 (Proyecto 1) — extractor de features en vivo desde el RS del autocontour,
en el tomografo (CT-sim), antes de que exista el plan.

Reusa los helpers DICOM de data/preprocess.py (cargar_ct, cargar_estructuras,
contornos_a_mascara) SIN downsample ni recorte — todo en la grilla NATIVA del CT, para
no repetir el bug de escala de voxel ya documentado en el proyecto (spacing nativo
aplicado a una mascara downsampleada da un volumen equivocado y paciente-dependiente,
ver CLAUDE_CODE_CONTEXT_070826.md / scripts/compute_overlap_real.py). Aca no hace falta
ese cuidado porque nunca se downsamplea: se rasteriza directo sobre la imagen del CT tal
cual viene.

Los nombres EXACTOS de estructura del autocontour se leen de configs/config_p1.yaml
(PENDIENTE que Pablo los complete — sin eso, extraer_features() falla con un error
explicito, no un valor silenciosamente mal).

Uso:
    python scripts/extract_features_live.py --carpeta <ruta con CT+RS DICOM>
    python scripts/extract_features_live.py --test-consistencia
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "data"))

from preprocess import cargar_ct, cargar_estructuras, contornos_a_mascara  # noqa: E402

CONFIG_PATH = _REPO_ROOT / "configs" / "config_p1.yaml"
DATASET_CSV = _REPO_ROOT / "data" / "dataset_p1.csv"

FEATURE_COLS_FINAL = [
    "VolRectum_cc", "VolBladder_cc", "VolPTV_cc",
    "Solap_PTV_Rectum_cc", "Solap_PTV_Bladder_cc",
    "overlap_rel_recto", "overlap_rel_vejiga",
]


def cargar_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _mascara_de(estructuras: dict, nombre: str, imagen_ct):
    if nombre not in estructuras:
        disponibles = sorted(estructuras.keys())
        raise KeyError(
            f"No se encontro la estructura '{nombre}' en el RS. "
            f"Estructuras disponibles: {disponibles}. "
            f"Revisar configs/config_p1.yaml (seccion 'estructuras')."
        )
    return contornos_a_mascara(estructuras[nombre], imagen_ct)


def extraer_features(carpeta_dicom: Path, config: dict = None) -> dict:
    """Extrae las 7 features geometricas desde CT+RS DICOM en la grilla NATIVA (sin
    downsample). Devuelve dict {feature: valor} en el mismo orden/nombres que
    data/dataset_p1.csv, listo para pasar a los modelos de models/proyecto1/."""
    config = config or cargar_config()
    nombres = config["estructuras"]
    faltantes = [k for k, v in nombres.items() if not v]
    if faltantes:
        raise ValueError(
            f"configs/config_p1.yaml: faltan nombres de estructura para {faltantes}. "
            f"Pablo tiene que completarlos con los nombres exactos del autocontour antes "
            f"de poder correr esto en un caso real."
        )

    imagen_ct = cargar_ct(carpeta_dicom)
    estructuras = cargar_estructuras(carpeta_dicom)

    mask_ptv = _mascara_de(estructuras, nombres["ptv"], imagen_ct)
    mask_rectum = _mascara_de(estructuras, nombres["rectum"], imagen_ct)
    mask_bladder = _mascara_de(estructuras, nombres["bladder"], imagen_ct)

    sx, sy, sz = imagen_ct.GetSpacing()  # mm, orden XYZ (sitk)
    voxel_vol_cc = (sx * sy * sz) / 1000.0

    vol_ptv_cc = float(mask_ptv.sum()) * voxel_vol_cc
    vol_rectum_cc = float(mask_rectum.sum()) * voxel_vol_cc
    vol_bladder_cc = float(mask_bladder.sum()) * voxel_vol_cc
    solap_rectum_cc = float(np.logical_and(mask_ptv, mask_rectum).sum()) * voxel_vol_cc
    solap_bladder_cc = float(np.logical_and(mask_ptv, mask_bladder).sum()) * voxel_vol_cc

    return {
        "VolRectum_cc": vol_rectum_cc,
        "VolBladder_cc": vol_bladder_cc,
        "VolPTV_cc": vol_ptv_cc,
        "Solap_PTV_Rectum_cc": solap_rectum_cc,
        "Solap_PTV_Bladder_cc": solap_bladder_cc,
        "overlap_rel_recto": solap_rectum_cc / vol_rectum_cc if vol_rectum_cc > 0 else float("nan"),
        "overlap_rel_vejiga": solap_bladder_cc / vol_bladder_cc if vol_bladder_cc > 0 else float("nan"),
    }


def correr_test_consistencia(config: dict = None, tol: float = None):
    """Compara features extraidas en vivo contra las ya calculadas en
    data/dataset_p1.csv, para una carpeta con varios pacientes (subcarpetas = AnonID)
    ya presentes en el dataset de entrenamiento. Umbral de exchangeabilidad: <1% de
    diferencia relativa (ver PROMPT_claude_code_proyecto ML Tomo.md, Tarea 7)."""
    import pandas as pd

    config = config or cargar_config()
    tol = tol if tol is not None else config.get("tolerancia_relativa", 0.01)
    carpeta_root = config.get("carpeta_test_consistencia")
    if not carpeta_root:
        print("[ABORTADO] configs/config_p1.yaml: 'carpeta_test_consistencia' esta vacio. "
              "Hace falta una carpeta con CT+RS DICOM de varios pacientes YA presentes en "
              "data/dataset_p1.csv (ninguna sobrevive en disco ahora mismo, ver docstring "
              "del yaml) para poder correr este chequeo.")
        return

    carpeta_root = Path(carpeta_root)
    df = pd.read_csv(DATASET_CSV).set_index("AnonID")

    resultados = []
    for sub in sorted(carpeta_root.iterdir()):
        if not sub.is_dir() or sub.name not in df.index:
            continue
        try:
            feats_live = extraer_features(sub, config)
        except Exception as e:
            print(f"  [{sub.name}] ERROR extrayendo: {e}")
            continue
        fila = df.loc[sub.name]
        max_rel_diff = 0.0
        detalle = {}
        for feat in FEATURE_COLS_FINAL:
            v_csv = float(fila[feat])
            v_live = feats_live[feat]
            rel_diff = abs(v_live - v_csv) / abs(v_csv) if v_csv != 0 else abs(v_live)
            detalle[feat] = {"csv": v_csv, "live": v_live, "rel_diff": rel_diff}
            max_rel_diff = max(max_rel_diff, rel_diff)
        ok = max_rel_diff < tol
        resultados.append({"AnonID": sub.name, "ok": ok, "max_rel_diff": max_rel_diff, "detalle": detalle})
        print(f"  [{sub.name}] max_rel_diff={max_rel_diff:.4f}  {'OK' if ok else 'FALLO'}")

    if not resultados:
        print("[SIN DATOS] No se encontraron subcarpetas que matcheen AnonIDs de data/dataset_p1.csv.")
        return

    n_ok = sum(1 for r in resultados if r["ok"])
    print(f"\n{n_ok}/{len(resultados)} pacientes dentro de tolerancia ({tol:.1%}).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", type=Path, default=None, help="Carpeta con CT+RS DICOM de un paciente")
    ap.add_argument("--test-consistencia", action="store_true")
    args = ap.parse_args()

    if args.test_consistencia:
        correr_test_consistencia()
        return

    if args.carpeta is None:
        print("Uso: --carpeta <ruta CT+RS DICOM>  o  --test-consistencia")
        return

    feats = extraer_features(args.carpeta)
    for k, v in feats.items():
        print(f"  {k:24s} {v:.4f}")


if __name__ == "__main__":
    main()
