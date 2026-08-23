"""Wrapper de la herramienta de tomografo sobre el pipeline ya construido y validado
del Proyecto 1 (Tareas 6-7). NO reimplementa extraccion de features ni logica de
modelo: llama directo a scripts/extract_features_live.py y scripts/infer_tomografo.py.

Funcion publica unica: procesar_paciente(carpeta) -> dict.
"""

import contextlib
import json
import logging
import os
import shutil
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import pydicom

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from extract_features_live import extraer_features, cargar_config  # noqa: E402
from infer_tomografo import predecir_paciente  # noqa: E402

log = logging.getLogger("tomografo_tool.pipeline")

REGISTROS_DIR = Path(__file__).resolve().parent / "registros"


def identificar_paciente(carpeta: Path) -> str:
    """Identificador visible del paciente: PatientID del header DICOM (CT o RS),
    con fallback al nombre de la carpeta si no se puede leer ningun header. Busca
    recursivamente porque el tomografo suele exportar con subcarpetas (por paciente,
    a veces una mas por serie)."""
    for f in sorted(carpeta.rglob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            pid = getattr(ds, "PatientID", None)
            if pid:
                return str(pid).strip()
        except Exception:
            continue
    return carpeta.name


@contextlib.contextmanager
def _carpeta_plana(carpeta: Path):
    """extract_features_live.extraer_features() (y los helpers de data/preprocess.py
    que usa por debajo) esperan CT+RS mezclados en UN solo nivel de carpeta -- asi
    esta organizado el dataset con el que se entreno el Proyecto 1. En la practica el
    tomografo exporta con subcarpetas (una por paciente, a veces otra mas por serie
    CT/RS dentro de esa). En vez de tocar esa logica ya validada, se arma -si hace
    falta- una carpeta temporal plana con hardlinks (o copia si el hardlink no es
    posible, p.ej. recurso de red) a todos los .dcm encontrados recursivamente, y se
    le pasa esa carpeta plana. Si `carpeta` ya viene plana, se usa tal cual -- sin
    copiar nada."""
    # Un solo glob("*") + filtro por sufijo en minusculas -- si se buscan "*.dcm" y
    # "*.DCM" por separado, en Windows (filesystem case-insensitive) cada archivo
    # matchea las DOS veces y se duplica en la carpeta plana (CT con slices repetidos,
    # confirmado en pruebas: "Non uniform sampling" de SimpleITK).
    archivos = [p for p in carpeta.rglob("*") if p.is_file() and p.suffix.lower() == ".dcm"]
    directos = [p for p in carpeta.glob("*") if p.is_file() and p.suffix.lower() == ".dcm"]
    if len(archivos) == len(directos):
        yield carpeta
        return

    with tempfile.TemporaryDirectory(prefix="tomografo_tool_") as tmp:
        tmp_path = Path(tmp)
        vistos = set()
        for i, f in enumerate(archivos):
            nombre = f.name
            if nombre in vistos:
                nombre = f"{i:04d}_{nombre}"
            vistos.add(nombre)
            dest = tmp_path / nombre
            try:
                os.link(f, dest)
            except OSError:
                shutil.copy2(f, dest)
        log.info("Carpeta %s no estaba plana (%d archivos en subcarpetas); armada "
                  "carpeta temporal plana con %d archivos", carpeta, len(archivos), len(vistos))
        yield tmp_path


def procesar_paciente(carpeta: Path) -> dict:
    """Punto de entrada unico de la herramienta. Extrae features (Tarea 7 en vivo),
    corre inferencia (modelos/umbrales del Proyecto 1), arma el dict de resultado,
    lo guarda en registros/ y lo devuelve. No lanza excepciones hacia el llamador:
    en caso de error, devuelve un dict con "estado": "error" para que el watcher y
    la ruta manual de Flask puedan mostrarlo sin crashear."""
    carpeta = Path(carpeta)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    patient_id = identificar_paciente(carpeta)

    try:
        config = cargar_config()
        with _carpeta_plana(carpeta) as carpeta_plana:
            feats = extraer_features(carpeta_plana, config)
        resultado_modelo = predecir_paciente(feats)
        por_constraint = resultado_modelo["por_constraint"]
        por_oar = resultado_modelo["por_oar"]

        resultado = {
            "estado": "ok",
            "patient_id": patient_id,
            "timestamp": timestamp,
            "carpeta": str(carpeta),
            "features": feats,
            "recto": {
                "zona": por_oar["recto"]["zona"],
                "V65_pred": por_constraint["RV65"]["V_pred"],
                "V55_pred": por_constraint["RV55"]["V_pred"],
            },
            "vejiga": {
                "zona": por_oar["vejiga"]["zona"],
                "V65_pred": por_constraint["BV65"]["V_pred"],
            },
        }
    except Exception as e:
        log.error("Error procesando %s: %s", carpeta, e, exc_info=True)
        resultado = {
            "estado": "error",
            "patient_id": patient_id,
            "timestamp": timestamp,
            "carpeta": str(carpeta),
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    _guardar_registro(resultado)
    return resultado


def _guardar_registro(resultado: dict) -> Path:
    REGISTROS_DIR.mkdir(parents=True, exist_ok=True)
    nombre = f"{resultado['patient_id']}_{resultado['timestamp']}.json"
    # Sanear el nombre de archivo (PatientID real puede traer caracteres raros)
    nombre = "".join(c if c.isalnum() or c in "._-" else "_" for c in nombre)
    ruta = REGISTROS_DIR / nombre
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    log.info("Registro guardado: %s", ruta)
    return ruta
