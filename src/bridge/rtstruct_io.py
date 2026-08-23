"""Carga de RTStruct via rt-utils, aislada del gotcha de series contaminada.

`RTStructBuilder.create_from` (rt-utils) escanea TODOS los archivos del directorio
que tengan `pixel_array` para reconstruir la serie de imagen -- el RD (RTDOSE)
tambien tiene PixelData, asi que si se le pasa la carpeta completa del paciente
(CT+RS+RP+RD, como vienen en dicom_pilot/ o en los DICOM exportados de Eclipse) el
RD se cuela como si fuera un corte mas de la serie (confirmado empiricamente: 140
"slices" en vez de 139 CT reales para PT_003fe2bb84986507). Por eso siempre se arma
una carpeta temporal con SOLO los CT.*.dcm antes de invocar rt-utils.

Usado por `scripts/substruct_derisk_mask.py` (Bloque 0) y `scripts/split_oar_by_dose.py`
(Bloque 1, HANDOFF_substructuras_dosis.md).
"""
import glob
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from pydicom.uid import generate_uid
from rt_utils import RTStructBuilder


def carpeta_solo_ct(dicom_dir: Path, tmp_dir: Path) -> Path:
    """Copia (hardlink si es posible) unicamente los CT.*.dcm de `dicom_dir` a
    `tmp_dir`. Cae a copia normal si el hardlink no es posible (p.ej. `dicom_dir`
    en un recurso de red UNC)."""
    dicom_dir = Path(dicom_dir)
    ct_files = sorted(glob.glob(str(dicom_dir / "CT.*.dcm")))
    if not ct_files:
        raise FileNotFoundError(f"No hay archivos CT.*.dcm en {dicom_dir}")
    for f in ct_files:
        dest = tmp_dir / Path(f).name
        try:
            os.link(f, dest)  # hardlink: instantaneo, sin duplicar espacio
        except OSError:
            shutil.copy2(f, dest)
    return tmp_dir


def cargar_rtstruct_ct_only(dicom_dir: Path, rs_path: str = None):
    """Devuelve un `RTStruct` (rt-utils) cargado SOLO con la serie CT real (sin
    RD/RP contaminando), listo para `get_roi_mask_by_name`/`add_roi`. El resultado
    es seguro de usar despues de esta llamada (pydicom ya leyo todo a memoria) --
    no depende de la carpeta temporal, que se borra al salir de la funcion."""
    dicom_dir = Path(dicom_dir)
    if rs_path is None:
        rs_files = glob.glob(str(dicom_dir / "RS.*.dcm"))
        if not rs_files:
            raise FileNotFoundError(f"No hay RS*.dcm en {dicom_dir}")
        rs_path = rs_files[0]
    with tempfile.TemporaryDirectory(prefix="ct_only_") as tmp:
        ct_only_dir = carpeta_solo_ct(dicom_dir, Path(tmp))
        rtstruct = RTStructBuilder.create_from(str(ct_only_dir), rs_path)
    return rtstruct


def regenerar_uids(rtstruct, series_description: str):
    """Genera SOPInstanceUID/SeriesInstanceUID nuevos para que Eclipse no rechace
    el RS como duplicado del original ya importado (`RTStructBuilder.create_from`
    conserva los UIDs del RS leido con `dcmread` sin tocarlos -- confirmado por
    Pablo al intentar importar el de Bloque 0). Mismo patron que
    `write_rd_dicom.py::escribir_rd_sintetico` usa para el RD sintetico.
    NO toca FrameOfReferenceUID -- ese es el que determina la alineacion espacial
    y debe seguir apuntando al mismo Frame of Reference que la CT real."""
    ds = rtstruct.ds
    nuevo_sop = generate_uid()
    ds.SOPInstanceUID = nuevo_sop
    ds.file_meta.MediaStorageSOPInstanceUID = nuevo_sop
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesDescription = series_description
    now = datetime.now()
    ds.InstanceCreationDate = now.strftime("%Y%m%d")
    ds.InstanceCreationTime = now.strftime("%H%M%S.%f")
    if hasattr(ds, "StructureSetDate"):
        ds.StructureSetDate = now.strftime("%Y%m%d")
    if hasattr(ds, "StructureSetTime"):
        ds.StructureSetTime = now.strftime("%H%M%S.%f")
