"""Carga paciente + geometría de arcos reales desde DICOM, para usar como
punto de partida del mimicking (PIPELINE_KBP_PDRT.md, sección A3/B paso 3).

Decisión de diseño (A3): la geometría de los arcos (ángulos de gantry,
isocentro, colimador) sale de un plan real ya tratado, usado como plantilla
de partida para la optimización — no se arma una geometría de arco
"estándar" desde cero (no hay técnica estándar en el centro, ver
CLAUDE_CODE_CONTEXT_070826.md). Alcance inicial del proyecto: próstata VMAT
2 arcos (los casos de 3 arcos se descartaron en el análisis de dataset por
motivos ajenos a esto — igual el código de acá no asume 2 a la fuerza).
"""
from pathlib import Path

import pydicom
import torch

from pydosert.data.loaders import load_dicom

STRUCT_NAMES_DEFAULT = ["PTV_High", "Bladder", "Rectum", "FemoralHead_L", "FemoralHead_R", "BODY"]


def patch_rp_missing_ssd(rp_path: Path, tmp_path: Path) -> Path:
    """`fetch_plan_data` (pydosert) exige SourceToSurfaceDistance en TODOS los
    control points, pero Eclipse sólo lo exporta en el primero de cada arco
    dinámico (Type 3 opcional en DICOM; no cambia el cálculo real de dosis,
    que se hace por ray-tracing en la CT, no con este escalar). Se completa
    por forward-fill desde el CP anterior que sí lo tenga, en una copia
    temporal - el RP original no se modifica.
    """
    ds = pydicom.dcmread(str(rp_path))
    patched = False
    for beam in ds.BeamSequence:
        last_ssd = None
        for cp in beam.ControlPointSequence:
            if hasattr(cp, "SourceToSurfaceDistance"):
                last_ssd = cp.SourceToSurfaceDistance
            elif last_ssd is not None:
                cp.SourceToSurfaceDistance = last_ssd
                patched = True
    if patched:
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(str(tmp_path))
        return tmp_path
    return rp_path


def find_dicom_files(patient_dir: Path):
    patient_dir = Path(patient_dir)
    rs = next(patient_dir.glob("RS.*.dcm"))
    rp = next(patient_dir.glob("RP.*.dcm"))
    rd = sorted(patient_dir.glob("RD.*.dcm"))
    if not rd:
        raise FileNotFoundError(f"No encontré RD.*.dcm en {patient_dir}")
    return patient_dir, rs, rp, rd


def load_patient_and_arcs(
    patient_dir,
    struct_names=STRUCT_NAMES_DEFAULT,
    new_spacing: tuple = (2.0, 2.0, 2.0),
    crop_volume: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    tmp_dir=None,
):
    """Carga el paciente (CT/RS/RD reales) y la geometría de sus arcos VMAT
    reales (uno por arco dinámico del RP — el pip `pydosert` ya descarta
    solo los campos estáticos de verificación/portal, que no tienen MLCX).

    Devuelve (patient, arcos) — `arcos` es una lista de `BeamSequence`, una
    por arco (normalmente 2 para el alcance de este proyecto, pero no se
    asume: si un plan real tiene 3, se devuelven los 3).
    """
    patient_dir = Path(patient_dir)
    _, rs_path, rp_path, rd_paths = find_dicom_files(patient_dir)
    tmp_dir = Path(tmp_dir) if tmp_dir is not None else patient_dir
    rp_path = patch_rp_missing_ssd(rp_path, tmp_dir / f"_RP_ssd_patched_{patient_dir.name}.dcm")

    patient, arcos = load_dicom(
        str(patient_dir),
        dose_path=[str(p) for p in rd_paths],
        plan_path=[str(rp_path)],
        struct_path=str(rs_path),
        struct_names=struct_names,
        new_spacing=new_spacing,
        crop_volume=crop_volume,
        device=device,
        dtype=dtype,
    )
    if len(arcos) != 2:
        print(
            f"[build_beams] Aviso: {patient_dir.name} tiene {len(arcos)} arco(s) VMAT "
            "cargado(s), no 2 (alcance inicial del proyecto es próstata 2 arcos)."
        )
    return patient, arcos
