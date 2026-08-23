"""Escribe un RP DICOM real con los leaves/MU/jaws optimizados por
`src/planning/mimicking.py` (PIPELINE_KBP_PDRT.md paso 5) — PyDoseRT sólo lee
DICOM, así que el RP optimizado se construye acá con pydicom, usando el RP
real del paciente como template y copiando todo lo demás sin tocar.

Sólo se sobreescriben, por control point, los tags que `mimicking.py`
realmente optimiza:
  - BeamLimitingDevicePositionSequence MLCX -> LeafJawPositions (leaves)
  - BeamLimitingDevicePositionSequence ASYMY -> LeafJawPositions (jaws Y)
  - CumulativeMetersetWeight
y a nivel de haz:
  - FractionGroupSequence[0].ReferencedBeamSequence[].BeamMeterset (MU total)

Gantry, colimador, isocentro y SSD quedan iguales al plan real — mimicking.py
no los optimiza (`OptimizableArc` los copia fijos del `BeamSequence` real).

Además se genera un SOPInstanceUID/SeriesInstanceUID nuevos para el plan
(Eclipse no permite importar un RP con el mismo SOPInstanceUID que uno ya
existente en la base) y se resetea ApprovalStatus a UNAPPROVED (el plan real
está aprobado/tratado; este es nuevo, no aprobado). StudyInstanceUID y
FrameOfReferenceUID NO se tocan — el plan optimizado sigue perteneciendo al
mismo estudio/CT del paciente. `write_optimized_rtplan` devuelve el nuevo
SOPInstanceUID para que un RD que lo referencie (`scripts/write_rd_dicom.py
--referenced-plan-uid`) apunte al plan correcto.

Mapeo control-point <-> índice de tensor: idéntico al que usa
`pydosert.data.utils.dicom_utils.fetch_plan_data` para leer el RP (por eso
`load_patient_and_arcs` devuelve los arcos en ese mismo orden/subset) — sólo
los control points con una entrada MLCX cuentan como beam_data, y dentro de
esos, la entrada ASYMY (si existe) comparte el mismo índice.
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import generate_uid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

RTPLAN_LABEL_SUFFIX = "_PDRT"  # RTPlanLabel es VR 'SH', maximo 16 caracteres


def _mlcx_and_asymy(cps):
    mlcx = asymy = None
    for seq in cps.BeamLimitingDevicePositionSequence:
        if seq.RTBeamLimitingDeviceType == "MLCX":
            mlcx = seq
        elif seq.RTBeamLimitingDeviceType == "ASYMY":
            asymy = seq
    return mlcx, asymy


def _is_dynamic_beam(beam) -> bool:
    """Mismo criterio que fetch_plan_data: al menos un control point con
    una entrada MLCX (los beams de setup/verificación no la tienen)."""
    for cps in beam.ControlPointSequence:
        if "BeamLimitingDevicePositionSequence" not in cps:
            continue
        if any(s.RTBeamLimitingDeviceType == "MLCX" for s in cps.BeamLimitingDevicePositionSequence):
            return True
    return False


def write_optimized_rtplan(template_rp_path, optim_arcs: list, out_rp_path) -> Path:
    """
    Args:
        template_rp_path: RP.*.dcm real del paciente (no se modifica, se lee
            y se guarda una copia aparte en `out_rp_path`).
        optim_arcs: lista de `OptimizableArc` (mimicking.py), en el MISMO
            ORDEN que devolvió `load_patient_and_arcs` para ese mismo RP
            (que a su vez sigue el orden de `ds.BeamSequence`, filtrando los
            beams estáticos sin MLCX).
        out_rp_path: ruta de salida para el RP optimizado.

    Returns:
        tuple[Path, str]: Path al RP escrito, y el SOPInstanceUID nuevo del
        plan (para pasarlo como `--referenced-plan-uid` a write_rd_dicom.py).
    """
    ds = pydicom.dcmread(str(template_rp_path))

    nuevo_sop = generate_uid()
    ds.SOPInstanceUID = nuevo_sop
    ds.SeriesInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = nuevo_sop
    now = datetime.now()
    ds.InstanceCreationDate = now.strftime("%Y%m%d")
    ds.InstanceCreationTime = now.strftime("%H%M%S.%f")
    if hasattr(ds, "RTPlanLabel"):
        max_base_len = 16 - len(RTPLAN_LABEL_SUFFIX)
        ds.RTPlanLabel = str(ds.RTPlanLabel)[:max_base_len] + RTPLAN_LABEL_SUFFIX
    if hasattr(ds, "ApprovalStatus"):
        ds.ApprovalStatus = "UNAPPROVED"

    dynamic_beams = [beam for beam in ds.BeamSequence if _is_dynamic_beam(beam)]
    if len(dynamic_beams) != len(optim_arcs):
        raise ValueError(
            f"El RP template tiene {len(dynamic_beams)} beams dinámicos (con MLCX), "
            f"pero se pasaron {len(optim_arcs)} optim_arcs — deben corresponder 1 a 1, "
            f"en el mismo orden que devolvió load_patient_and_arcs."
        )

    beam_meterset_refs = {
        str(ref.ReferencedBeamNumber): ref
        for ref in ds.FractionGroupSequence[0].ReferencedBeamSequence
    }

    for beam, oa in zip(dynamic_beams, optim_arcs):
        bs = oa.to_beam_sequence()
        leaf_positions = bs.leaf_positions.detach().cpu().numpy()  # [CP, N, 2] (left, right) mm
        jaw_positions = bs.jaw_positions.detach().cpu().numpy()    # [CP, 2] (lower, upper) mm
        mus = bs.mus.detach().cpu().numpy()                        # [CP] MU incremental

        cum_mu = np.cumsum(mus)
        total_mu = float(cum_mu[-1])
        cum_weight = cum_mu / total_mu
        cum_weight[0] = 0.0  # convención DICOM: el primer CP arranca en peso 0

        idx = 0
        for cps in beam.ControlPointSequence:
            if "BeamLimitingDevicePositionSequence" not in cps:
                continue
            mlcx, asymy = _mlcx_and_asymy(cps)
            if mlcx is None:
                continue

            left = leaf_positions[idx, :, 0]
            right = leaf_positions[idx, :, 1]
            mlcx.LeafJawPositions = [float(v) for v in np.concatenate([left, right])]

            if asymy is not None:
                asymy.LeafJawPositions = [float(v) for v in jaw_positions[idx]]

            if "CumulativeMetersetWeight" in cps:
                cps.CumulativeMetersetWeight = float(cum_weight[idx])

            idx += 1

        if idx != leaf_positions.shape[0]:
            raise ValueError(
                f"Beam {beam.BeamNumber}: {idx} control points con MLCX en el RP template, "
                f"pero el arco optimizado tiene {leaf_positions.shape[0]} — no corresponden."
            )

        ref = beam_meterset_refs.get(str(beam.BeamNumber))
        if ref is not None and hasattr(ref, "BeamMeterset"):
            ref.BeamMeterset = total_mu

    out_rp_path = Path(out_rp_path)
    out_rp_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(out_rp_path), write_like_original=False)
    return out_rp_path, nuevo_sop
