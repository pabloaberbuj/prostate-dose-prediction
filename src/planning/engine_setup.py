"""Motor de dosis de PyDoseRT comisionado para el 6X (PIPELINE_KBP_PDRT.md,
sección B paso 3). Envuelve `PDRT.MachineConfig`/`PDRT.DoseEngine` apuntando
al resultado del comisionamiento (`commissioning/machine_config_6MV.json`,
ver CLAUDE_CODE_CONTEXT_070826.md sección "Comisionamiento PDRT 6X...").
"""
from pathlib import Path

import pydosert as PDRT
import torch

DEFAULT_MACHINE_CONFIG = (
    Path(__file__).resolve().parent.parent.parent / "commissioning" / "machine_config_6MV.json"
)
DEFAULT_KERNEL_SIZE = 25  # mismo valor que examples/rtplan.ipynb (voxels del kernel pencil-beam)


def load_machine_config(machine_config_path=DEFAULT_MACHINE_CONFIG) -> "PDRT.MachineConfig":
    """Config comisionada, sin motor asociado — para leer límites de máquina
    (maximum_leaf_speed, maximum_dose_rate, etc.) desde afuera del engine,
    por ejemplo para regularización de entregabilidad en mimicking.py.
    """
    return PDRT.MachineConfig(preset=str(machine_config_path))


def build_dose_engine(
    patient,
    machine_config_path=DEFAULT_MACHINE_CONFIG,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    beam_template=None,
    auto_calibrate: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> "PDRT.DoseEngine":
    """Arma un DoseEngine comisionado, con la grilla de dosis calzada a la del
    `patient` (mismo `resolution`/`density_image.shape` que devuelve
    `load_dicom` — así el target de mimicking, la CT y el motor comparten
    grilla sin resamplear nada extra).
    """
    config = load_machine_config(machine_config_path)
    engine = PDRT.DoseEngine(
        config,
        kernel_size,
        patient.resolution,
        patient.density_image.shape,
        beam_template=beam_template,
        auto_calibrate=auto_calibrate,
        dtype=dtype,
        device=device,
    )
    return engine
