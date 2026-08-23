import sys
from pathlib import Path
import torch

REPO = Path(r"c:\Pablo\ProstateDoseProject\repo")
sys.path.insert(0, str(REPO))
from src.planning.build_beams import load_patient_and_arcs
from src.planning.engine_setup import load_machine_config
from src.planning.mimicking import OptimizableArc, deliverability_loss

DIR = r"c:\Pablo\ProstateDoseProject\dicom_pilot\PT_003fe2bb84986507"
patient, arcs = load_patient_and_arcs(DIR, device=torch.device("cpu"), dtype=torch.float32)
machine_config = load_machine_config()
ptv_mask = patient.structures["PTV_High"]

for lm in [5.0, 3.0, 2.0]:
    optim_arcs = [
        OptimizableArc(
            a, ptv_conform_start=True, ptv_mask=ptv_mask, resolution=patient.resolution,
            leaf_widths=machine_config.leaf_widths, leaf_margin_mm=lm, jaw_margin_mm=2.0,
            leaf_max_step_mm=None, machine_config=machine_config, optimize_jaws=False,
        )
        for a in arcs
    ]
    dl = deliverability_loss(optim_arcs, machine_config)
    dg = dl["diag"]
    print(f"leaf_margin_mm={lm}: INIT (sin ningun paso de optimizacion) -- "
          f"exceso_max={dg['leaf_viol_max_mm']:.3f}mm ({dg['leaf_viol_n']} pares)  "
          f"uso_max={100*dg['leaf_step_frac_max']:.1f}%  p90={dg['leaf_step_p90_mm']:.2f}mm  "
          f"deliv_raw={float(dl['total']):.6f}")
