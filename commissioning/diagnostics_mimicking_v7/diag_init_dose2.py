import sys
from pathlib import Path
import torch

REPO = Path(r"c:\Pablo\ProstateDoseProject\repo")
sys.path.insert(0, str(REPO))
from src.planning.build_beams import load_patient_and_arcs, STRUCT_NAMES_DEFAULT
from src.planning.engine_setup import load_machine_config, build_dose_engine
from src.planning.mimicking import OptimizableArc, dvh_loss, compute_dvh_targets
from src.bridge.unet_to_target import unet_dose_to_pdrt_grid

DIR = r"c:\Pablo\ProstateDoseProject\dicom_pilot\PT_003fe2bb84986507"
PRED = r"c:\Pablo\ProstateDoseProject\repo\results\pilot_rd\pred_PT_003fe2bb84986507.npz"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

patient, arcs = load_patient_and_arcs(DIR, struct_names=STRUCT_NAMES_DEFAULT, device=DEV, dtype=torch.float32)
machine_config = load_machine_config()
ptv_mask = patient.structures["PTV_High"]
density_image = patient.density_image.to(device=DEV, dtype=torch.float32).unsqueeze(0)

dose_target_np, _ = unet_dose_to_pdrt_grid(DIR, PRED, 78.0, 39)
dose_target = torch.from_numpy(dose_target_np).to(device=DEV, dtype=torch.float32)
rx_frac_gy = 78.0 / 39

engine = build_dose_engine(patient, beam_template=arcs[0], device=DEV, dtype=torch.float32)
engine.eval()

dvh_targets = compute_dvh_targets(dose_target, patient, rx_frac_gy)
print(f"target D98={dvh_targets['ptv_d98_gy']:.3f}Gy (raw={dvh_targets['ptv_d98_gy_raw']:.3f}) "
      f"V70={dvh_targets['rectum_v70_target_pct']:.1f}% V50={dvh_targets['bladder_v50_target_pct']:.1f}%")

for lm in [5.0, 2.0]:
    optim_arcs = [
        OptimizableArc(
            a, ptv_conform_start=True, ptv_mask=ptv_mask, resolution=patient.resolution,
            leaf_widths=machine_config.leaf_widths, leaf_margin_mm=lm, jaw_margin_mm=2.0,
            leaf_max_step_mm=None, machine_config=machine_config, optimize_jaws=False,
        )
        for a in arcs
    ]
    with torch.no_grad():
        dose_pred = None
        for oa in optim_arcs:
            bs = oa.to_beam_sequence()
            d = engine.compute_dose(bs, density_image=density_image, beam_chunk_size=1)
            dose_pred = d if dose_pred is None else dose_pred + d
        dose_pred = dose_pred[0]
    dh = dvh_loss(dose_pred, patient, dvh_targets)
    d = dh["diag"]
    print(f"leaf_margin_mm={lm}: INIT (dosis cruda, sin optimizar) -- "
          f"D98={d['ptv_d98_actual_gy']:.3f}/{d['ptv_d98_target_gy']:.3f}Gy  "
          f"V70={d['rectum_v70_actual_pct']:.1f}/{d['rectum_v70_target_pct']:.1f}%  "
          f"V50={d['bladder_v50_actual_pct']:.1f}/{d['bladder_v50_target_pct']:.1f}%  "
          f"dvh_raw={float(dh['total']):.4f}")
