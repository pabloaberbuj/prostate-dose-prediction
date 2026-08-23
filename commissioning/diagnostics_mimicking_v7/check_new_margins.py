import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(r"c:\Pablo\ProstateDoseProject\repo")
sys.path.insert(0, str(REPO))
from src.planning.build_beams import load_patient_and_arcs
from src.planning.engine_setup import load_machine_config
from src.planning.ptv_conforming_init import compute_ptv_conforming_aperture

sys.path.insert(0, r"C:\Users\MEVATE~1\AppData\Local\Temp\claude\c--Pablo-ProstateDoseProject\970561e7-7351-4ce3-9f9f-79b199f3b6ab\scratchpad")
from bev_check import cobertura, leer_rp, ptv_puntos
import glob

DIR = r"c:\Pablo\ProstateDoseProject\dicom_pilot\PT_003fe2bb84986507"
patient, arcs = load_patient_and_arcs(DIR, device=torch.device("cpu"), dtype=torch.float32)
machine_config = load_machine_config()
ptv_mask = patient.structures["PTV_High"]

rs = glob.glob(DIR + r"\RS*.dcm")[0]
pts = ptv_puntos(rs)
rp = glob.glob(DIR + r"\RP.*.dcm")[0]
arcos_c, bounds = leer_rp(rp)
sg, sc = 1, 1

LEAF_MARGIN_NEW = 2.0
EXTRA_NEW = 3.0

for i, a in enumerate(arcs):
    leaf_init, jaw_init = compute_ptv_conforming_aperture(
        ptv_mask, a.gantry_angles, a.collimator_angles, iso_center=a.iso_center, sid=a.sid,
        resolution=patient.resolution, leaf_widths=machine_config.leaf_widths, field_size=a.field_size,
        leaf_margin_mm=LEAF_MARGIN_NEW, jaw_margin_mm=2.0, leaf_max_step_mm=None,
        max_leaf_speed_mm_s=float(machine_config.maximum_leaf_speed),
        max_gantry_speed_deg_s=float(machine_config.maximum_gantry_angle_speed),
        speed_safety=0.80,
    )
    gantry_deg = torch.rad2deg(a.gantry_angles).numpy()
    coll_deg = float(torch.rad2deg(a.collimator_angles[0]))
    iso_real = arcos_c[i]["iso"]
    n = leaf_init.shape[1]

    # init puro (arranque)
    L_init = torch.cat([leaf_init[..., 0], leaf_init[..., 1]], dim=1).numpy()
    arco_init = dict(leaf=L_init, gantry=gantry_deg, iso=iso_real, coll=coll_deg)
    cub0, fue0 = cobertura(arco_init, bounds, pts, sg, sc)

    # peor caso: el optimizador usa TODO el margen extra permitido para abrir mas
    leaf_max = leaf_init.clone()
    leaf_max[..., 0] -= EXTRA_NEW
    leaf_max[..., 1] += EXTRA_NEW
    L_max = torch.cat([leaf_max[..., 0], leaf_max[..., 1]], dim=1).numpy()
    arco_max = dict(leaf=L_max, gantry=gantry_deg, iso=iso_real, coll=coll_deg)
    cub1, fue1 = cobertura(arco_max, bounds, pts, sg, sc)

    print(f"arco {i}: init nuevo (margen={LEAF_MARGIN_NEW}mm)      -> cobertura {cub0*100:5.1f}%  fuera {fue0*100:5.1f}%")
    print(f"arco {i}: peor caso (+extra={EXTRA_NEW}mm de margen)   -> cobertura {cub1*100:5.1f}%  fuera {fue1*100:5.1f}%")

print("\nreferencia: plan clinico ~7.0%/6.7% fuera; v6 (margenes viejos 5+10mm) init 25.7%/25.8%, final optimizado 33.8%/32.9%")
