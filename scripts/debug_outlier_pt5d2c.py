"""Diagnostico puntual del paciente PT_5d2c6ee9551e7804 (test, NArcos=2) que diverge
solo en exp_hipo_001b_baseline_clean (mae_ptv=78.7, mae_rectum=41.4, mae_bladder=42.4)
pero predice bien en zero-shot exp002 y en exp_hipo_002b_finetune_clean.

Corre los 3 checkpoints sobre el mismo NPZ y compara estadisticas de la dosis predicha
(min/max/mean dentro de cada mascara) + genera figura DVH+slice para los 3, para
distinguir "mapa de dosis roto" (fallo real del modelo) de un problema aguas abajo del
computo de DVH/evaluate.

Uso:
    python scripts/debug_outlier_pt5d2c.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.lightning_module import DosePredictionModule
from evaluate import cargar_modelo, figura_paciente, dvh_metrics

ANONID = "PT_5d2c6ee9551e7804"
NPZ_PATH = Path(r"C:\Pablo\ProstateDoseProject\processed_hipo") / f"{ANONID}.npz"

CHECKPOINTS = {
    "zero_shot_exp002":   ("checkpoints/exp002_unet2d_psdm/epoch=191.ckpt", "configs/exp002_unet2d_psdm.yaml"),
    "finetune_clean":     ("checkpoints/exp_hipo_002b_finetune_clean/epoch=028.ckpt", "configs/exp_hipo_002b_finetune_clean.yaml"),
    "baseline_clean":     ("checkpoints/exp_hipo_001b_baseline_clean/epoch=337.ckpt", "configs/exp_hipo_001b_baseline_clean.yaml"),
}

OUT_DIR = Path("results/debug_outlier_PT_5d2c6ee9551e7804")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_patient_npz(path: Path) -> dict:
    data = np.load(str(path), allow_pickle=True)
    meta = json.loads(str(data['meta'][0]))
    out = {
        'ct':           torch.from_numpy(np.array(data['ct'], dtype=np.float32)),
        'dose':         torch.from_numpy(np.array(data['dose'], dtype=np.float32)),
        'ptv_mask':     torch.from_numpy(np.array(data['ptv_mask'], dtype=np.uint8)).float(),
        'body_mask':    torch.from_numpy(np.array(data['body_mask'], dtype=np.uint8)).float(),
        'rectum_mask':  torch.from_numpy(np.array(data['rectum_mask'], dtype=np.uint8)).float(),
        'bladder_mask': torch.from_numpy(np.array(data['bladder_mask'], dtype=np.uint8)).float(),
        'psdm_ptv':     torch.from_numpy(np.array(data['psdm_ptv'], dtype=np.float32)),
        'psdm_rectum':  torch.from_numpy(np.array(data['psdm_rectum'], dtype=np.float32)),
        'psdm_bladder': torch.from_numpy(np.array(data['psdm_bladder'], dtype=np.float32)),
        'anonid': meta.get('anonid', path.stem),
    }
    data.close()
    return out


def main():
    sample = load_patient_npz(NPZ_PATH)
    print(f"Paciente: {ANONID}")
    print(f"  n_slices: {sample['ct'].shape[0]}")
    print(f"  dose real rango: [{sample['dose'].min():.2f}, {sample['dose'].max():.2f}] (%rx)")
    for m in ['ptv_mask', 'body_mask', 'rectum_mask', 'bladder_mask']:
        print(f"  {m} voxels: {int(sample[m].sum())}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for nombre, (ckpt_path, cfg_path) in CHECKPOINTS.items():
        cfg = OmegaConf.load(cfg_path)
        model, device = cargar_modelo(ckpt_path, cfg)

        batch = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
        with torch.no_grad():
            x = model._build_input(batch)
            pred = model(x)
        dose_pred = pred[0].cpu().numpy()
        dose_real = sample['dose'].numpy()
        ptv = sample['ptv_mask'].numpy()
        rectum = sample['rectum_mask'].numpy()
        bladder = sample['bladder_mask'].numpy()
        body = sample['body_mask'].numpy()
        ct = sample['ct'].numpy()

        print(f"\n=== {nombre} ({ckpt_path}) ===")
        print(f"  dose_pred rango global: [{dose_pred.min():.2f}, {dose_pred.max():.2f}]  mean={dose_pred.mean():.2f}")
        for etiqueta, mask in [("BODY", body), ("PTV", ptv), ("Rectum", rectum), ("Bladder", bladder)]:
            roi_pred = dose_pred[mask > 0]
            roi_real = dose_real[mask > 0]
            print(f"  {etiqueta:8s} pred: mean={roi_pred.mean():7.2f} min={roi_pred.min():7.2f} max={roi_pred.max():7.2f}"
                  f"   | real: mean={roi_real.mean():7.2f} min={roi_real.min():7.2f} max={roi_real.max():7.2f}")

        figura_paciente(
            ANONID, ct, dose_real, dose_pred, body, ptv, rectum, bladder,
            OUT_DIR / f"{nombre}_{ANONID}.png",
        )
        print(f"  Figura guardada: {OUT_DIR / f'{nombre}_{ANONID}.png'}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
