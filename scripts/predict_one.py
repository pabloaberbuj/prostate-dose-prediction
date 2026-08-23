"""
Inferencia standalone para 1 paciente — piloto RD DICOM.

No usa DoseDataModule (evita el padding en Z pensado para batchear >1 paciente,
innecesario acá) ni evaluate.py completo (que siempre itera todo el test set).
Carga el NPZ igual que DosePatientDataset._load_npz + __getitem__ (mismo dtype,
mismo cast a float32) y llama directo a model._build_input / model.forward para
garantizar que el orden de canales coincide exactamente con el de entrenamiento.

Verificado contra el checkpoint (2026-08-07): pese a que
configs/exp002_unet2d_psdm.yaml trae `model.context_slices: 3` (residual sin usar
de otro experimento — _build_input lee `cfg.data.context_slices`, ausente en este
yaml/checkpoint), el checkpoint real de exp002 fue entrenado con
`inc.block.0.weight.shape == (16, 5, 3, 3)` → 5 canales, sin contexto axial.
Ese es el comportamiento que este script reproduce (no hay bug que evitar, ya
que _build_input cae al default context_slices=1 en ambos casos).

Uso:
    python scripts/predict_one.py \
        --patient-id PT_003fe2bb84986507 \
        --checkpoint checkpoints/exp002_unet2d_psdm/epoch=191.ckpt \
        --config configs/exp002_unet2d_psdm.yaml \
        --output-dir results/pilot_rd
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import cargar_modelo  # noqa: E402


def cargar_npz_paciente(npz_path: Path) -> dict:
    """Replica DosePatientDataset._load_npz + el cast a float32 de __getitem__."""
    data = np.load(str(npz_path), allow_pickle=True)
    meta = json.loads(str(data["meta"][0]))
    sample = {
        "ct":           torch.from_numpy(np.array(data["ct"],           dtype=np.float32)),
        "dose":         torch.from_numpy(np.array(data["dose"],         dtype=np.float32)),
        "ptv_mask":     torch.from_numpy(np.array(data["ptv_mask"],     dtype=np.float32)),
        "body_mask":    torch.from_numpy(np.array(data["body_mask"],    dtype=np.float32)),
        "rectum_mask":  torch.from_numpy(np.array(data["rectum_mask"],  dtype=np.float32)),
        "bladder_mask": torch.from_numpy(np.array(data["bladder_mask"], dtype=np.float32)),
        "psdm_ptv":     torch.from_numpy(np.array(data["psdm_ptv"],     dtype=np.float32)),
        "psdm_rectum":  torch.from_numpy(np.array(data["psdm_rectum"],  dtype=np.float32)),
        "psdm_bladder": torch.from_numpy(np.array(data["psdm_bladder"], dtype=np.float32)),
    }
    data.close()
    return sample, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", required=True, help="Ej. PT_003fe2bb84986507")
    parser.add_argument("--checkpoint", default="checkpoints/exp002_unet2d_psdm/epoch=191.ckpt")
    parser.add_argument("--config", default="configs/exp002_unet2d_psdm.yaml")
    parser.add_argument("--processed-dir", default=None,
                        help="Override cfg.data.processed_dir; default = el del config")
    parser.add_argument("--output-dir", default="results/pilot_rd")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.processed_dir is not None:
        cfg.data.processed_dir = args.processed_dir

    torch.set_float32_matmul_precision("high")

    model, device = cargar_modelo(args.checkpoint, cfg)

    # Chequeo de canales: falla rápido y claro si el checkpoint no matchea in_channels.
    conv1 = model.model.inc.block[0]
    print(f"Conv1 in_channels real del checkpoint: {conv1.in_channels} "
          f"(cfg.model.in_channels={cfg.model.in_channels})")
    if conv1.in_channels != cfg.model.in_channels:
        raise RuntimeError(
            f"Mismatch: cfg.model.in_channels={cfg.model.in_channels} pero el "
            f"checkpoint espera {conv1.in_channels} canales de entrada."
        )

    npz_path = Path(cfg.data.processed_dir) / f"{args.patient_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No existe {npz_path}")

    sample, meta = cargar_npz_paciente(npz_path)
    print(f"Paciente: {args.patient_id}  n_slices={meta['n_slices']}  "
          f"spacing_mm={meta['spacing_mm']}  z_range={meta['z_range']}")

    # Batch dim = 1 (el modelo requiere (B, Z, H, W) por canal de entrada)
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}

    with torch.no_grad():
        x = model._build_input(batch)      # (1, Z, C, H, W)
        pred = model(x)                    # (1, Z, H, W)
    dose_pred = pred[0].cpu().numpy().astype(np.float32)

    print(f"dose_pred: shape={dose_pred.shape} min={dose_pred.min():.2f} "
          f"max={dose_pred.max():.2f} (% Rx)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"pred_{args.patient_id}.npz"
    np.savez_compressed(
        str(out_path),
        dose_pred=dose_pred,
        ptv_mask=sample["ptv_mask"].numpy().astype(np.uint8),
        body_mask=sample["body_mask"].numpy().astype(np.uint8),
        rectum_mask=sample["rectum_mask"].numpy().astype(np.uint8),
        bladder_mask=sample["bladder_mask"].numpy().astype(np.uint8),
        meta=np.array([json.dumps(meta)]),
    )
    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
