"""
Smoke test del pipeline de entrenamiento.

Verifica que:
  1. El DataModule carga correctamente desde NPZ.
  2. El modelo hace forward sin errores.
  3. La loss se calcula y backprop funciona.
  4. Los callbacks generan figuras sin romper.

NO entrena de verdad: corre solo 1-2 batches.

Uso:
    python scripts/smoke_test.py --config configs/exp001_unet2d_baseline.yaml
"""

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datamodules.dose_datamodule import DoseDataModule
from src.models.lightning_module import DosePredictionModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--processed-dir", default=None,
                        help="Override cfg.data.processed_dir (ruta a NPZs)")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.processed_dir is not None:
        cfg.data.processed_dir = args.processed_dir
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(42)

    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"Lightning: {pl.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 1) DataModule
    print("[1] Construyendo DataModule...")
    dm = DoseDataModule(cfg)
    dm.setup()
    print(f"  Train: {len(dm.train_ds)} pacientes")
    print(f"  Val:   {len(dm.val_ds)} pacientes")
    print(f"  Test:  {len(dm.test_ds)} pacientes")

    # 2) Cargar un batch
    print("\n[2] Cargando primer batch de train...")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print(f"  Claves batch: {list(batch.keys())}")
    print(f"  ct shape:     {batch['ct'].shape}")
    print(f"  dose shape:   {batch['dose'].shape}")
    print(f"  ct rango:     [{batch['ct'].min():.2f}, {batch['ct'].max():.2f}]")
    print(f"  dose rango:   [{batch['dose'].min():.2f}, {batch['dose'].max():.2f}]")

    # 3) Modelo
    print("\n[3] Construyendo modelo...")
    model = DosePredictionModule(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parámetros: {n_params:,}")

    # 4) Forward
    print("\n[4] Forward pass...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    x = model._build_input(batch)
    print(f"  Input al modelo: {x.shape} (B, Z, C, H, W)")
    pred = model(x)
    print(f"  Predicción:      {pred.shape}")
    print(f"  Pred rango:      [{pred.min():.2f}, {pred.max():.2f}]")

    # 5) Loss y backprop
    print("\n[5] Loss + backprop...")
    losses = model.loss_fn(pred, batch['dose'], batch['body_mask'],
                            {'ptv': batch['ptv_mask'], 'rectum': batch['rectum_mask'],
                             'bladder': batch['bladder_mask']})
    print(f"  Loss total: {losses['total'].item():.4f}")
    print(f"  Loss MAE:   {losses['mae'].item():.4f}")
    if 'moment' in losses:
        print(f"  Loss Mom:   {losses['moment'].item():.4f}")

    losses['total'].backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f"  Grad norm:  {grad_norm:.4f}")

    # 6) Trainer fast_dev_run
    print("\n[6] Trainer.fit con fast_dev_run=True...")
    trainer = pl.Trainer(
        fast_dev_run=2,
        accelerator="auto",
        devices=1,
        precision=cfg.training.precision,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=dm)

    print("\n" + "=" * 60)
    print("SMOKE TEST OK")
    print("=" * 60)


if __name__ == "__main__":
    main()
