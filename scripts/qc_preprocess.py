"""
Script principal de entrenamiento.

Uso:
    python scripts/train.py --config configs/exp001_unet2d_baseline.yaml
    python scripts/train.py --config configs/exp001_unet2d_baseline.yaml --fast-dev-run
    python scripts/train.py --config configs/exp001_unet2d_baseline.yaml --no-wandb
"""

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

# Asegurar que src/ está en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datamodules.dose_datamodule import DoseDataModule
from src.models.lightning_module import DosePredictionModule
from src.callbacks.logging_callbacks import DVHLoggingCallback, SliceLoggingCallback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-dev-run", action="store_true",
                        help="1 batch para debug rápido")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path a checkpoint para reanudar")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Desactivar logging a W&B (usa CSVLogger local)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    torch.set_float32_matmul_precision('high')
    pl.seed_everything(cfg.experiment.seed, workers=True)

    # DataModule
    datamodule = DoseDataModule(cfg)

    # Modelo
    model = DosePredictionModule(cfg)

    # Logger
    logger = False
    if not args.no_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger
            logger = WandbLogger(
                project=cfg.logging.wandb_project,
                entity=cfg.logging.wandb_entity,
                name=cfg.experiment.name,
                config=OmegaConf.to_container(cfg, resolve=True),
                save_dir="lightning_logs",
            )
        except Exception as e:
            print(f"[WARN] W&B no disponible ({e}). Usando CSVLogger.")
            from pytorch_lightning.loggers import CSVLogger
            logger = CSVLogger("lightning_logs", name=cfg.experiment.name)
    else:
        from pytorch_lightning.loggers import CSVLogger
        logger = CSVLogger("lightning_logs", name=cfg.experiment.name)

    # Callbacks
    ckpt_dir = Path("checkpoints") / cfg.experiment.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    monitor = cfg.training.monitor_metric

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="{epoch:03d}",
            monitor=monitor,
            mode=cfg.training.monitor_mode,
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=True,
        ),
        EarlyStopping(
            monitor=monitor,
            mode=cfg.training.monitor_mode,
            patience=cfg.training.early_stopping_patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        DVHLoggingCallback(
            every_n_epochs=cfg.logging.log_dvh_every_n_epochs,
            num_samples=cfg.logging.num_visual_samples,
        ),
        SliceLoggingCallback(
            every_n_epochs=cfg.logging.log_visual_every_n_epochs,
            num_samples=cfg.logging.num_visual_samples,
        ),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.gradient_clip_val,
        logger=logger,
        callbacks=callbacks,
        deterministic=False,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
