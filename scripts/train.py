"""
Script principal de entrenamiento.

Uso:
    python scripts/train.py --config configs/exp001_unet2d_baseline.yaml
    python scripts/train.py --config configs/exp001_unet2d_baseline.yaml --fast-dev-run
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

# TODO: imports propios
# from src.datamodules.dose_datamodule import DoseDataModule
# from src.models.lightning_module import DosePredictionModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path al YAML de config")
    parser.add_argument("--fast-dev-run", action="store_true", help="1 batch para debug")
    parser.add_argument("--resume", type=str, default=None, help="Path a checkpoint a reanudar")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    pl.seed_everything(cfg.experiment.seed, workers=True)

    # ------------------- DataModule -------------------
    # datamodule = DoseDataModule(cfg)
    raise NotImplementedError("DataModule pendiente — etapa 3")

    # ------------------- Modelo -------------------
    # model = DosePredictionModule(cfg)

    # ------------------- Logger -------------------
    logger = WandbLogger(
        project=cfg.logging.wandb_project,
        entity=cfg.logging.wandb_entity,
        name=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        save_dir="lightning_logs",
    )

    # ------------------- Callbacks -------------------
    callbacks = [
        ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.experiment.name}",
            filename="{epoch:03d}-{val/mae:.4f}",
            monitor=cfg.training.monitor_metric,
            mode=cfg.training.monitor_mode,
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor=cfg.training.monitor_metric,
            mode=cfg.training.monitor_mode,
            patience=cfg.training.early_stopping_patience,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        # TODO: DVHLoggingCallback (etapa 3)
        # TODO: VisualLoggingCallback (etapa 3)
    ]

    # ------------------- Trainer -------------------
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.gradient_clip_val,
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        fast_dev_run=args.fast_dev_run,
    )

    # trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
