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

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

# Asegurar que src/ está en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datamodules.dose_datamodule import DoseDataModule
from src.models.lightning_module import DosePredictionModule
from src.callbacks.logging_callbacks import DVHLoggingCallback, SliceLoggingCallback, EpochSummaryCallback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-dev-run", action="store_true",
                        help="1 batch para debug rápido")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path a checkpoint para reanudar (restaura trainer/optimizer/epoch)")
    parser.add_argument("--init-weights", type=str, default=None,
                        help="Path a checkpoint del cual cargar SOLO los pesos del modelo "
                             "(init desde otro experimento, ej. finetuning — NO resume el "
                             "estado del trainer/optimizer/epoch, a diferencia de --resume)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Desactivar logging a W&B (usa CSVLogger local)")
    parser.add_argument("--processed-dir", default=None,
                        help="Override cfg.data.processed_dir (ruta a NPZs)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.processed_dir is not None:
        cfg.data.processed_dir = args.processed_dir

    torch.set_float32_matmul_precision('high')
    if torch.cuda.is_available():
        # Techo duro de VRAM por proceso. En Windows/WDDM, un alloc que excede
        # la VRAM fisica NO tira OutOfMemoryError limpio por defecto -- el
        # driver empieza a paginar a RAM del sistema ("shared GPU memory") y
        # puede colgar toda la PC durante minutos (incidente real durante el
        # sondeo de memoria de exp_normo_3dunet). Con el cap, el allocator de
        # PyTorch respeta el limite y tira OOM limpio en vez de pagear.
        # Override opcional via env var (ej. datasets con pacientes de mas cortes
        # que requieren mas margen) -- default sigue siendo 0.88 si no se setea.
        import os as _os
        _mem_fraction = float(_os.environ.get("PROSTATE_CUDA_MEM_FRACTION", "0.88"))
        torch.cuda.set_per_process_memory_fraction(_mem_fraction, device=0)
    pl.seed_everything(cfg.experiment.seed, workers=True)

    # DataModule
    datamodule = DoseDataModule(cfg)

    # Modelo
    model = DosePredictionModule(cfg)

    if args.init_weights is not None:
        # Carga SOLO los pesos (state_dict) de otro checkpoint como inicialización
        # (ej. finetuning desde un experimento en otro dataset) — a diferencia de
        # --resume, no toca optimizer/scheduler/epoch: el training arranca en epoch 0
        # con el LR/horizonte de ESTA config.
        _orig_torch_load = torch.load
        def _load_no_weights_only(*a, **kw):
            kw['weights_only'] = False
            return _orig_torch_load(*a, **kw)
        torch.load = _load_no_weights_only
        print(f"Cargando pesos iniciales desde {args.init_weights} (init, NO resume)")
        init_ckpt = torch.load(args.init_weights, map_location='cpu')
        torch.load = _orig_torch_load
        resultado = model.load_state_dict(init_ckpt['state_dict'], strict=True)
        print(f"  state_dict cargado sin faltantes ni sobrantes: {resultado}")

    # Logger
    logger = False
    if not args.no_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger
            tags = getattr(cfg.logging, 'tags', None)
            logger = WandbLogger(
                project=cfg.logging.wandb_project,
                entity=cfg.logging.wandb_entity,
                name=cfg.experiment.name,
                tags=list(tags) if tags is not None else None,
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
        EpochSummaryCallback(every_n_epochs=getattr(cfg.logging, 'print_every_n_epochs', 1)),
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

    if args.resume is not None:
        # PyTorch 2.6+ cambió weights_only=True por default, rompiendo el resume de
        # checkpoints que incluyen OmegaConf DictConfig (mismo fix que evaluate.py).
        _orig_torch_load = torch.load
        def _load_no_weights_only(*a, **kw):
            kw['weights_only'] = False
            return _orig_torch_load(*a, **kw)
        torch.load = _load_no_weights_only

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)

    # Test con el mejor checkpoint (por val/mae) — deja test/* en el summary de W&B de
    # este mismo run, comparable directamente en la tabla del proyecto. El análisis
    # completo (OpenKBP scores, constraints, figuras) sigue haciéndose aparte con
    # scripts/evaluate.py, que es más rico que este test_step simple.
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()