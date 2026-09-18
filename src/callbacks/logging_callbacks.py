"""
Callbacks personalizados para logging a W&B:
- DVHLoggingCallback: cada N epochs loguea DVHs comparativos de N pacientes de val.
- SliceLoggingCallback: cada N epochs loguea cortes axiales (real vs predicha vs diff).
"""

import io
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf

# Paleta cíclica para DVHLoggingCallback — antes 'PTV'=blue/'Rectum'=red/'Bladder'=cyan
# hardcodeado (3 colores fijos); con anatomías de más de 3 estructuras hace falta un
# esquema que no dependa de nombres fijos. Los primeros 3 colores reproducen el orden
# de próstata para no cambiar el aspecto visual de los runs existentes.
_DVH_COLOR_CYCLE = ['blue', 'red', 'cyan', 'green', 'orange', 'purple', 'brown']


def _calcular_dvh(dosis_3d: np.ndarray, mascara: np.ndarray,
                  n_bins: int = 150) -> tuple:
    """Devuelve (bins, vol_acumulado_pct)."""
    dosis_roi = dosis_3d[mascara > 0]
    if len(dosis_roi) == 0:
        return np.array([0, 150]), np.array([100, 0])
    bins = np.linspace(0, 130, n_bins)
    vol_pct = np.array([100.0 * (dosis_roi >= b).sum() / len(dosis_roi)
                        for b in bins])
    return bins, vol_pct


class EpochSummaryCallback(pl.Callback):
    """Imprime a stdout un resumen de metricas clave cada `every_n_epochs` epochs.

    Complementa la progress bar de Lightning: cuando la salida se redirige a un archivo
    (ej. `... | Tee-Object -FilePath log.txt` en PowerShell) la progress bar no se refresca
    bien porque depende de retorno de carro, y monitorear el run se hace dificil. Esto
    imprime una linea nueva por epoch, siempre legible en un log plano.
    """

    def __init__(self, every_n_epochs: int = 1):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        def g(k):
            v = trainer.callback_metrics.get(k)
            return f"{v.item():.4f}" if v is not None else "—"

        lr = trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else float('nan')
        print(
            f"[epoch {trainer.current_epoch + 1:03d}/{trainer.max_epochs}] "
            f"train/mae={g('train/mae')}  val/mae={g('val/mae')}  "
            f"val/mae_ptv_pct={g('val/mae_ptv_pct')}  val/mae_rectum_pct={g('val/mae_rectum_pct')}  "
            f"val/mae_bladder_pct={g('val/mae_bladder_pct')}  val/dvh_score={g('val/dvh_score')}  "
            f"lr={lr:.2e}",
            flush=True,
        )


class DVHLoggingCallback(pl.Callback):
    """Loguea N DVHs comparativos a W&B cada `every_n_epochs`."""

    def __init__(self, every_n_epochs: int = 5, num_samples: int = 3, structures: list = None):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        # Estructuras no-body a graficar (ver anatomy schema, src/config/anatomy.py).
        # train.py siempre pasa `structures=cfg.anatomy_schema.structures` (lista de
        # DictConfig con .key/.role); default None solo por si algún llamador viejo
        # (tests, notebooks) no las pasa todavía — reproduce el próstata hardcodeado
        # de antes del refactor.
        if structures is None:
            structures = OmegaConf.create([
                {'key': 'ptv', 'role': 'target'},
                {'key': 'rectum', 'role': 'oar'},
                {'key': 'bladder', 'role': 'oar'},
            ])
        self.structures = [s for s in structures if s.role != 'body']

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        if not hasattr(pl_module, '_val_first_batch'):
            return
        batch = pl_module._val_first_batch
        pred   = batch['pred'].numpy()    # (B, Z, H, W)
        target = batch['target'].numpy()
        masks  = {s.key: batch['batch'][f'{s.key}_mask'].numpy() for s in self.structures}
        anonids = batch['batch']['anonid']

        n = min(self.num_samples, pred.shape[0])
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        for i in range(n):
            ax = axes[0, i]
            for s, color in zip(self.structures, itertools.cycle(_DVH_COLOR_CYCLE)):
                nombre = s.key.capitalize()
                mask_v = masks[s.key][i]
                if mask_v.sum() == 0:
                    continue
                bins_r, vol_r = _calcular_dvh(target[i], mask_v)
                bins_p, vol_p = _calcular_dvh(pred[i],   mask_v)
                ax.plot(bins_r, vol_r, color=color, linewidth=1.5, label=f'{nombre} real')
                ax.plot(bins_p, vol_p, color=color, linewidth=1.5, linestyle='--',
                        label=f'{nombre} pred')
            ax.set_xlim(0, 130)
            ax.set_ylim(0, 105)
            ax.set_xlabel('Dosis (% pres)', fontsize=8)
            ax.set_ylabel('Volumen (%)', fontsize=8)
            ax.set_title(anonids[i] if isinstance(anonids[i], str) else anonids[i][0], fontsize=8)
            ax.legend(fontsize=6, loc='upper right')
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)

        plt.tight_layout()
        if (trainer.logger is not None and hasattr(trainer.logger, 'experiment')
                and hasattr(trainer.logger.experiment, 'log')):
            trainer.logger.experiment.log({
                "val/dvh_comparativo": wandb.Image(fig),
                "epoch": trainer.current_epoch,
            })
        plt.close(fig)


class SliceLoggingCallback(pl.Callback):
    """Loguea cortes axiales (real vs predicho vs diff) a W&B cada N epochs."""

    def __init__(self, every_n_epochs: int = 10, num_samples: int = 3,
                 structures: list = None, primary_target: str = 'ptv'):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        # train.py siempre pasa structures=cfg.anatomy_schema.structures y
        # primary_target=cfg.anatomy_schema.primary_target; defaults = próstata
        # (comportamiento pre-refactor) por si algún llamador viejo no los pasa.
        if structures is None:
            structures = OmegaConf.create([{'key': 'body', 'role': 'body'}])
        self.body_key = next((s.key for s in structures if s.role == 'body'), 'body')
        self.primary_target = primary_target

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        if not hasattr(pl_module, '_val_first_batch'):
            return
        batch = pl_module._val_first_batch
        pred   = batch['pred'].numpy()
        target = batch['target'].numpy()
        ct     = batch['batch']['ct'].numpy()
        body   = batch['batch'][f'{self.body_key}_mask'].numpy()
        anonids = batch['batch']['anonid']

        n = min(self.num_samples, pred.shape[0])
        fig, axes = plt.subplots(n, 3, figsize=(11, 3.5 * n), squeeze=False)
        for i in range(n):
            # Encontrar corte axial con más presencia del target primario
            target_v = batch['batch'][f'{self.primary_target}_mask'][i].numpy()
            slices_con_target = np.where(target_v.sum(axis=(1, 2)) > 0)[0]
            z = slices_con_target[len(slices_con_target)//2] if len(slices_con_target) > 0 else pred.shape[1]//2

            real_s = target[i, z]
            pred_s = pred[i,   z]
            diff_s = pred_s - real_s

            for col, (arr, titulo, vmin, vmax, cmap) in enumerate([
                (real_s, 'Real',     0, 110, 'jet'),
                (pred_s, 'Predicha', 0, 110, 'jet'),
                (diff_s, 'Pred-Real', -20, 20, 'RdBu_r'),
            ]):
                ax = axes[i, col]
                ax.imshow(ct[i, z], cmap='gray', vmin=-1, vmax=1)
                arr_masked = np.where(body[i, z] > 0, arr, np.nan)
                im = ax.imshow(arr_masked, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.6)
                plt.colorbar(im, ax=ax, fraction=0.046)
                ax.set_title(f"{anonids[i] if isinstance(anonids[i], str) else anonids[i][0]} z={z} — {titulo}",
                             fontsize=8)
                ax.axis('off')

        plt.tight_layout()
        if (trainer.logger is not None and hasattr(trainer.logger, 'experiment')
                and hasattr(trainer.logger.experiment, 'log')):
            trainer.logger.experiment.log({
                "val/cortes": wandb.Image(fig),
                "epoch": trainer.current_epoch,
            })
        plt.close(fig)
