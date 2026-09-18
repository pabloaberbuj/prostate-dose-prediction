"""
LightningModule para predicción de dosis 2D con sampling por paciente.

Estrategia:
- Recibe un batch de volúmenes (B, Z, H, W).
- Procesa cortes en paralelo aplanando: (B*Z, C, H, W).
- Calcula loss sobre todo el volumen.
- Permite calcular DVH/MomentLoss por paciente porque tiene el volumen completo.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.checkpoint import checkpoint

from src.models.unet2d import build_model
from src.losses.losses import CombinedLoss
from src.config.anatomy import load_anatomy


class DosePredictionModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        # Guardar config (excluye paths para no contaminar el checkpoint)
        self.save_hyperparameters({"cfg": dict(cfg) if hasattr(cfg, 'keys') else cfg})
        self.cfg = cfg

        self.model = build_model(cfg)
        self.loss_fn = CombinedLoss(cfg)

        # Estructuras de la anatomía (ver src/config/anatomy.py) — reemplaza los nombres
        # hardcodeados 'ptv'/'rectum'/'bladder' que tenía este módulo antes del refactor
        # a anatomy schema. train.py/evaluate.py/evaluate_hipo.py ya cargan el config
        # con load_experiment_config() (que agrega cfg.anatomy_schema); este fallback
        # es solo para callers que todavía instancian con OmegaConf.load() directo
        # (algunos scripts auxiliares fuera del alcance de este refactor) — reproduce
        # el comportamiento de próstata de antes en vez de romper con AttributeError.
        anatomy_schema = cfg.get("anatomy_schema", None) if hasattr(cfg, "get") else None
        if anatomy_schema is None:
            anatomy_schema = load_anatomy("prostate")
        self.structures = anatomy_schema.structures
        self.primary_target = anatomy_schema.primary_target

        # Tracking de mejores métricas
        self.best_val_mae = float('inf')

    # ─── Forward y construcción de inputs ─────────────────────────────────────
    @staticmethod
    def _shift_z(t: torch.Tensor, offset: int) -> torch.Tensor:
        """
        Desplaza un volumen (B, Z, H, W) a lo largo de Z para dar contexto axial.
        offset=-1 → cada posición ve el corte anterior (z-1); offset=+1 → el siguiente (z+1).
        Bordes replicados (el corte extremo se repite), consistente con el padding de PSDM/CT.
        """
        if offset == 0:
            return t
        k = abs(offset)
        if offset < 0:
            pad = t[:, :1].expand(-1, k, -1, -1)
            return torch.cat([pad, t[:, :-k]], dim=1)
        pad = t[:, -1:].expand(-1, k, -1, -1)
        return torch.cat([t[:, k:], pad], dim=1)

    def _build_input(self, batch: dict) -> torch.Tensor:
        """
        Construye el tensor de entrada según la config.
        Devuelve (B, Z, C, H, W) donde C depende de los inputs activos × context_slices.

        Contexto axial (context_slices > 1): cada canal base se repite desplazado en Z
        (z-k, ..., z, ..., z+k), agregando gradiente axial sin cambiar Z ni el target
        (que sigue siendo el corte z sin desplazar). Orden: offset-mayor, canal-menor
        (todos los canales de z-1, luego todos los de z, luego todos los de z+1).
        """
        context_slices = getattr(self.cfg.data, 'context_slices', 1)
        k = (context_slices - 1) // 2

        # Estructuras no-body en el orden declarado por el anatomy schema (para
        # "prostate" es [ptv, rectum, bladder] — mismo orden que el bloque
        # hardcodeado que reemplaza esto, así que el orden de canales del modelo
        # ya entrenado no cambia).
        no_body = [s for s in self.structures if s.role != 'body']
        body_key = next((s.key for s in self.structures if s.role == 'body'), None)

        canales = []
        for offset in range(-k, k + 1):
            canales.append(self._shift_z(batch['ct'], offset).unsqueeze(2))

            if self.cfg.data.inputs.use_body_mask and body_key is not None:
                canales.append(self._shift_z(batch[f'{body_key}_mask'], offset).unsqueeze(2))

            if self.cfg.data.inputs.use_psdm:
                for s in no_body:
                    key = f'psdm_{s.key}'
                    if key in batch:
                        canales.append(self._shift_z(batch[key], offset).unsqueeze(2))
            else:
                for s in no_body:
                    if getattr(self.cfg.data.inputs, f'use_{s.key}_mask', False):
                        canales.append(self._shift_z(batch[f'{s.key}_mask'], offset).unsqueeze(2))

        # Concatenar a lo largo del canal: (B, Z, C, H, W)
        return torch.cat(canales, dim=2)

    def forward(self, x_volumen: torch.Tensor) -> torch.Tensor:
        """
        x_volumen: (B, Z, C, H, W)
        Devuelve dosis predicha: (B, Z, H, W)

        arch == 'unet3d': el modelo consume el volumen completo (convs 3D
        cruzan Z) — se permuta a (B, C, Z, H, W), a diferencia de 'unet2d'
        que aplana B*Z y corre cortes 2D independientes en paralelo.
        """
        B, Z, C, H, W = x_volumen.shape
        if self.cfg.model.arch == "unet3d":
            x_3d = x_volumen.permute(0, 2, 1, 3, 4)   # (B, C, Z, H, W)
            pred_3d = self.model(x_3d)                # (B, 1, Z, H, W)
            pred = pred_3d.squeeze(1)                 # (B, Z, H, W)
            return pred
        # Aplanar B y Z para procesar todos los cortes en paralelo
        x_2d = x_volumen.view(B * Z, C, H, W)
        if self.training and torch.is_grad_enabled():
            # Gradient checkpointing: recomputa el forward completo del U-Net
            # durante el backward en vez de retener las activaciones de las
            # B*Z "copias" simultaneas (pacientes con muchos cortes, ej. Z=80,
            # agotaban la VRAM con batch_size=1 -- ver CLAUDE_CODE_CONTEXT.md).
            # Resultado numerico identico, solo cambia el tradeoff memoria/compute.
            pred_2d = checkpoint(self.model, x_2d, use_reentrant=False)
        else:
            pred_2d = self.model(x_2d)  # (B*Z, 1, H, W)
        pred = pred_2d.view(B, Z, H, W)
        return pred

    # ─── Steps ────────────────────────────────────────────────────────────────
    def _shared_step(self, batch: dict, stage: str) -> dict:
        x = self._build_input(batch)            # (B, Z, C, H, W)
        pred = self(x)                          # (B, Z, H, W)
        target = batch['dose']                  # (B, Z, H, W)
        body_mask = batch['body_mask']

        struct_masks = {
            s.key: batch[f'{s.key}_mask']
            for s in self.structures
            if s.role != 'body' and f'{s.key}_mask' in batch
        }
        losses = self.loss_fn(pred, target, body_mask, struct_masks)

        # Métricas adicionales
        with torch.no_grad():
            metrics = self._calcular_metricas(pred, target, struct_masks, body_mask)

        # Logging
        bs = x.size(0)
        self.log(f"{stage}/loss",    losses['total'], on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs)
        self.log(f"{stage}/mae",     losses['mae'],   on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs)
        if 'moment' in losses:
            self.log(f"{stage}/moment", losses['moment'], on_step=False, on_epoch=True,
                     batch_size=bs)
        if 'dvh' in losses:
            self.log(f"{stage}/dvh_loss", losses['dvh'], on_step=False, on_epoch=True,
                     batch_size=bs)

        for nombre, valor in metrics.items():
            self.log(f"{stage}/{nombre}", valor, on_step=False, on_epoch=True, batch_size=bs)

        dvh_score = self._calcular_dvh_score(pred, target, struct_masks)
        self.log(f"{stage}/dvh_score", dvh_score, on_step=False, on_epoch=True,
                 prog_bar=(stage == "val"), batch_size=bs)

        return {'loss': losses['total'], 'pred': pred, 'target': target, 'batch': batch}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")['loss']

    def validation_step(self, batch, batch_idx):
        out = self._shared_step(batch, "val")
        # Guardar primer batch de val para callback visual
        if batch_idx == 0 and not hasattr(self, '_val_first_batch'):
            self._val_first_batch = {
                'pred':   out['pred'].detach().cpu(),
                'target': out['target'].detach().cpu(),
                'batch':  {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                          for k, v in batch.items()},
            }
        return out['loss']

    def on_validation_epoch_end(self):
        # Resetear el primer batch para la próxima época
        if hasattr(self, '_val_first_batch'):
            del self._val_first_batch

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")['loss']

    # ─── Métricas DVH simples ─────────────────────────────────────────────────
    @staticmethod
    def _calcular_metricas(pred: torch.Tensor, target: torch.Tensor,
                           struct_masks: dict, body_mask: torch.Tensor) -> dict:
        """Devuelve un dict de métricas escalares."""
        metricas = {}
        # MAE global dentro de BODY (sin pesar)
        diff_body = (pred - target).abs() * body_mask
        denom_body = body_mask.sum().clamp(min=1.0)
        metricas['mae_body_pct'] = (diff_body.sum() / denom_body).item()

        for nombre, mask in struct_masks.items():
            denom = mask.sum()
            if denom < 1:
                continue
            diff = (pred - target).abs() * mask
            metricas[f'mae_{nombre}_pct'] = (diff.sum() / denom).item()
            # Dmean error por estructura (en % de prescripción)
            pred_mean   = (pred * mask).sum() / denom
            target_mean = (target * mask).sum() / denom
            metricas[f'dmean_err_{nombre}'] = (pred_mean - target_mean).abs().item()
        return metricas

    def _calcular_dvh_score(self, pred: torch.Tensor, target: torch.Tensor,
                            struct_masks: dict) -> torch.Tensor:
        """
        DVH score estilo OpenKBP: promedio de |Δ| entre pred y target sobre
        D2/D95/D99 (target primario) y Dmean/Dmax (OARs). Mismo criterio que
        evaluate.py, para poder comparar runs de entrenamiento (val/dvh_score)
        contra resultados de test sin recalcular todo post-hoc.
        Asume batch_size=1 (decisión de diseño por VRAM) — si hay más de un
        paciente en el batch, mezcla sus voxeles al calcular percentiles.
        """
        errores = []
        for nombre, mask in struct_masks.items():
            roi = mask > 0
            if roi.sum() < 1:
                continue
            pred_roi = pred[roi].float()   # torch.quantile no soporta float16 (AMP)
            tgt_roi  = target[roi].float()
            if nombre == self.primary_target:
                for q in (0.98, 0.05, 0.01):  # D2, D95, D99
                    errores.append((torch.quantile(pred_roi, q)
                                     - torch.quantile(tgt_roi, q)).abs())
            else:
                errores.append((pred_roi.mean() - tgt_roi.mean()).abs())  # Dmean
                errores.append((pred_roi.max()  - tgt_roi.max()).abs())   # Dmax
        if not errores:
            return pred.new_tensor(float('nan'))
        return torch.stack(errores).mean()

    # ─── Optimizer ────────────────────────────────────────────────────────────
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        if self.cfg.scheduler.name == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.cfg.scheduler.max_epochs)
            return {"optimizer": opt, "lr_scheduler": sched}
        return opt
