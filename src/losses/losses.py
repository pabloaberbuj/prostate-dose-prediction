"""
Losses para predicción de dosis.
- MAE enmascarada por BODY (ignora background).
- MomentLoss sobre PTV y OARs (Jhanwar et al. 2022).
"""

import torch
import torch.nn as nn


class MaskedMAELoss(nn.Module):
    """MAE limitada a voxeles dentro del BODY."""
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                body_mask: torch.Tensor) -> torch.Tensor:
        # pred, target, body_mask: (B, 1, H, W) o (B, Z, H, W)
        diff = (pred - target).abs() * body_mask
        denom = body_mask.sum().clamp(min=1.0)
        return diff.sum() / denom


class MomentLoss(nn.Module):
    """
    Loss basada en momentos M_p del DVH (Jhanwar et al. 2022).
    Para cada estructura y orden p:
        M_p(D) = (mean(D_v^p))^(1/p),  v ∈ estructura
    L = sum_p sum_struct (M_p(pred) - M_p(target))^2
    """
    def __init__(self, structures_moments: dict):
        """
        structures_moments: dict {nombre_estructura: lista_de_órdenes_p}
            ej: {'ptv': [1,2,10], 'rectum': [1,2,10], 'bladder': [1,2,10]}
        """
        super().__init__()
        self.structures_moments = structures_moments

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                masks: dict) -> torch.Tensor:
        """
        pred, target: (B, ...) tensor de dosis
        masks: dict {nombre_estructura: tensor_máscara}
        """
        total = pred.new_zeros(())
        for nombre, orders in self.structures_moments.items():
            if nombre not in masks:
                continue
            roi = masks[nombre] > 0
            if roi.sum() < 1:
                continue
            # Seleccionar los voxeles de la ROI una sola vez (mask antes de pow, no
            # después): la ROI es una fracción chica del volumen (1, Z, H, W), y
            # pow(p) sobre el volumen completo para descartar el resto vía máscara
            # desperdicia cómputo/memoria de GPU proporcional al volumen entero.
            pred_roi   = pred[roi].clamp(min=0.0)
            target_roi = target[roi].clamp(min=0.0)
            for p in orders:
                m_pred   = self._compute_moment(pred_roi,   p)
                m_target = self._compute_moment(target_roi, p)
                total = total + (m_pred - m_target) ** 2
        return total

    @staticmethod
    def _compute_moment(dose_roi: torch.Tensor, p: int) -> torch.Tensor:
        """M_p = (mean(d^p))^(1/p) sobre los voxeles ya seleccionados de la ROI."""
        mean_dp = dose_roi.pow(p).mean()
        return mean_dp.clamp(min=1e-8).pow(1.0 / p)


class CombinedLoss(nn.Module):
    """MAE + λ · MomentLoss."""
    def __init__(self, cfg):
        super().__init__()
        self.mae = MaskedMAELoss()
        self.use_moment = cfg.loss.use_moment_loss
        if self.use_moment:
            self.moment = MomentLoss(cfg.loss.moment_structures)
            self.moment_weight = cfg.loss.moment_weight
        self.mae_weight = cfg.loss.mae_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                body_mask: torch.Tensor, struct_masks: dict = None) -> dict:
        mae_val = self.mae(pred, target, body_mask)
        out = {'mae': mae_val, 'total': self.mae_weight * mae_val}
        if self.use_moment and struct_masks is not None:
            mom_val = self.moment(pred, target, struct_masks)
            out['moment'] = mom_val
            out['total'] = out['total'] + self.moment_weight * mom_val
        return out
