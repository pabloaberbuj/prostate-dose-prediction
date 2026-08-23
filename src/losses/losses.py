"""
Losses para predicción de dosis.
- MAE enmascarada por BODY (ignora background).
- MomentLoss sobre PTV y OARs (Jhanwar et al. 2022).
- DifferentiableDVHLoss (Nguyen et al. 2020, sin componente adversarial) — ver
  RESUMEN_prior_terma_arcos.md para el detalle técnico/receta. Corrige el mismatch
  de MomentLoss (momentos de potencia vs. percentiles clínicos) atacando
  directamente puntos V(D) del DVH real vía una aproximación sigmoide.
"""

import numpy as np
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


class DifferentiableDVHLoss(nn.Module):
    """
    DVH diferenciable vía sigmoide (Nguyen et al. 2020, Med Phys 47(3):837-849),
    sin componente adversarial (ver justificación en RESUMEN_prior_terma_arcos.md:
    el paper mismo reporta que MSE+DVH ya captura la mayor parte de la mejora sobre
    MSE+DVH+ADV, y ADV casi duplica el tiempo de entrenamiento).

    Para cada estructura s y cada umbral de dosis D (muestreado denso alrededor
    del "hombro" del DVH, no del plateau bajo — ver `structures_d_bins`):

        V_approx_pred(D) = mean_voxels∈s[ sigmoid(pendiente_s · (dosis_pred − D)) ]
        V_exact_gt(D)    = mean_voxels∈s[ 1{dosis_gt >= D} ]   (SIN gradiente — target)

        L_DVH = mean_s mean_D | V_approx_pred(D) − V_exact_gt(D) |     (L1)

    V en fracción [0,1] (no %) y D en % de prescripción (misma escala que el resto
    del proyecto) — la unidad de V solo afecta la escala de λ, no la física.

    La pendiente del sigmoide (`pendiente_s`, en 1/%Rx) se ata al paso de bin de
    cada estructura: pendiente_s = k / paso_bin, con k tal que el ancho de
    transición del sigmoide (10%-90%, = 2·ln(9)/pendiente_s) sea ~1 bin. Verificado
    empíricamente contra el DVH exacto del GT antes de usarla en entrenamiento (ver
    scripts/calibrar_dvh_loss.py) — el valor final de k y el sesgo medido quedan
    documentados en configs/exp_hipo_003_dvhloss.yaml y en el log de calibración.
    """
    def __init__(self, structures_d_bins: dict, sigmoid_slopes: dict):
        """
        structures_d_bins: dict {nombre_estructura: lista_de_D_en_pct_rx}
        sigmoid_slopes: dict {nombre_estructura: pendiente_s (1/%Rx)}
        """
        super().__init__()
        self.structures = list(structures_d_bins.keys())
        self.sigmoid_slopes = dict(sigmoid_slopes)
        for nombre, bins in structures_d_bins.items():
            # persistent=False: son bins de dosis fijos derivados de la config, no
            # estado aprendido — NO deben aparecer en state_dict() (si no, romperia
            # la carga strict=True de --init-weights desde checkpoints anteriores
            # a esta loss, ej. exp002 normo, que no tienen estos buffers).
            self.register_buffer(f"d_bins_{nombre}", torch.tensor(list(bins), dtype=torch.float32),
                                  persistent=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, masks: dict) -> tuple:
        """
        pred, target: (B, Z, H, W) dosis en % prescripcion.
        masks: dict {nombre_estructura: mascara binaria misma forma}
        Devuelve (loss_total_promedio_sobre_estructuras, dict_detalle_por_estructura).
        """
        total = pred.new_zeros(())
        detalle = {}
        n_terms = 0
        for nombre in self.structures:
            if nombre not in masks:
                continue
            roi = masks[nombre] > 0
            if roi.sum() < 1:
                continue
            pred_roi = pred[roi]                 # (N,)
            target_roi = target[roi].detach()     # (N,) sin gradiente — es el GT

            d_bins = getattr(self, f"d_bins_{nombre}")  # (Nbins,)
            s = self.sigmoid_slopes[nombre]

            v_approx_pred = torch.sigmoid(s * (pred_roi.unsqueeze(1) - d_bins.unsqueeze(0))).mean(dim=0)
            v_exact_gt = (target_roi.unsqueeze(1) >= d_bins.unsqueeze(0)).float().mean(dim=0)

            l1_struct = (v_approx_pred - v_exact_gt).abs().mean()
            total = total + l1_struct
            detalle[nombre] = l1_struct.detach()
            n_terms += 1

        if n_terms > 0:
            total = total / n_terms
        return total, detalle


class CombinedLoss(nn.Module):
    """MAE + λ_moment · MomentLoss + λ_dvh · DifferentiableDVHLoss."""
    def __init__(self, cfg):
        super().__init__()
        self.mae = MaskedMAELoss()
        self.use_moment = cfg.loss.use_moment_loss
        if self.use_moment:
            self.moment = MomentLoss(cfg.loss.moment_structures)
            self.moment_weight = cfg.loss.moment_weight
        self.mae_weight = cfg.loss.mae_weight

        self.use_dvh = bool(getattr(cfg.loss, 'use_dvh_loss', False))
        if self.use_dvh:
            structures_d_bins = {
                nombre: list(np.arange(cfg_s.d_min_pct, cfg_s.d_max_pct + 1e-6, cfg_s.d_step_pct))
                for nombre, cfg_s in cfg.loss.dvh_structures.items()
            }
            sigmoid_slopes = {
                nombre: cfg.loss.dvh_sigmoid_k / cfg_s.d_step_pct
                for nombre, cfg_s in cfg.loss.dvh_structures.items()
            }
            self.dvh = DifferentiableDVHLoss(structures_d_bins, sigmoid_slopes)
            self.dvh_weight = cfg.loss.dvh_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                body_mask: torch.Tensor, struct_masks: dict = None) -> dict:
        mae_val = self.mae(pred, target, body_mask)
        out = {'mae': mae_val, 'total': self.mae_weight * mae_val}
        if self.use_moment and struct_masks is not None:
            mom_val = self.moment(pred, target, struct_masks)
            out['moment'] = mom_val
            out['total'] = out['total'] + self.moment_weight * mom_val
        if self.use_dvh and struct_masks is not None:
            dvh_val, dvh_detalle = self.dvh(pred, target, struct_masks)
            out['dvh'] = dvh_val
            for nombre, val in dvh_detalle.items():
                out[f'dvh_{nombre}'] = val
            out['total'] = out['total'] + self.dvh_weight * dvh_val
        return out
