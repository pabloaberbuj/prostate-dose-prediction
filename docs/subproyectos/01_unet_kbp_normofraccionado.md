---
tags: [prostatedoseproject, subproyecto, unet, kbp, normofraccionado]
estado: cerrado
---

# U-Net / KBP — Próstata normofraccionado

## Qué es
Pipeline de predicción de dosis 3D (implementado como U-Net 2D con contexto axial) para próstata normofraccionado (78Gy/39fx): DICOM → NPZ preprocesado → entrenamiento con PyTorch Lightning → evaluación DVH. Es la serie de experimentos original del proyecto (exp001–exp006) y la base sobre la que se construyó todo lo demás.

## Estado actual
**CERRADO** — dado por terminado explícitamente ("ESTADO FINAL — TODO CERRADO" en `CLAUDE_CODE_CONTEXT.md`). Ganador: `exp002` (U-Net 2D + PSDM). Rebaseline posterior tras el fix de CT: `exp002_unet2d_psdm_ctfix_fov34`.

## Realizado
- Serie de ablaciones exp001→exp006 sobre inputs (máscaras vs. PSDM), arquitectura y loss.
  - **exp002** (ganador): U-Net 2D + PSDM.
  - exp003: saltado.
  - exp004 (2.5D, 3 cortes de contexto): sin mejora sobre exp002.
  - exp005: descartado.
  - exp006 (MomentLoss): sin mejora neta; sweep de λ (`exp006_sweep_lambda*`) tampoco cambió la conclusión.
- Auditoría de bug de overlap (cálculo `Math.Min` en vez de intersección real) y de fuga train/val/test en `splits_v1.json` — ambos corregidos (ver [[../aprendizajes_transversales/bugs_criticos_resueltos]]). Consecuencia: **exp001 no se puede re-analizar** (checkpoint no sobrevivió).
- Bug crítico de carga de CT (cargaba la serie RD) descubierto y corregido en agosto 2026 — forzó re-generar todo el dataset y re-decidir el FOV de recorte (34cm definitivo). Nueva línea base: `exp002_ctfix_fov34`.
- `exp_normo_3dunet`: comparación 2D vs 3D — 3D no mejora, piso de OAR ("hombro" de la curva) es real incluso con arquitectura 3D. Ver [[../aprendizajes_transversales/hipotesis_descartadas]].

## Pendientes
| Pendiente | Bloqueador |
|---|---|
| exp004 con CT real (2.5D ctfix) | Explícitamente no re-entrenado; queda como "experimento futuro solo si un revisor lo pide" |

## Aprendizajes propios
- El CT real no aporta señal por sobre PSDM+máscaras, ni en FOV 50cm ni en 34cm — ver [[../aprendizajes_transversales/hipotesis_descartadas]].
- MSE uniforme sobre BODY concentra 89% del gradiente en dosis baja — ver [[../aprendizajes_transversales/decisiones_diseno_modelo]].
- Gradient checkpointing resuelve OOM en pacientes de Z grande sin subir el cap de VRAM — ver [[../aprendizajes_transversales/decisiones_diseno_modelo]].

## Archivos clave
- `configs/exp001_unet2d_baseline.yaml` … `configs/exp006_sweep_lambda1.yaml`, `configs/exp002_unet2d_psdm_ctfix_fov34.yaml`, `configs/exp_normo_3dunet.yaml`
- `checkpoints/exp002_unet2d_psdm_ctfix_fov34/`
- `scripts/train.py`, `scripts/evaluate.py`
- `src/models/unet2d.py`, `src/models/unet3d.py`, `src/models/lightning_module.py`
- `src/datamodules/dose_datamodule.py`
- `src/losses/losses.py`
- `data/preprocess.py`
