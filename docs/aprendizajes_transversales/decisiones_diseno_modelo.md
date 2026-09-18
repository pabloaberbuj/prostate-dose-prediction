---
tags: [prostatedoseproject, aprendizajes, diseno-modelo]
estado: vigente
---

# Decisiones de diseño del modelo

Decisiones tomadas con evidencia experimental propia, no supuestos de diseño de entrada. Relevantes para cualquiera que agregue una anatomía nueva o retome el ajuste de loss.

## MSE gasta 89% del peso en dosis baja

Un MSE uniforme sobre toda la máscara BODY concentra ~89% del peso del gradiente en voxeles por debajo del 20% de la dosis de prescripción, y solo ~0.6% por encima del 90%. Es la explicación más probable de por qué los planes salen poco modulados (el optimizador prioriza acertar el fondo de baja dosis, no la zona clínicamente relevante). Pendiente de acción: ponderar por banda de dosis o usar una loss que no trate todos los voxeles por igual (ver `MomentLoss`/`DifferentiableDVHLoss` en `src/losses/losses.py`, que sí son dict-driven por estructura).

## 3D no supera a 2D (`exp_normo_3dunet`)

Pasar de U-Net 2D a 3D no mejoró resultados; el piso de OAR es real incluso con arquitectura 3D (ver [[hipotesis_descartadas]]). Cierra, por ahora, la pregunta de si la arquitectura 3D es la palanca que falta.

## Serie de barrido de loss DVH (exp_hipo_003/003b/003c)

Concluye que hace falta un λ **diferenciado por estructura** en la loss DVH (no un único λ global) — diseñado pero no implementado todavía. Candidato natural para retomar junto con el punto de ponderación por banda de dosis.

## Gradient checkpointing para pacientes de Z grande

CUDA OOM en pacientes con más cortes (mayor Z) se resuelve con gradient checkpointing en el forward de `lightning_module.py` (ver método `forward`, comentario sobre `torch.utils.checkpoint.checkpoint`), **no** subiendo el cap de VRAM. Resultado numérico idéntico, solo cambia el tradeoff memoria/cómputo.

## losses.py ya es genérico — precedente para la generalización anatómica

`CombinedLoss`/`MomentLoss`/`DifferentiableDVHLoss` reciben las estructuras como dict `{nombre: ...}` desde config — es el módulo del pipeline de entrenamiento que **no** necesitó cambios para el refactor de generalización a otras localizaciones (ver instructivo de CLI). El acoplamiento real a próstata estaba en quién arma esos dicts (`lightning_module.py`, `dose_datamodule.py`), no en la loss en sí.
