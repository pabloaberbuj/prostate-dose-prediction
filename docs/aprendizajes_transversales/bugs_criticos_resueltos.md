---
tags: [prostatedoseproject, aprendizajes, bugs]
estado: resuelto
---

# Bugs críticos resueltos

Tres bugs que en su momento parecían límites del problema (arquitectura, física, o el propio motor de dosis) y resultaron ser errores de implementación. Se documentan juntos porque comparten un patrón: cada uno generó semanas de investigación sobre una hipótesis equivocada antes de encontrarse la causa real.

## 1. CT corrupto — serie DICOM equivocada

`cargar_ct()` cargaba, en el 100% de los pacientes (normo + hipo), la serie **RD** (dosis) en vez de la serie **CT** real, sin ningún error visible porque ambas series tienen geometría compatible. Confirmado y corregido en agosto 2026.

- Consecuencia: todo el dataset preprocesado hasta ese momento (`processed/`, `processed_hipo/`) quedó inválido y se re-generó completo en `processed_ctfix_34` / `processed_hipo_ctfix`.
- El fix forzó además re-decidir el FOV de recorte in-plane (quedó en 34cm, ver [[decisiones_diseno_modelo]]).
- Confirmado 3 veces por separado que, ya con CT real disponible, no aporta señal por sobre el PSDM solo — sí acelera convergencia ~1.8x.
- Ver [[../subproyectos/01_unet_kbp_normofraccionado]] y [[../subproyectos/02_unet_kbp_hipofraccionado]] para el detalle de cómo esto re-fundó ambas series de experimentos.

## 2. Bug de overlap (Math.Min) + leak train/val/test

Auditoría encontró dos problemas independientes en el pipeline de máscaras/splits de la serie normofraccionado (exp001–exp006):

- Un bug de cálculo de overlap entre estructuras usaba `Math.Min` en vez de una intersección real de máscaras.
- `splits_v1.json` tenía fuga de pacientes entre val/test (mismo paciente en más de un split).

Ambos corregidos. Como consecuencia, **exp001 no se puede re-analizar** porque no sobrevivió su checkpoint — cualquier comparación contra exp001 debe tratarse como no reproducible.

## 3. "Gap PDRT vs AAA" — falso positivo (DEBUNKED)

Se creyó por un tiempo que había una discrepancia sistemática entre el motor propio (PyDoseRT) y el motor comercial (AAA) al recomputar el mismo plan — hipótesis que motivó revisar el comisionamiento completo. La causa real: el RD usado para la comparación era el **target de entrenamiento de la U-Net**, no la salida real de PDRT (error de qué archivo se estaba leyendo, no un problema físico). Una vez corregido, PDRT y AAA concuerdan bien. La causa raíz real de las diferencias que sí existían era una combinación de early-stopping y ponderación de MSE por banda de dosis — ver [[decisiones_diseno_modelo]].

Ver [[../subproyectos/04_mimicking_pdrt_planificacion]] y [[../subproyectos/06_comisionamiento_pdrt]].
