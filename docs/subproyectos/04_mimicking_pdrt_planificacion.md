---
tags: [prostatedoseproject, subproyecto, mimicking, pdrt, planificacion, bridge]
estado: bloqueado
---

# Mimicking / PDRT — Planificación (Proyecto 2 / KBP)

## Qué es
Pipeline downstream que toma la dosis 3D predicha por la U-Net y la convierte en un plan VMAT real y entregable: un "bridge" traduce la salida de la red a estructuras RTSTRUCT/target, y un optimizador de mimicking ajusta MLC/pesos de arco para reproducir esa dosis objetivo usando el motor de dosis propio (PyDoseRT), terminando en un RTPLAN DICOM real.

## Estado actual
**En progreso, con un bloqueo activo**: la v7 del optimizador de mimicking quedó congelada (LBFGS frozen) sin resolver desde 2026-08-16.

## Realizado
- Comisionamiento de PyDoseRT completo y validado (ver [[06_comisionamiento_pdrt]]).
- Pipeline completo U-Net → PDRT → mimicking → RTPLAN → Eclipse confirmado end-to-end en un paciente piloto (RD sintético importado correctamente en Eclipse).
- El "gap PDRT vs AAA" que motivó una revisión completa del comisionamiento resultó ser un falso positivo (RD de comparación leía el target de entrenamiento de la U-Net, no la salida real de PDRT) — ver [[../aprendizajes_transversales/bugs_criticos_resueltos]]. La causa real de las diferencias que sí existían: early-stopping + ponderación de MSE por banda de dosis (ver [[../aprendizajes_transversales/decisiones_diseno_modelo]]).
- v6 del mimicking: el MLC abre 33.8% fuera de la PTV (sin resolver).
- v7: se probaron un piso de D98 y márgenes más ajustados — **ambos, de forma independiente, disparan el freeze de LBFGS**; ambos fixes fueron revertidos, la causa raíz no está aislada.

## Pendientes
| Pendiente | Bloqueador |
|---|---|
| Resolver el freeze de LBFGS en v7 | Causa no aislada del todo — bloqueo activo más importante del subproyecto |
| Revisar código de mimicking en busca de bugs | Pedido explícito de Pablo, sin ejecutar desde 2026-08-16 |
| Resolver cuello de botella de tiempo de PDRT (4h/paciente) | Marcado como inaceptable; se ataca solo después de resolver el freeze — ver [[../aprendizajes_transversales/infraestructura_windows_gpu]] para contexto de por qué no asumir que es solo la VM compartida |
| Diseñar nuevas pruebas de margen | Pendiente, tercer paso pedido por Pablo |
| PTV-aware MLC/jaw init (conform+5mm, jaws +5mm X/+2mm Y) en vez de closed/max-open | Mejora de cold-start pendiente, no bloqueante |
| Límite de MU/grado sin modelar en `machine_config` | Ver [[06_comisionamiento_pdrt]] |

## Aprendizajes propios
- El "gap PDRT vs AAA" y la contaminación nodal son ejemplos del mismo patrón: una hipótesis de límite físico que resultó ser un bug de comparación — ver [[../aprendizajes_transversales/bugs_criticos_resueltos]] e [[../aprendizajes_transversales/hipotesis_descartadas]].

## Archivos clave
- `src/planning/mimicking.py`, `build_beams.py`, `engine_setup.py`, `write_rtplan.py`, `ptv_conforming_init.py`
- `src/bridge/unet_to_target.py`, `rtstruct_io.py`, `mask_morph.py`
- `scripts/run_mimicking.py`, `run_write_rtplan.py`, `write_substruct_rs.py`, `split_oar_by_dose.py`, `substruct_derisk_mask.py`
- `prompts viejos\PROMPT_NUEVA_CONVERSACION_mimicking_v7.md`
