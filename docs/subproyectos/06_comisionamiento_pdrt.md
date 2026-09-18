---
tags: [prostatedoseproject, subproyecto, comisionamiento, pdrt]
estado: completo
---

# Comisionamiento PDRT

## Qué es
Validación del motor de dosis propio (PyDoseRT) contra el motor comercial de referencia (AAA) y calibración de sus parámetros (kernel size, límites cinemáticos de máquina) — es el trabajo que da sustento físico a todo lo que corre downstream en [[04_mimicking_pdrt_planificacion]].

## Estado actual
**Completo y validado**.

## Realizado
- Sandbox de comisionamiento sobre 7 pacientes: gamma pasando 96.6–99%.
- `kernel_size=55` (recomendado por el paper original de PyDoseRT como el de mejor precisión) probado explícitamente en esta comisión: **empeora** el acuerdo en PTV/dosis alta y resulta ~3x más lento — se descarta, se mantiene `kernel_size=25`. Ver [[../aprendizajes_transversales/hipotesis_descartadas]].
- `machine_config` no tenía límites cinemáticos reales configurados (pydosert caía en defaults silenciosos) — corregido con los valores reales de la máquina.
- El "gap PDRT vs AAA" que motivó revisar todo el comisionamiento resultó ser un falso positivo de comparación (RD equivocado, no un problema físico) — ver [[../aprendizajes_transversales/bugs_criticos_resueltos]].

## Pendientes
| Pendiente | Bloqueador |
|---|---|
| Modelar el límite de MU/grado en `machine_config` | Sin modelar explícitamente todavía — no bloqueante para el uso actual, pero es un límite real de la máquina no representado |

## Aprendizajes propios
Ver [[../aprendizajes_transversales/hipotesis_descartadas]] (kernel_size), [[../aprendizajes_transversales/bugs_criticos_resueltos]] (gap PDRT/AAA).

## Archivos clave
- `commissioning/run_commissioning_pipeline.py`, `recompute_check.py`, `qc_report.py`
- `commissioning/diagnostics_mimicking_v7/`, `commissioning/sandbox_results/`
- `commissioning/LICENSE_PyDoseRT_upstream`
- `papers/PyDoseRT paper.md`
- `datos comisionamiento/viejos/` (datos crudos antiguos, fuera de `repo\`)
