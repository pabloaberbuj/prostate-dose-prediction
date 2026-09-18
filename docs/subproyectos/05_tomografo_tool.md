---
tags: [prostatedoseproject, subproyecto, tomografo-tool, produccion]
estado: en-produccion
---

# Tomógrafo Tool

## Qué es
Aplicación web (Flask) que corre en producción clínica junto al tomógrafo, consumiendo el modelo del [[03_proyecto1_ml_clasico|Proyecto 1 (ML clásico)]] para dar, en tiempo real, un semáforo verde/naranja/rojo de cumplimiento de constraints por paciente. Es la única pieza de este proyecto que corre en producción activa hoy, no un experimento.

## Estado actual
**En producción**, monitoreada de forma continua.

## Realizado
- App completa: `app.py` (servidor), `pipeline.py` (procesamiento), `watcher.py` (monitoreo de carpeta), `templates/`/`static/` (UI).
- Tarea programada (scripts `.ps1`) para mantener el watcher activo.
- Registro de resultados por paciente en `registros/` (JSON), en producción.
- `MONITOREO.md` documenta la operación diaria de la herramienta.

## Pendientes
| Pendiente | Bloqueador |
|---|---|
| Extender a pacientes hipo | Depende de que se desbloquee la Tarea 7 del Proyecto1 (extractor en vivo desde DICOM) — ver [[03_proyecto1_ml_clasico]] |

## Aprendizajes propios
Ninguno propio de este subproyecto más allá de los ya documentados en [[03_proyecto1_ml_clasico]], del cual consume directamente el modelo y el pipeline de features.

## Archivos clave
- `tomografo_tool/app.py`, `pipeline.py`, `watcher.py`
- `tomografo_tool/config.yaml`
- `tomografo_tool/templates/`, `tomografo_tool/static/`
- `tomografo_tool/registros/`
- `tomografo_tool/MONITOREO.md`
