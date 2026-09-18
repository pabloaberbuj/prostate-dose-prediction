---
tags: [prostatedoseproject, indice]
estado: vigente
---

# ProstateDoseProject — Índice

Mapa de estado del proyecto, pensado para orientarse rápido sin tener que leer `CLAUDE_CODE_CONTEXT.md` completo (2556 líneas, sigue existiendo como fuente histórica de detalle línea a línea).

## Qué es este proyecto

Predicción de dosis 3D (U-Net, knowledge-based planning) para radioterapia de próstata, con dos líneas de trabajo que se separaron cuando se descubrió que un modelo de ML clásico iguala a la U-Net en la tarea de clasificar cumplimiento de constraints:

- **Proyecto 1**: clasificación de cumplimiento vía ML clásico (features geométricas escalares) → corre hoy en producción en el tomógrafo.
- **Proyecto 2 (KBP)**: predicción de dosis 3D completa vía U-Net → plan VMAT real entregable, vía un motor de dosis propio (PyDoseRT) y un optimizador de mimicking.

## Estado por subproyecto

| Subproyecto | Estado | Doc |
|---|---|---|
| U-Net/KBP — normofraccionado | Cerrado | [[subproyectos/01_unet_kbp_normofraccionado]] |
| U-Net/KBP — hipofraccionado | Completo (v4, baseline vigente) | [[subproyectos/02_unet_kbp_hipofraccionado]] |
| Proyecto 1 — ML clásico | Tareas 1-6 completas, Tarea 7 bloqueada | [[subproyectos/03_proyecto1_ml_clasico]] |
| Mimicking / PDRT — planificación | Bloqueado (v7 optimizador congelado) | [[subproyectos/04_mimicking_pdrt_planificacion]] |
| Tomógrafo Tool | En producción | [[subproyectos/05_tomografo_tool]] |
| Comisionamiento PDRT | Completo | [[subproyectos/06_comisionamiento_pdrt]] |

## Aprendizajes transversales

Aplican a más de un subproyecto — se documentan una sola vez y se linkean desde cada doc de subproyecto:

- [[aprendizajes_transversales/bugs_criticos_resueltos]] — CT corrupto, overlap+leak, falso gap PDRT/AAA
- [[aprendizajes_transversales/hipotesis_descartadas]] — arc-ceiling, kernel_size=55, piso de OAR
- [[aprendizajes_transversales/decisiones_diseno_modelo]] — ponderación de loss, 3D vs 2D, gradient checkpointing
- [[aprendizajes_transversales/infraestructura_windows_gpu]] — reglas operativas de esta máquina

## Pendiente más urgente del proyecto

El bloqueo activo de mayor prioridad hoy es la v7 del optimizador de mimicking (congelado, LBFGS frozen) — ver [[subproyectos/04_mimicking_pdrt_planificacion]]. Es el paso que traba todo el Proyecto 2 (KBP) downstream de una U-Net que ya funciona.

## Código y CLI

El pipeline de entrenamiento (U-Net) se está generalizando para poder entrenar otras localizaciones anatómicas además de próstata, desde línea de comandos. Ver `CLI_INSTRUCTIVO.md` (en `repo\docs\`) una vez completado el refactor correspondiente.
