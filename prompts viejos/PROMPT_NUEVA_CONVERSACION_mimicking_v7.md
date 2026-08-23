# Prompt para arrancar la próxima conversación (Claude Code)

Copiar y pegar esto como primer mensaje:

---

Leé `CLAUDE_CODE_CONTEXT_070826.md`, sección final "Proyecto 2 — KBP mimicking: v6 en Eclipse,
apertura excesiva, freeze de LBFGS SIN RESOLVER (2026-08-15/16)".

Contexto en una frase: v6 convergió en entrenamiento pero el plan real (Eclipse/AAA) salió
malo porque el MLC abre ~34% de su área fuera del PTV (vs 7% del plan clínico) — confirmado
que PDRT también predice ese punto caliente, no es un gap de motor. Al intentar arreglarlo
(bajar `leaf_margin_mm`/`aperture_extra_mm` + corregir un artefacto de D98) el optimizador
(LBFGS) quedó clavado — 7 smoke tests aislaron que NO es por `aperture_extra_mm` ni por el
valor de D98 solo, pero la combinación de márgenes más ajustados también rompe algo sin
identificar todavía.

Los 3 pasos pendientes, en orden:
1. Revisar `src/planning/mimicking.py` y `src/planning/ptv_conforming_init.py` en busca de
   bugs/interacciones que expliquen el freeze (no asumir que es sólo tuning) — ver candidatos
   señalados en la sección del archivo de contexto.
2. Atacar el cuello de botella de tiempo (cada paso de LBFGS tarda 340-1800s; un smoke test de
   5 pasos cuesta 1-2h reales) — esto es lo que hace inviable seguir iterando a ciegas.
3. Recién después, diseñar las próximas pruebas (aislar margen vs extra por separado con D98
   crudo; probar `--optimizer adam` en vez de LBFGS, dado el precedente ya documentado de que
   las proyecciones duras — `project_leaf_velocity`, y ahora sospecho `project_aperture_bounds`
   — no son compatibles con LBFGS).

Los scripts de diagnóstico de la sesión anterior están en
`commissioning/diagnostics_mimicking_v7/` (bev_check.py es el método de validación BEV
independiente, reusable). El código de mimicking está actualmente REVERTIDO al estado que
converge (D98 crudo, sin el fix de margen) — no relanzar una corrida larga sin antes resolver
el punto 1.
