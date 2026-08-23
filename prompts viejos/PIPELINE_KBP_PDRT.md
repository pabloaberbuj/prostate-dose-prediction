# Pipeline KBP con PyDoseRT — de la dosis U-Net al preplan RTPLAN

## Cómo usar este documento

Dos fases, en orden:

1. **Chat rápido de diseño en Claude.ai** (sección A) — resolver las decisiones abiertas y
   definir cómo se ordena el proyecto CONTRA el repo real antes de escribir código.
2. **Handoff a Claude Code** (sección B) — implementación, una vez cerradas las decisiones.

No pasar a la fase 2 sin cerrar la fase 1. Es la misma disciplina de trabajo que en la serie
normo/hipo: diseño en Claude.ai, implementación en Claude Code.

**Prerrequisito duro:** el comisionamiento (`COMISIONAMIENTO_PDRT.md`) tiene que estar validado
(recálculo vs AAA con gamma OK) antes de que el mimicking tenga sentido. Se puede escribir el
esqueleto del código en paralelo, pero no validar resultados de mimicking sin motor comisionado.

---

# SECCIÓN A — Chat de diseño en Claude.ai (hacer primero)

Objetivo: cerrar estas decisiones y dejarlas escritas antes de tocar código. Traer a ese chat el
repo real (o su árbol de directorios) para decidir integración concreta, no en abstracto.

## A1. ¿Repo nuevo o subcarpeta del repo existente?

El proyecto de predicción (U-Net) ya existe. El KBP consume su salida (dosis 3D) pero es un
dominio distinto (optimización de entrega, no predicción). Decidir:

- **Opción 1 — subcarpeta `kbp/` en el repo actual.** Comparte utilidades de I/O de NPZ, splits,
  configs. Riesgo: mezclar dos dominios con dependencias distintas (PDRT trae pydicom,
  SimpleITK, pymedphys; el repo de predicción no las necesita).
- **Opción 2 — repo nuevo `prostate-kbp` que importa la salida de la U-Net como artefacto.**
  Limpio, dependencias separadas. La U-Net exporta dosis predicha a disco (NPZ/DICOM) y el KBP
  la consume. Acopla por datos, no por código.

Recomendación a discutir: **Opción 2** si el KBP va a crecer (writer RTPLAN, comisionamiento,
comparación RapidPlan); Opción 1 si es exploración corta. Decidir en el chat viendo el repo real.

## A2. ¿En qué formato entrega la U-Net la dosis al KBP?

La U-Net predice en grilla 256×256 (dosis relativa, D95=100%). PDRT trabaja en la grilla física
del paciente (mm reales, `patient.resolution`). Definir el puente:

- ¿Se re-escala la dosis 256×256 a la grilla nativa del CT? ¿Con qué interpolación?
- ¿Dosis relativa (% prescripción) o absoluta (Gy)? PDRT calibra en Gy → hay que llevar la
  predicción a Gy con la prescripción del caso.
- **Ojo con el bug de escala de voxel ya documentado:** `patient.resolution` en PDRT es spacing
  físico real; no arrastrar el error de multiplicar spacing nativo por grilla downsampleada.

## A3. Manejo de los 2 arcos

`load_dicom` devuelve `beam_sequence` indexado (un elemento por arco). Definir:

- ¿Se optimizan los 2 arcos conjuntamente (una loss, ambos `beam_sequence` con `requires_grad`)
  o secuencial?
- ¿Se toma la geometría de arco (ángulos, colimador) de un plan real como plantilla, o se fija
  una configuración estándar de 2 arcos? (Relacionado con el hallazgo de que NO hay técnica
  estándar en el centro: 90 casos 2-arcos, 88 casos 3-arcos.)
- **Alcance inicial: restringir a casos de 2 arcos**, donde la U-Net anda bien (MAE body 1.83,
  AUC altas). Los 3-arcos catastróficos son target poco confiable — no arrancar por ahí.

## A4. Función de mimicking exacta

El ejemplo mimetiza dosis prescrita uniforme. Nosotros mimetizamos la dosis U-Net 3D. Definir:

- ¿Mimicking global (todo el body) o ponderado por región (PTV/OAR/resto con pesos)?
- ¿Sumar objetivos DVH (`dvh_percentile_objective`) sobre recto/vejiga además del mimicking, o
  mimicking puro? (El paper de loss DVH que tiene Pablo es relevante acá — enfoque híbrido
  dose+DVH ya identificado como mejora prioritaria.)
- Términos de entregabilidad obligatorios para VMAT: `leaf_speed_reg`, `mu_rate_reg`,
  `jaw_speed_reg`. Definir pesos iniciales.

## A5. Criterio de éxito del preplan

¿Cómo se mide que el mimicking "salió bien"? Antes de codificar la evaluación:

- Gamma dosis-mimética vs dosis-U-Net (¿qué criterio?).
- DVH de recto/vejiga del preplan vs. DVH predicho vs. DVH del plan clínico real.
- Y el test real: tras importar a Eclipse y recalcular con AAA, ¿el plan recalculado cumple
  constraints? Comparar contra RapidPlan (en el otro chat) y contra el plan clínico manual.

## Entregable de la fase A

Un documento de decisiones cerradas (A1–A5) + árbol de directorios acordado del proyecto. Ese
documento es el input de la sección B.

---

# SECCIÓN B — Handoff a Claude Code (después de cerrar A)

## Rol para Claude Code

Especialista en computer vision / deep learning con conocimientos de radioterapia. Implementación
concisa, explicar qué se cambió y por qué. Preguntar antes de cambios estructurales. Este archivo
+ `CLAUDE_CODE_CONTEXT.md` (sección Proyecto 2) + el documento de decisiones de la fase A son el
contexto.

## Estructura de código propuesta (ajustar según decisión A1)

```
prostate-kbp/                      # o kbp/ dentro del repo actual
├── configs/
│   └── kbp_pdrt_2arc.yaml         # geometría de arco, pesos de loss, params optimizador
├── src/
│   ├── bridge/
│   │   └── unet_to_target.py      # dosis U-Net 256×256 → target en grilla física PDRT (A2)
│   ├── planning/
│   │   ├── build_patient.py       # CT+RS → PDRT.Patient (o vía load_dicom)
│   │   ├── build_beams.py         # geometría de 2 arcos → BeamSequence (A3)
│   │   ├── mimicking.py           # loss de mimicking + entregabilidad (A4), loop LBFGS
│   │   └── engine_setup.py        # DoseEngine con machine_config comisionado
│   ├── io/
│   │   └── write_rtplan.py        # beam_sequence optimizado → RTPLAN DICOM (pydicom)
│   └── eval/
│       └── evaluate_preplan.py    # gamma, DVH comparativo, cumplimiento (A5)
├── machine/
│   └── machine_config_6x.json     # salida del comisionamiento
└── scripts/
    ├── recompute_check.py         # validación recálculo vs AAA (test de aceptación motor)
    ├── run_mimicking.py           # pipeline completo un paciente
    └── batch_mimicking.py         # sobre el test set
```

## Orden de implementación sugerido

1. **`recompute_check.py` primero** (NO el mimicking). Cargar plan real con `load_dicom`,
   recalcular con `engine.forward`, comparar vs AAA con gamma. Esto valida el comisionamiento y
   la instalación antes de invertir en el resto. Patrón: `examples/rtplan.ipynb`.
2. **`unet_to_target.py`** — el puente de dosis (decisión A2). Testeable aislado: cargar una
   predicción U-Net guardada, re-escalar a grilla física, verificar unidades/escala.
3. **`build_beams.py` + `engine_setup.py`** — geometría de 2 arcos + motor comisionado.
4. **`mimicking.py`** — la loss (A4) y el loop LBFGS. Patrón: `examples/optimization.ipynb`.
   Empezar con mimicking puro sobre 1 paciente 2-arcos; agregar entregabilidad y DVH después.
5. **`write_rtplan.py`** — writer pydicom. Usar un RP de Eclipse como plantilla; reemplazar
   ControlPointSequence con leaves+MU optimizados. (Patrón conceptual: `write_rt_plan_vmat` de
   PortPy.) Confirmar mapeo de 60 leaf pairs / anchos del Millennium.
6. **`evaluate_preplan.py`** — métricas de A5.

## Notas de implementación críticas (de la lectura del código de PDRT)

- **Optimizador:** `torch.optim.LBFGS(..., line_search_fn="strong_wolfe")` con `closure()`. NO
  Adam. El line search es crítico para robustez (lo dice el propio código).
- **Parametrización de leaves:** vía `make_ordered_pairs` (sigmoid center+width) → aperturas
  válidas por construcción. MU vía `softplus`. Copiar este patrón, no optimizar posiciones crudas.
- **`patient.resolution` en mm físicos reales** — cuidado con el bug de escala de voxel conocido.
- **Entregabilidad VMAT:** sumar `leaf_speed_reg`/`mu_rate_reg`/`jaw_speed_reg` de
  `pydosert.objectives.losses`. Sin ellas el arco puede no ser entregable.
- **PDRT es solo reader de DICOM** — el writer lo hacemos nosotros (paso 5).
- **Instalación:** `pip install pydosert`. Python 3.11–3.13. Deps: torch, pydicom, SimpleITK,
  pymedphys (solo para gamma).
- **README de PDRT tiene un error:** menciona `rtplan_test_1arc.ipynb` (no existe); el notebook
  real de carga DICOM es `rtplan.ipynb`.

## Al terminar cada tarea

Actualizar `CLAUDE_CODE_CONTEXT.md` (sección Proyecto 2) con hallazgos, y volver a Claude.ai con
métricas para análisis y decisión del próximo paso.

---

## Frente IMRT alternativo (si se quiere validar concepto antes de comisionar PDRT)

Con `pyRadPlan` (Python puro, máquina genérica, sin comisionar): genera Dij de fotones
(`calc_dose_influence`, ~20s), objetivo `SquaredDeviation` apuntado a la dosis U-Net → fluencia →
export a formato Eclipse → "Import Optimal Fluence" en la GUI (sin ESAPI de escritura) → leaf
sequencing + AAA. Camino end-to-end funcional hoy en Eclipse 13.6 para equipos de campos
estáticos. Menor barrera de entrada, pero es IMRT, no VMAT. Ver detalle en `CLAUDE_CODE_CONTEXT.md`.
