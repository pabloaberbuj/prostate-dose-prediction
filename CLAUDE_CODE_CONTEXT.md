# Prompt de contexto para Claude Code — Prostate Dose Prediction

> **Nota de unificación (2026-08):** este documento reemplaza a `CLAUDE_CODE_CONTEXT_1.md`
> (rama vieja, sin actualizar desde antes del entrenamiento hipo — se puede borrar). Si algo
> de este archivo contradice a ese, este es el vigente.
>
> **⚠️ ALERTA CERRADA — "techo de geometría de arcos":** la conclusión de que existía un
> límite físico de modulación angular (ver hallazgo (a), sección hipofraccionado) quedó
> **descartada**. Era contaminación de dataset por planes con irradiación ganglionar pélvica,
> no un límite del modelo. Ver sección "⚠️ CORRECCIÓN — contaminación nodal, techo de arcos
> DESCARTADO" más abajo para el detalle completo. No retomar el arc-prior TERMA motivado por
> este hallazgo — si se retoma, que sea por mejora de DVH/dosis completo en el proyecto KBP,
> no por este techo (que no existe) ni por clasificación (donde el ML clásico ya gana, ver
> sección "Baseline ML clásico").
>
> **⚠️⚠️ BUG CRÍTICO DE CT — RESUELTO (2026-08):** `cargar_ct()` cargaba la serie RD (dosis)
> en vez del CT real en TODOS los NPZ generados hasta agosto 2026 (normo + hipo). El canal CT
> era dosis remuestreada, no anatomía. Ya está corregido + se re-decidió el FOV (recorte 34cm).
> Las conclusiones RELATIVAS entre experimentos NO se invalidan (todos compartían el mismo CT
> corrupto). Nueva línea base: **exp002_ctfix_fov34**. Ver sección "BUG CRÍTICO DE CT +
> corrección de FOV" al final del documento para el detalle completo. TODO NPZ generado antes
> del fix es obsoleto — re-preprocesar desde DICOM antes de usar.

## Rol

Sos un especialista en computer vision y deep learning con conocimientos de radioterapia.
Trabajamos en un proyecto de predicción de distribución de dosis 3D para pacientes de próstata tratados con VMAT.
Respondé de forma concisa y directa. Cuando hagas cambios al código explicá brevemente qué cambiaste y por qué.
Si necesitás más contexto sobre decisiones de diseño preguntá antes de cambiar algo estructural.

---

## Workflow Claude.ai ↔ Claude Code

Este archivo es el handoff entre las dos herramientas. División de tareas:

**Claude.ai (diseño y análisis):**
- Decisiones de diseño y arquitectura.
- Análisis de resultados y métricas.
- Discusión sobre próximas etapas.
- Cuando algo no funciona y hay que pensar qué cambiar.

**Claude Code (implementación):**
- Debugging iterativo (error → fix → error → fix).
- Implementar algo nuevo que ya está diseñado en Claude.ai.
- Optimización de performance (profiling, benchmarking).
- Refactoring de código existente.
- Cualquier tarea donde el ciclo editar/correr/ver resultado sea el cuello de botella.

Al terminar una tarea en Claude Code, actualizar este archivo con los hallazgos relevantes y volver a Claude.ai con metrics.csv + summary.json (si aplica) para análisis y decisión del próximo paso.

---

## Contexto clínico

- Pacientes de próstata VMAT normofraccionados (39 fx × 2 Gy = 78 Gy, algoritmo AAA, 6X).
- El modelo predice distribución de dosis 3D voxel-wise en dosis relativa (% de prescripción, D95(PTV) = 100%).
- Uso clínico: predecir si recto y vejiga cumplirán los constraints de DVH antes de planificar.
- Estructuras: PTV (PTV, PTV_High, PTV_High2, PTV_Abr22, PTVp), Rectum, Bladder, BODY.
- Constraints clínicos normofraccionado: Rectum V70<25%, V65<35%; Bladder V70<35%, V65<50%.
- Constraints hipofraccionado (28 fx, ~70-78Gy): AÚN NO DEFINIDOS EN YAML — pendiente.

---

## Estado actual del proyecto — Serie normofraccionado (CERRADA)

### Dataset (completado)
- 385 pacientes usables. Split: 265 train / 60 val / 60 test.
- Splits en `data/splits/splits_v1.json`.
- ⚠️ Estratificación por terciles de "solapamiento PTV-Recto" — VER BUG CRÍTICO abajo.

### Preprocesado (completado)
- NPZs en `C:\Pablo\ProstateDoseProject\processed\` (fuera del repo).
- Cada NPZ: ct [-1,1], dose (% prescripción), masks binarias, PSDM en cm/15.
- Pipeline en `data/preprocess.py`.

### exp001 — Baseline 2D U-Net con masks (COMPLETADO)
- in_channels=5 (CT+BODY+masks). MAE body 2.39, rectum 5.07, bladder 4.19. DVH score 1.40.

### exp002 — PSDM 2D (COMPLETADO — ganador de inputs)
- in_channels=5 (CT+BODY+PSDM×3). MAE body 1.77, rectum 3.75, bladder 2.16. DVH score 1.15.
- Mejora fuerte vs exp001: −26% rectum, −48% bladder.

### exp003 — saltado (PSDM ya reemplaza masks, 8ch innecesario)

### exp004 — 2.5D PSDM 3 cortes (COMPLETADO — sin mejora)
- in_channels=15, context_slices=3. MAE body 1.77, rectum 3.72, bladder 2.14. DVH score 1.17.
- Verificado: el modelo usa los canales de contexto (pesos no nulos), pero no aportan señal.
- Conclusión: la anatomía de cortes vecinos no informa la geometría angular de los arcos VMAT.
  exp005 (5 cortes) descartado por la misma razón — no tiene sentido escalar algo que no aporta.

### exp006 — MomentLoss (COMPLETADO — cerrado, sin mejora neta)
- Sweep de λ (0.001, 0.01, 0.1) a 40 epochs (con fix de scheduler.max_epochs=200 fijo,
  independiente de trainer.max_epochs, para que el LR en época N sea comparable a exp002):
  MAE empeora monótono con λ; dvh_score plano dentro del ruido en todo el rango.
- Entrenamiento completo con λ=0.001 (el menos invasivo): estadísticamente equivalente a
  exp002 en test (MAE rectum 3.95 vs 3.75, PTV 1.15 vs 1.41 — mejor en PTV, algo peor en
  rectum, sin cambio neto en DVH score). Convergencia más rápida en las primeras ~80 épocas,
  mismo punto de llegada final.
- Conclusión: MomentLoss (M1/M2/M10) no mejora las métricas clínicas relevantes (DVH score,
  MAE de OARs). Hipótesis: mismatch entre momentos de potencia y percentiles de DVH
  (D2/D95/D99) que son los que realmente importan para los constraints. Vía cerrada.

### Conclusión de la serie de ablaciones normofraccionado
Mejor modelo: **exp002** (U-Net 2D + PSDM + MAE puro). MAE rectum ~3.75%, DVH score ~1.15,
acuerdo de constraints 95-100%.

Los artefactos radiales residuales (visibles en mapas de diferencia, zona de dosis media-baja)
son un techo estructural de este enfoque: el modelo no tiene información de la modulación
angular real de los arcos (dose rate por control point), solo anatomía. Ni 2.5D ni MomentLoss
lo resuelven porque atacan ejes ortogonales al problema (contexto axial anatómico; forma
global de la distribución). Se necesitaría un input de geometría de entrega (arc prior
TERMA-style) o encoder polar CON ese input adicional — ninguno de los dos sirve solo.

**Decisión: pausar exploración de arquitecturas (cascada, dual-encoder, polar) en este dataset.**
Ninguna ataca el cuello de botella real (falta de información de entrega). Retomarlas más
adelante, junto con el arc prior, en el dataset que vaya a ser el de largo plazo.

---

## ⚠️ BUG CRÍTICO — cálculo de solapamiento PTV-OAR

**Confirmado por el usuario.** El código en `extract_dicom_csharp/Program.cs`:

```csharp
private static double CalcularSolapamiento(Structure s1, Structure s2)
{
    if (s1 == null || s2 == null) return double.NaN;
    try { return Math.Min(s1.Volume, s2.Volume); }
    catch { return double.NaN; }
}
```

Esto NO calcula intersección geométrica — devuelve el volumen del órgano más chico. En próstata,
el PTV casi siempre es mayor que el Recto, así que "Solapamiento" ≈ Volumen del Recto en la
gran mayoría de los casos (confirmado empíricamente: 174/179 filas en el CSV hipofraccionado
tienen Solap_PTV_Rectum_cc == VolRectum_cc exactamente).

**Afecta a ambos datasets** (normofraccionado y el nuevo hipofraccionado), porque usan el mismo
código de extracción.

**Consecuencia en el dataset normofraccionado:** la estratificación de `splits_v1.json` por
"terciles de solapamiento" es en la práctica casi un proxy de tamaño de recto, no de
solapamiento geométrico real. Esto NO invalida las comparaciones agregadas entre experimentos
(MAE global, DVH score) porque se calculan sobre el test set completo. SÍ afecta la
interpretación causal de los análisis estratificados `by_solap_rectum` en `error_analysis.json`
de exp001/002/004/006: lo que atribuimos a "ambigüedad angular por bajo solapamiento" puede ser
en gran parte el mismo efecto que "recto pequeño → peor MAE" (ya visto en `by_vol_bladder` /
tamaño de órgano). Sigue siendo un hallazgo real (el PSDM ayuda más en casos difíciles), pero
la causa atribuida hay que revisarla.

## Auditoría de overlap (COMPLETADA — Claude Code, 2026-07-09)

### Resultados

**Script:** `scripts/compute_overlap_real.py`. Recalcula overlap real (intersección
`ptv_mask & rectum_mask`) desde los 385 NPZ en `processed/`, cruza con `VolPTV_cc`
de `metricas_planes.csv` y compara terciles.

**⚠️ Hallazgo adicional (no anticipado en el paso 1):** las máscaras del NPZ están
downsampleadas in-plane a 256×256, pero `meta['spacing_mm']` guardado es el spacing
*nativo* de la CT (pre-downsample) — multiplicarlo directo por voxels del array
256×256 da un volumen equivocado por un factor `(tamaño_original/256)²`, que además
**varía por paciente** (se midió razón real/esperado entre 1.1× y 4.2× en una
muestra). El script corrige esto calibrando el volumen por voxel con
`meta['vol_ptv_cc']` (volumen nativo de PTV, calculado en `preprocess.py` ANTES de
downsamplear) ÷ voxels de PTV en la máscara ya downsampleada. Sin esta corrección,
el overlap real habría salido sobreestimado y con ruido paciente-a-paciente.

**Validación de la réplica:** los límites de tercil recalculados con la fórmula
vieja (`Solap_PTV_Rectum_cc / VolPTV_cc × 100`, bug Math.Min) — `[11.49, 32.99,
47.41, 100.0]` — coinciden con los guardados en `splits_v1.json` (`[11.5, 33.0,
47.4, 100.0]`), confirmando que la fórmula y el universo de pacientes replican
correctamente la estratificación original.

**Magnitud del bug confirmada:** `Solap_PTV_Rectum_cc == VolRectum_cc` exacto en
374/385 filas de `metricas_planes.csv` (97%). Overlap real medio = 8.0 cc vs. Solap
"viejo" (≈volumen de recto) con media mucho mayor — el bug sobreestima el
solapamiento en casi un orden de magnitud en la mayoría de los casos. Un paciente
(`PT_4ae30f255aafce1a`) tiene overlap real = 0.0 cc (PTV y recto no se tocan) pero
el valor viejo marcaba 46.35 cc de "solapamiento".

**⚠️ Segundo hallazgo, no relacionado al bug de overlap — leak train/val/test en
`splits_v1.json`:** `PT_505fcdbdbb6553b6` está simultáneamente en `val` y en
`test` (confirmado: `train∩val=∅`, `train∩test=∅`, `val∩test={PT_505fcdbdbb6553b6}`).
Es el único caso (n=1 de 384 pacientes únicos, sobre 385 slots train+val+test).
Efecto: ese paciente pudo influir en la selección de checkpoint/early stopping
(vía val) y también se reporta como resultado de test — su métrica de test no es
un hold-out limpio en exp001/002/004/006. Impacto esperado en las métricas
agregadas es mínimo (1/60 del test set) pero es un problema metodológico real.
**No se corrigió `splits_v1.json`** — se dedupe solo internamente en
`compute_overlap_real.py` para no arrastrar filas repetidas al CSV de salida.
Decisión pendiente con Claude.ai: dejarlo documentado como limitación conocida,
o corregir el split (sacar a ese paciente de val o de test) — lo segundo
invalidaría/requeriría re-anotar las métricas de test ya reportadas para los 4
experimentos (efecto probablemente despreciable en agregado, pero no gratis).

**Tabla de confusión (tercil viejo vs. nuevo, n=384 pacientes únicos):**

| viejo \ nuevo | bajo | medio | alto |
|---|---|---|---|
| bajo  | 62 | 42 | 24 |
| medio | 36 | 46 | 46 |
| alto  | 30 | 40 | 58 |

**Pacientes que cambian de tercil: 218/384 (56.8%)** → cambio grande, no marginal.

**Consecuencia (según regla del Paso 2):** no hace falta re-entrenar nada (splits
train/val/test no dependen del overlap para la partición en sí). Se re-corrió
`analyze_errors.py` con las etiquetas de tercil corregidas — ver resultados abajo.

**Archivos generados:**
- `data/overlap_real_normo.csv` — por paciente: `overlap_ptv_rectum_cc_real`,
  `vol_ptv_cc_npz`, `VolPTV_cc`, `VolRectum_cc`, `Solap_PTV_Rectum_cc`,
  `pct_solap_viejo_bug`, `pct_solap_real`, `tercil_viejo`, `tercil_nuevo`.
- `data/overlap_real_confusion_terciles.csv` — tabla de confusión de arriba.

### Resultados re-análisis `analyze_errors.py` con overlap real (completado — Claude Code, 2026-07-09)

`scripts/analyze_errors.py` ahora acepta `--overlap-real-csv data/overlap_real_normo.csv`
(opcional, no rompe el uso anterior): si se pasa, reemplaza `pct_solap_rectum` y
`tercil_solap_rect` por los valores reales (columna `tercil_nuevo`) en vez de
recalcularlos con la fórmula bugueada. `error_analysis.json` ahora incluye
`overlap_solap_rectum_fuente` para dejar registrado qué versión se usó.

Re-corrido para los 3 experimentos con checkpoint disponible, output en
`results/<exp>_test/analysis_overlap_real/`:

| Exp | by_solap_rectum (VIEJO, bug) | by_solap_rectum (NUEVO, real) |
|---|---|---|
| exp002 | bajo mae=4.32 dvh=1.37 → medio 3.97/1.30 → alto 2.96/0.79 (monótono) | bajo(n=17) mae=3.38 dvh=0.96 · medio(n=27) mae=4.08 dvh=1.24 · alto(n=16) mae=3.58 dvh=1.21 (sin patrón claro) |
| exp004 | (mismo patrón que exp002, no reproducido en detalle acá) | bajo mae=3.52 · medio mae=3.96 · alto mae=3.41 |
| exp006 | (mismo patrón) | bajo mae=3.89 · medio mae=4.15 · alto mae=3.57 |

**Conclusión confirmada:** con el bug, había una tendencia monótona clara
("menos solapamiento → peor MAE/DVH"), que es la que motivó la hipótesis de
"ambigüedad angular por bajo solapamiento" en la conclusión de exp002/004/006.
Con el overlap real, esa tendencia **desaparece** (el grupo "medio" es el peor en
los tres experimentos, no el "bajo") — consistente con lo que ya se sospechaba:
el efecto observado con el bug era en gran parte tamaño de recto (`Solap ≈
VolRectum`), no ambigüedad geométrica real. **La hipótesis de "ambigüedad angular
por bajo solapamiento" queda descartada** para esta serie; el hallazgo válido que
sobrevive es el de `by_vol_bladder`/tamaño de órgano (no depende del bug).

**exp001 NO se pudo re-analizar:** no existe `results/exp001_test/test_metrics.csv`
ni checkpoint guardado en `checkpoints/` (solo logs de entrenamiento en
`lightning_logs/exp001_unet2d_baseline/`). Los números de exp001 en este documento
(MAE body 2.39, rectum 5.07, etc.) no se pueden desagregar por tercil sin
reevaluar el modelo, y no hay checkpoint para hacerlo sin re-entrenar. Si hace
falta ese desagregado, avisar para decidir si se re-entrena exp001 (es el
baseline, entrenamiento corto) o se documenta como no disponible.

Los `analysis_overlap_real/` conviven con los `analysis/` viejos (no se
sobrescribieron) para no perder la traza de qué venía del bug.

### Conclusión de la auditoría — narrativa corregida

La hipótesis de "ambigüedad angular por bajo solapamiento" (que motivaba la lectura de
exp002/004/006) **queda descartada**: con overlap real el patrón monótono desaparece (el
grupo "medio" pasa a ser el peor, no el "bajo"). El efecto observado con el bug era en gran
parte tamaño de recto (`Solap ≈ VolRectum`). El hallazgo válido que sobrevive es el de
`by_vol_bladder` / tamaño de órgano (no depende del bug): **el PSDM ayuda más en estructuras
pequeñas / de geometría de borde compleja**, no por solapamiento angular. Usar esta narrativa
en cualquier reporte o paper futuro.

---

## Verificación PSDM + corrección de leak (COMPLETADO — Claude Code, 2026-07-09)

### Tarea A — Verificar si el PSDM tiene el mismo bug de escala de voxel

**RESULTADO:** Sin bug. El PSDM se calcula sobre la máscara nativa con spacing nativo antes
del downsample a 256×256. El resize posterior solo interpola el mapa ya en cm. Verificado
empíricamente en 6 pacientes: diferencia en borde de estructura 0.10-0.19 cm (error normal
de interpolación). La serie normofraccionada queda cerrada sin reservas. exp002 sigue siendo
el ganador.

### Tarea B — Corrección de leak train/val/test

**RESULTADO:** `PT_505fcdbdbb6553b6` sacado de test (queda en val). Backup en
`splits_v1_backup.json`. Re-evaluado exp002/004/006 con test=59 → `results/<exp>_test59/`.
Deltas 0.003-0.13 en todas las métricas: despreciable, no altera orden entre experimentos
ni conclusiones. exp001 sin checkpoint → no re-evaluable, documentado como no disponible.

---

## ESTADO FINAL — Serie normofraccionada (TODO CERRADO)

Ambas tareas completadas. No hay nada pendiente en la serie normofraccionada.
El trabajo continúa en el **dataset hipofraccionado** — ver `HIPOFX_KICKOFF.md` para el
contexto de arranque del nuevo chat.

Métricas de referencia finales (test=59, exp002 — el ganador):
- MAE body: 1.77, PTV: 1.41, Rectum: 3.75, Bladder: 2.16
- DVH score: 1.15
- Constraints: Rectum V70=95%, V65=100%; Bladder V70=98.3%, V65=100%

---

## Estructura del código

```
prostate-dose-prediction/
├── configs/
│   ├── exp001_unet2d_baseline.yaml
│   ├── exp002_unet2d_psdm.yaml
│   ├── exp004_unet25d_3ctx.yaml
│   └── exp006_moment_loss.yaml
├── data/
│   ├── preprocess.py
│   ├── qc_preprocess.py
│   ├── splits/splits_v1.json
│   ├── metricas_planes.csv              # normofraccionado
│   ├── metricas_planes_hipofx.csv       # hipofraccionado, 180 pacientes (con issues, ver arriba)
│   └── extract_dicom_csharp/Program.cs  # bug conocido en CalcularSolapamiento
├── src/
│   ├── datamodules/dose_datamodule.py
│   ├── models/
│   │   ├── unet2d.py
│   │   └── lightning_module.py          # _shift_z y _build_input con context_slices
│   ├── losses/losses.py                 # MAE + MomentLoss
│   └── callbacks/logging_callbacks.py
└── scripts/
    ├── train.py
    ├── smoke_test.py
    ├── evaluate.py
    ├── analyze_errors.py
    └── profile_step.py
```

---

## Decisiones de diseño — NO cambiar sin consultar

- Sampling por paciente (no por corte suelto).
- num_workers=0 obligatorio en Windows.
- batch_size=1 por VRAM.
- Bilinear upsample + conv 3x3 (no transposed conv).
- GroupNorm (no BatchNorm con batch=1).
- Dosis en % prescripción normalizada a D95(PTV)=100% (normofraccionado).
- PSDM en cm dividido por 15cm → rango [-1,1] aprox.
- Cache float16 en RAM, conversión a float32 en __getitem__.
- Test set sin cache.
- torch.set_float32_matmul_precision('high') en todos los scripts.
- base_features=16 para todos los experimentos en RTX A2000 12GB.
- context_slices implementado en lightning_module.py via _shift_z y _build_input.
- cache_train=false si la VM de W7 está activa.
- scheduler.max_epochs siempre fijo al horizonte de entrenamiento completo (200), independiente
  de trainer.max_epochs, para que cualquier corte temprano (sweeps) sea comparable en LR.
- ⚠️ Máscaras NPZ downsampleadas a 256×256 pero `meta['spacing_mm']` es spacing NATIVO. Para
  cualquier cálculo de volumen/distancia física sobre las máscaras, calibrar el volumen por
  voxel con `meta['vol_ptv_cc']` nativo ÷ voxels de PTV en la máscara downsampleada. NO
  multiplicar spacing nativo × voxels del array 256×256 (da error de escala variable por
  paciente 1.1×-4.2×). Ver `scripts/compute_overlap_real.py`. Pendiente verificar si el PSDM
  arrastra este sesgo (Tarea A).

---

## Hardware y entorno

- GPU: NVIDIA RTX A2000 12GB, CUDA 13.0, Driver 581.42.
- RAM: 32GB. VM W7 consume 2-4GB → cerrar antes de entrenar.
- Windows 11, PowerShell, .venv en raíz del repo.
- W&B proyecto: prostate-dose-prediction.
- Kaggle (T4 16GB, Linux) para pruebas rápidas → num_workers=4 allá.

---

## Plan de ablaciones — normofraccionado (CERRADO)

| Exp | Variable | Estado |
|-----|----------|--------|
| 001 | Baseline masks 2D | ✓ completado |
| 002 | PSDM 2D (5ch) | ✓ completado — ganador |
| 003 | saltado | — |
| 004 | 2.5D PSDM (3 ctx) | ✓ completado — sin mejora |
| 005 | 2.5D PSDM (5 ctx) | descartado |
| 006 | MAE + MomentLoss | ✓ completado — sin mejora, cerrado |

Arquitecturas pendientes (cascada, dual-encoder, polar) y arc prior geométrico: **pausadas**,
retomar en la etapa hipofraccionado junto con el arc prior.

## Dataset hipofraccionado — serie de experimentos (2026-07-10 a 2026-07-12)

### Pipeline construido
- Splits estratificados: `data/splits/splits_hipo_v1.json` (116/27/36, estrato 2D de
  compliance operativo RV65/RV55/BV65 — script `repo/scripts/make_splits_hipo.py`).
- Preprocesado a NPZ: `repo/data/preprocess_hipo.py` → `processed_hipo/`. Bug encontrado y
  corregido: la lista de prioridad global de alias de PTV elegía mal la estructura para
  `PT_5b54e7add30325f0` (el RS tenía `PTV_High` Y `PTV_High1` simultáneamente, distintas
  estructuras) — se cambió a usar el `NombrePTV` del CSV C# como fuente de verdad
  por-paciente, con la lista de prioridad solo como fallback.
- GT DVH sobre grilla 256×256 (NO los flags del C#/ESAPI — nativo, nunca re-interpolado):
  `repo/data/compute_gt_dvh_hipo.py` → `data/gt_dvh_hipo_256.csv`. Cross-check contra C#
  D95norm: 16 flips en 3 constraints, todos <1pp de distancia al umbral (ruido de grilla,
  no bug).
- `scripts/evaluate.py --dataset hipo` (dispatch a `scripts/evaluate_hipo.py`, módulo
  paralelo, no toca el path normo): MAE por estructura, puntos DVH (bias/MAE), y
  clasificación clínica de los 3 constraints operativos (Rectum V65<15/V55<25, Bladder
  V65<15 — únicos con señal suficiente, ver D5). Umbral operativo de capa 3 calibrado en
  VAL (sensibilidad≥0.90), CONGELADO, aplicado a test — evita la circularidad de calibrar
  y evaluar sobre el mismo test set. Bootstrap (1000 resamples) para IC95 de capa1/capa3.
  Desglose adicional por NArcos (2 vs 3 arcos) — ver hallazgo (a) abajo.
- `scripts/train.py`: nuevo flag `--init-weights` (carga SOLO el `state_dict` del modelo
  desde otro checkpoint, para finetuning cross-dataset — a diferencia de `--resume`, que
  restaura el estado completo del trainer/optimizer/epoch).

### Experimentos corridos (test hipo, n=36, mismo umbral val-frozen)

| Métrica | zero-shot exp002 | baseline desde cero (exp_hipo_001) | finetune (exp_hipo_002) |
|---|---|---|---|
| MAE body | 4.00 | 4.42 | **3.90** |
| MAE rectum | 7.09 | 7.23 | **6.44** |
| MAE bladder | 6.78 | 7.60 | **6.44** |
| PTV_V70Gy bias | -5.31pp | -14.00pp | **-5.37pp** |
| AUC RV65 | 0.877 | 0.729 | 0.842 |
| AUC RV55 | 0.889 | 0.835 | **0.889** |
| AUC BV65 | 0.975 | 0.949 | **0.982** |

**Ganador: exp_hipo_002_finetune** (init = pesos completos de exp002/normo, LR=1e-5, full
finetune sin freezing, horizonte corto de 100 épocas — activó early stopping en la época 34,
mejor checkpoint en la época 4 con val/mae=2.78). Igualó o mejoró al zero-shot en casi todas
las métricas, y crucialmente preservó el prior de cobertura de PTV (V70Gy bias -5.37pp vs.
-14.00pp del baseline desde cero) — confirma que entrenar desde cero con solo 116 casos
heterogéneos no alcanza para aprender ese prior por sí solo, mientras que partir de exp002 y
adaptar con LR bajo sí lo conserva.

El baseline desde cero, pese a escalar el horizonte del scheduler (`scheduler.max_epochs=460`,
verificado self-consistent con el estado embebido en el propio checkpoint — descartado el
confound de LR-piso) para igualar el presupuesto de updates de gradiente de exp002, quedó
peor en AUC en los 3 constraints operativos. Confirma la decisión de fondo: exp002 ya estaba
cerca de una buena solución en el mismo espacio de salida (dosis relativa, D95=100%, mismos
canales) — mejor adaptar que reaprender desde cero con pocos casos.

### Hallazgos clave (revisados — ver corrección más abajo)

**(a) [SUPERADO — ver "Corrección: contaminación nodal" más abajo] Techo de geometría de arcos.**
Los mismos 4 pacientes (`PT_00dc878545e331b2`, `PT_4eb8a1dbe930c7d5`, `PT_7f59629d324c8f85`,
`PT_e4ca9f90c174e15a`) fallaban catastróficamente (mae_body 16-22%, vs. ~2-3% del resto) en
las TRES corridas independientes (zero-shot, baseline, finetune), sin excepción. Se atribuyó
a un límite físico del modelo (falta de información de modulación angular real de los arcos).
**Esta conclusión quedó descartada** — ver la sección siguiente. Se deja el razonamiento
original como registro histórico, no como hallazgo vigente.

**(b) Tensión de inferencia — el número de arcos no se conoce al momento de predecir.**
Este punto sigue siendo válido independientemente de (a): en el tomógrafo de planificación
(antes de que exista el plan VMAT) no se sabe cuántos arcos va a usar el físico. La
herramienta, en uso clínico real, predice "cumplimiento bajo técnica estándar" — cualquier
desviación de esa técnica estándar en el plan real es inherentemente impredecible en ese
momento con el input actual (solo anatomía). Sigue siendo relevante como limitación general
del punto de uso, con independencia de si hay o no un techo de arcos real.

**(c) No hay "técnica estándar" definida en el centro.** Distribución real de NArcos en el
dataset completo (n=179, previo a la limpieza nodal): 90 con 2 arcos, 88 con 3 arcos, 1 con 4
arcos — prácticamente 50/50. Dato descriptivo que se mantiene; ya no motiva por sí solo un
arc-prior (ver corrección abajo).

---

## ⚠️ CORRECCIÓN — contaminación nodal, techo de arcos DESCARTADO (2026-07, posterior)

**Hallazgo:** 28 de los 179 pacientes del dataset hipo incluían áreas ganglionares pélvicas
(planes con geometría de tratamiento completamente distinta a próstata-sola). **Los 4 casos
"catastróficos" del hallazgo (a) estaban DENTRO de esos 28.** IDs completos de los 28
excluidos:

```
PT_48f5d541d1345fac, PT_b69afd5aac626652, PT_bb3e683b982b7e10, PT_3567c33a62b3639b,
PT_186d1e95997c4d4c, PT_4eb8a1dbe930c7d5, PT_d0ec99de816f43a2, PT_8ce99f6c2510ac4d,
PT_71418878dbbe4c8f, PT_093bfd0af80b588e, PT_0456deafecd33ff1, PT_dda8bf75509cd629,
PT_9371aae1bc4a76ac, PT_e18d2f582312c202, PT_c8a9e92ed2c8657c, PT_00dc878545e331b2,
PT_7f59629d324c8f85, PT_0b1dfdffc25ab05c, PT_e4ca9f90c174e15a, PT_64bdc6f122da6e84,
PT_9303de5015932f5a, PT_a52308a14b30db33, PT_b9b5d34c32f39343, PT_072f15b8e53a9cee,
PT_13171608b4e247ab, PT_e12c3b41a2f2f22c, PT_6b7e2ece0dc9ec3c, PT_6954073bb0493b6e
```

**Por qué el análisis anatómico anterior no lo detectó:** el `VolPTV_cc` de estos pacientes es
normal (el `PTV_High` seleccionado es el de próstata; el PTV nodal es una estructura separada
que no entra en las máscaras del modelo). La contaminación vive en la **dosis** (baño nodal
que se deposita cerca de la vejiga, que el modelo próstata-sola no puede predecir), no en la
geometría de las máscaras — por eso el chequeo de volúmenes/overlap/MU no encontró firma.

**Dataset limpio: 151 pacientes** (no 179). CSV:
`metricas_planes_hipofx_D95norm_clean.csv`. Splits regenerados con la misma metodología
(estrato 2D compliance operativo, Hamilton, 65/15/20, seed=42) → `splits_hipo_v2_clean.json`
(98/22/31).

**Factor de escala de épocas recalculado** (no reusar el 0.44 viejo): pasos-de-gradiente-
totales constante, no épocas — con `batch_size=1`, pasos/época = n(train). factor =
train_normo(265) / train_hipo_nuevo(98) = 2.7041 → `max_epochs=540`, `early_stopping.patience`
escalado en la misma proporción (≈80). Aplicar este razonamiento cada vez que cambie el
tamaño del train set de cualquier serie.

### Re-evaluación sobre dataset limpio (n_test=31)

| Métrica | zero-shot exp002 | baseline desde cero (clean) | finetune (clean) |
|---|---|---|---|
| MAE body | 2.36 | 2.95 | **2.24** |
| MAE PTV | 1.78 | 5.62* | **1.74** |
| MAE rectum | 7.04 | 7.62 | **6.35** |
| MAE bladder | 4.16 | 5.34 | **3.71** |
| PTV_V70Gy bias | -2.30pp | -16.53pp | **-0.52pp** |
| AUC RV65 | 0.952 | 0.681 | 0.919 |
| AUC RV55 | 0.881 | 0.786 | 0.839 |
| AUC BV65 | 0.952 | 0.857 | 0.933 |
| Casos arc-limited (mae_body>8) | 0 | 1 (NArcos=**2**) | 0 |

\* MAE PTV del baseline dominado por 1 outlier (`PT_5d2c6ee9551e7804`, overlap PTV-recto en
percentil 96 — divergencia real del modelo, confirmada en 4 checkpoints, no bug de eval; sin
ese paciente MAE PTV baseline ≈3.18). Modelo entrenado desde cero, sin el prior de normo, no
cubre bien la cola de la distribución anatómica con solo 98 casos.

**El techo de arcos queda descartado.** Con el dataset limpio, los casos "arc-limited"
desaparecen casi del todo (4→0/1/0), y el único remanente es de **2 arcos**, no 3 —
contradice directamente la hipótesis original. Era contaminación de dataset, no un límite
físico de modulación angular. Buen ejemplo de correlación espuria por confusor no controlado
(NArcos correlacionaba con "tiene ganglios", no con dificultad de modulación per se).

**El finetuning sigue ganando** (bias V70 casi nulo, mejor MAE en las 4 estructuras) — esa
conclusión es robusta a la limpieza, se mantiene igual o más fuerte.

### Corrección de calibración RV65 — split balanceado

Detectado: `Flag_Rectum_V65Gy_lt15_D95norm` tenía solo 2 positivos en val (9%) vs. 10 en test
(32%) sobre `splits_hipo_v2_clean.json` — calibración de umbral operativo frágil (cae en
fallback clínico). Corregido con 3 swaps train↔val dentro de la misma celda del estrato 2D
(sin tocar test) → `splits_hipo_v2_clean_balanced.json` (val RV65 sube a 5/22, 22.7%). Es el
split **vigente** para toda evaluación de capa 3 de la serie hipo; `splits_hipo_v2_clean.json`
(no balanceado) queda como referencia histórica, no usar para reportar sens/esp de RV65.

**Nota de rigor pendiente:** los checkpoints se seleccionaron (early stopping) con el split NO
balanceado. Solo se re-evaluó (no re-entrenó) con el balanceado — afecta la calibración del
umbral post-hoc, no la selección de pesos. Se estima que el efecto de reentrenar con el
balanceado sería mínimo (3 pacientes movidos sobre 98+22), pero no está verificado
formalmente. Si se publica, considerar reentrenar los tres desde cero con el split balanceado
por prolijidad.

### Resultados finales — referencia vigente (test=31, split balanceado)

| Constraint | AUC finetune [IC95] | Sens/Esp finetune (umbral val-frozen) |
|---|---|---|
| Rectum V65Gy<15 | 0.919 [0.804–0.991] | 0.70 / 0.90 |
| Rectum V55Gy<25 | 0.839 [0.673–0.951] | 1.00 / 0.50 |
| Bladder V65Gy<15 | 0.933 [0.842–1.0] | 0.80 / 0.81 |

`exp_hipo_002b_finetune_clean` es el modelo de referencia de la serie hipo. AUC/MAE/bias no
cambian entre split balanceado y no-balanceado (no dependen del umbral); solo cambia el punto
de operación de capa 3, sobre todo en RV65.

---

## Baseline ML clásico (control metodológico) — hallazgo que reordena el proyecto

Se entrenó un baseline de ML clásico (regresión logística + gradient boosting) sobre 7
features geométricas escalares, TODAS conocibles en tomógrafo antes de existir el plan:
`VolRectum_cc`, `VolBladder_cc`, `VolPTV_cc`, `Solap_PTV_Rectum_cc`, `Solap_PTV_Bladder_cc`,
`overlap_rel_recto`, `overlap_rel_vejiga`. Mismo split (`splits_hipo_v2_clean_balanced.json`),
mismo protocolo de calibración val-frozen, mismo bootstrap que la red.

| Constraint | Regresión logística | Gradient Boosting | U-Net finetune |
|---|---|---|---|
| RV65 | **0.981** | **0.986** | 0.919 |
| RV55 | **0.917** | **0.923** | 0.839 |
| BV65 | 0.938 | 0.905 | 0.933 |

**El clásico iguala o supera a la U-Net en clasificación de constraints.** Feature
importance: `overlap_rel_recto` domina RV65/RV55, `overlap_rel_vejiga` domina BV65 (casi un
orden de magnitud sobre la siguiente feature) — el cumplimiento de un constraint de OAR está
gobernado casi enteramente por el overlap relativo PTV-OAR, señal escalar que no necesita un
mapa de dosis 3D.

**Matiz importante (no solo "el clásico gana"):** en el punto de operación (no en AUC), el
clásico calibra umbrales muy laxos por escasez de positivos en val (misma limitación que
afecta a la red) — especificidad tan baja como 0.24–0.38 en RV65 pese al AUC alto. El AUC
mayor no garantiza mejor matriz de confusión con este tamaño de dataset.

**Consecuencia — el proyecto se separa en dos líneas con objetivos distintos:**

1. **Herramienta de tomógrafo (clasificación de cumplimiento) → camino ML clásico.** La U-Net
   NO se justifica acá: mismo resultado (o peor) a mucho mayor costo. Ver
   `KICKOFF_1_herramienta_clasica_tomografo.md`.
2. **KBP (predicción de dosis 3D / preplan) → camino U-Net, con otra métrica de éxito.** La
   U-Net se justifica por lo que el clásico no puede hacer: DVH completo, mapa espacial usable
   como target de optimización, escenarios de escalado post-hoc. **La métrica principal para
   esta línea NO es AUC de clasificación** (ahí pierde estructuralmente contra el clásico) —
   es calidad del DVH/dosis completo, comparado contra RapidPlan y contra la dosis entregada
   real. Ver `KICKOFF_2_kbp_rapidplan.md`.

Esto también resta motivación al arc-prior TERMA como estaba planteado: se justificaba en
parte por el techo de arcos (descartado) y por mejorar clasificación (donde ya no compite,
el clásico gana esa pelea). Si se retoma, evaluarlo por si mejora el DVH/dosis completo para
el objetivo KBP, no por clasificación.

### Archivos generados (esta corrección)
- `metricas_planes_hipofx_D95norm_clean.csv`, `splits_hipo_v2_clean.json`,
  `splits_hipo_v2_clean_balanced.json`.
- `results/exp_hipo_001b_baseline_clean_test_hipo_v2_balanced/`,
  `results/exp_hipo_002b_finetune_clean_test_hipo_v2_balanced/`,
  `results/exp002_unet2d_psdm_test_hipo_v2_balanced/` (zero-shot).
- `results/baseline_clasico/metrics_summary.json`.

### Archivos generados (serie original, previa a la corrección)
- `configs/exp_hipo_001_baseline.yaml`, `configs/exp_hipo_002_finetune.yaml`.
- `results/exp002_unet2d_psdm_test_hipo/`, `results/exp_hipo_001_baseline_test_hipo/`,
  `results/exp_hipo_002_finetune_test_hipo/` — superados por los resultados sobre dataset
  limpio arriba; se conservan solo como traza histórica.

---

## Proyecto 2 — KBP vía dosis 3D → preplan entregable (INVESTIGACIÓN, 2026-07)

Etapa nueva. La U-Net deja de justificarse por clasificación de constraints (eso lo resuelve
el modelo clásico de ML — ver Proyecto 1) y pasa a justificarse por lo que solo ella da: la
**distribución de dosis 3D completa** como target de optimización para un KBP. La pregunta de
esta etapa es cómo convertir esa dosis predicha en un preplan (VMAT o IMRT) importable a
Eclipse, sin depender de ESAPI de escritura.

### Decisión de herramienta — resultado del relevamiento (verificado en código)

Se relevó el ecosistema open-source (matRad, PortPy/ECHO-VMAT, pyRadPlan, open-kbp-opt,
pyanno4rt, OpenTPS, PyDoseRT). Hallazgos clave verificados clonando los repos:

- **matRad:** VMAT solo en rama `dev_VMAT` de **2018**, sin export DICOM de RTPLAN (el master
  dice explícitamente `Only IMRT support (DYNAMIC), no VMAT`). Sirve solo para generar Dij de
  fotones sin Eclipse. Licencia: discrepancia interna (LICENSE=BSD-3, README=GPLv3). MATLAB.
- **PortPy/ECHO-VMAT:** mimicking voxel-wise + VMAT SCP + export, PERO la Dij se extrae de
  Eclipse vía API → cuello de botella para pacientes nuevos con Eclipse 13.6 read-only.
- **pyRadPlan (DKFZ):** Python puro, Apache-2.0, activo (v0.4.1, jun-2026). Motor pencil-beam
  de fotones **funcional** (`calc_dose_influence`, verificado corriendo TG119: Dij en ~20s,
  máquina "Generic", solver SciPy out-of-the-box). Hace **IMRT (fluencia)**, NO VMAT. Objetivo
  `SquaredDeviation` = base para mimicking voxel-wise. → **candidato para el frente IMRT.**

- **★ PyDoseRT (PDRT) — herramienta elegida para el frente VMAT.**
  `github.com/UMU-DDI/PyDoseRT` · **MIT** · activo (commit may-2026) · `pip install pydosert`.
  Paradigma distinto y superior para nuestro stack: **motor de dosis diferenciable en PyTorch**
  (pencil-beam convolution con heterogeneidad/scatter/penumbra). NO usa Dij precomputada — el
  forward pass es `parámetros MLC → fluencia → convolución → dosis`, todo diferenciable, y el
  gradiente fluye hasta las posiciones de leaf y MU. **Elimina el problema de la Dij** (el motor
  ES el grafo PyTorch) y resuelve entregabilidad VMAT vía regularización mecánica en la loss.
  VMAT/IMRT/static nativo. GPU. Import DICOM (CT/RS/RP/RD). MIT.

### PDRT — API real (verificada en `examples/optimization.ipynb` y `examples/rtplan.ipynb`)

Cuatro piezas:

1. **Patient** — `PDRT.Patient(ct_tensor, resolution=(mm,mm,mm))` + `add_mask(name, tensor)`.
   OJO: `resolution` en mm físicos reales — conecta con el bug de escala de voxel ya conocido
   (usar spacing correcto, no el nativo aplicado a grilla downsampleada).
2. **BeamSequence** — `PDRT.BeamSequence.create(gantry_angles, n_leaf_pairs=60, field_size,
   iso_center, collimator_angles, sid, ...)`. Un arco VMAT = gantry_angles discretizados en
   control points (ej. `np.linspace(-170,170,40)`). `n_leaf_pairs=60` = nuestro Millennium.
3. **DoseEngine** — `PDRT.DoseEngine(machine_config, dose_grid_spacing, dose_grid_shape,
   beam_template, auto_calibrate=True, kernel_size=15)`. `MachineConfig(tpr_20_10,
   mean_photon_energy_MeV, number_of_leaf_pairs)` — parámetros del comisionamiento (ejemplo trae
   10MV; nosotros 6X). Es un `nn.Module` estándar (`.train()`, `.forward()`).
4. **Loop de optimización** — se optimizan `leaf_positions`, `mus`, `jaw_positions` (params del
   beam_sequence). El **dose mimicking contra la U-Net entra acá**, cambio de una línea:
   ```python
   dose_pred = engine.forward(leaf_positions=..., mus=..., jaw_positions=..., density_image=...)
   loss_mimicking = torch.mean((dose_pred[External] - dose_unet[External])**2)  # ← target U-Net
   ```

**Carga desde DICOM real** (`rtplan.ipynb`, caso de próstata):
```python
from pydosert.data.loaders import load_dicom
patient, beam_sequence = load_dicom(ct_folder, dose_path=[rd], plan_path=[rp],
    struct_path=rs, struct_names=["PTV","Bladder","Rectum","FemoralHead_L","FemoralHead_R","Body"])
beam_sequence = beam_sequence[0]   # plan_path es lista; cada elem = un arco/haz
# beam_sequence.leaf_positions: [n_control_points, n_leaf_pairs, 2] (0=left, 1=right)
# beam_sequence.mus:            [n_control_points]  (MU acumulado)
```

### Detalles de código no obvios (anotar — ahorran sorpresas)

- **Optimizador: `torch.optim.LBFGS` con `line_search_fn="strong_wolfe"`.** El propio código lo
  marca "critical — without this it's much less robust". NO es Adam. Usa `closure()`.
- **Aperturas válidas por construcción:** `make_ordered_pairs` parametriza leaves vía sigmoid
  (center + width) → garantiza left<right. MU vía `softplus` → siempre positivo.
- **Entregabilidad dinámica del arco** (leaf/MU/jaw speed entre control points): losses
  `leaf_speed_reg`, `mu_rate_reg`, `jaw_speed_reg` en `src/pydosert/objectives/losses.py`. El
  ejemplo simple NO las usa; para VMAT real HAY que sumarlas o el plan puede tener saltos de
  leaf no entregables.
- **Objetivos DVH incluidos:** `dvh_percentile_objective(dose, mask, percentile)`. El código
  trae ejemplos comentados con Bladder/Rectum/FemoralHead (próstata).

### Límites reales de PDRT (honestos)

- **Comisionamiento OBLIGATORIO** (no hay máquina genérica lista). Decisión: se hace con
  planes de phantom de agua generados en Eclipse (dosis AAA), NO con mediciones físicas del
  equipo — coherencia con el ground truth del proyecto (AAA es la referencia en todo el
  pipeline). La validación posterior con pacientes reales es solo un chequeo de que el sandbox
  no se rompe en heterogeneidad, no una búsqueda de fidelidad clínica (PDRT nunca sale del loop
  U-Net→Eclipse). Ver `COMISIONAMIENTO_PDRT.md`.
- **Solo READER de DICOM, NO writer de RTPLAN.** Verificado: `dicom_utils.py` solo tiene
  `load_*`. El writer (leaves+MU+CP → RTPLAN sobre plantilla Eclipse) lo construimos con pydicom.
  Pablo confirmó que no es problema (patrón conocido, tipo `write_rt_plan_vmat` de PortPy).
- **Recién salido (may-2026), poco battle-tested.** Somos early adopters.
- **Error en el README de PDRT:** lista `rtplan_test_1arc.ipynb` que NO existe. El notebook real
  de carga DICOM es `rtplan.ipynb`.

### Pipeline resultante (VMAT, PDRT)

```
CT + RS (DICOM, ya extraídos)
  → U-Net → dosis 3D predicha (target)
  → load_dicom → geometría de arco (2 arcos) desde RP real como plantilla de beam_sequence
  → PDRT DoseEngine (comisionado 6X): optimización LBFGS de leaves+MU
       loss = mimicking voxel-wise vs dosis U-Net + leaf/MU/jaw speed reg (entregabilidad)
  → beam_sequence optimizado → writer pydicom → RTPLAN DICOM
  → Import Eclipse → recálculo AAA/AXB → validación
```

### Frente IMRT (alternativa de menor comisionamiento, con pyRadPlan)

Para validar el concepto rápido sin comisionar PDRT: `pyRadPlan (Dij+FMO, máquina genérica) +
SquaredDeviation apuntado a dosis U-Net → fluencia → export .txt → Eclipse "Import Optimal
Fluence" (GUI, NO requiere ESAPI escritura) → leaf sequencing + AAA`. Camino end-to-end más
allanado en Eclipse 13.6 para equipos de campos estáticos. Confirmar formato de export de
fluencia (pyRadPlan vs. `get_eclipse_fluence` de PortPy).

### Estrategia de despliegue por centro

- **Eclipse 15.6 / 18 SIN RapidPlan** → mejor caso: la herramienta ES el KBP, sin competencia,
  y ESAPI de escritura habilita automatización real. **Arrancar el KBP acá.**
- **Equipos IMRT static (cualquier versión)** → segundo frente, técnicamente el más limpio
  (fluencia-vía-GUI sin ESAPI escritura).
- **VMAT + RapidPlan + Eclipse 13.6** → peor caso (competís con RapidPlan donde es fuerte;
  depende de verificar si el PO acepta arrancar en paso 4 desde dosis intermedia). Dejar al final.

### Pendientes de verificación de Pablo (dependen de SU Eclipse, no verificables desde acá)

- ¿PO de Eclipse 13.6 acepta arrancar en "paso 4" desde una dosis 3D externa como intermedia?
- ¿Toma un RTPLAN VMAT con aperturas como base plan y recalcula sin re-optimizar?
- ¿MLC HD120 vs Millennium 120 — el writer necesita mapeo de leaves distinto? (PortPy hardcodea 60).

### Piloto — dosis U-Net → RD DICOM sintético → import en Eclipse (COMPLETADO, 2026-08-07)

Antes de comisionar PDRT, se hizo una prueba mucho más barata: tomar 1 paciente de test normo
(`PT_003fe2bb84986507`, `exp002`), escribir la dosis predicha directo como RTDOSE (RD) DICOM
(no como XML de objetivos DVH, que era el piloto planeado en `HANDOFF_kbp_dvh_to_xml.md`), e
intentar importarlo en Eclipse. **Resultado: import correcto**, dosis alineada con la anatomía
y DVH calculable sobre las estructuras reales (Rectum/Bladder/PTV_High) — ver captura en
`results/pilot_rd/Dosis importada.png`.

**Scripts nuevos:** `scripts/predict_one.py` (inferencia standalone de 1 paciente, sin pasar por
todo el test set — reusa `cargar_modelo()` de `evaluate.py`) y `scripts/write_rd_dicom.py`
(clona un RD real del paciente como template DICOM y reemplaza el `PixelData` con la dosis
predicha, resampleada al grid del RD).

**Hallazgo clave — puente % relativo (grid 256×256) → Gy en grid físico real, sin poder confiar
en `meta['spacing_mm']` del NPZ:** para este paciente `meta['spacing_mm']=[2.5,2.5,3.0]` no
coincide con la CT real exportada hoy (0.976563mm) — evidencia de que la CT que
`preprocess.py` vio en la corrida original (hace meses) no era la misma serie/resolución que la
CT de hoy en Eclipse. Además, el spacing in-plane del grid 256 resultó **anisotrópico**
(`sp_x≈1.60mm` vs `sp_y≈1.09mm` para este paciente, ratio ~1.47) — confirmado que
`preprocess.py` NO recorta en X/Y (solo hace `downsample_inplane` 512→256 con `fy`/`fx`
calculados por separado; sin crop en el código, en ninguna de las 3 versiones en git); la
anisotropía sale de que la CT nativa que se usó originalmente aparentemente no era cuadrada (o
era otra serie), dato que no quedó guardado en ningún lado (ni NPZ, ni código, ni docs) y que no
se pudo recuperar — el proyecto C# `extract_dicom_csharp` (candidato más probable a tener este
paso) es un proyecto separado, nunca versionado en este repo (confirmado: no existe ni un solo
`.cs` en el historial de git).

**Solución adoptada — calibración empírica en vez de reconstruir la transformación perdida:**
en `write_rd_dicom.py::calibrar_grid_256_inplane`, el spacing y origen del grid 256 (por eje,
X e Y por separado) se calibran contra el contorno real del PTV en el RS del paciente (siempre
disponible): `sp_eje = extensión_real_mm(RS) / extensión_píxeles(máscara 256)`, origen resuelto
para que el centro de ambas extensiones coincida. No depende de ningún supuesto sobre la CT
original (tamaño, si hubo crop, qué serie se usó) — por eso generaliza a cualquier paciente sin
necesitar recuperar esa metadata perdida. Validado empíricamente: con esta calibración el
import en Eclipse quedó bien alineado. **Recomendación para el proyecto KBP completo:**
promover esta función a `src/bridge/unet_to_target.py` (estructura propuesta en
`PIPELINE_KBP_PDRT.md`) — es el puente dosis-U-Net→grid-físico real que ese documento dejaba
como pregunta abierta (sección A2).

**Segundo hallazgo — responde una pregunta pendiente de `PIPELINE_KBP_PDRT.md` (sección
"Pendientes de verificación de Pablo"):** se probó usar el RD sintético como dosis base sobre
un plan vacío nuevo, para continuar optimizando en Eclipse desde ahí. **No funcionó — Eclipse
13.6 no lo permite.** Confirma que el camino de "arrancar en paso 4 desde una dosis 3D externa"
está cerrado; el flujo tiene que ser el completo con PyDoseRT (mimicking voxel-wise → leaves/MU
→ RTPLAN completo vía pydicom → import del plan con dosis, no de una dosis suelta). Esto es
consistente con el diseño ya documentado en `PIPELINE_KBP_PDRT.md`, ahora confirmado
empíricamente antes de invertir en comisionar PDRT.

**Otros detalles técnicos del piloto:**
- El modelo (`exp002`) no tiene activación final → dosis predicha con ruido negativo cerca de
  0%. Al castear a `uint32` (formato de pixel del RD) sin clipear, esos negativos hacían
  *wraparound* a ~4.3×10⁹ (dosis "sintética" saliendo en cientos de miles de Gy) — bug
  encontrado y corregido clipeando a 0 antes de escribir.
- El `ReferencedRTPlanSequence` del RD real apunta al plan tratado (con dosis asociada,
  intocable en Eclipse) — hubo que reapuntarlo a un plan vacío nuevo creado por Pablo para
  poder importar (`write_rd_dicom.py --referenced-plan-uid`).
- La alineación en Z (recorte `meta['z_range']`, índices de corte nativo) SÍ resultó exacta
  sin ningún ajuste — verificado que el grid Z del RD real coincide frame-a-frame con los
  cortes de la CT real exportada (mismo step de 3mm, mismo origen).

### Comisionamiento PDRT 6X + chequeo de sandbox (COMPLETADO, 2026-08-10)

**Datos de entrada:** perfiles X/Y, diagonal 40x40 (4 profundidades) y output factors (28
tamaños de campo, incl. asimétricos) exportados por Pablo desde Eclipse sobre phantom de agua,
en `datos comisionamiento/` (los archivos viejos — PDD y 2 perfiles con sólo 2 campos abiertos,
posición sin centrar — quedaron en `datos comisionamiento/viejos/`, no se usan salvo el PDD para
derivar `tpr_20_10` a mano).

**Pipeline armado:** `commissioning/convert_eclipse_to_json.py` parsea el formato propio de
Eclipse (una curva por columna, posición ya centrada en 0 y dosis sin normalizar en los archivos
nuevos — los viejos requerían offset manual, no reusable) al JSON que espera el toolkit de
comisionamiento de PyDoseRT. Ese toolkit (`commissioning/toolkit/`, `commissioning/conversion/`,
`commissioning/run_commissioning_pipeline.py`, `commissioning/machine_config_base_varian.json`)
se vendorizó desde el repo real de GitHub (MIT) porque el paquete pip `pydosert` no lo incluye
(sólo trae el motor de dosis, no los scripts de ejemplo/comisionamiento).

**Hallazgo (a) — bug de doble conteo de divergencia en `jaw_scale`, CORREGIDO.**
`fit_geometric_penumbra` calcula `jaw_scale = semiancho_medido / semiancho_nominal` en el campo y
profundidad de referencia (100x100 a 10cm) — pero el semiancho medido a esa profundidad YA
incluye la magnificación geométrica normal por divergencia del haz, y `_field_size_scaled` la
vuelve a aplicar al simular a cualquier profundidad → divergencia contada dos veces, campos
simulados ~10% más grandes que los medidos, error creciente con el tamaño de campo (verificado
independiente de head-scatter: apagando `head_scatter_magnitude` el resultado no cambiaba).
**Corrección** (documentada con comentario en `run_commissioning_pipeline.py` y `qc_report.py`):
dividir `jaw_scale` por la magnificación en la profundidad de referencia después de
`fit_geometric_penumbra`. Resultado verificado con `qc_report.py` (métricas cuantitativas: tamaño
de campo por cruce al 50%, penumbra 20-80%, RMSE, gamma 1D — se agregó porque el reporte visual
`report.png` del toolkit no alcanzaba para juzgar la calidad del fit a ojo):

| Campo (10x10 + 20x20, las 4 profundidades) | Antes | Después |
|---|---|---|
| Error de tamaño de campo | 11–25 mm | 0.8–1.4 mm |
| RMSE en zona de campo | 4.0–7.0% | 1.3–3.2% |
| Gamma 1D 3%/3mm | 40–89% | 83–100% |

**Hallazgo (b) — límite físico del MLC para 40x40, NO corregible por comisionamiento.** Incluso
con `jaw_scale` corregido, el campo 40x40 simulado sigue sin caer nunca por debajo del 50% de
dosis (~600mm de "ancho" en vez de ~440mm): el semiancho requerido a profundidad (~220mm con
divergencia real) supera el recorrido máximo de las láminas del Millennium 120 modelado
(`leaf_boundaries` en `machine_config_base_varian.json` llega sólo a ±200mm) — no es un mal
ajuste, es pedirle al modelo un campo que su MLC no puede abrir. Sin impacto práctico esperado
(los campos VMAT reales de próstata están muy por debajo de 40x40); los datos de 40x40 siguen
sirviendo para output factors y la forma del diagonal cerca del eje, no para juzgar el borde de
campo completo.

**`tpr_20_10` real (Pablo, del PDD) = 0.679** — reemplaza el placeholder 0.67 de
`machine_config_base_varian.json`.

**Resultado final:** `commissioning/machine_config_6MV.json` (motor comisionado para el 6X).

**Chequeo de sandbox** (`commissioning/recompute_check.py` un paciente,
`commissioning/run_sandbox_batch.py` batch) — recalcula planes VMAT reales con el motor
comisionado y compara contra la dosis AAA real (RD de Eclipse) con gamma 3D. Corrido sobre **7
pacientes** DICOM reales en `dicom_pilot/` (todos terminaron siendo de 2 arcos, ninguno de 3).

Dos bugs de terceros resueltos en el camino (documentados con comentario en el código):
- `pydosert` (pip) `fetch_plan_data` exige `SourceToSurfaceDistance` en todos los control
  points, pero Eclipse sólo lo exporta en el primero de cada arco dinámico (Type 3 opcional en
  DICOM, normal para VMAT) → `patch_rp_missing_ssd()` completa por forward-fill en una copia
  temporal del RP; no afecta el cálculo real de dosis (ray-tracing sobre la CT, no ese escalar).
- `pymedphys.gamma` no corre en Python 3.13 (su backend `interpolation`/econforge importa `cgi`,
  eliminado del stdlib) → `gamma_3d()` propio en `recompute_check.py`, vectorizado por ventana de
  búsqueda local (ambas dosis ya están en la misma grilla, no hace falta interpolar entre
  grillas distintas — más simple que el caso general que resuelve pymedphys).

**Resultados (7 pacientes, media / rango):**

| Métrica | Media | Rango |
|---|---|---|
| Gamma 5%/5mm | 99.0% | 96.7–99.9% |
| Gamma 3%/3mm | 96.6% | 91.6–98.7% |
| diff dosis media PTV_High | −0.7% | −1.2% a −0.1% |
| diff dosis media Rectum | −4.5% | −3.7% a −5.1% |
| diff dosis media Bladder | −6.5% | −4.2% a −8.8% |
| diff dosis media Fémures | ~0%, sin sesgo | −4.9% a +3.6% |
| diff dosis media BODY (dominado por scatter) | −4.3% | −1.9% a −7.3% |

**Conclusión — sandbox validado, con un sesgo sistemático documentado (no un error de
geometría):** el PTV (dosis alta, dentro del haz) prácticamente calca AAA en los 7 pacientes.
Rectum, Bladder y BODY subestiman de forma consistente (mismo signo en los 7 casos, no ruido) —
patrón esperado de un modelo pencil-beam simplificado (PDRT) comparado contra AAA: acierta la
región de dosis alta/primaria (donde se comisionó con los perfiles de agua) pero subestima el
scatter fuera de campo, que es justo donde vive la dosis a recto/vejiga en próstata. Los fémures
no muestran sesgo (van para los dos lados), consistente con que el efecto es de scatter/campo
más que de heterogeneidad ósea específicamente. Caso más débil: `PT_678e11b9a4cb95e9` (gamma
3%/3mm=91.6%, Bladder −8.8%) — sin ser alarmante, queda de referencia. Por diseño del proyecto
(ver más arriba, "por qué NO se valida contra heterogeneidades" y "PDRT nunca sale del sandbox
U-Net→Eclipse") este sesgo se documenta y no bloquea avanzar al mimicking — cualquier preplan que
salga de PDRT se recalcula siempre con AAA en Eclipse antes de usarse.

Archivos: `commissioning/convert_eclipse_to_json.py`, `commissioning/qc_report.py`,
`commissioning/recompute_check.py`, `commissioning/run_sandbox_batch.py`,
`commissioning/machine_config_6MV.json`, `commissioning/sandbox_results/` (dosis por paciente +
`summary.csv`), `commissioning/reports/commissioning/report.png` (dashboard del fit).

### Documentos handoff generados

- `COMISIONAMIENTO_PDRT.md` — qué exportar de dosimetría y cómo mapearlo al pipeline de PDRT.
- `PIPELINE_KBP_PDRT.md` — estructura del proyecto de código para Claude Code (previo: chat
  rápido de diseño con Claude.ai para ordenar el repo).

---

## Referencias clave

- DoseDiff (Zhang et al. 2024): PSDM + multi-encoder.
- HD U-Net (Nguyen et al. 2019): U-Net con conexiones densas por nivel.
- Moment Loss (Jhanwar et al. 2022): MAE + momentos M1/M2/M10. λ=0.01 en el paper — no
  replicado en este dataset (ver exp006).
- OpenKBP (Babier et al. 2021): dose score y DVH score como métricas estándar.

---

## Validación cuantitativa del arc-prior TERMA — analisis angular (COMPLETADO, 2026-08-12)

Diagnóstico puro (sin entrenar, sin tocar configs/splits) para reemplazar la evidencia
"visual" del artefacto radial (ver `RESUMEN_prior_terma_arcos.md`) por una métrica
espectral y testear las 4 hipótesis que motivarían retomar el arc-prior. Script:
`scripts/analisis_angular.py`. Salidas en `results/analisis_angular/`
(`metricas_angulares_por_paciente.csv`, `summary.json`, `plots/`).

**Dataset:** val+test de `splits_hipo_v2_clean_balanced.json` (22+31=53 pacientes, dataset
limpio, sin contaminación nodal). Predicciones regeneradas por inferencia con
`checkpoints/exp_hipo_002b_finetune_clean/epoch=028.ckpt` (el mismo checkpoint de
referencia de la serie hipo).

**⚠️ Hallazgo de infraestructura (no anticipado):** el CSV
`metricas_planes_hipofx_D95norm_clean.csv` (con `NArcos`, `MUs`, overlap/volúmenes) vivía
en `C:\Pablo\ProstateDoseProject\dicoms hipofx\`, carpeta que **ya no existe en disco**
(limpieza de espacio — la carpeta de DICOMs crudos hipo tampoco existe más). Consecuencia:
- `VolPTV/Rectum/Bladder_cc` y `overlap_rel_recto/vejiga` se **recalcularon directo desde
  las máscaras del NPZ** (mismo patrón de calibración de vóxel que
  `scripts/compute_overlap_real.py`) — de hecho más correcto que el CSV viejo (evita el
  bug `Math.Min` ya documentado).
- **`MUs` y `NArcos` NO están disponibles para este cohorte.** El control de "modulación
  general" de H3 (vs. `MUs`) y la **estratificación transversal por NArcos (2 vs 3
  arcos) NO se pudieron ejecutar** — no hay ningún otro archivo en el proyecto con esas
  columnas para este split. Si se necesitan a futuro, hay que re-derivar `NArcos` desde
  los RP DICOM (que tampoco existen ya en disco) o desde una nueva exportación.

**Convención angular (verificada empíricamente, no asumida):** θ=0° hacia anterior,
centro = centroide del PTV por corte. Verificado sobre 270 slices/15 pacientes: fila del
centroide de Rectum > fila del centroide de Bladder en el 100% de los casos → fila
creciente = posterior (consistente con LPS estándar). La dirección de giro (a qué lado
queda "izquierda") no se verificó — no afecta ninguna métrica usada (ver docstring del
script). `amp_estrias` = RMS de amplitud de armónicos n=4–12 (FFT sobre 180 muestras
angulares, muestreadas por interpolación bilineal sobre círculos concéntricos al PTV, r∈
{1.5,2.0,2.5,3.0}×R_eq, descartando/interpolando ángulos fuera de BODY), normalizado por
la dosis media del círculo (real) — mismo denominador para real/predicha/residuo, así son
comparables en escala (el residuo tiene media ≈0, normalizarlo por su propia media
hubiera sido un error de escala, corregido durante la implementación).

### Resultados (headline r=2.0×R_eq; ver `summary.json` para las 4 radios completas)

**H1 — ¿hay estructura angular en la dosis real? → CONFIRMADA.**
`amp_estrias` real = 0.112 ± 0.030 vs. control null (mismos valores, orden angular
barajado) = 0.083 ± 0.021 (razón 1.35×, Wilcoxon p=6.8e-9). Efecto crece con el radio
(1.36× en r=1.5 hasta 1.52× en r=3.0, p<2e-10 en todos). La banda baja (n=1-3, forma del
cuerpo, confounder) tiene amplitud ~2× mayor que la banda media — consistente con que hay
señal real en la banda de interés por encima del ruido de forma.

**H2 — ¿las estrías siguen al OAR (evitación anatómica) o son firma mecánica? →
NO CONCLUYENTE, con hallazgo del confounder explícito.** θ_recto y θ_vejiga son
extremadamente estereotipados entre pacientes (circstd 3.0° y 5.0° respectivamente,
Rayleigh p~1e-22 — la próstata casi no varía de geometría relativa recto/vejiga entre
pacientes). θ_min (ángulo del mínimo de la banda-estrías) SÍ es no-uniforme a nivel
poblacional (Rayleigh p=3.6e-8 en r=2.0, más fuerte aún en r=1.5 p=7e-17) y su media
circular (164-176° según radio) cae cerca de θ_recto (180°) — pero con circstd=62-99°
(21× más disperso que θ_recto). La correlación entre la desviación de θ_min respecto a su
media poblacional y la desviación de θ_recto respecto a la suya es baja y no significativa
(Spearman rho=0.18-0.20, p=0.14-0.20 en r=1.5/2.0). **Conclusión honesta: el dataset no
permite separar "mecánico" de "anatómico" con este método — no porque falte estructura,
sino porque el OAR casi no varía de posición entre pacientes (recto siempre posterior,
vejiga siempre anterior, geometría de próstata estereotipada). Tampoco es firma puramente
mecánica: si lo fuera, θ_min debería ser tan fijo como θ_recto (circstd~3°), y en cambio
tiene 20-30× más dispersión.** La señal es más fuerte y más consistente cerca del PTV
(r=1.5-2.0×R_eq) y se degrada con la distancia (r=2.5: Rayleigh p=0.01; r=3.0: p=0.067, ya
no significativo) — sugiere que la modulación angular es un efecto de campo cercano que se
diluye en el scatter de fondo lejos del blanco.

**H3 — ¿la amplitud crece en planes forzados? → CONFIRMADA (parcialmente — sin el control
de MUs).** `amp_estrias` correlaciona con `overlap_rel_recto` (rho=0.30, p=0.027),
`overlap_rel_vejiga` (rho=0.33, p=0.016) y `VolPTV_cc` (rho=0.39, p=0.004) en r=2.0 — más
fuerte aún en r=1.5 (overlap_rel_recto rho=0.72, p=1.4e-9). Los no-cumplidores tienen
`amp_estrias` mayor que los cumplidores en los 3 constraints operativos en r=1.5
(RV65 p=0.0009, RV55 p=0.0036, BV65 p=0.024) y BV65 se mantiene significativo hasta r=2.5.
**No se pudo controlar por MUs** (dato no disponible, ver arriba) — la correlación con
overlap podría estar parcialmente confundida con modulación general si ambos covarían con
MUs; queda pendiente si se recupera esa columna.

**H4 — ¿el modelo captura la estructura? → CONFIRMADA (el punto que más importa para
decidir el arc-prior).** El modelo suaviza sistemáticamente: `amp_estrias` predicha es
0.52× la real en r=2.0 (0.36-0.77× según radio, Wilcoxon p<2e-10 en todos). El residuo
(pred−real) retiene 0.85-0.95× de la amplitud de la banda-estrías real en r=2.0-3.0 (0.60×
en r=1.5) — es decir, la mayor parte de la estructura que el modelo no reproduce queda
literalmente en el error. El ángulo del mínimo del residuo coincide angularmente con el
ángulo del mínimo real más de lo esperable por azar (test de Rayleigh sobre la diferencia
angular, p=0.003 en r=2.0, p<1e-4 en r=2.5/3.0, p=4.8e-11 en r=1.5) — el error NO es
angularmente aleatorio, está anclado a donde está la estructura real. Y la amplitud del
residuo escala con la amplitud real por paciente (Spearman rho=0.85-0.94, p<1e-15 en
r=2.0-3.0): el modelo falla más, específicamente en la banda angular de interés, donde hay
más estructura que reproducir. El espectro de amplitud promedio (`espectro_fft_promedio.png`,
panel log) muestra visualmente esto: real y residuo caminan casi juntos desde n≈3 en
adelante, mientras la predicha cae mucho más rápido — el modelo actúa como un filtro
pasa-bajos angular.

### Conclusión para la decisión de arquitectura

> **⚠️ CORRECCIÓN (2026-08-12, posterior — ver sección "Métrica DVH-curva-completa" más
> abajo):** la lectura original de esta sección ("las 4 hipótesis juntas justifican el
> arc-prior") **quedó revisada tras mirar el desglose por radio con más cuidado**. Se deja
> el razonamiento original como registro (útil para ver qué cambió), pero la conclusión
> vigente es la de la caja de abajo.

~~Las 4 hipótesis, juntas, dan la señal más fuerte a favor del arc-prior TERMA que se tuvo
hasta ahora — pero por una razón distinta a la original (techo de arcos, ya descartado) o
a clasificación (donde el ML clásico ya gana, ver más arriba): el modelo demostrablemente
suaviza una estructura angular real, medible, que crece en los casos de peor cumplimiento,
y el residuo se concentra donde esa estructura vive.~~ (superado — ver corrección).

**Pendiente si se retoma el arc-prior:** recuperar `NArcos`/`MUs` (re-derivar de RP DICOM
o re-exportar) para (a) el control de modulación general en H3 y (b) la estratificación
2 vs 3 arcos que verificaría si la firma cambia con el muestreo angular mecánico — sin eso,
no se puede descartar del todo que parte de la señal de H1/H3 sea sensibilidad al número
de arcos en vez de (o además de) dificultad geométrica real. Este pendiente sigue vigente
independientemente de la corrección de abajo.

---

## ⚠️ CORRECCIÓN — H3 y H4 NO coinciden angularmente → apoya loss de DVH, no el arc-prior (2026-08-12)

Mirando el desglose por radio de `results/analisis_angular/summary.json` (ya generado, no
hubo que re-correr nada) en vez de solo el headline r=2.0, aparece un patrón que la
conclusión anterior no capturaba: **H3 (estructura ligada al OAR) y H4 (residuo grande del
modelo) viven en radios DISTINTOS, casi sin superposición.**

| Radio (×R_eq PTV) | H3 — correlación amp_estrias vs. `overlap_rel_recto` | H4 — cuánto de la amplitud real captura el modelo (`ratio_pred_vs_real`) |
|---|---|---|
| r=1.5 (cerca del PTV) | rho=0.72, **p=1.4e-9** (fuerte) | 0.766 → el modelo **ya captura ~77%** |
| r=2.0 | rho=0.30, p=0.027 (débil) | 0.523 → captura ~52% |
| r=2.5 | rho=0.11, p=0.44 (NO sig.) | 0.385 → captura ~39% |
| r=3.0 (lejos, dosis baja) | rho=0.11, p=0.41 (NO sig.) | 0.366 → captura ~37% (el peor punto, el residuo es más grande acá) |

**Lectura:** la parte de la estructura angular que SÍ está ligada al OAR (overlap PTV-recto/
vejiga, la señal que un canal de geometría de arcos ayudaría a explicar mejor) vive **cerca
del PTV (r1.5)** — y ahí el modelo **ya la captura razonablemente bien** (77% de la
amplitud). La parte donde el modelo **falla más** (residuo más grande, solo ~37-39%
capturado) vive **lejos del PTV, en la zona de dosis baja (r2.5-3.0)** — y ahí la
correlación con `overlap_rel_recto/vejiga` **deja de ser significativa**. Es decir: el
residuo grande no es principalmente el residuo ligado a evitación de OAR — es una
estructura angular más genérica (probablemente relacionada a forma de arco/geometría de
entrega en general, no a la posición específica del OAR de cada paciente), en una zona de
dosis donde el impacto clínico directo (constraints de recto/vejiga, que operan en dosis
media-alta, V65/V55) es menor.

**Esto cambia el argumento para el arc-prior:** un canal de geometría de arcos (tipo
Barragán-Montero/TERMA) se justificaría mejor si el residuo grande estuviera DONDE está la
señal ligada al OAR (cerca del PTV) — ahí un canal de "qué ángulos atraviesan el OAR"
tendría paridad causal directa con el error del modelo. Pero el residuo grande está lejos,
donde la correlación con el OAR se diluye. Un input de geometría de arcos no tiene una
razón clara para arreglar específicamente esa zona lejana si no está ligada a la anatomía
del paciente — podría ser tan relevante como cualquier otra fuente de suavizado genérico del
modelo (kernel size, capacidad, receptive field). **En cambio, la línea base de DVH-curva-
completa (ver sección siguiente) muestra que el error donde SÍ importa clínicamente —
recto y vejiga, banda de dosis media (40-80% Rx) — es justamente donde el modelo peor
reproduce el DVH** (mean|ΔV| recto banda media = 8.36pp, la banda dominante). Una loss que
fuerce fidelidad de DVH ataca directamente esa banda, sin necesitar inventar qué parte del
residuo es anatómica y cuál es genérica — mientras que el arc-prior necesitaría esa
distinción para justificarse y, con los datos actuales, no la tiene clara.

**Decisión:** el diagnóstico conjunto (angular + DVH-curva-completa) apoya priorizar
**exp007 (loss híbrida dosis+DVH, Nguyen et al. 2020 — ver receta ya documentada en
`RESUMEN_prior_terma_arcos.md`)** antes que el arc-prior. El arc-prior TERMA (el canal
literal de dose-rate por control point) sigue sin evidencia que lo justifique
específicamente. Ver más abajo la nota sobre la clase más amplia de priors de geometría de
entrega, que NO queda cerrada, solo condicionada.

---

## Métrica DVH-curva-completa — línea base pre-exp007 (COMPLETADO, 2026-08-12)

Vara de comparación KBP real (no solo 3 puntos de constraints operativos) para poder medir
si exp007 (loss DVH) mejora algo, y para localizar en qué banda de dosis vive el error de
cada estructura. Evaluación pura (sin entrenar, sin tocar configs/splits). Script:
`scripts/dvh_curva_completa.py`. Salidas en `results/dvh_curva_completa/`
(`dvh_fidelity_por_paciente.csv`, `summary.json`, `plots/`).

**Dataset:** SOLO test (n=31) de `splits_hipo_v2_clean_balanced.json` — a diferencia del
análisis angular (que usaba val+test para más n en un diagnóstico exploratorio), acá es una
métrica de evaluación, mismo cohorte de test que se reporta en el resto del proyecto.
Checkpoint: el mismo de referencia, `exp_hipo_002b_finetune_clean/epoch=028.ckpt`.

**Normalización:** GT ya viene con D95(PTV_real)=100% (de `preprocess_hipo.py`). La dosis
PREDICHA se renormaliza a D95(PTV_pred)=100% ANTES de calcular su DVH (factor de renorm
medio=1.001, std=0.015, rango 0.992-1.077 en los 31 pacientes — casi todos muy cerca de 1,
consistente con el bias de escala ya conocido y chico del finetune) — esto aísla fidelidad
de FORMA del sesgo de escala global. Se clipea la dosis predicha a 0 antes de renormalizar
(el modelo no tiene activación final, puede dar ruido negativo cerca de 0%, mismo fix que
`write_rd_dicom.py` ya aplicaba). **Sanity checks: todos OK** (V(0)=100% en las 4
estructuras × 31 pacientes, DVH monótona no creciente, D95(PTV_pred) post-renorm dentro de
±0.5pp de 100 en el 100% de los casos).

### Resultado — mean|ΔV(D)| (grilla 0-110% Rx, paso 1%, 111 bins), pp de volumen

| Estructura | Global | Banda baja (0-40%) | Banda media (40-80%) | Banda alta (80-110%) | Banda dominante |
|---|---|---|---|---|---|
| PTV     | 1.11 ± 0.98 | 0.00 | 0.00 | **4.12** | alta (cobertura/hombro cerca de Rx) |
| Rectum  | 5.03 ± 3.42 | 3.76 | **8.36** | 2.34 | **media (40-80%)** |
| Bladder | 2.79 ± 2.61 | 3.24 | **3.37** | 1.40 | media (40-80%), muy cerca de baja |
| BODY    | 0.57 ± 0.27 | **1.14** | 0.30 | 0.14 | baja (magnitud chica en términos absolutos) |

**La banda que domina el error de Rectum y Bladder es la de dosis MEDIA (40-80% Rx) — la
misma zona donde vive la "estrella" del análisis angular (banda espectral n=4-12) y donde
operan los constraints clínicos reales (V65/V55).** Rectum en banda media (8.36pp) es
~2.2× peor que en banda baja (3.76pp) y ~3.6× peor que en banda alta (2.34pp) — la firma es
clara y específica de OAR, no de BODY en general (BODY es chica en todas las bandas, y su
banda dominante es la baja, patrón distinto). Esto es la confirmación directa (vara DVH,
independiente de la vara espectral angular) de que el blur del modelo en dosis media-baja
tiene consecuencia clínica medible en fidelidad de DVH de OAR — no es solo un artefacto
visual/espectral sin traducción a DVH.

**Vara vieja (OpenKBP) al lado, misma base de comparación (dosis renormalizada):**
`dose_score_openkbp` = 2.23 ± 0.83 (≈ MAE body ya reportado, 2.24, consistente — el factor
de renorm medio ≈1 no mueve la aguja), `dvh_score_openkbp` = 2.06 ± 1.37. Estos 2-3 números
escalares no distinguen banda de dosis ni estructura — la métrica nueva es la que permite
decir "el problema es Rectum, banda media" en vez de solo "hay 2.06 de error DVH en algún
lado".

**Overlay de 3 pacientes (bajo/medio/alto mean|ΔV| de Rectum):** el caso "alto"
(`PT_f894582c126c65e3`, mean|ΔV| recto=19.08pp) es el MISMO paciente identificado como
"alta amp_estrias" en el análisis angular — cross-validación entre las dos varas,
independientes una de otra, sobre el mismo caso peor: el DVH real de recto cae abruptamente
entre 20-60% Rx (un escalón pronunciado) que la predicha reproduce como una caída mucho más
gradual y desplazada ~20-30pp de dosis — el mismo modo de falla (el modelo no reproduce un
perfil de dosis con estructura fina/abrupta en la zona media) visto desde dos ángulos
distintos (espectral y DVH).

### Archivos generados
- `results/dvh_curva_completa/dvh_fidelity_por_paciente.csv` (124 filas = 31 pacientes × 4
  estructuras: mean|ΔV| global + 3 bandas, factor de renorm, chequeos de sanidad).
- `results/dvh_curva_completa/summary.json` (agregados + vara OpenKBP de referencia).
- `results/dvh_curva_completa/plots/`: `dvh_medio_<estructura>.png` (DVH medio±std real vs.
  predicha + ΔV(D) medio±std, por estructura) y `overlay_3_pacientes_{rectum,bladder}.png`.

---

## TERMA descartado — clase amplia de priors de geometría de entrega: CONDICIONADA, no cerrada

Con la corrección de arriba, **el arc-prior TERMA literal (dose-rate por control point,
Barragán-Montero 2019 tal cual) queda sin evidencia específica que lo justifique** — la
parte del residuo donde el modelo peor performa (r2.5-3, dosis baja) no correlaciona con
overlap PTV-OAR, así que no hay razón clara para pensar que un input de "qué OAR atraviesa
cada ángulo" resolvería justo esa zona. La prioridad pasa a exp007 (loss DVH).

Pero esto **no cierra la clase más amplia de priors de geometría de entrega** (BEV masks,
PDD por ángulo de gantry, fluencia-por-ángulo — variantes de la misma familia de idea de
Barragán-Montero, adaptadas a VMAT en vez de a haces estáticos). Queda **CONDICIONADA**, no
descartada: si exp007 (loss DVH) **no** resuelve la banda media de Rectum/Bladder (medida
acá como línea base: 8.36pp y 3.37pp respectivamente) o si el error en la zona lejana/dosis
baja (r2.5-3 del análisis angular) sigue siendo grande y clínicamente relevante después de
la loss DVH, ahí sí valdría la pena revisitar un input de geometría de entrega — pero
re-enfocado específicamente a esa zona lejana/genérica (no a la evitación de OAR cerca del
PTV, que el modelo ya captura razonablemente bien sin ayuda adicional). Orden de decisión:
1) correr exp007, 2) re-medir esta misma línea base (mismo script, mismo split) sobre el
checkpoint de exp007, 3) si la banda media de OAR no mejora sustancialmente, recién ahí
evaluar un prior de geometría de entrega re-enfocado.

---

## Extensión — signo del error de DVH: blur simétrico vs. bias direccional (COMPLETADO, 2026-08-13)

Extensión chica de la línea base anterior, mismo checkpoint/split/renorm/bins. Objetivo:
separar, en la banda media [40-80%Rx] de Rectum/Bladder, cuánto del error es blur simétrico
(forma S, una loss de DVH sigmoide sola alcanza) vs. bias direccional (mono-signo, hace
falta un término de gradiente/edge además). Script: `scripts/dvh_signed.py`. Salidas:
`results/dvh_curva_completa/dvh_fidelity_por_paciente_signed.csv`, `summary_signed.json`,
`plots/delta_v_con_signo_por_estructura.png`, `plots/delta_v_rectum_cumple_vs_no.png`.

**Convención de signo:** ΔV(D) = V_pred(D) − V_real(D). ΔV>0 = PESIMISTA (predice más
volumen a esa dosis, OAR más caliente que la realidad). ΔV<0 = OPTIMISTA (el caso peligroso
para un target de PDRT: se "comería" violaciones reales).

### ⚠️ Hallazgo principal — el ratio poblacional bajo es un ARTEFACTO DE CANCELACIÓN, no blur real

El discriminador poblacional (`ratio_bias = |mean_signed ΔV|_media / mean|ΔV|_media`) da
0.24 (Rectum) y 0.20 (Bladder) — a primera vista, "domina blur simétrico". **Pero el mismo
ratio calculado POR PACIENTE (no sobre la curva promedio) da 0.78±0.33 (Rectum) y
0.97±0.12 (Bladder)** — casi lo opuesto. La resolución: `signed_mean_media` por paciente
tiene std=9.77pp (Rectum) / 4.99pp (Bladder), con solo 48%/45% de los pacientes
compartiendo el signo del promedio poblacional. **Cada paciente individual tiene un sesgo
direccional fuerte y consistente dentro de su propia banda media — pero la DIRECCIÓN
(pesimista u optimista) varía de paciente a paciente, y esas direcciones opuestas se
cancelan al promediar la población.** El ratio poblacional bajo no es evidencia de blur
simétrico dentro de cada paciente — es un promedio de sesgos direccionales opuestos.
`summary_signed.json` incluye este chequeo automatizado (`nota_poblacional_vs_per_paciente`
por estructura) precisamente para no repetir esta lectura errónea en el futuro.

### La curva poblacional tampoco es una S — es una JOROBA de un solo signo

Sub-bandas 40-60% y 60-80%: Rectum +2.36 / +1.68pp, Bladder +0.67 / +0.65pp — **mismo signo
en ambas mitades**, sin cruce por cero dentro de la banda media (0 cruces significativos
40-80 en las 3 estructuras). La curva poblacional real: sube desde ~0 en D=20% hasta un
pico +2.5pp (Rectum, D≈55-60%) / +1.4pp (Bladder, D≈35%), baja de nuevo, y solo cruza cero
cerca de D≈95-105%, con una leve zona negativa en dosis muy baja (D<20%) y muy alta
(D>95%). **No es un hombro desplazado hacia un lado (shift) — es una redistribución de
volumen desde los extremos (dosis muy baja y muy alta, donde ΔV es levemente negativo)
hacia el centro (30-90%, positivo)**, la misma firma de "regresión a la media"/suavizado ya
vista en el análisis angular (el modelo actúa como filtro pasa-bajos) y en la métrica sin
signo (línea base anterior).

**Nota — PTV en banda media es degenerado, no un hallazgo real:** tanto `mean|ΔV|` como
`|signed_mean|` son ~0 (real y predicha saturan a ~100% de volumen en esa banda, casi sin
señal) — el ratio=1.00 que sale ahí es división por un número casi nulo, no "bias
direccional". Ignorar esa fila del reporte automático.

### Rectum, cumple vs. no-cumple RV65 (real) — el riesgo peligroso NO se observa

- **Cumple (n=21):** signed_mean_media=−0.27pp (casi neutro), ratio_bias=0.04 → error chico
  y ahí sí mayormente simétrico.
- **No-cumple (n=10):** signed_mean_media=+6.82pp (**PESIMISTA**), ratio_bias=0.62,
  7/10 pacientes pesimistas (incluye el peor caso del cohorte, `PT_f894582c126c65e3`,
  +32.6pp, el mismo paciente identificado como "alta amp_estrias" en el análisis angular).

**El riesgo que se quería descartar explícitamente — que el modelo fuera OPTIMISTA
justo en los casos que en la realidad no cumplen (subestimaría violaciones reales) — NO se
observa.** Es al revés: en los casos difíciles el modelo tiende a sobreestimar dosis de
recto en la banda media. Es el sesgo más seguro posible para un target de PDRT (sobre-
advertir es mejor que subestimar una violación real). `ALERTA_optimista_en_no_cumple` en
`summary_signed.json` está en `false`.

### Implicación para exp007

Como la loss DVH (sigmoide, Nguyen et al. 2020) se calcula por paciente/muestra durante el
entrenamiento (no sobre una curva ya promediada de la población) y es una loss de
**valores** (matching V(D) en varios umbrales de dosis, sensible a un hombro desplazado, no
solo a su pendiente), en principio debería atacar directamente este sesgo direccional
por-paciente sin necesitar agregar un término de gradiente/edge separado — a diferencia de
lo que un `ratio_bias` poblacional mal leído podría sugerir. La pregunta queda abierta
empíricamente: al re-medir esta misma métrica sobre el checkpoint de exp007, mirar el
**ratio_bias POR PACIENTE** (no solo el poblacional, que esconde el problema por
cancelación) y si la joroba se achica en amplitud sin dejar de ser joroba (blur) o si además
se corrige la dispersión inter-paciente de `signed_mean_media` (el verdadero indicador de si
el sesgo direccional por-paciente mejoró) — ver resultado real de exp007/exp_hipo_003_dvhloss
más abajo, que responde esta pregunta empíricamente.

---

## exp_hipo_003_dvhloss (= "exp007") — PRIMER INTENTO, resultado NEGATIVO a λ₀ calibrado (COMPLETADO, 2026-08-14)

Primera tarea de ENTRENAMIENTO de la serie DVH-loss (sí se tocaron configs). Objetivo:
agregar la loss DVH diferenciable (Nguyen et al. 2020, sigmoide, SIN adversarial — receta en
`RESUMEN_prior_terma_arcos.md`) sobre el baseline de referencia (002b), para atacar el
corrimiento direccional del hombro del DVH de OAR medido en `summary_signed.json`.

### Setup (match limpio con 002b — confirmado leyendo `logs/exp_hipo_002b_finetune_clean.log`, no asumido)

- `configs/exp_hipo_003_dvhloss.yaml`: copia exacta de `exp_hipo_002b_finetune_clean.yaml`,
  único bloque cambiado es `loss`. Init-weights (`checkpoints/exp002_unet2d_psdm/epoch=191.ckpt`),
  split (`splits_hipo_v2_clean.json`, el NO-balanceado — igual que 002b), LR=1e-5, horizonte
  (100 epochs) y early stopping (patience=30) idénticos.
- Loss implementada en `src/losses/losses.py` (`DifferentiableDVHLoss`, conectada a
  `CombinedLoss`): `L_total = L_MAE + λ·L_DVH`, `L_DVH` solo sobre PTV/Rectum/Bladder, sigmoide
  `V_approx_pred(D) = mean[sigmoid(s·(dosis_pred−D))]` vs. `V_exact_gt(D)` (sin gradiente),
  L1 promediada sobre estructuras y bins. D muestreado en el hombro (OAR 40-95%Rx paso 2.5%,
  PTV 90-107%Rx paso 1%), pendiente `s = k/paso_bin` con `k=4.394` (ancho de transición
  10-90% = 1 bin). **Bug real encontrado y corregido en el camino:** los bins de dosis
  registrados como buffers rompían la carga `strict=True` de `--init-weights` (el checkpoint
  de exp002 no tiene esos buffers) — arreglado con `persistent=False` (son constantes
  derivadas de la config, no estado aprendido, no deben viajar en el `state_dict`).

### Calibración previa (`scripts/calibrar_dvh_loss.py`, obligatoria antes de entrenar)

- **Chequeo de sesgo del sigmoide** (V_approx vs. V_exacto sobre dosis GT, n=10 pacientes de
  train): PTV 0.304pp medio (max por-bin 3.54pp, cerca de D=100% donde la dosis de PTV está
  muy concentrada), Rectum 0.033pp, Bladder 0.016pp — los 3 << 0.5pp de tolerancia, `k=4.394`
  se dejó sin ajustar.
- **λ₀** (medido sobre 5 batches de train, modelo AL INIT = pesos de exp002 normo, igual que
  arranca el entrenamiento real): `L_MAE_mean=1.9869`, `L_DVH_mean=0.039005` →
  **λ₀ = 1.9869/0.039005 = 50.94**. Detalle completo (los 2 términos crudos por batch) en
  `results/calibracion_dvh_loss.json`.
- Entrenamiento: 1 corrida a λ₀ (sin barrer, como pedía la tarea). Early stopping en época 51,
  mejor checkpoint por `val/mae` = **época 21** (`checkpoints/exp_hipo_003_dvhloss/epoch=021.ckpt`,
  val/mae=2.044 — vs. 1.986 de 002b, ~3% peor ya en val).

### Resultado — tabla comparativa 002b vs. 003 (test=31, split balanceado, las 4 varas pedidas)

`scripts/comparar_002b_vs_003.py` → `results/exp_hipo_003_dvhloss_test_hipo_v2_balanced/comparacion_002b_vs_003.json`.

**Vara 1 — mean|ΔV| banda media OAR (tenía que BAJAR):**

| Estructura | 002b | 003 | Δ | ¿mejoró? |
|---|---|---|---|---|
| Rectum  | 8.36pp | **9.10pp** | +0.74pp (+8.9%) | **NO — empeoró** |
| Bladder | 3.37pp | 3.27pp | −0.10pp (−2.9%) | Sí, marginal |

**Vara 2 — MAE espacial + dose_score (NO tenía que subir):**

| | 002b | 003 | Δ |
|---|---|---|---|
| MAE body | 2.239 | 2.298 | +0.059 (empeoró) |
| MAE PTV | 1.742 | 1.877 | +0.134 (empeoró) |
| MAE Rectum | 6.349 | **7.419** | **+1.070 (+17%, empeoró notablemente)** |
| MAE Bladder | 3.707 | 3.761 | +0.054 (empeoró, casi neutro) |
| dose_score_openkbp | 2.239 | 2.298 | +0.059 (empeoró) |

**Las 4 métricas espaciales empeoraron** — el escenario que la tarea pedía explícitamente
vigilar ("no comprar DVH degradando el mapa") **ocurrió**, y con margen notable en Rectum.

**Vara 3 — signo en no-cumple RV65 (el conteo de casos optimistas duros NO tenía que aumentar):**

| | 002b | 003 |
|---|---|---|
| signed_mean_media no-cumple | +6.82pp (pesimista) | +5.55pp (pesimista, magnitud menor) |
| frac. optimistas en no-cumple | 0.30 | **0.40** |
| n casos optimistas (de 10) | 3 | **4 — AUMENTÓ** |
| Alerta "optimista en promedio en no-cumple" | false | false (no se invirtió el promedio) |

El promedio del grupo no-cumple se mantiene pesimista en ambos (sin alerta), pero el conteo
de casos individuales optimistas (el riesgo concreto que se quería vigilar) subió de 3 a 4/10.

**Vara 4 — constraints operativos (control, NO debían romperse):**

| Constraint | AUC 002b | AUC 003 | ¿se rompió? |
|---|---|---|---|
| RV65 | 0.919 | **0.838** | Sí (−0.081) |
| RV55 | 0.839 | **0.720** | Sí (−0.119) |
| BV65 | 0.933 | 0.938 | No (+0.005, dentro de ruido) |

### Conclusión — λ₀ calibrado por balance de magnitud fue DEMASIADO ALTO en este dataset

**Resultado neto: negativo.** El objetivo primario (bajar mean|ΔV| banda media de Rectum) no
solo no se logró — empeoró (+8.9%). Bladder mejoró marginalmente (−2.9%) pero a costa de MAE
espacial peor en las 4 estructuras (Rectum +17%) y AUC de RV65/RV55 rotos (−0.08 y −0.12).
El conteo de casos "optimistas duros" en no-cumple subió de 3 a 4/10 — tampoco se cumplió esa
condición. Esto es consistente con `λ₀` (calibrado solo por balance de MAGNITUD de gradiente
al init, `L_MAE/L_DVH` a λ=1) siendo demasiado agresivo una vez que el entrenamiento avanza:
`L_MAE` naturalmente se achica a medida que el modelo mejora, mientras que `L_DVH` — una loss
de forma, no de valor absoluto de error — puede seguir empujando fuerte y terminar dominando
la optimización, sacrificando fidelidad voxel-a-voxel (y hasta DVH de Rectum especificamente)
por ajustar los bins del sigmoide. **No es evidencia de que la loss DVH esté mal
implementada o sea mala idea** (el chequeo de sesgo del sigmoide fue impecable, la
implementación matchea la receta del paper) — es evidencia de que `λ₀` (un punto de partida
razonable, no un valor final) resultó demasiado alto para ESTE dataset/arquitectura.

**Responde la pregunta que había quedado abierta en la sección anterior** ("¿la joroba se
achica sin dejar de ser joroba, o se corrige la dispersión inter-paciente de
`signed_mean_media`?"): NO se corrigió — de hecho la dispersión inter-paciente de
`signed_mean_media` en Rectum **aumentó** (std 9.77pp → 10.34pp) y el ratio_bias per-paciente
se mantuvo igual de alto (0.78 → 0.80). El sesgo direccional por-paciente sigue sin explicarse
ni corregirse con esta loss a este λ.

### Próximo paso — NO barrer a ciegas, bajar λ y volver a correr UNA vez

Tal como se pidió, esta corrida fue a λ₀ sin barrer. El resultado indica que el próximo intento
debería usar un λ bien más chico (ej. λ₀/5 ≈ 10, o λ₀/10 ≈ 5) para que `L_DVH` actúe como una
guía suave sin dominar `L_MAE` — y recién ahí, si el resultado es mixto, considerar un barrido
formal (2-3 valores) en vez de otro tiro único. Mantener las mismas 4 varas + el chequeo de
sesgo del sigmoide (ya validado, no hace falta recalibrar) para cualquier corrida futura.
Archivos de esta corrida: `configs/exp_hipo_003_dvhloss.yaml`,
`checkpoints/exp_hipo_003_dvhloss/`, `results/exp_hipo_003_dvhloss_test_hipo_v2_balanced/`,
`results/exp_hipo_003_dvhloss_dvh_curva_completa/`, `results/calibracion_dvh_loss.json`.

---

## exp_hipo_003b_dvhloss_lambda10 (λ₀/5=10) — recuperación PARCIAL (COMPLETADO, 2026-08-14)

Segundo intento, pedido explícito: "avancemos con lambda/5=10 y vayamos viendo parametros
cada 5 epochs para evaluar en early stop manual si requiere una reducción más agresiva de
lambda". Setup idéntico a exp_hipo_003_dvhloss salvo `loss.dvh_weight: 10.0` (config
`configs/exp_hipo_003b_dvhloss_lambda10.yaml`).

**Monitoreo durante el entrenamiento** (`scripts/watch_lambda_run.py`, lee el estado
embebido en `last.ckpt` — `epoch`, `ModelCheckpoint.current_score`,
`EarlyStopping.best_score/wait_count` — más confiable que parsear el log de texto, que no
imprime una línea limpia por época cuando no hay mejora): la trayectoria de val/mae fue,
desde el principio, mejor que la corrida a λ₀=50.94 y muy cercana a la de 002b (época 0:
2.120 vs. 2.132 de λ₀ vs. 2.069 de 002b; época 10: 2.018, ya mejor que el MEJOR valor que
logró λ₀=50.94 en toda su corrida). **No hizo falta ninguna intervención manual** — el
entrenamiento corrió hasta el final solo (early stopping automático en época 57, mejor
checkpoint época 27, val/mae=2.005 — vs. 1.986 de 002b y 2.044 de λ₀=50.94).

### Resultado — comparación de 3 vías (002b vs. λ=50.94 vs. λ=10)

`scripts/comparar_series_dvhloss.py` → `results/comparacion_series_dvhloss.json`.

**Vara 1 — mean|ΔV| banda media OAR:**

| Estructura | 002b | λ=50.94 | λ=10 |
|---|---|---|---|
| Rectum  | 8.36pp | 9.10pp (peor) | **8.58pp** (recupera casi todo, queda +2.6% vs 002b) |
| Bladder | 3.37pp | 3.27pp | **3.19pp — MEJOR que 002b** (−5.3%) |

**Vara 2 — MAE espacial + dose_score:**

| | 002b | λ=50.94 | λ=10 |
|---|---|---|---|
| MAE body | 2.239 | 2.298 | 2.266 (recupera ~55% del daño) |
| MAE PTV | 1.742 | 1.877 | 1.810 (recupera ~50%) |
| MAE Rectum | 6.349 | 7.419 (+17%) | 6.870 (**+8.2% vs 002b** — recupera ~50% del daño, no lo cierra) |
| MAE Bladder | 3.707 | 3.761 | **3.702 — practicamente igual a 002b** |
| dose_score_openkbp | 2.239 | 2.298 | 2.266 |

**Vara 3 — signo en no-cumple RV65:**

| | 002b | λ=50.94 | λ=10 |
|---|---|---|---|
| signed_mean_media no-cumple | +6.82pp | +5.55pp | **+4.61pp** (magnitud del sesgo "seguro" sigue achicándose — tendencia monótona con λ menor) |
| n casos optimistas (de 10) | 3 | 4 | **4 — sin recuperar**, se estancó en el mismo valor que λ=50.94 |

**Vara 4 — constraints operativos (AUC):**

| Constraint | 002b | λ=50.94 | λ=10 |
|---|---|---|---|
| RV65 | 0.919 | 0.838 | 0.867 (recupera ~36% del daño, queda −0.052) |
| RV55 | 0.839 | 0.720 | 0.762 (recupera ~35%, queda −0.077) |
| BV65 | 0.933 | 0.938 | **0.943 — MEJOR que 002b**, mejora monótona con λ menor |

### Conclusión — recuperación parcial, con una asimetría Rectum/Bladder que vale la pena anotar

**λ=10 es claramente mejor que λ₀=50.94 en las 4 varas, confirmando el diagnóstico anterior**
(λ₀ dominaba la optimización). Pero **no recupera del todo el terreno perdido frente a 002b,
y lo hace de forma DESPAREJA entre estructuras**: en **Bladder, λ=10 iguala o supera a 002b**
en las 3 métricas que la tocan (mean|ΔV| media, MAE espacial, AUC BV65) — la loss DVH está
funcionando como se esperaba ahí. En **Rectum, queda un residuo consistente** (+2.6% en
mean|ΔV| media, +8.2% en MAE espacial, −0.052/−0.077 en AUC RV65/RV55) — mejor que λ=50.94
pero no neutral. El conteo de casos optimistas en no-cumple tampoco volvió a 3/10.

Es la "mejora parcial esperable" que anticipaba la tarea original ("hay piso irreducible: la
varianza de carving plan-específico no es observable desde anatomía") — pero el patrón
Rectum-peor/Bladder-mejor sugiere que no es solo piso irreducible: puede que Rectum necesite
un λ aún más chico que Bladder (son términos separados en la misma loss promediada, no hay
nada que fuerce que el λ óptimo sea igual para ambas estructuras), o que el término D-sampling
de Rectum (40-95%Rx paso 2.5%) interactúe distinto con su geometría típica (estructura más
chica, más pegada al PTV) que el de Bladder.

### Decisión pendiente (no ejecutada — requiere indicación de Pablo)

No se lanzó una tercera corrida sin pedido explícito. Opciones abiertas para la próxima
iteración, en orden de esfuerzo creciente:
1. Aceptar λ=10 como compromiso (Bladder mejora, Rectum casi neutro, mucho mejor que λ=50.94
   en todo) y seguir con otra línea de trabajo.
2. Probar un λ intermedio más chico (ej. λ=5) con el mismo monitoreo cada 5 épocas, para ver
   si Rectum sigue recuperando terreno sin que Bladder empiece a perderlo.
3. Separar el λ por estructura (`dvh_weight` per-estructura en vez de global) — cambio de
   diseño más grande en `DifferentiableDVHLoss`/`CombinedLoss`, solo si (2) no alcanza.

Archivos de esta corrida: `configs/exp_hipo_003b_dvhloss_lambda10.yaml`,
`checkpoints/exp_hipo_003b_dvhloss_lambda10/`,
`results/exp_hipo_003b_dvhloss_lambda10_test_hipo_v2_balanced/`,
`results/exp_hipo_003b_dvhloss_lambda10_dvh_curva_completa/`,
`results/comparacion_series_dvhloss.json`, `scripts/watch_lambda_run.py`,
`scripts/comparar_series_dvhloss.py`.

---

## exp_hipo_003c_dvhloss_lambda5 (λ=5) — CIERRA la serie de barrido: el patrón NO es monótono, justifica λ diferenciado (COMPLETADO, 2026-08-14)

Tercer y último intento de este barrido manual, pedido explícito: correr λ=5 (~λ₀/10) para
"evaluar si mejoran ambos [Rectum y Bladder] o solo recto y a partir de ahí largar con
lambda diferenciado o seguir afinando". GPU ocupada por otro proceso al momento de pedirlo —
se preparó el config sin lanzar, se monitoreó con `nvidia-smi` (2 lecturas libres consecutivas
para evitar falso positivo por un dip) y se lanzó automáticamente al liberarse (~1h después,
como estimó Pablo). Setup idéntico a 003b salvo `loss.dvh_weight: 5.0`
(`configs/exp_hipo_003c_dvhloss_lambda5.yaml`).

**Entrenamiento:** trayectoria de val/mae la mejor de las 3 corridas con loss DVH en casi
todo el recorrido (época 10: 2.013, ya mejor que 002b en el mismo punto). Early stopping en
época 76, mejor checkpoint época 46, **val/mae=1.9933 — prácticamente empata con el mejor de
002b (1.986)**, la diferencia más chica de las 3 corridas con loss DVH.

### Resultado — comparación de 4 vías (002b / λ=50.94 / λ=10 / λ=5)

`scripts/comparar_series_dvhloss.py` (ahora con las 4 corridas) →
`results/comparacion_series_dvhloss.json`.

**Vara 1 — mean|ΔV| banda media OAR:**

| Estructura | 002b | λ=50.94 | λ=10 | λ=5 |
|---|---|---|---|---|
| Rectum  | 8.36pp | 9.10pp | **8.58pp (mejor)** | 8.77pp (peor que λ=10) |
| Bladder | 3.37pp | 3.27pp | 3.19pp | **3.18pp (mejor)** |

**Vara 2 — MAE espacial + dose_score:**

| | 002b | λ=50.94 | λ=10 | λ=5 |
|---|---|---|---|---|
| MAE body | 2.239 | 2.298 | 2.266 | 2.261 |
| MAE PTV | 1.742 | 1.877 | **1.810 (mejor)** | 1.829 |
| MAE Rectum | 6.349 | 7.419 | 6.870 | 6.852 (mejora marginal sobre λ=10) |
| MAE Bladder | 3.707 | 3.761 | 3.702 | **3.693 (mejor que TODAS, incl. 002b)** |
| dose_score | 2.239 | 2.298 | 2.266 | 2.261 |

**Vara 3 — signo no-cumple RV65:**

| | 002b | λ=50.94 | λ=10 | λ=5 |
|---|---|---|---|---|
| signed_mean_media no-cumple | +6.82 | +5.55 | **+4.61 (mínimo)** | +6.14 (vuelve a subir) |
| n casos optimistas (de 10) | 3 | 4 | 4 | 4 |

**El conteo de casos optimistas quedó CLAVADO en 4/10 en las 3 corridas con loss DVH,
sin importar λ (50.94, 10 o 5)** — no responde al ajuste de λ en este rango; no vuelve al
3/10 de 002b con ningún λ probado.

**Vara 4 — constraints operativos (AUC):**

| Constraint | 002b | λ=50.94 | λ=10 | λ=5 |
|---|---|---|---|---|
| RV65 | 0.919 | 0.838 | 0.867 | **0.871 (mejor, sigue mejorando)** |
| RV55 | 0.839 | 0.720 | **0.762 (mejor)** | 0.744 (peor que λ=10) |
| BV65 | 0.933 | 0.938 | **0.943 (mejor)** | 0.938 (vuelve al nivel de 002b) |

### Conclusión — el patrón NO es monótono, y es DISTINTO por estructura: justifica λ diferenciado

**Bladder mejora de forma consistente y casi monótona a medida que λ baja** (mean|ΔV| media,
MAE espacial, y en general se mantiene igual o mejor que 002b en las 4 métricas que la
tocan, incluso con λ=50.94 que fue catastrófico para Rectum). Su óptimo, dentro del rango
probado, está en λ=5 o más abajo — no se llegó a ver el punto donde Bladder empieza a
empeorar.

**Rectum, en cambio, tiene un punto óptimo NO monótono alrededor de λ=10**: mean|ΔV| media y
AUC RV55 son mejores en λ=10 que en λ=5 (aunque λ=5 sigue siendo mejor que λ=50.94 en casi
todo). Bajar de 10 a 5 no sigue mejorando Rectum — lo empeora levemente en 2 de las 4
métricas que lo tocan.

**Esto responde exactamente la pregunta que motivó esta corrida** ("¿mejoran ambos o solo
recto?"): **NO mejoran igual — tienen óptimos distintos**, y ninguno de los 2 (Bladder
todavía bajando, Rectum ya subiendo de nuevo a λ=5) está sincronizado con el otro. Un λ
GLOBAL único no puede ser óptimo para las dos estructuras a la vez. **Esto justifica pasar a
λ diferenciado por estructura** (`dvh_weight` como dict `{ptv: ..., rectum: ..., bladder:
...}` en vez de un escalar único) en vez de seguir afinando un λ global — seguir bajando el
global arriesga perder lo ganado en Rectum (ya se vio, 10→5) a cambio de una mejora en
Bladder que probablemente se puede conseguir igual o mejor con su propio λ más bajo,
independiente del de Rectum.

**El conteo de casos optimistas en no-cumple (vara 3, el riesgo de seguridad que más importa)
no respondió a NINGÚN λ probado (siempre 4/10)** — esto sugiere que ese problema específico
no es principalmente una cuestión de balance de loss, sino de otra fuente (varianza
plan-específica no observable desde anatomía, como anticipaba la tarea original, o
simplemente un subconjunto de pacientes con un patrón distinto que ninguna de estas 3
corridas tocó). No asumir que λ diferenciado lo va a resolver — es una hipótesis a probar,
no una certeza.

### Decisión pendiente (no ejecutada — requiere indicación de Pablo)

**Cambio de diseño propuesto para la próxima iteración:** `dvh_weight` per-estructura en
`DifferentiableDVHLoss`/`CombinedLoss` (ej. `rectum: 10.0, bladder: 3.0` o similar — punto de
partida sugerido, no calibrado) en vez de un único escalar. Requiere:
1. Extender el config (`loss.dvh_weight` pasa de escalar a dict por estructura).
2. Modificar `CombinedLoss.forward` para ponderar cada término de `DifferentiableDVHLoss`
   por separado en vez de promediarlos con el mismo peso.
3. Un nuevo entrenamiento (con el mismo monitoreo cada 5 épocas ya probado) y las mismas 4
   varas para comparar contra las 4 corridas ya hechas.

No lanzado sin indicación explícita — es un cambio de diseño (no solo un hiperparámetro), y
ya se juntaron 4 corridas completas (~10h de GPU en total) para llegar a este diagnóstico.

Archivos de esta corrida: `configs/exp_hipo_003c_dvhloss_lambda5.yaml`,
`checkpoints/exp_hipo_003c_dvhloss_lambda5/`,
`results/exp_hipo_003c_dvhloss_lambda5_test_hipo_v2_balanced/`,
`results/exp_hipo_003c_dvhloss_lambda5_dvh_curva_completa/`,
`results/comparacion_series_dvhloss.json` (ahora con las 4 corridas).

---

## Diagnóstico de causa del hombro medio-bajo de OAR — D1-D5 (COMPLETADO, 2026-08-15)

> **⚠️ CORRECCIÓN DE FRAMING:** el "piso irreducible: la varianza de carving plan-específico
> no es observable desde anatomía" mencionado en la sección de `exp_hipo_003b` (y cualquier
> lectura previa tipo "piso de anatomía") queda reemplazado por el hallazgo de abajo: el piso
> **no es de anatomía, es de CONSISTENCIA DE GENERACIÓN del ground truth** (RapidPlan
> homogéneo en normo vs. planificación manual idiosincrática en hipo). Es una distinción
> importante: "anatomía" sugeriría que ni con infinitos datos se podría predecir mejor
> (límite físico); "generación" dice que el límite es del PROCESO que produjo el GT hipo, no
> de la anatomía en sí — replanificar con un generador consistente (RapidPlan/KBP en hipo, o
> un protocolo manual estandarizado) debería mover ese piso.

Tarea de DIAGNÓSTICO puro (sin entrenar, sin tocar configs/splits) para decidir la CAUSA del
error del hombro medio-bajo de OAR (banda media [40-80%Rx] de Rectum/Bladder, ya cuantificado
en `results/dvh_curva_completa/summary.json`: Rectum 8.36pp, Bladder 3.37pp — la banda
dominante de error en ambas estructuras). Se separaron 4 hipótesis con evidencia independiente
cada una. Scripts en `scripts/diagnostico_piso/`, salidas en `results/diagnostico_piso/`
(`summary.json` con el veredicto agregado + un JSON/CSV por diagnóstico + `plots/`).

### D1 — ¿Gap de generalización? (`d1_gap_train_test.py`)

Inferencia de `exp_hipo_002b_finetune_clean` sobre TRAIN hipo (98 casos, reusando
`dvh_curva_completa.py` sin modificarlo, vía un splits.json temporal con `"test"` apuntando a
los IDs de train) comparado contra el test ya establecido (n=31, resultado reusado tal cual de
`results/dvh_curva_completa/summary.json`).

| Estructura | mean\|ΔV\| media train | mean\|ΔV\| media test | ratio test/train | Veredicto |
|---|---|---|---|---|
| Rectum  | 6.77pp | 8.36pp | 1.23 | dentro de "similar" — NO es generalización |
| Bladder | 2.55pp | 3.37pp | 1.32 | apenas fuera — gap de generalización marginal |

**Lectura:** ambos ratios están cerca de 1 (Rectum claramente, Bladder al borde del umbral
usado, 1.25). El modelo se equivoca casi tanto en datos que YA VIO en entrenamiento como en
test — no es un problema de "no generalizó", es que ni siquiera ajustó bien el train en esta
banda. Esto descarta que más N/augmentation sea la palanca principal para este hombro
específico (no seria underfitting-por-falta-de-datos si el modelo ya no resuelve el train).

### D2 — ¿Información 3D que el 2D descarta? (`d2_coherencia_axial.py`)

El modelo es 2D puro (`lightning_module.py::forward` aplana B×Z y corre convs 2D sin ningún
kernel que cruce Z — confirmado leyendo el código, no asumido). Se testeó si esto introduce
discontinuidad axial espuria (o pierde estructura axial real) específicamente en la banda
media, sobre test hipo (n=31): 3 métricas por paciente/estructura — perfil D(z) en un pixel XY
FIJO (centroide 3D del órgano), perfil de dosis media por corte, y perfil de %volumen del
corte en banda media [40,80]%Rx (esta última liga directo al hombro clínico) — para cada una,
"rugosidad" = mean(\|diff en z\|), real vs. predicha (renormalizada igual que
`dvh_curva_completa.py`).

| Estructura | rugosidad real (vfrac) | rugosidad pred | ratio pred/real | Wilcoxon p | rho(perfil real, perfil pred) |
|---|---|---|---|---|---|
| Rectum  | 6.78 | 6.24 | 0.98 | 0.018 | 0.80 |
| Bladder | 4.99 | 4.86 | 1.00 | 0.555 | 0.94 |

**Lectura:** el ratio de rugosidad está prácticamente en 1.0 en ambas estructuras (Rectum
tiene un p significativo pero el efecto es trivial, ratio=0.98 — por eso el veredicto usa un
umbral de magnitud además del p-valor, no solo significancia estadística con n=31). La
correlación real-vs-predicha del perfil en Z es alta (0.80-0.94): el modelo 2D, aunque procesa
cada corte independiente, YA sigue razonablemente bien la modulación axial real de la banda
media. **No hay evidencia de que un U-Net 3D esté motivado por ESTE hombro específico** —
distinto del hallazgo ya cerrado sobre la "estrella lejana" (que sí es in-plane, ver análisis
angular) y consistente con exp004 (2.5D con contexto axial, sin mejora — pero por una vía de
evidencia totalmente distinta a la de acá).

### D3 — Sonda uno-a-muchos por dataset (`d3_sonda_vecinos.py`) — LA EVIDENCIA MÁS FUERTE

Puramente sobre GT (sin modelo). Para normo (n=384, RapidPlan) e hipo (n=151, limpio de
contaminación nodal, manual) POR SEPARADO: 7 features geométricas recalculadas desde máscaras
NPZ (las mismas del "Baseline ML clásico": VolPTV/Rectum/Bladder_cc, Solap_PTV_Rectum/
Bladder_cc, overlap_rel_recto/vejiga), estandarizadas y usadas para k=3 vecinos más cercanos
por paciente (KNN separado por dataset, nunca mezclado). Para cada paciente + sus 3 vecinos
(grupo de 4), se mide el spread (std) del %V(D) REAL de Rectum en banda media [40,80]%Rx,
promediado sobre los bins de la banda.

| Dataset | spread medio (pp) | n |
|---|---|---|
| normo (RapidPlan) | 3.95 ± 2.03 | 384 |
| hipo (manual) | 6.37 ± 1.98 | 151 |

**Ratio hipo/normo = 1.61, Mann-Whitney p=2.8e-30.** Entre anatomías CASI IDÉNTICAS (vecinos
más cercanos en el espacio de 7 features), el DVH real de recto varía 61% más en hipo que en
normo. Si el límite fuera de anatomía, el spread debería ser parecido en ambos datasets para
anatomías igual de parecidas — no lo es. **Esta es la pieza de evidencia más fuerte y con más
potencia estadística de las 4 (n=535 pacientes combinados, p altísimamente significativo).**

Solapamiento de pacientes entre datasets (mismo AnonID = mismo hash de HC, verificado): 15
pacientes están en ambos `processed/` y `processed_hipo/` (clean). La sonda es intra-dataset
por diseño — el KNN de cada dataset nunca ve al otro, así que ningún paciente cruza vecinos
entre datasets ni se cuenta dos veces dentro de un mismo cálculo (ver nota completa en
`d3_sonda_vecinos.json`).

### D4 — Subset pareado, mismo paciente dos generadores (`d4_subset_pareado.py`)

**⚠️ El conteo real de pacientes en ambos datasets es 15, no ~21** como se estimaba al
plantear la tarea (la estimación original no descontaba a los 28 pacientes con contaminación
nodal excluidos del hipo "clean", ni la posibilidad de NPZ faltante en alguno de los dos
preprocesados).

Para esos 15 pacientes (misma anatomía física, plan RapidPlan-normo Y plan manual-hipo):
comparación de la DISPERSIÓN inter-paciente (std, NO las medias — confundidas por
fraccionamiento 78Gy/39fx normo vs. 70Gy/28fx hipo) del hombro rectal banda media.

| Métrica | std hipo | std normo | ratio hipo/normo | Levene p |
|---|---|---|---|---|
| shoulder banda media (%V, 40-80%Rx) | 7.19 | 4.99 | 1.44 | 0.531 (NS) |
| EQD2 Dmean Rectum (Gy, α/β=3) | 3.29 | 2.91 | 1.13 | 0.901 (NS) |

**Lectura:** misma dirección que D3 (hipo más disperso que normo) pero sin significancia
propia — n=15 tiene poca potencia para un test de Levene. **No contradice a D3, lo corrobora
direccionalmente sin poder confirmarlo de forma independiente.** No se sobre-interpreta este
resultado por sí solo dado el tamaño de muestra.

### D5 — Contexto: banda media en normo (`d5_banda_media_normo.py`)

`dvh_curva_completa.py` reusado sin modificar (monkeypatch de `PROCESSED_DIR`) sobre exp002 +
test59 normo (mismo checkpoint/config de referencia de la serie normofraccionada).

| Estructura | banda media normo | banda media hipo | ratio hipo/normo |
|---|---|---|---|
| Rectum  | 4.07pp | 8.36pp | 2.05 |
| Bladder | 1.36pp | 3.37pp | 2.48 |

**Puramente contextual (confundido por fraccionamiento/constraints distintos, no decisivo por
sí solo):** el mismo enfoque (2D + PSDM + MAE) YA tiene un hombro de banda media en normo —
o sea, no es exclusivo de hipo/manual — pero a la MITAD de magnitud que en hipo. Consistente
con (no prueba por sí solo) la lectura de D3/D4: parte del hombro es arquitectura/approach
(presente en ambos datasets), y una parte adicional sustancial es específica del
dataset/generador hipo.

### Veredicto agregado (`results/diagnostico_piso/summary.json`)

**El hombro medio-bajo de OAR NO es principalmente gap de generalización (D1) ni información
3D que el 2D descarte (D2).** La causa dominante, con la evidencia más fuerte y de mayor
potencia estadística (D3), es un **PISO DE CONSISTENCIA DE GENERACIÓN del ground truth**:
RapidPlan (normo) genera planes reproducibles entre anatomías parecidas; la planificación
manual (hipo) no. D4 corrobora la misma dirección en el subset pareado sin alcanzar
significancia propia (n=15). D5 confirma que el mismo approach ya tiene este hombro en normo,
pero a mitad de magnitud — consistente con una combinación de piso-de-approach (menor,
presente en ambos) + piso-de-generación-hipo (mayor, específico de la planificación manual).

**Implicación práctica para las 3 palancas en juego:**
- **3D U-Net:** NO motivado por este hombro específico (D2). Seguiría sin motivación clara
  salvo que aparezca evidencia distinta en otra vía (ver nota ya existente sobre la clase
  amplia de priors de geometría de entrega, que queda igual de condicionada que antes).
- **Más datos/augmentation (N):** retorno esperado marginal para este hombro (D1) — el modelo
  no está claramente underfitting por falta de datos en esta banda.
- **Replanificación consistente del GT hipo** (que Pablo ya empezó en paralelo, fuera de este
  diagnóstico): es la palanca con más potencial según esta evidencia — ataca directamente la
  causa dominante identificada (D3, la más robusta estadísticamente).

### Archivos generados
- `scripts/diagnostico_piso/{d1_gap_train_test,d2_coherencia_axial,d3_sonda_vecinos,
  d4_subset_pareado,d5_banda_media_normo,resumen_final}.py`.
- `results/diagnostico_piso/summary.json` (veredicto agregado, el archivo a leer primero),
  `d1_gap_train_test.json`, `d2_coherencia_axial.json` (+ `..._por_paciente.csv`),
  `d3_sonda_vecinos.json` (+ `d3_{normo,hipo}_por_paciente.csv`), `d4_subset_pareado.json`
  (+ `..._por_paciente.csv`), `d5_banda_media_normo.json`, `plots/` (z-profiles D2, scatter
  spread D3, boxplot varianza pareada D4), `d1_train_inferencia/` y
  `d5_normo_dvh_curva_completa/` (resultados completos de las corridas de
  `dvh_curva_completa.py` reusadas para D1/D5).

---

## Proyecto 2 — KBP mimicking: v6 en Eclipse, apertura excesiva, freeze de LBFGS SIN RESOLVER (2026-08-15/16)

**Continúa el Proyecto 2 (KBP vía PDRT, sección arriba) después de `mimicking.py` implementado
y de varias corridas de optimización (v1-v6) no documentadas en detalle en este archivo — el
detalle completo de v1-v5 vive en la memoria de Claude Code (`project_mimicking_pipeline_status.md`),
no acá. Esta sección cubre v6 (última corrida real probada en Eclipse) y la investigación de
v7 (falló, sin resolver — hay que retomarla).**

### Estado de v6 — convergió en entrenamiento, PERO el plan real es malo

`run_write_rtplan.py` con `dvh_weight=1.0` (términos D98 PTV "at_least"/V70 Recto/V50 Vejiga
"at_most" contra el DVH del target de U-Net) + `deliv_weight=50.0` + `ptv_conform_start` +
`project_aperture`, 200 pasos, convergió bien en las métricas de entrenamiento (mse y DVH
estabilizados). Pablo lo importó en Eclipse y recalculó con AAA — **el plan real es
notoriamente peor que el clínico**: Dmax 135%, punto caliente >110% en la zona inferoanterior
(13.8 cm³ >110%, centrado ~44/17/8mm del centro del PTV), muy mala conformación de la isodosis
del 100%, y MLC visiblemente abierto por fuera del PTV.

**Confirmado que NO es un gap PDRT-vs-AAA (pregunta de Pablo, respondida):** se recalculó la
dosis de v6 con PDRT fresco (no leyendo el RD viejo, ver landmine de archivos RD mal
etiquetados en `project_pdrt_aaa_gap_exploitation.md`/memoria) y coincide casi exactamente con
AAA — D98 81.4% (PDRT) vs 79.6% (AAA), ratio de dosis integral 0.9898. **La dosis que predice
PDRT para esa apertura TAMBIÉN tiene el punto caliente** — se confirmó visualmente
(`analiza_hotspot_v6.py` + imagen comparativa PDRT/AAA en el corte más caliente): son
prácticamente indistinguibles. No es un artefacto de AAA ni una discrepancia de motores — el
problema es geométrico (la apertura en sí), y ambos motores lo reflejan igual.

### Causa geométrica encontrada: el MLC abre ~5x más área fuera del PTV que el plan clínico

Con un método de proyección BEV independiente (geometría IEC propia, sin reusar código de
`ptv_conforming_init.py` — `commissioning/diagnostics_mimicking_v7/bev_check.py`, convención de
signo validada contra el plan clínico real) se midió, sobre el RP real optimizado:

| Plan | Área de MLC abierta por fuera de la silueta real del PTV |
|---|---|
| Clínico (Eclipse) | 7.0% / 6.7% (arco0/arco1) |
| v6 (RP final, 200 pasos) | **33.8% / 32.9%** |

**¿Es seguro que se debe SOLO a `leaf_margin_mm`/`aperture_extra_mm` (pregunta de Pablo — respuesta: NO, sólo parcialmente):**
- Medido con el INIT conformado (antes de CUALQUIER paso de optimización, `leaf_margin_mm=5`/
  `aperture_extra_mm=10`, los valores viejos): ya abre **25.7%/25.8%** por sí solo — es decir,
  ~76% del gap final de v6 (33.8%) YA EXISTE en el arranque, antes de tocar el optimizador.
  El optimizador sólo suma ~8 puntos más en 200 pasos.
- Se descartaron dos hipótesis sobre CÓMO el optimizador agranda esos 8 puntos (medido
  directo contra el RP real): (a) pares cerrados por el init "descubriendo" el margen extra
  como loophole — descartado, quedaron ≤4.44mm; (b) pares abiertos por el init creciendo
  mucho más allá de su bound — descartado, crecimiento modesto (media 4.8mm) y en promedio
  las aperturas se achicaron, no crecieron.
- Barrido de `leaf_margin_mm` (con el init solo, sin optimizar, `check_new_margins.py`):
  5mm→25.7%, 3mm→21.5%, 2mm→19.3%, 1mm→17.3%, **0mm→15.5%**. Es decir, **incluso con margen
  CERO queda un ~15.5% de área fuera del PTV que NO se explica por estos dos parámetros** —
  un "piso estructural" probablemente de la aproximación rectangular por fila de lámina a una
  silueta curva del PTV, o de algo en el suavizado de velocidad
  (`_rate_limit_sequence`/`ptv_conforming_init.py`) que todavía no se aisló. **No está
  confirmado qué es exactamente ese piso de ~15%** — es la primera cosa a investigar en la
  próxima sesión (pedido de Pablo, punto 1: "revisar código en busca de bugs o señales que no
  estemos perdiendo").

### Intento de fix (2026-08-16) — ROMPIÓ el optimizador, quedó revertido, SIN RESOLVER

Se intentó: (a) bajar `leaf_margin_mm` 5→2mm y `APERTURE_EXTRA_MARGIN_MM` 10→3mm
(`src/planning/mimicking.py`), y (b) corregir el D98 objetivo de `compute_dvh_targets` (crudo
81.5% de Rx/fx por un artefacto de borde de 1.8% del volumen del PTV, ver
`chk_target_dentro_ptv.py` de la sesión anterior — casi seguro un límite de z_range del U-Net,
no cobertura real). Se relanzó v7 (200 pasos, mismo resto de config que v6) y **el optimizador
quedó clavado (LBFGS) desde el paso 2**: `mse` idéntico bit-a-bit por 36+ pasos, PTV
sobredosificado (D98 real ~185% del target, vs el target mismo) y Recto muy por encima del
límite (V70 79% vs 43%) — congelado ahí, sin mejorar ni empeorar. Se cortó el proceso
(~4h de GPU perdidas).

**Investigación de la causa (7 smoke tests de 5 pasos, ~1-2h reales cada uno — este es
exactamente el cuello de botella de tiempo que Pablo pidió atacar, punto 2):**

| # | leaf_margin_mm | aperture_extra_mm | D98 objetivo | `--project-aperture` | Resultado |
|---|---|---|---|---|---|
| 1 | 2 | 3 | 95% (piso fijo) | sí | Congelado |
| 2 | 2 | 8 | 95% (piso fijo) | sí | Congelado (idéntico a #1 — descarta `aperture_extra_mm`) |
| 3 | 2 | 3 | 86.6% (exclusión bordes) | sí | Congelado (descarta que fuera el piso fijo específicamente) |
| 4 | 2 | 3 | 86.6% | **no** | Congelado (descarta `--project-aperture` por completo) |
| 5 | 5 | 3 | 86.6% | sí | Congelado (descarta que alcance con revertir sólo el margen) |
| 6 | **5** | **10** | **81.5% (crudo, = receta EXACTA de v6)** | sí | **✅ Convergió normal** (mse bajando 2.743→2.709→2.687→2.651→2.637) |
| 7 | 2 | 3 | 81.5% (crudo) | sí | **Congelado otra vez** (idéntico a #1-5 en los valores finales) |

**Conclusión honesta: NO está resuelto.** La prueba #6 (receta exacta de v6) prueba que el
código actual SÍ puede reproducir el comportamiento bueno de v6. Pero la prueba #7 (sólo
cambiando `leaf_margin_mm`/`aperture_extra_mm` a los valores nuevos, con el D98 objetivo
IDÉNTICO al de v6) también se congeló — así que subir el D98 objetivo NO es la única causa
como se pensó tras las pruebas #1-4; **apretar `leaf_margin_mm`+`aperture_extra_mm` juntos
TAMBIÉN rompe algo, por un mecanismo todavía no aislado** (no se probó margen=2/extra=10 ni
margen=5/extra=3 con D98 crudo por separado — quedó pendiente). El fix de D98 (ambas
variantes: piso fijo y exclusión de bordes) quedó REVERTIDO en el código
(`compute_dvh_targets` usa el valor crudo, `d98_gy = d98_gy_raw`, con el docstring
documentando todo este historial) — el artefacto de borde del D98 sigue sin corregirse.

### Los 3 próximos pasos que pidió Pablo (sin ejecutar todavía)

1. **Revisar el código de nuevo buscando bugs/señales que se estén perdiendo** — no asumir que
   el problema es sólo de tuning de hiperparámetros. Candidatos a mirar: `project_aperture_bounds`
   (¿el clamp `.copy_()` con `torch.no_grad()` corrompe el estado interno de LBFGS de alguna
   forma sutil, similar al problema ya documentado con `project_leaf_velocity`+LBFGS?),
   `_rate_limit_sequence`/`ptv_conforming_init.py` (¿el piso estructural de ~15% viene de acá?),
   la interacción entre `dvh_Dp_loss`("at_least") y `dvh_Vx_loss`("at_most") cuando ambos están
   muy violados a la vez desde un arranque de MU uniforme masivamente sobredosificado.
2. **Mejorar el cuello de botella de tiempo de PDRT** (aunque el bottleneck esté dentro de
   PyDoseRT mismo, no en nuestro código) — cada paso de LBFGS tarda 340-1800s según VM/estado
   de convergencia interna (ver `project_shared_vm_timing_variability` en memoria), y un smoke
   test de sólo 5 pasos cuesta 1-2 horas reales. Esto hizo que aislar la causa del freeze
   tomara ~12+ horas de GPU en un solo día. Sin resolver esto, cualquier iteración empírica
   futura (barrido de hiperparámetros, debugging) va a seguir siendo extremadamente lenta. Ya
   está anotado como pendiente en `project_iteration_time_pending.md` (memoria) pero SIN
   empezar — Pablo lo pidió explícitamente como prioridad para la próxima sesión, antes de
   seguir con más pruebas de margen.
3. **Pensar nuevas pruebas** una vez resuelto (o al menos entendido) el freeze — candidatas ya
   identificadas pero no ejecutadas: margen=2/extra=10 y margen=5/extra=3 por separado (D98
   crudo) para terminar de aislar qué combinación específica rompe LBFGS; investigar si
   `--optimizer adam` (en vez de LBFGS default) es más robusto a estas proyecciones duras,
   dado el precedente ya documentado de que `project_leaf_velocity` tuvo el mismo tipo de
   incompatibilidad con LBFGS y sólo funcionó bien combinado con Adam.

### Archivos generados / relevantes para retomar

- `src/planning/mimicking.py` — `compute_dvh_targets` (D98 revertido a crudo, docstring con
  todo el historial de los 2 intentos de fix fallidos), `APERTURE_EXTRA_MARGIN_MM=3.0` (dejado
  en el valor nuevo, pero NO USADO en ninguna corrida larga exitosa todavía),
  `leaf_margin_mm` default 2.0 (ídem).
- `commissioning/diagnostics_mimicking_v7/` (copiados del scratchpad de la sesión, que es
  efímero y NO sobrevive entre sesiones): `bev_check.py`/`bev_check_v6.py` (método de
  validación BEV independiente, reusable para cualquier plan), `diag_init_dose2.py` (dosis
  cruda del init sin optimizar, por margen), `diag_freeze.py` (chequeo de
  `deliverability_loss` del init puro, sin optimizar), `check_new_margins.py` (barrido de
  margen), `analiza_hotspot_v6.py`/`valida_v6.py` (comparación PDRT vs AAA decisiva de v6).
- `commissioning/sandbox_results/PT_003fe2bb84986507_mimicking/RP_v6_dvh.dcm` — RP de v6
  (el que se probó en Eclipse, referencia para comparar contra cualquier v7/v8 futuro).
- Memoria de Claude Code (`project_mimicking_pipeline_status.md`) tiene el detalle completo de
  v1-v6 y de esta investigación, con más profundidad que este resumen — leer ahí primero si
  hace falta más contexto de cómo se llegó a `ptv_conform_start`/`deliverability_loss`/etc.

### Recomendación de herramienta para la próxima sesión

Los 3 puntos pedidos por Pablo (revisar código, mejorar el bottleneck de tiempo, diseñar
pruebas) son justo el tipo de tarea que este archivo ya asigna a **Claude Code** en la sección
"Workflow Claude.ai ↔ Claude Code" de arriba: debugging iterativo, profiling/optimización de
performance, y cualquier tarea donde el ciclo editar/correr/ver-resultado sea el cuello de
botella — los tres puntos lo son. Claude.ai no puede leer el código fuente, correr los smoke
tests, ni perfilar tiempos reales de PDRT. Conviene arrancar la próxima conversación en Claude
Code, sobre este mismo repo.

---

## exp_normo_3dunet — U-Net 3D sobre normo, A/B de arquitectura vs exp002 (COMPLETADO, 2026-08-18/19)

Pregunta primaria: ¿el 3D baja el |ΔV| del hombro de OAR (banda media 40-80%Rx) por debajo del
piso-de-vecinos de ~4pp de normo (D3, `results/diagnostico_piso`)? Normo (RapidPlan, GT
consistente) elegido a propósito como sustrato limpio para aislar beneficio de arquitectura del
ruido de generación que domina en hipo (ver diagnóstico D1-D5 arriba).

**⚠️ Nota de discrepancia sin resolver:** la tarea original pedía chequear si el 3D "sube el rho
de rugosidad Z desde el 0.03-0.18 que tenía el 2D". Ese número no existe en el repo — el D2 ya
completado (arriba, hipo test=31) midió rho=0.80-0.99 en las 3 sub-métricas, y concluyó
explícitamente que el 3D NO está motivado por este hombro. No se encontró ningún 0.03-0.18 en
ningún análisis del proyecto (lo más cercano es un rho=0.18-0.20 de un análisis totalmente
distinto, la correlación θ_min/θ_recto del análisis angular). Se procedió igual usando los
números reales de D2 como referencia — si Claude.ai tenía otro número en mente, revisar.

### Bloque 1 — Sondeo de memoria (`scripts/probe_mem_3d.py`, `results/probe_mem_3d.json`)

Barrido de 24 configs (2 volúmenes × 3 base_features × AMP on/off × grad_checkpointing on/off),
cada una en subproceso aislado, `torch.cuda.max_memory_allocated/reserved` + tiempo/paso.

| Volumen | shape(D,H,W) | Resultado |
|---|---|---|
| full (mismo FOV que exp002) | 80×256×256 | Solo entra con **base_features=16** (24/32 → OOM o timeout en cualquier combo AMP/checkpoint) |
| crop (bbox real PTV∪Rectum∪Bladder+2cm, peor caso train) | 80×240×128 | Entra con **base_features hasta 32**, incluso sin AMP ni checkpoint |

El crop se calculó con datos reales (no un valor de paper): bbox por paciente calibrado con el
mismo método de `compute_overlap_real.py` (voxel real vía `vol_ptv_cc`, no spacing nativo crudo
sobre grilla downsampleada), worst-case sobre los 265 de train. Z ya viene recortado en
`preprocess.py` (recorte ROI+margen, 37-77 cortes nativos, 80 tras redondeo a múltiplo de 8).

**Config elegida: volumen full, base_features=16, AMP=True, sin grad_checkpointing** (7.77GB
reservados de 12GB, 0.52s/paso — la combinación más rápida que entra con margen; el
checkpointing no hace falta, solo cuesta tiempo). Se descartó el crop pese a permitir
base_features más alto porque introduciría un segundo confound (FOV distinto) en la pregunta
2D-vs-3D — este experimento prioriza el A/B limpio por sobre exprimir más capacidad.

**⚠️ Incidente de GPU documentado (para no repetir):** en Windows/WDDM, un proceso que excede la
VRAM física NO tira `OutOfMemoryError` limpio — el driver empieza a paginar a RAM del sistema
("shared GPU memory"), lo que colgó la PC entera ~17 minutos durante el sondeo (RAM a 0GB
libres de 31.6GB). Causa raíz: subprocesos huérfanos de un `TaskStop` que no propaga a procesos
nativos de Windows lanzados desde bash. Fix aplicado en `probe_mem_3d.py` Y en `train.py`:
`torch.cuda.set_per_process_memory_fraction(0.88, device=0)` al arrancar — con esto, exceder el
cap tira OOM limpio en vez de pagear. Verificado explícitamente contra el caso que colgó antes
(ahora falla en segundos, no en minutos). Aplicar este mismo cap a cualquier script nuevo que
toque CUDA en este proyecto.

### Bloque 2 — Entrenamiento

`configs/exp_normo_3dunet.yaml`: arch=unet3d, in_channels=5 (CT+BODY+PSDM×3, PSDM ya volumétrico
nativo — verificado, sin cambios necesarios), depth=4, mismo split que exp002
(`splits_v1.json`, test=59), mismos inputs/normalización. `scheduler.max_epochs=200` SIN
reescalar (mismo n_train=265 que exp002 → mismos steps/época, no aplica la regla de reescalado
por tamaño de dataset). `--no-wandb` (CSVLogger local) para evitar riesgo de prompt de auth
interactivo colgando un run desatendido — métricas completas en
`lightning_logs/exp_normo_3dunet/version_0/metrics.csv`.

Corrió las 200 épocas completas (sin activar early_stopping, patience=30) en ~17h15m
(13:22→06:37), compartiendo GPU con `run_mimicking.py` al principio y al final (se esperó a que
terminara antes de arrancar, y de nuevo no se tocó nada durante una corrida manual de
`evaluate.py` de Pablo en paralelo — ambas coexistieron sin problema gracias al cap de memoria).
Ritmo real ~5-7 min/época (más lento que la estimación sintética de 2.3 min/época del sondeo,
por I/O de disco con `cache_train=false` + overhead de Lightning/logging no capturado en el
sondeo). Mejor checkpoint: `checkpoints/exp_normo_3dunet/epoch=182.ckpt` (val/mae=2.129,
val/dvh_score=1.771 — la curva ya estaba prácticamente plana desde la época ~167, diferencias
de milésimas entre los últimos ~30 checkpoints candidatos).

### ⚠️ Bug encontrado y corregido — evaluación 3D con Z nativo sin padding

`analisis_angular.inferir_dosis` (reusada por `dvh_curva_completa.py` y `d2_coherencia_axial.py`)
cargaba el NPZ nativo (Z real del paciente, sin padding) y lo pasaba directo al modelo — correcto
para 2D (cada corte se procesa independiente, Z no afecta el forward) pero **INVÁLIDO para 3D**:
el modelo se entrenó SIEMPRE con Z padeado a un tamaño fijo (`DoseDataModule._pad_or_crop_z`,
80 cortes, padding simétrico con CT=-1/PSDM=1/masks=0). Verificado en un paciente de prueba:
MAE_BODY=7.44 con Z nativo sin padding vs. **1.596 con el padding correcto** (idéntico al
`evaluate.py` oficial, que sí pasa por el DataModule). La primera corrida de
`dvh_curva_completa.py` sobre el 3D dio números disparatados (Rectum banda media 6.48pp, Bladder
banda baja 12.53pp) por este bug — descartados, no reflejan al modelo real.

**Fix:** `inferir_dosis(model, device, arrays, pad_z_to=None)` — nuevo parámetro opcional, pública
`calcular_n_slices_target(processed_dir, splits_path)` que replica el cálculo del DataModule.
`dvh_curva_completa.py` y `d2_coherencia_axial.py` ahora detectan `cfg.model.arch=="unet3d"` y
pasan `pad_z_to` automáticamente; para 2D el comportamiento es idéntico a antes (verificado:
re-corrida de `d5_banda_media_normo.py` reproduce exactamente 4.07pp/1.36pp). **Cualquier script
nuevo que evalúe un modelo 3D fuera del DataModule tiene que pasar `pad_z_to`** — si no, los
números van a estar silenciosamente mal (sin error, sin warning, solo peor de lo real).

### Bloque 3 — Resultados finales (test=59, normo)

| Métrica | exp002 (2D) | exp_normo_3dunet (3D) | Piso-de-vecinos (D3) |
|---|---|---|---|
| **|ΔV| banda media Rectum** ← PRIMARIA | 4.07pp | **4.02pp** | 3.95pp |
| |ΔV| banda media Bladder | 1.36pp | 1.33pp | (no medido en D3) |
| D2 rho (vfrac) Rectum | 0.91 | 0.90 | — |
| D2 rho (vfrac) Bladder | 0.97 | 0.98 | — |
| MAE body | 1.77 | 1.886 | — |
| MAE PTV | 1.41 | 1.381 | — |
| MAE Rectum | 3.75 | 3.957 | — |
| MAE Bladder | 2.16 | 2.411 | — |
| DVH score (OpenKBP) | 1.15 | 1.363 | — |
| Rectum V70 / V65 (acuerdo) | 95% / 100% | 94.9% / 100% | — |
| Bladder V70 / V65 (acuerdo) | 98.3% / 100% | 100% / 100% | — |

(D2 corrido por primera vez sobre normo para ambos modelos — antes solo existía sobre hipo. 2D
normo: `results/diagnostico_piso/d2_normo_exp002/`. 3D: `results/diagnostico_piso/d2_normo_3dunet/`.
DVH-curva-completa: `results/exp_normo_3dunet_dvh_curva_completa/`. Evaluate estándar:
`results/exp_normo_3dunet_test/`.)

### Veredicto

**El 3D NO baja el hombro por debajo del piso — se estanca exactamente en él** (4.02pp vs
3.95pp del piso, contra 4.07pp del 2D — diferencia de 0.05pp, ruido puro). **El rho de
coherencia axial tampoco sube** (0.90 vs 0.91 Rectum, 0.98 vs 0.97 Bladder — sin cambio). En MAE
espacial y DVH score el 3D queda **levemente peor** que el 2D en la mayoría de las estructuras
(PTV es la única que mejora un poco), sin ninguna ganancia que compense el costo (~17h vs. las
horas que toma exp002 en 2D, más el trabajo de infraestructura nueva).

**Conclusión: el piso es real incluso con GT consistente (RapidPlan) y arquitectura que sí ve
la Z completa.** No hay señal anatómica oculta que el 2D estuviera perdiendo — consistente con
el diagnóstico D1-D5 (el hombro es mayormente piso de consistencia de generación del GT, no de
arquitectura ni de información 3D descartada). Esto cierra la pregunta de arquitectura 3D para
esta serie: no hay razón, con esta evidencia, para preferir 3D sobre 2D en el pipeline KBP.
Si se retoma 3D en el futuro, que sea por otra motivación (no este hombro específico, ya
descartado dos veces — hipo y normo).

### Archivos generados
- `src/models/unet3d.py`, `configs/exp_normo_3dunet.yaml`.
- `scripts/probe_mem_3d.py` + `results/probe_mem_3d.json`.
- Fix de dispatch 3D: `src/models/unet2d.py::build_model`, `src/models/lightning_module.py::forward`.
- Fix de memoria (cap WDDM): `scripts/probe_mem_3d.py`, `scripts/train.py`.
- Fix de evaluación 3D (Z padding): `scripts/analisis_angular.py::inferir_dosis` +
  `calcular_n_slices_target` (nueva), `scripts/dvh_curva_completa.py`,
  `scripts/diagnostico_piso/d2_coherencia_axial.py` (también generalizado para aceptar
  checkpoint/config/splits/processed_dir/out_dir como parámetros, antes hardcodeado a hipo).
- `checkpoints/exp_normo_3dunet/epoch=182.ckpt`, `lightning_logs/exp_normo_3dunet/version_0/`.
- `results/exp_normo_3dunet_test/`, `results/exp_normo_3dunet_dvh_curva_completa/`,
  `results/diagnostico_piso/d2_normo_exp002/`, `results/diagnostico_piso/d2_normo_3dunet/`.

---

## Proyecto 1 — herramienta ML clasica de tomografo, Tareas 1-6 COMPLETADAS, Tarea 7 bloqueada (COMPLETADO, 2026-08-19)

Implementacion completa segun `PROMPT_claude_code_proyecto ML Tomo.md`. Bloqueo inicial
resuelto por Pablo: `metricas_planes_hipofx_D95norm.csv` (199 filas, con `Status`) no
existia en ningun lado del disco (la carpeta `dicoms hipofx\` que lo contenia se borro en
la limpieza de espacio ya documentada) — Pablo lo repuso en `repo/data/`. Verificado
contra el prompt: 195 pacientes unicos tras dedup (133 Approved/38 Rejected/24
UnApproved), coincide exacto.

### Tarea 1 (`scripts/prep_data_p1.py`)
Dedup + 1 paciente mas descartado (`PT_f4bd8360f920284c`, Rejected — extraccion de
Rectum/Bladder totalmente vacia, no solo el flag) -> **194 pacientes utilizables**.
Split `data/splits/splits_hipo_v3.json` (135/26/33): val/test SOLO de poblacion natural
(Approved+Rejected, n=170), UnApproved (24) entero a train. Estratificacion 2D de
compliance operativo (recto_op_fail=RV65_fail OR RV55_fail; vejiga_op_fail=BV65_fail),
reparto Hamilton por celda — MISMO metodo que `scripts/make_splits_hipo.py` (no el
vector completo de 3 flags: una celda de esa version tiene solo 2 miembros, insuficiente
para garantizar representacion en las 3 particiones a la vez). Val/test con eventos de
falla de los 3 constraints (val: RV65=9, RV55=6, BV65=7; test: RV65=10, RV55=7, BV65=8).
`data/dataset_p1.csv`: 7 features + fail_{RV65,RV55,BV65} + value_{RV65,RV55,BV65}
(continuo, para Tarea 3) + Status + split.

### Tareas 2-3 (`scripts/train_p1_clf.py`, `scripts/train_p1_reg.py`)
Clasificacion: LogReg (desplegar) vs HistGradientBoosting (techo). AUC test: RV65
0.952/0.961, RV55 0.885/0.896, BV65 1.000/0.970 — logreg y gb dentro de rango esperado
entre si en los 3. Regresion de severidad: **Ridge gano en los 3 constraints** (no hizo
falta caer a HGB) — correlacion cerca-del-umbral en test 0.92 (RV65), 0.83 (RV55), 0.98
(BV65).

### Tarea 4 (`scripts/calibrate_p1.py`) — semaforo por OAR, congelado en val
Score combinado pesimista para Recto (`max(P_fail_RV65, P_fail_RV55)`, label=OR de los 2
flags — mismo criterio que el estrato del split); Vejiga usa BV65 directo. Umbral verde:
Recto sens_val=0.889 (objetivo 0.85), Vejiga sens_val=1.000 (objetivo 0.90, banda alta
por ROC discreto con n chico). Delta naranja/rojo = mediana del margen (valor predicho −
umbral clinico) entre los no-verde de val: Recto=2.62pp, Vejiga=2.10pp. Guardado en
`models/proyecto1/thresholds.json`.

### Tarea 5 (`scripts/eval_p1.py`) — resultado en test, umbrales SIN tocar
**Hallazgo real, no oscurecido:** la sensibilidad de la frontera verde de RECTO cae de
0.889 (val) a **0.500 (test, 6TP/6FN de 12 fallas reales)**, pese a AUC test=0.921. Vejiga
se mantiene fuerte (AUC=1.000, sens=1.000, esp=0.880). Verificado que NO es un bug: el
score combinado de 6 pacientes que fallan en test cae en la banda 0.26-0.65, por debajo
del umbral 0.6652 calibrado en val — es la misma inestabilidad de calibracion con N chico
ya documentada en el proyecto (`results/baseline_clasico/`: "el AUC alto no garantiza
buena matriz de confusion con este tamano de dataset"), ahora del lado de sensibilidad en
vez de especificidad. **No se recalibro contra test** (rompería el protocolo val-frozen);
queda documentado como limitacion real del punto de operacion actual, no como bug a
arreglar. Resultado completo en `results/proyecto1_v3/metrics_summary.json` (AUC por
constraint individual + AUC/sens/esp/matriz de zonas por OAR, bootstrap 1000, seed=42).

### Tarea 6 (`scripts/build_manifest_p1.py`)
Verificado que los 6 joblib (`clf_*`, `reg_*`) + `thresholds.json` cargan sin re-fitear
nada. `models/proyecto1/manifest.json` con orden de features, modelos usados por
constraint, metadata del split y notas de pendientes.

### Tarea 7 (`scripts/extract_features_live.py`, `scripts/infer_tomografo.py`) — BLOQUEADA, scaffold funcional
Codigo completo y probado con features ya conocidas (smoke test: `predecir_paciente()`
reproduce el mismo semaforo que `eval_p1.py` para 5 pacientes de test). Reusa
`cargar_ct`/`cargar_estructuras`/`contornos_a_mascara` de `data/preprocess.py` SIN
downsample (grilla nativa, evita la clase de bug de escala de voxel ya conocida en el
proyecto). **Dos bloqueos reales, no ejecutables desde este chat:**
1. `configs/config_p1.yaml` tiene los nombres de estructura del autocontour en `null`
   (PENDIENTE Pablo) — `extraer_features()` falla explicito si no estan.
2. El test de consistencia (Tarea 7) necesita CT+RS DICOM de una muestra de pacientes ya
   en `data/dataset_p1.csv` — **ninguna carpeta de DICOMs crudos hipo sobrevive en disco**
   (confirmado: `dicoms hipofx\` y la carpeta fallback de `preprocess_hipo.py` ya no
   existen). Hace falta que Pablo aporte una carpeta nueva (aunque sea de unos pocos
   pacientes) para poder cerrar esta tarea.

### Archivos generados
- `data/metricas_planes_hipofx_D95norm.csv` (repuesto por Pablo), `data/dataset_p1.csv`,
  `data/splits/splits_hipo_v3.json`.
- `scripts/prep_data_p1.py`, `scripts/train_p1_clf.py`, `scripts/train_p1_reg.py`,
  `scripts/calibrate_p1.py`, `scripts/eval_p1.py`, `scripts/build_manifest_p1.py`,
  `scripts/extract_features_live.py`, `scripts/infer_tomografo.py`.
- `configs/config_p1.yaml` (estructuras en `null`, pendiente Pablo).
- `models/proyecto1/{clf,reg}_{RV65,RV55,BV65}.joblib`, `thresholds.json`, `manifest.json`.
- `results/proyecto1_v3/metrics_summary.json`.

---

## Proyecto 1 — calibracion por CV + IC del punto de operacion (COMPLETADO, 2026-08-19)

Implementado segun `SPEC_calibracion_cv_p1.md` (spec de Claude.ai en respuesta al hallazgo
de sens=0.50 en test de v3). Diagnostico confirmado: el umbral de v3 se calibro sobre val
(n_pos=9 en Recto), no el modelo — el AUC alto (0.921) ya indicaba buen ordenamiento.

**Scripts NUEVOS, v3 queda intacto como referencia** (`calibrate_p1.py`/`eval_p1.py`/
`thresholds.json`/`results/proyecto1_v3/` sin tocar):
- `scripts/calibrate_p1_cv.py` -> `models/proyecto1/thresholds_cv.json`.
- `scripts/eval_p1_cv.py` -> `results/proyecto1_v3_cv/metrics_summary.json`.

**Cambios implementados (los 4 de la spec):**
1. Pool de calibracion = natural (Approved+Rejected) de train+val agrupados (n=137),
   UnApproved excluido del calculo del umbral (sigue en train para fitear el modelo
   desplegado, sin cambios ahi).
2. Umbral = mediana de 5-fold StratifiedKFold (refit de LogReg/Ridge SOLO para estimar
   el umbral, nunca el modelo final).
3. Dos variantes de la frontera verde calibradas y reportadas lado a lado, SIN elegir
   ganadora mirando test: A=probabilidad del clasificador (status quo), B=margen de la
   regresion de severidad. `variante_activa="A"` marcada en el manifest — decision de
   cambiar a B pendiente de Pablo/Claude.ai.
4. Objetivo de sensibilidad de RECTO subido a 0.95 (antes 0.85), Vejiga se mantiene en
   0.90 — eleccion deliberada por costo asimetrico de FN de recto.

**Resultado — el fix funciono, sensibilidad de Recto en test recuperada:**

| | v3 (umbral sobre val, n_pos=9) | v3_cv variante A | v3_cv variante B |
|---|---|---|---|
| Umbral (escala prob/margen) | 0.6652 | 0.3003 (mediana CV, IQR [0.264,0.390]) | −1.33 (mediana CV, IQR [−1.56,−0.45]) |
| Sens test Recto | **0.500** (6/12) | **0.833** (10/12) | **0.833** (10/12) |
| Esp test Recto | 0.905 | 0.810 | 0.714 |
| AUC test Recto | 0.921 | 0.921 | 0.909 |

Vejiga ya era fuerte en v3 y mejora un poco mas: sens=1.000 en las 3 variantes, esp sube
de 0.880 (v3) a 0.920 (v3_cv, ambas variantes).

**Bootstrap del umbral (Cambio 3, diagnostico honesto):** IC95 del umbral de Recto
variante A = [0.175, 0.431] — un rango amplio que confirma que el 0.6652 de v3 (fuera de
este rango) nunca tuvo la precision que aparentaba con n_pos=9. Vejiga: IC95=[0.203,0.695].

**Variante A vs B — sin ganador claro en Recto (reportado, no decidido aca):** A tiene
mejor especificidad que B a igual sensibilidad (0.810 vs 0.714) en este test — contrario
a la hipotesis de la spec de que B seria mas estable para recto. En Vejiga, A y B dan
resultados practicamente identicos. Queda para Claude.ai decidir si vale la pena cambiar
a B con esta evidencia (no parece justificado por ahora).

**Delta (naranja/rojo) recalibrado sobre el pool ampliado** (antes solo val) — extension
no pedida explicitamente por la spec pero consistente con reducir varianza de N chico:
Recto Delta≈1.8-1.9pp, Vejiga Delta≈1.7-1.9pp segun variante.

### Archivos generados
- `scripts/calibrate_p1_cv.py`, `scripts/eval_p1_cv.py`.
- `models/proyecto1/thresholds_cv.json`.
- `results/proyecto1_v3_cv/metrics_summary.json`.

---

## Complejidad VMAT desde el RP real, como arbitro generacion-vs-anatomia (2026-08-19)

Tarea de Pablo en dos partes sobre el hombro medio-bajo de OAR (D3, ver
`results/diagnostico_piso/summary.json`): (A) extraer 3 metricas de complejidad VMAT del
RTPLAN (RP) DICOM real; (B) usarlas como arbitro de si el spread de DVH entre vecinos
anatomicos (D3) es de GENERACION (agresividad de plan) o de ANATOMIA no capturada por las
7 features geometricas.

**⚠️ ALCANCE — SOLO normofx.** Al arrancar la tarea se detecto que solo habia RP DICOM
reales para 7-8 pacientes en disco (`dicom_pilot/`, del piloto de comisionamiento —
resultaron ser todos pacientes normo, ninguno hipo), muy lejos de los n=384 normo/n=151
hipo que D3 usa. Pablo indico explicitamente: la tarea se hace SOLO con normofx. Fuente
real (386 pacientes, CT/RS/RP/RD reales de Eclipse, AnonID=nombre de carpeta, coincide
1:1 con `splits_v1.json`): `\\10.100.0.252\centro_de_datos2018\101_Cosas de\PABLO\CNN
Prostata\dicoms normofx`. Los RP de hipofx NO estan disponibles localmente -> Validacion
1 (pareado manual-hipo vs RapidPlan-normo, mismo paciente) y Validacion 2 (terciles de
complejidad dentro de hipo) de la tarea original quedan **bloqueadas**, no ejecutadas.

**Formulas — de papers reales, no reconstruidas de memoria.** Pablo proveyo los papers
(`papers/Metricas/`: Masi et al. 2013 Med Phys, McNiven et al. 2010 citado en el Appendix
de Masi, Nguyen & Chan 2020, Park et al. 2014) despues de que se detectara que
`VMAT1.md` no existia localmente. MCSv (LSV Eq. A2, AAV Eq. A3, agregacion Eq. A4 del
Appendix de Masi) implementado directo de esas ecuaciones — con una correccion respecto
de la unica implementacion open-source encontrada (`pymedphys._experimental.
plancomplexity`): esa libreria calcula el rango posmax de LSV sobre las 60 leaves
completas en vez de solo las N activas dentro del jaw Y (Eq. A1 de Masi es explicita:
"posmax(CP) = ⟨max(pos_n∈N) − min(pos_n∈N)⟩" sobre N = leaves activas) — se implemento
la version correcta, restringida a N. SAS10 (Crowe et al. 2014 / Younge et al. 2016, via
Nguyen&Chan 2020 Sec 2.B.5) y MU Factor (mismo Sec 2.B.1, MU_por_fraccion/dosis_por_
fraccion_cGy, algebraicamente igual a MU_curso/dosis_curso) confirmados por texto directo
de esos papers, no solo por busqueda web.

**Millennium 120 — NO hardcodeado.** Los leaf-boundaries se LEEN de cada RP
(`BeamLimitingDeviceSequence[MLCX].LeafPositionBoundaries`) y se VALIDAN contra el patron
esperado (40 pares centrales de 5mm + 20 externos de 10mm, -200..+200mm) — confirmado
real en el DICOM de `PT_003fe2bb84986507`. Jaws (ASYMX/ASYMY) tienen carry-forward
implementado (Eclipse solo los redefine en CP0 de cada arco; MLCX se redefine en cada
CP). Beams de SETUP (`TreatmentDeliveryType="SETUP"`, ej. campos ANT/LAT DER de imagen
portal) se excluyen del conteo de arcos y de la suma de MU.

### Parte A — `scripts/extraer_complejidad_rp.py`

**386/386 RP parseados OK, 0 fallidos**, todos con MCSv y SAS10 dentro de [0,1] (chequeo
de rango pasado sin necesidad de clipear nada). Valores en linea con literatura publicada
(Masi: MCSv medio 0.41, rango 0.19-0.65; Nguyen&Chan: MU Factor mediana 2.76-3.63,
SAS10 mediana ~0.2-0.3):

| Metrica | mean | std | min | max |
|---|---|---|---|---|
| MU_factor (MU/cGy) | 3.42 | 0.43 | 2.12 | 6.36 |
| MCSv | 0.300 | 0.039 | 0.173 | 0.484 |
| SAS10 | 0.248 | 0.071 | 0.042 | 0.479 |

**NArcos recuperado (pendiente historico, ver H1/H3 arc-prior mas arriba en este
documento): 310 pacientes con 2 arcos, 73 con 3 arcos, 2 con 1 arco, 1 con 4 arcos** —
contradice lo que se penso al mirar solo el piloto de comisionamiento ("los 7 pacientes
de dicom_pilot terminaron siendo todos de 2 arcos, ninguno de 3") -- eso era un artefacto
de esa muestra chica (n=7), NO representativo del cohorte real (~19% de los planes normo
son de 3 arcos). Si se retoma el arc-prior/analisis angular, la estratificacion 2 vs 3
arcos ahora SI se puede hacer para normo (queda pendiente para hipo, sin RP).

Salida: `results/complejidad_arbitro/data/complejidad_rp.csv` (AnonID, dataset, MU_factor,
MCSv, SAS10, NArcos, MU_total, dosis_por_fraccion_cgy, n_fracciones_dicom,
en_splits_v1) + `extraccion_rp_summary.json`.

### Parte B — `scripts/arbitro_complejidad.py`

Metodologia de complejidad RESIDUAL (no cruda): cada metrica se regresiona (Linear
Regression) contra las 7 features geometricas de D3 -> el residuo es la parte de
complejidad NO explicada por la anatomia ya matcheada en D3. Reusa el KNN de D3
(k=3, mismo espacio de 7 features, mismo codigo `d3_sonda_vecinos.cargar_dataset`) para
tener los MISMOS grupos de vecinos anatomicos — 384/384 pacientes del cohorte D3 normo
tenian RP parseado (interseccion perfecta), dando 808 pares (paciente, vecino) unicos
evaluables.

**Resultado — arbitro esencialmente NULO dentro de normo:**

| Metrica | R2(metrica~anatomia) | spearman(Δresidual, ΔDVH) | spearman(Δcruda, ΔDVH) |
|---|---|---|---|
| MU_factor | 0.112 | rho=0.031, p=0.38 | rho=0.032, p=0.36 |
| MCSv | 0.208 | rho=0.030, p=0.39 | rho=0.065, p=0.065 |
| SAS10 | 0.272 | rho=0.078, **p=0.027** | rho=0.062, p=0.077 |

Solo SAS10-residual llega a p<0.05, pero con rho=0.078 (n=808 pares) es un efecto
minusculo, estadisticamente detectable solo por el n grande, no un tamaño de efecto
relevante en la practica. Las otras dos metricas ni siquiera cruzan p<0.05.

**Interpretacion (⚠️ NO responde la pregunta de fondo de la tarea, ver alcance):** dentro
de normo, la complejidad de plan (residual o cruda) practicamente no explica el poco
spread de DVH que queda entre vecinos anatomicos — consistente con D3 (RapidPlan ya es
un generador homogeneo, spread chico, y lo poco que queda no se explica por agresividad
de plan tampoco). Esto NO prueba ni descarta la hipotesis central de D3/D4 (que el spread
GRANDE de hipo es de generacion) — esa prueba directa necesita RP de hipofx, que no
estan disponibles. Lo que si aporta: dentro del cohorte donde se pudo medir, no hay
evidencia de que "mas agresividad de plan matematicamente medida" sea la palanca — el
mecanismo de D3 para hipo probablemente pasa mas por heterogeneidad de CRITERIO/objetivo
clinico entre planificadores humanos que por una diferencia de modulacion pura del MLC
detectable con estas 3 metricas (hipotesis a confirmar el dia que haya RP de hipofx).

Salida: `results/complejidad_arbitro/summary.json`,
`results/complejidad_arbitro/plots/arbitro_scatter_residual_vs_dvh.png`.

**Pendiente si se consiguen RP de hipofx:** correr Validacion 1 (Levene de varianza de
complejidad, pareado mismo-paciente manual-hipo vs RapidPlan-normo) y Validacion 2
(terciles de complejidad dentro de hipo vs spread de DVH) tal como especifico la tarea
original — son las pruebas que SI atacan directo la pregunta de fondo.

---

## Proyecto 1 — reentrenamiento con 6 casos UnApproved nuevos (COMPLETADO, 2026-08-20)

Pablo agrego 6 pacientes UnApproved nuevos (normo replanificados como hipo, sinteticos)
a `metricas_planes_hipofx_D95norm.csv` (199->205 filas crudas). Verificado ANTES de
correr nada: los 6 AnonID (`PT_07d90a40bf044570`, `PT_3ed3e17c3ed80180`,
`PT_433758aa83f996eb`, `PT_4ce363c7537c9c61`, `PT_6707459d5a5b0a83`,
`PT_fe4f867670743b59`) no colisionan con ninguno de los 195 AnonID unicos previos
(reconstruidos desde `dataset_p1.csv` + el paciente descartado por datos faltantes);
ningun paciente viejo desaparecio; sin nulos en las 6 filas nuevas.

Recorrida la cadena completa (Tareas 1-6 + CV) sin cambios de codigo — el diseno ya
garantizaba por construccion los 2 requisitos de Pablo:
1. **UnApproved solo a train:** `prep_data_p1.py` ya arma val/test SOLO de poblacion
   natural: al ser exclusivamente sintetico el aporte, val/test quedaron **identicos**
   (n=26/33, mismos AnonID) — solo train crecio 135->141 (24->30 UnApproved).
2. **UnApproved fuera del pool de calibracion:** `calibrate_p1_cv.py` ya filtra
   `Status != "UnApproved"` al armar el pool — confirmado que `pool_calibracion.n=137`
   se mantuvo exactamente igual antes/despues.

**Resultado — estable, sin sorpresas:** clasificacion/regresion practicamente sin
cambio (AUC test RV65=0.952, RV55=0.879, BV65=1.000, igual dentro de ruido). v3_cv
en test: Recto variante A sens=0.833/esp=0.810 (identico a antes), variante B
sens **subio** de 0.833 a 0.917 (esp bajo a 0.714) — pura variacion del refit del
modelo desplegado con 6 filas sinteticas mas, dentro de lo esperable, no un cambio de
conclusion. Vejiga sin cambios (sens=1.000 ambas variantes).

Todos los artefactos (`data/dataset_p1.csv`, `data/splits/splits_hipo_v3.json`,
`models/proyecto1/*.joblib`, `thresholds.json`, `thresholds_cv.json`, `manifest.json`,
`results/proyecto1_v3/`, `results/proyecto1_v3_cv/`) sobreescritos con los datos
nuevos — no se generaron archivos aparte porque es la misma serie, solo mas datos de
train.

---

## exp_hipo_003_finetune_v3 — U-Net sobre dataset hipo AMPLIADO (Rejected+UnApproved), overnight (COMPLETADO, 2026-08-20)

**No confundir con `data/splits/splits_hipo_v3.json`** (Proyecto 1, ML clasico, seccion
anterior) — este experimento usa `data/splits/splits_hipo_v3_unet.json`, split distinto,
misma fuente de datos.

### Dataset
CSV `data/metricas_planes_hipofx_D95norm.csv` ampliado a 201 pacientes unicos (205 filas
crudas, 4 duplicados de AnonID resueltos: 2 exactos, y 2 con 2 planes RP/RD reales en la
misma carpeta DICOM — `PT_70368fdeed2777a4`, `PT_8b1aa3d35e1b468b` — resueltos comparando
`meta['s_D95']` del NPZ ya existente contra `FactorNorm_D95` de cada fila candidata).
Preprocesados 51 pacientes nuevos sobre `processed_hipo/` (150 ya existian) desde
`20260819_2248_ptta_hipo_todos_sin_LN` — 0 errores de extraccion de PTV, **1 exclusion real**:
`PT_f4bd8360f920284c` (Rejected) tiene Rectum/Bladder ausentes tanto en el CSV como en el
RS (mascaras vacias en el NPZ) — excluido del split, pendiente que Pablo revise si se puede
re-extraer. QC en muestra de 9 pacientes (3 por Status) OK; PSDM excede [-1,1] en la cola
positiva (normalizacion ÷15cm sin clip, esperado, igual que en normo).

### Split — `splits_hipo_v3_unet.json` (200 pac., 130/30/40)
Estratificacion 2D: cumple/no-cumple (AND estricto RV65∧RV55∧BV65, flags D95norm) x tercil
de `Solap_PTV_Rectum_cc` (**confirmado por Pablo: el bug de `CalcularSolapamiento` ya esta
corregido en el C#, esta columna es overlap real** — no recalcular desde mascaras esta vez).
No hicieron falta swaps train↔val: val salio con 15/30 no-cumplidores de entrada. Subgrupo
UnApproved (normos replanificados) quedo repartido train=17/val=5/test=8.

### Config — `configs/exp_hipo_003_finetune_v3.yaml`
Igual a exp_hipo_002/002b (init completo desde `exp002_unet2d_psdm/epoch=191.ckpt`, LR=1e-5,
full finetune) salvo el horizonte: a diferencia de 002/002b (que dejaban max_epochs=100 FIJO
sin escalar, ver razonamiento en esa seccion), esta vez el pedido explicito fue escalar
tambien el finetune. factor=265/130=2.0385 aplicado sobre la base de 002/002b (100/30) →
max_epochs=204, patience=61. Early stopping activo en epoch ~101, mejor checkpoint en
epoch=040 (val/mae=1.811).

### ⚠️ Bug encontrado y corregido — OOM en el paciente con mas cortes (gradient checkpointing)
El primer intento crasheo con `CUDA OutOfMemoryError` de forma reproducible (mismo punto
exacto en 3 intentos, incluso subiendo el cap de VRAM y con `expandable_segments`) al
procesar el paciente de train con mas cortes (Z=78→80 padded) — el mismo padding-target que
ya usaba `exp_hipo_002b_finetune_clean` sin problema, asi que NO es un problema de tamano de
dataset/split, probablemente deriva de un torch/cuDNN mas nuevo que aumento el overhead de
memoria por-op desde julio (no confirmado, no vale la pena perseguirlo mas). Fix real:
gradient checkpointing en el forward del U-Net (`src/models/lightning_module.py`,
`DosePredictionModule.forward`, solo si `self.training`) — resultado numerico identico,
soluciono el OOM sin tocar batch_size/base_features/el cap de VRAM (que se dejo en 0.88,
default; se agrego override opcional via env var `PROSTATE_CUDA_MEM_FRACTION` en `train.py`
por si hace falta en el futuro, pero NO se uso en la corrida final). Ver memoria
`feedback_gradient_checkpointing_bigZ_oom` para el detalle completo del diagnostico.

### Resultados — test=40 (split v3_unet, umbral val-frozen)

| Métrica | Valor |
|---|---|
| MAE body [IC95] | 2.46 [1.99–3.22] |
| MAE PTV [IC95] | 1.79 [1.51–2.17] |
| MAE rectum [IC95] | 5.62 [4.78–6.65] |
| MAE bladder [IC95] | 3.60 [2.89–4.68] |
| RV65 — AUC [IC95] | 0.882 [0.760–0.969] |
| RV55 — AUC [IC95] | 0.809 [0.658–0.936] |
| BV65 — AUC [IC95] | 0.973 [0.892–1.0] |
| RV65 — Sens/Esp (umbral val-frozen) | 0.714 / 0.846 |
| RV55 — Sens/Esp (umbral val-frozen) | 0.692 / 0.630 |
| BV65 — Sens/Esp (umbral val-frozen) | 1.000 / 0.607 |

**Subgrupo UnApproved (normos replanificados como hipo, n=8 en test)** — generaliza bien,
AUC igual o mejor que el agregado: RV65=0.933, RV55=0.875, BV65=1.000. MAE body=1.92,
rectum=4.24, bladder=3.38. Ver `subgroup_unapproved_metrics.json`.

**1 caso arc-limited:** `PT_fe24a2bdb79bd38a` (TreatmentApproved, NArcos=3, mae_body=16.14,
mae_bladder=19.22) — geometria/CSV normales (VolPTV=271cc, s_D95=1.01, cumple todo), sin
señal de bug de extraccion. Encaja con el patron viejo de "techo de arcos" (modelo sin info
de modulacion angular real) MÁS QUE con contaminacion nodal (este dataset ya es "sin_LN").
Pendiente de que Claude.ai decida si vale la pena revisar mas a fondo o dejarlo como
outlier conocido de la serie.

### Archivos generados
`data/splits/splits_hipo_v3_unet.json`, `data/gt_dvh_hipo_256_v3.csv`,
`configs/exp_hipo_003_finetune_v3.yaml`, `scripts/make_splits_hipo_v3_unet.py`,
`scripts/qc_audit_hipo_v3.py`, `scripts/handoff_extras_hipo_v3.py`,
`scripts/error_analysis_hipo_v3.py`,
`results/exp_hipo_003_finetune_v3_test_hipo_v3/` (`test_metrics.csv`, `summary.json`,
`error_analysis.json`, `subgroup_unapproved_metrics.json`, `worst_cases.csv`,
`best_cases.csv`, `plots/`).

---

## ⚠️⚠️ BUG CRÍTICO DE CT + corrección de FOV — re-fundación del pipeline (2026-08, RESUELTO)

**El más grave del proyecto.** `cargar_ct()` en `data/preprocess.py` cargaba la serie RD
(dosis) en vez del CT real, en TODOS los pacientes (normo + hipo). El "canal CT" de todos los
NPZ generados hasta esta fecha era dosis remuestreada, NO anatomía.

### Diagnóstico (confirmado, 4 verificaciones)
- **Causa raíz:** `GetGDCMSeriesFileNames(carpeta)` sin filtro de modalidad. Cada carpeta de
  paciente tiene CT+RD+RS+RP juntos; GDCM devolvía la serie del RD (1 archivo) en vez del CT
  (132-153 archivos). El viejo parche del "CT lee 4D" era en realidad el RD mal leído — nunca
  fue la CT, y se parcheó el síntoma sin ver la causa.
- **V1:** la serie que cargaba GDCM era Modality=RTDOSE, confirmado por pydicom en 4 pacientes.
- **V2:** el spacing del RD 4D (2.5,2.5,3.0) coincide EXACTO con `meta.spacing_mm` de todos los
  NPZ → prueba definitiva de que el "CT" guardado era el RD.
- **V3:** universal — 613/614 pacientes (99.8%) con >90% de vóxeles CT saturados dentro de BODY.
- **V4:** exp002 le daba norma 1.04 al canal CT (comparable a los otros) → el modelo "intentaba"
  usar un canal que era basura casi constante.

### Qué NO se invalida
Las conclusiones RELATIVAS entre experimentos siguen válidas: TODOS compartían el mismo CT
corrupto, así que las comparaciones (PSDM>masks, finetune>scratch, 2.5D sin mejora, MomentLoss
sin mejora) fueron justas. Lo que cambia es la interpretación: la señal venía del PSDM
(geometría de estructuras), no del CT.

### Fix aplicado
`cargar_ct()` reescrito: selecciona la serie con Modality=='CT' vía `GetGDCMSeriesIDs` +
verificación por pydicom; elimina el parche 4D; agrega aserciones fail-fast (dimensión 3D,
rango HU plausible min<-500/max>200). `preprocess_hipo.py` importa `cargar_ct` → el fix aplica
a ambos pipelines.

### Efecto colateral — cambio de FOV (resuelto con recorte)
El CT real es camilla-a-camilla (FOV ~500-600mm) vs el RD que casualmente estaba recortado
cerca del PTV. Sin recorte axial, la anatomía tratada quedaba en ~20-30% del frame 256×256
(perdía 2-3x resolución sobre OARs). Se decidió **recorte en plano de caja cuadrada fija
centrada en centroide PTV**. Mediciones (44 pacientes) descartaron el offset anterior (el
cuello de botella es ancho de caderas lateral, no asimetría AP).

### Comparación de FOV (exp002 con CT real, test=59)
| Métrica | FOV 50cm (1.95mm/px) | FOV 34cm (1.33mm/px) |
|---|---|---|
| MAE rectum | 3.95 | **3.63** |
| rectum_D15 bias | -3.96 | **-2.42** |
| rectum_D10 bias | -3.09 | **-1.77** |
| PTV D95/D98/D99 MAE | 1.29/1.88/2.24 | **0.87/0.99/1.13** |
| PTV D2 (hotspot) MAE | **0.64** | 1.19 |
| rectum_V70 agreement | **100%** | 94.9% |
| DVH score | 1.39 | 1.41 (neutro) |

### Decisiones finales (NO reabrir)
1. **FOV definitivo: caja 34cm, cuadrada, centrada en centroide PTV, offset 0, 256×256
   (1.328 mm/px).** 34cm mejora predicción de OARs y cobertura de PTV (lo que importa
   clínicamente) a costa del hotspot de PTV (D2, que no es constraint). DVH score neutro entre
   34 y 50; se elige 34 por resolución sobre OARs.
2. **El CT real NO mejora la performance** — confirma que el PSDM ya capturaba toda la señal
   geométrica útil. En pelvis (tejido homogéneo, sin heterogeneidades tipo pulmón) la densidad
   del CT aporta poco; la geometría de estructuras (PSDM) domina.
3. **El CT real SÍ acelera la convergencia ~1.8x** (val/loss llega al piso en ~15k steps vs
   ~27k con CT corrupto). Beneficio secundario: horizontes de entrenamiento más cortos posibles.
4. **Nueva línea base normo: exp002_ctfix_fov34.** Los runs viejos (CT corrupto) quedan como
   referencia histórica, NO como baseline de comparación directa.

### Implicancia — exp004 (2.5D) NO se reentrenó
Si el CT real no aporta señal ni en 2D, es muy improbable que el contexto axial de un canal sin
señal aporte en 2.5D. exp004-ctfix queda como experimento futuro solo si un revisor lo pide.

---

## PRÓXIMA ETAPA — Hipofraccionado sobre pipeline corregido (re-arranque)

Ver `HIPOFX_KICKOFF.md` para el detalle completo. Todo el trabajo hipo previo
(exp_hipo_001/002/003, ML clásico, splits) se hizo con CT corrupto y FOV sin recortar → se
rehace sobre el pipeline corregido. Paquete de trabajo:

1. Re-extraer/re-preprocesar dataset hipo desde DICOM (cargar_ct corregido + FOV 34cm) →
   carpeta nueva `processed_hipo_ctfix34/`. Descartar NPZ hipo viejos.
2. Rehacer split estratificado (cumple/no-cumple + overlap real del C# corregido).
3. **Rehacer ML clásico sobre el split nuevo** — OBLIGATORIO: (a) mismo split que la U-Net para
   comparabilidad; (b) las features de overlap cambiaron con el C# corregido, y el overlap era
   la feature dominante. Es barato (sin GPU).
4. Fine-tuning U-Net desde **exp002_ctfix_fov34** (NO desde el exp002 viejo de CT corrupto).
5. Evaluar: métricas completas + U-Net vs ML clásico (mismo split) + desglose por status.

Decisiones abiertas heredadas (normalización D98 vs D95 en planes de mala cobertura, balance de
clases, casos subóptimos zona gris) — ver HIPOFX_KICKOFF.md, resolver con el CSV re-extraído.

---

## Hipo sobre pipeline corregido — pasos 1 y 2 completados (2026-08-25)

### Paso 1 (re-preprocesado) — ya estaba hecho, no documentado hasta ahora
`processed_hipo_ctfix/` generado 2026-08-23 desde 230 candidatos
(`data/splits/splits_hipo_ctfix_all230.json` — placeholder de preprocesado, todo en
`train`, NO es el split real). 29 pacientes "no encontrado" en el DICOM root (fuera de
la cohorte hipo real) + 2 fallos transitorios por `Unable to allocate ... MiB`
(`PT_48486760d06b6f9d`, `PT_4ce363c7537c9c61` — falta de RAM momentánea, no problema de
datos, ver `feedback_shared_vm_timing_variability`) reprocesados con
`--only <id1> <id2>` → OK. Total 201 NPZ únicos disponibles.

### D1/D2/D3 (kickoff) resueltos con `metricas_planes_hipofx_D95norm.csv` (205 filas,
no afectado por el bug de CT — es DVH calculado por el C#)
- **D2 (balance):** 49.5% no-cumplidores (101/204, AND estricto RV65∧RV55∧BV65
  D95norm) — muy distinto del ~12% del dataset viejo. Muy desparejo por Status:
  Rejected 94.7% no cumple, UnApproved 75.8%, TreatmentApproved 31.6%.
- **D1 (normalización):** factor D95 acotado incluso en planes mal cubiertos (máx 1.59,
  3 pacientes >1.2); factor D98 se dispara en planes sub-cubiertos (máx **4.64**, 7
  pacientes >1.2) porque D98 es una cola sensible al ruido de cobertura. Confirma que
  normalizar por D95 (ya lo que hace el pipeline "D95norm") es correcto — D98 borraría
  la señal de incumplimiento.
- **D3:** no analizado en detalle esta sesión (pendiente si hace falta para el análisis
  de casos zona gris).

### Exclusión manual — outlier `PT_5b54e7add30325f0`
TreatmentApproved con `s_D95=1.8626` (máximo de toda la cohorte), D95 crudo=37.58 Gy y
`dose_max=196.64 Gy` — no es cobertura real mala sino probable artefacto de extracción
(PTV_High mal asignado o plan parcial). **Decisión de Pablo: excluir del dataset**, no
investigar más por ahora.

### Exclusión manual — `PT_a0f9d9d98bbb8c81` (⚠️ verificar)
HIPOFX_KICKOFF.md lo documenta como plan normo colado (39fx/78Gy) a descartar, pero el
CSV actual muestra 28fx/70Gy para este AnonID — posible que ya haya sido corregido/
re-extraído desde que se escribió el kickoff, no verificado esta sesión. Se excluyó por
precaución siguiendo la decisión escrita. **Pendiente: confirmar con Pablo si ya se
puede reincorporar.**

### Paso 2 (split estratificado) — `splits_hipo_ctfix_v4.json` (198 pac., 128/30/40)
Generado con `scripts/make_splits_hipo_ctfix_v4.py` (mismo método 2D que
`make_splits_hipo_v3_unet.py`: cumple/no-cumple x tercil `Solap_PTV_Rectum_cc`).
201 NPZ − 1 excluido por falla de extracción (`PT_f4bd8360f920284c`, Rectum/Bladder
ausentes en RS, ya conocido) − 2 exclusiones manuales de arriba = 198. No hicieron
falta swaps train↔val (val salió con 15/30 no-cumplidores de entrada, sobre el mínimo
de 5). Terciles overlap: t1=6.45cc, t2=10.65cc.

| Split | n | no_cumple | TreatmentApproved | Rejected | UnApproved |
|---|---|---|---|---|---|
| train | 128 | 64 | 82 | 27 | 19 |
| val | 30 | 15 | 21 | 4 | 5 |
| test | 40 | 20 | 29 | 5 | 6 |

### ⚠️ Bug encontrado en el propio paso 1 — FOV real era 50cm, no 34cm
El preprocesado del 2026-08-23 (arriba) en realidad corrió con `INPLANE_CROP_MM=500.0`
(confirmado en el log: "recorte de 50cm"; en meta NPZ: `crop_lado_mm: 500.0`). La
constante en `data/preprocess.py` nunca se actualizó a 340 tras la decisión "NO
reabrir" — normo pasaba `--crop-mm 340` explícito en su propio CLI, pero
`preprocess_hipo.py` importaba la constante compartida directo, sin exponer override.
**Fix:** se agregó `--crop-mm` a `preprocess_hipo.py` (default 340.0, desacoplado del
default de `preprocess.py`/normo). NPZ de 500mm respaldados en
`processed_hipo_ctfix_OLD_500mm_bak/` (no borrados). Re-corrido completo con
`--crop-mm 340 --workers 3` (bajado de 6 por poca RAM libre, ver
[[project-shared-vm-timing-variability]]) — 201 OK, mismos 29 "no encontrado", 0 OOM,
0 errores de clipping duro de OAR/PTV (200/201 con BODY clipeado lateral, esperado a
34cm). Split reconstruido: mismo resultado exacto (mismos 201 candidatos, mismas
exclusiones) — `splits_hipo_ctfix_v4.json` no cambió.

### Paso 3 (ML clásico) — hecho, ver [[project-hipo-ctfix-v4-split]] para detalle
AUC test: RV65=0.968, RV55=0.853, BV65=0.969 (mejoró mucho vs la corrida vieja,
consistente con el dataset mucho más balanceado). Evaluado sobre el test COMPLETO del
split nuevo (mezcla de Status), no filtrado a "población natural" como hacía el
protocolo viejo — decisión deliberada para que la comparación con la U-Net sea sobre
el mismo test set exacto.

### Paso 4 (U-Net finetune) — exp_hipo_004_finetune_ctfix_v4, LANZADO 2026-08-25
Init desde `checkpoints/exp002_unet2d_psdm_ctfix_fov34/epoch=127.ckpt` (mejor checkpoint
por val/mae=1.9042, confirmado via bookkeeping de ModelCheckpoint — NO el
exp002_unet2d_psdm viejo de CT corrupto). LR=1e-5, full finetune. Horizonte escalado
con la misma convención que exp_hipo_003 (base finetune 100/30 epochs, no el
max_epochs=200 propio de exp002_ctfix_fov34): factor = n_train_normo (265,
splits_v1.json) / n_train_hipo (128, splits_hipo_ctfix_v4.json) = 2.0703 →
max_epochs=207, patience=62. Config: `configs/exp_hipo_004_finetune_ctfix_v4.yaml`.
Corriendo en background con el `.venv` del repo (torch 2.12.1+cu130 — el miniconda
`base` sólo tiene torch CPU, y `cnn_prostata` no tiene pytorch_lightning instalado;
usar `.venv/Scripts/python.exe` para cualquier entrenamiento/eval de U-Net de acá en
adelante). GPU estaba libre (0% util) antes de lanzar, RAM libre ~6.5GB (VM compartida
consume el resto, normal).

### Paso 5 (evaluación completa) — TERMINADO, ver `RESULTADO_hipo_ctfix_v4.md`
exp_hipo_004 early-stopped en época 77/207 (mejor val/mae=1.876, época 15). Test (n=40):
MAE body=2.552, RV65 AUC=0.909, RV55 AUC=0.833, BV65 AUC=0.982. **El ML clásico
(LogReg, mismo split) iguala o supera a la U-Net en las 3 métricas operativas** (RV65
AUC=0.968, RV55=0.853, BV65=0.969) — conclusión principal de esta vuelta, ver el .md
para detalle completo y la comparación (orientativa, splits distintos) contra la
corrida vieja de CT corrupto.
