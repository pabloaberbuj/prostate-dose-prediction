# Prompt de contexto para Claude Code — Prostate Dose Prediction

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

## Tarea inmediata — Verificación PSDM + corrección de leak

Dos tareas acotadas. El punto 3 anterior (limpieza/re-extracción del CSV hipofraccionado) lo
hace Pablo por fuera — NO es tarea de Claude Code.

### Tarea A — Verificar si el PSDM tiene el mismo bug de escala de voxel

Contexto: la auditoría descubrió que las máscaras del NPZ están downsampleadas a 256×256 pero
`meta['spacing_mm']` es el spacing nativo (pre-downsample), lo que da volúmenes/distancias mal
escalados por un factor `(orig/256)²` que varía por paciente (1.1×-4.2×). El overlap real se
corrigió calibrando con `vol_ptv_cc` nativo.

**La pregunta:** el PSDM (input GANADOR de toda la serie, usado en exp002/004/006) se guarda en
cm en el NPZ. ¿Se calculó sobre la grilla NATIVA antes de downsamplear (→ correcto), o sobre la
grilla ya downsampleada usando el spacing nativo sin recalibrar (→ mismo sesgo de escala)?

Pasos:
1. Revisar en `data/preprocess.py` el orden exacto de operaciones: ¿en qué punto se computa el
   PSDM respecto del downsample a 256×256? ¿Qué spacing usa el cálculo de la distancia?
2. Si se computa sobre la grilla nativa y luego se downsamplea el PSDM ya en cm → probablemente
   OK (la distancia física no cambia con el downsample, solo se remuestrea el mapa). Confirmar
   que el remuestreo no reescala los valores.
3. Si se computa sobre la grilla downsampleada con spacing nativo → tiene el sesgo. Cuantificar
   el impacto: como el PSDM se clipea a ±15cm y se divide por 15, verificar si el error de
   escala cambia significativamente los valores DENTRO del rango no saturado, o si el clipping
   lo absorbe en la mayoría de los vóxeles. NO asumir que es despreciable — medirlo en algunos
   pacientes (comparar PSDM guardado vs PSDM recalculado con escala correcta).
4. Reportar a Claude.ai: ¿el PSDM usado en la serie está bien escalado o no? Si no lo está,
   ¿cuál es la magnitud del error y afecta las conclusiones (exp002 sigue siendo el ganador)?

**Importante:** esto determina si la serie normofraccionado se puede dar por cerrada del todo
o si hay que revisar algo. No entrenar nada todavía — es solo diagnóstico.

### Tarea B — Corregir el leak train/val/test

`PT_505fcdbdbb6553b6` está en `val` y `test` simultáneamente en `splits_v1.json`.

1. **Sacarlo de `test`, NO de `val`.** Razón: `val` ya se usó para early stopping / selección
   de checkpoint en los 4 experimentos; sacarlo de val invalidaría esa selección. Sacarlo de
   test solo reduce el test a 59 pacientes, sin afectar los checkpoints.
2. Actualizar `splits_v1.json` (guardar el original como `splits_v1_backup.json` por trazabilidad).
3. Re-correr `evaluate.py` con test=59 sobre los checkpoints existentes: exp002, exp004, exp006.
   Es evaluación, no entrenamiento. exp001 no tiene checkpoint → se documenta como no disponible.
4. Reportar a Claude.ai las métricas de test actualizadas (esperado: cambio despreciable, 1/60,
   pero hay que confirmar los números).

### Resultado Tarea A — PSDM correctamente escalado, NO tiene el bug (completado — Claude Code, 2026-07-09)

Revisado `data/preprocess.py` paso a paso: el PSDM se computa en el paso 7
(`calcular_psdm(ptv_mask, spacing_zyx_mm)`), usando las máscaras NATIVAS
(`ptv_mask`/`rectum_mask`/`bladder_mask`, variables locales de resolución
completa, ANTES del recorte axial y ANTES del downsample) y el spacing NATIVO
real (`ct_sitk.GetSpacing()`). El downsample a 256×256 ocurre después (paso 9),
y se aplica al mapa de distancia YA CALCULADO EN CM vía `scipy.ndimage.zoom`
(interpolación bilineal) — es un remuestreo puro del campo continuo, no vuelve a
multiplicar por ningún spacing. A diferencia del bug de overlap (que contaba
voxels de la máscara YA downsampleada y multiplicaba por el spacing nativo), el
PSDM nunca pasa por esa operación — no hay ningún punto del código donde se
mezcle spacing nativo con la grilla downsampleada.

**Verificación empírica** (no alcanzaba con la lectura del código, se midió en 6
pacientes al azar): se recalculó la distance transform directamente sobre la
máscara ya downsampleada (256×256), calibrando el spacing efectivo igual que en
`compute_overlap_real.py`, y se comparó contra el PSDM guardado. Diferencia
cerca del borde de la estructura (±2cm, la zona que importa clínicamente):
**MAE 0.10–0.19 cm** (~1-2mm) — es el error normal de interpolación al
resamplear una distance map, NO un sesgo sistemático. Lejos del borde la
diferencia crece (MAE global 1.1–1.7cm) pero eso es ESPERADO: la versión
recalculada sobre la grilla de 256px pierde precisión ahí porque tiene menos
resolución que la nativa; el PSDM guardado (calculado en la grilla nativa antes
de downsamplear) es la versión MÁS precisa de las dos, no al revés.

**Veredicto: el PSDM está bien escalado. La serie normofraccionado (exp002 como
ganador) no necesita revisión por este motivo — se puede dar por cerrada del
todo.** El bug de escala solo afectaba cálculos derivados que contaban voxels
sobre la máscara ya downsampleada (overlap real, y potencialmente cualquier
métrica de volumen futura calculada directo desde el NPZ) — no afecta ningún
input ni output real de los modelos entrenados.

### Resultado Tarea B — Leak corregido, test re-evaluado con n=59 (completado — Claude Code, 2026-07-09)

Confirmado el mejor checkpoint por `val/mae` (el que generó los `test_metrics.csv`
actuales) inspeccionando `best_model_path`/`best_model_score` dentro de cada
`.ckpt`: exp002 = `epoch=175.ckpt` (val/mae 1.9417), exp004 = `epoch=167.ckpt`
(val/mae 1.9297), exp006 = `epoch=142.ckpt` (val/mae 1.9140).

`data/splits/splits_v1.json` editado: `PT_505fcdbdbb6553b6` removido solo de
`test` (test 60→59, val se mantiene en 60). Original preservado en
`data/splits/splits_v1_backup.json`. Re-corrido `evaluate.py` para exp002/004/006
con el test corregido → `results/<exp>_test59/test_metrics.csv` + `summary.json`
(conviven con los `_test` viejos de n=60, no se sobrescribieron).

**Cambio confirmado despreciable, como se esperaba:**

| Métrica | exp002 (n=60→59) | exp004 (n=60→59) | exp006 (n=60→59) |
|---|---|---|---|
| mae_body | 1.771→1.766 | 1.766→1.763 | 1.767→1.763 |
| mae_rectum | 3.750→3.781 | 3.718→3.799 | 3.955→3.981 |
| mae_bladder | 2.164→2.149 | 2.137→2.118 | 2.179→2.160 |
| dvh_score | 1.154→1.210 | 1.170→1.303 | 1.165→1.170 |

Todos los deltas son del orden de ruido de 1 paciente sobre ~60 (el dvh_score es
la métrica más sensible por ser un promedio de pocos términos — en exp002/004
subió ~0.05-0.13, en exp006 casi no cambió — pero en ningún caso se altera el
orden relativo entre experimentos ni la conclusión "exp002 ganador"). **exp001
no se re-evaluó** (sin checkpoint, ver hallazgo previo).

### Handoff a Claude.ai

Volver con:
- Veredicto de Tarea A: **PSDM sin bug, serie normofraccionado cerrada sin reservas.**
- Métricas de test=59 actualizadas para exp002/004/006 (Tarea B) — cambio despreciable, no altera conclusiones.

Después de esto: definir constraints hipofraccionado en YAML y diseñar extracción DVH
(cumple/no-cumple), una vez que Pablo tenga el CSV hipofraccionado re-extraído.

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

## Próxima etapa — dataset hipofraccionado (EN CURSO)

1. Auditoría de overlap (este documento, arriba).
2. Definir constraints hipofraccionado en YAML.
3. Extracción DVH (ESAPI) para identificar cumple/no-cumple en los 179 pacientes.
4. Confirmar que los 21 pacientes pasados a normo tienen plan hipofraccionado calculado
   (con dosis 3D) accesible — usuario confirmó que están dentro del dataset de 180.
5. Preprocesado a NPZ (reusar pipeline existente).
6. Split estratificado por overlap real (no el bugueado) + cumple/no-cumple.
7. Entrenar desde cero (exp001-equivalente) sobre este dataset — no transfer learning del
   modelo normofraccionado, es un problema físico distinto (fraccionamiento, constraints,
   posiblemente márgenes de PTV distintos).

---

## Referencias clave

- DoseDiff (Zhang et al. 2024): PSDM + multi-encoder.
- HD U-Net (Nguyen et al. 2019): U-Net con conexiones densas por nivel.
- Moment Loss (Jhanwar et al. 2022): MAE + momentos M1/M2/M10. λ=0.01 en el paper — no
  replicado en este dataset (ver exp006).
- OpenKBP (Babier et al. 2021): dose score y DVH score como métricas estándar.
