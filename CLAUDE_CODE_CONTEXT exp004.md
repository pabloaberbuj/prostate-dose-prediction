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

Al terminar una tarea en Claude Code, actualizar este archivo con los hallazgos relevantes y volver a Claude.ai con `metrics.csv` + `summary.json` (si aplica) para análisis y decisión del próximo paso.

---

## Contexto clínico

- Pacientes de próstata VMAT normofraccionados (39 fx × 2 Gy = 78 Gy, algoritmo AAA, 6X).
- El modelo predice distribución de dosis 3D voxel-wise en dosis relativa (% de prescripción, D95(PTV) = 100%).
- Uso clínico: predecir si recto y vejiga cumplirán los constraints de DVH antes de planificar.
- Estructuras: PTV (PTV, PTV_High, PTV_High2, PTV_Abr22, PTVp), Rectum, Bladder, BODY.
- Constraints clínicos: Rectum V70<25%, V65<35%; Bladder V70<35%, V65<50%.

---

## Estado actual del proyecto

### Dataset (completado)
- 385 pacientes usables. Split: 265 train / 60 val / 60 test, estratificado por terciles solapamiento PTV-Recto.
- Splits en `data/splits/splits_v1.json`.

### Preprocesado (completado)
- NPZs en `C:\Pablo\ProstateDoseProject\processed\` (fuera del repo).
- Cada NPZ: ct [-1,1], dose (% prescripción), masks binarias, PSDM en cm/15.
- Pipeline en `data/preprocess.py`.

### exp001 — Baseline 2D U-Net con masks (COMPLETADO)

Config: `configs/exp001_unet2d_baseline.yaml`
- base_features=16, depth=4, in_channels=5 (CT+BODY+PTV+Rectum+Bladder masks)
- 152 epochs, ~3 min/epoch, early stop paciencia 30

Métricas test set (60 pacientes):
- MAE body: 2.39 ± 0.51%
- MAE PTV: 1.45 ± 0.42%
- MAE Rectum: 5.07 ± 1.76%
- MAE Bladder: 4.19 ± 1.49%
- Dose score: 2.39, DVH score: 1.40 ± 0.61
- Constraints: Rectum V70=95%, V65=100%; Bladder V70=100%, V65=100%

Peor caso: Rectum bajo solapamiento (MAE 5.79), Bladder pequeña (MAE 5.25).

### exp002 — PSDM 2D (COMPLETADO)

Config: `configs/exp002_unet2d_psdm.yaml`
- Mismo U-Net 2D, in_channels=5 (CT+BODY+PSDM_PTV+PSDM_Rectum+PSDM_Bladder)

Métricas test set (60 pacientes):
- MAE body: 1.77 ± 0.43% (−26% vs exp001)
- MAE PTV: 1.41 ± 0.78% (≈ igual)
- MAE Rectum: 3.75 ± 1.69% (−26%)
- MAE Bladder: 2.16 ± 1.01% (−48%)
- Dose score: 1.77, DVH score: 1.15 ± 0.68
- Constraints: Rectum V70=95%, V65=100%; Bladder V70=98.3%, V65=100%

Mejora estratificada (hipótesis confirmadas):
- Rectum bajo solapamiento: 5.79 → 4.32 (−25%)
- Bladder pequeña: 5.25 → 2.99 (−43%)

Hallazgos visuales: artefactos radiales en mapa de diferencia persisten pero atenuados. Causa: sigue siendo 2D, no ve contexto axial donde se entregan los arcos VMAT. → Motiva directamente exp004 (2.5D).

Outliers a monitorear: PT_9e1a8395db6704a5 (MAE rectum 12.48), PT_04b54f050dd5e064 (MAE body 3.85).

---

## Estructura del código

```
prostate-dose-prediction/
├── configs/
│   ├── exp001_unet2d_baseline.yaml
│   └── exp002_unet2d_psdm.yaml
├── data/
│   ├── preprocess.py
│   ├── qc_preprocess.py
│   ├── debug_geometry.py
│   ├── debug_norm.py
│   ├── splits/splits_v1.json
│   ├── metricas_planes.csv
│   └── extract_dicom_csharp/Program.cs
├── src/
│   ├── datamodules/dose_datamodule.py    # cache float16
│   ├── models/
│   │   ├── unet2d.py
│   │   └── lightning_module.py
│   ├── losses/losses.py                  # MAE + MomentLoss
│   └── callbacks/logging_callbacks.py
└── scripts/
    ├── train.py
    ├── smoke_test.py
    ├── evaluate.py
    ├── analyze_errors.py
    ├── check_psdm.py
    └── profile_step.py
```

---

## Tarea inmediata — exp004: U-Net 2.5D con PSDM (3 cortes de contexto)

**Objetivo:** agregar contexto axial al modelo para reducir los artefactos radiales residuales que se observan en exp002. Hereda los inputs PSDM de exp002 (ganador) y agrega cortes adyacentes (z-1, z, z+1) por canal.

**Hipótesis:** los artefactos radiales vienen de que el modelo 2D no tiene información de la geometría axial de los arcos VMAT. Dar contexto de cortes vecinos permite capturar gradientes axiales.

**Diseño decidido en Claude.ai:**
- 2.5D = U-Net 2D que recibe cortes adyacentes stackeados como canales extra.
- context_slices: 3 (target ± 1).
- Todos los canales reciben contexto: 5 canales base × 3 cortes = 15 in_channels.
  (Los 15 canales no son independientes — alta correlación axial entre cortes vecinos.
  La info nueva es el gradiente axial, ~5 dimensiones, no 10.)
- base_features=16 (mantener). El primer conv pasa de 5→16 a 15→16, impacto en VRAM negligible.
  Si los resultados no mejoran, considerar base_features=24 como ablación posterior (exp004b),
  pero NO hacerlo de entrada — primero probar que el contexto axial sirve con 16.
- Padding en bordes del volumen: replicar el corte extremo (pad="replicate"). En la práctica
  los cortes extremos rara vez son target porque el preproceso dejó margen arriba/abajo.
- Output: solo el corte target (z central). La loss se calcula sobre el slice target únicamente.
- Todo lo demás igual que exp002: PSDM inputs, MAE loss, Adam, GroupNorm, mismo split, cache.

### Paso 1 — Modificar el datamodule ✓ (implementado en lightning_module, no en el datamodule — ver nota)

**Nota de arquitectura (descubierta al implementar):** el plan asumía un `__getitem__` por `(patient, slice_idx)`,
pero `DosePatientDataset` en realidad devuelve el **volumen completo** por paciente (shape `(Z,H,W)` por canal),
y el ensamblado de canales de entrada (CT/BODY/PSDM según `cfg.data.inputs`) ya vive centralizado en
`DosePredictionModule._build_input` (`src/models/lightning_module.py`), que después aplana `(B,Z)` para
pasar por el U-Net 2D. No hace falta indexar por slice: el contexto axial se logra desplazando cada
volumen a lo largo de Z (`_shift_z`, con replicate padding en los bordes) dentro de `_build_input`.
Esto es equivalente en resultado al plan original pero no toca `dose_datamodule.py`. Implementado así:

- `DosePredictionModule._shift_z(t, offset)`: desplaza `(B,Z,H,W)` a lo largo de Z (offset=-1 hace que
  cada posición vea z-1, offset=+1 hace que vea z+1), replicando el corte extremo en los bordes.
- `_build_input` ahora itera `offset in range(-k, k+1)` con `k = (context_slices - 1) // 2`
  (`context_slices = getattr(cfg.data, 'context_slices', 1)`), y por cada offset agrega los mismos
  canales que antes (CT, BODY, PSDM_PTV/Rectum/Bladder según config) desplazados.
  Orden resultante: offset-mayor, canal-menor, igual al especificado originalmente
  (CT_z-1, BODY_z-1, PSDM_PTV_z-1, ..., CT_z, BODY_z, ..., CT_z+1, ...).
- Con `context_slices=1` (default), el loop tiene un solo offset=0 y `_shift_z` devuelve el tensor
  sin modificar, quedando idéntico a exp001/002. Verificado con smoke test (ver Paso 3).
- Dose target y masks para loss/métricas siguen sin desplazar (solo el corte z real), sin cambios.
- `dose_datamodule.py` queda intacto.

También se agregó soporte de `logging.tags` en `scripts/train.py` (antes no estaba cableado a `WandbLogger`).

### Paso 2 — Crear el config ✓

`configs/exp004_unet25d_3ctx.yaml` creado con `data.context_slices: 3`, `model.in_channels: 15`,
`cache_train: false` / `cache_val: false` (se dejó en false por si no se puede cerrar la VM de W7 —
igual que exp002), tags W&B `inputs:psdm`, `arch:unet25d`, `context:3`, `stage:5`.

### Paso 3 — Verificar retrocompatibilidad ✓

```
python scripts/smoke_test.py --config configs/exp002_unet2d_psdm.yaml
```
OK — input al modelo `[1, 80, 5, 256, 256]` (B,Z,C,H,W), idéntico a antes del cambio. Loss/backward sin errores.

### Paso 4 — Smoke test exp004 ✓

```
python scripts/smoke_test.py --config configs/exp004_unet25d_3ctx.yaml
```

- Input al modelo: `[1, 80, 15, 256, 256]` (B,Z,C,H,W) ✓
- Forward + backward sin errores, grad norm finito ✓
- Params: 1,813,745 vs 1,812,305 de exp002 (+1440, solo la primera conv 5→15 canales) — VRAM negligible ✓
- Verificado además con un chequeo numérico aislado de `_shift_z`: offset=-1 en la posición z trae el
  valor de z-1 (replicado en el borde), offset=+1 trae z+1 (replicado en el borde superior) — orden correcto.

Pendiente: correr smoke test también con `cache_train=true` momentáneamente no es necesario ya que
`_build_input` opera después del `DataLoader`, es independiente del cacheo.

### Paso 5 — Entrenamiento ✓ (COMPLETADO)

La VM de W7 quedó encendida (no se pudo cerrar); `cache_train=false`/`cache_val=false` ya estaban
seteados en el config para ese caso, así que no hubo que tocar nada.

Corrió 198 épocas (0-197) y frenó por early stopping (`val/mae` no mejoró en 30 chequeos,
mejor score 1.930). Run W&B: `run-20260701_200243-0ffwlfa1` (`lightning_logs/wandb/`).

**Anomalía de checkpointing detectada:** los `.ckpt` en disco dejaron de actualizarse en la
época 178 (10:31, `val/mae` en checkpoint ≈1.93) aunque el training siguió ~90 min más hasta la
época 197 (`val/mae` final 1.932, prácticamente igual). Causa probable: durante el chequeo de
código del turno anterior se hizo `torch.load` sobre `last.ckpt` mientras Lightning lo estaba
reemplazando (Windows no permite reemplazar un archivo con un handle de lectura abierto),
lo que probablemente rompió el guardado atómico de ahí en adelante sin frenar el training.
Como `val/mae` apenas se movió en ese tramo (1.930→1.932), el checkpoint de época 178 es
equivalente en calidad al final y se usó para evaluar. **Lección: no inspeccionar/leer archivos
de checkpoint con `torch.load` mientras un training está corriendo activamente — esperar a que
termine o a que se guarde el siguiente checkpoint.**

### Paso 6 — Evaluación ✓ (COMPLETADO)

```
python scripts/evaluate.py --checkpoint "checkpoints/exp004_unet25d_3ctx/epoch=178.ckpt" \
    --config configs/exp004_unet25d_3ctx.yaml --output-dir results/exp004_test
```

Resultados test set (60 pacientes) — comparado con exp002:

| Métrica | exp002 (PSDM 2D) | exp004 (PSDM 2.5D, 3 ctx) |
|---|---|---|
| MAE body | 1.77 ± 0.43% | 1.77 ± 0.43% |
| MAE PTV | 1.41 ± 0.78% | 1.34 ± 0.79% |
| MAE Rectum | 3.75 ± 1.69% | 3.72 ± 1.55% |
| MAE Bladder | 2.16 ± 1.01% | 2.14 ± 0.97% |
| Dose score | 1.77 | 1.77 |
| DVH score | 1.15 ± 0.68 | 1.17 |
| Rectum V70/V65 acuerdo | 95% / 100% | 95% / 100% |
| Bladder V70/V65 acuerdo | 98.3% / 100% | 98.3% / 100% |

**Prácticamente idéntico a exp002, sin mejora medible.** Se verificó que no es un problema de
wiring: el checkpoint real tiene `in_channels=15` (`model.inc.block.0.weight` shape `(16,15,3,3)`),
y los pesos de la primera conv para los canales z-1/z/z+1 tienen norma y media de |peso| comparables
(no se fueron a cero) — el modelo efectivamente usa los cortes de contexto, solo que no le aportan
señal útil para reducir el error.

**Análisis estratificado** (`analyze_errors.py`, `results/exp004_test/analysis/`):

| Subgrupo | exp002 | exp004 |
|---|---|---|
| MAE Rectum, solapamiento bajo | 4.32 ± 1.34 | 4.29 ± 1.29 |
| MAE Rectum, solapamiento medio | 3.97 ± 2.34 | 3.90 ± 2.11 |
| MAE Rectum, solapamiento alto | 2.96 ± 0.71 | 2.96 ± 0.57 |
| MAE Bladder, volumen bajo | 2.99 ± 1.28 | 2.88 ± 1.23 |
| MAE Bladder, volumen medio | 1.89 ± 0.47 | 1.94 ± 0.54 |
| MAE Bladder, volumen alto | 1.61 ± 0.45 | 1.60 ± 0.43 |

Los subgrupos que habían mejorado fuerte de exp001→exp002 (rectum bajo solapamiento −25%, bladder
pequeña −43%) **no mejoran más** de exp002→exp004 — las diferencias están dentro del ruido.

Outliers de exp002 monitoreados, sin resolverse:
- `PT_9e1a8395db6704a5`: MAE rectum 12.48 (exp002) → 11.16 (exp004). Mejora leve pero sigue siendo
  por lejos el peor caso del test set.
- `PT_04b54f050dd5e064`: MAE body 3.85 → 3.93 (sin cambio real).

Worst/best cases nuevos guardados en `results/exp004_test/analysis/{worst,best}_cases.csv`.

### Paso 7 — Handoff a Claude.ai

Volver con:
- `results/exp004_test/test_metrics.csv`, `results/exp004_test/summary.json`
- `results/exp004_test/analysis/error_analysis.json`, `worst_cases.csv`, `best_cases.csv`
- Figuras DVH + cortes en `results/exp004_test/figures/` y `results/exp004_test/analysis/figures/`
- Run W&B: `run-20260701_200243-0ffwlfa1`

Foco del análisis: el contexto axial de ±1 corte NO redujo el error (ver tabla arriba). Dos hipótesis
a discutir para decidir si vale la pena exp005 (5 cortes) o replantear el approach:
1. ±1 corte es insuficiente — cortes adyacentes están muy correlacionados, el gradiente axial real
   de los artefactos de arco puede necesitar más separación (exp005, 5 ctx, ±2).
2. El contexto de *anatomía* (CT/PSDM) no es donde vive la información que falta — ni CT ni PSDM
   codifican geometría de arco/gantry, que es la causa hipotética real de los artefactos radiales.
   Agregar más cortes de anatomía no ayudaría aunque se probara con más contexto.

---

## Decisiones de diseño — NO cambiar sin consultar

- Sampling por paciente (no por corte suelto).
- num_workers=0 obligatorio en Windows.
- batch_size=1 por VRAM.
- Bilinear upsample + conv 3x3 (no transposed conv).
- GroupNorm (no BatchNorm con batch=1).
- Dosis en % prescripción normalizada a D95(PTV)=100%.
- PSDM en cm dividido por 15cm → rango [-1,1] aprox.
- Cache float16 en RAM, conversión a float32 en __getitem__.
- Test set sin cache.
- torch.set_float32_matmul_precision('high') en todos los scripts.
- base_features=16 para todos los experimentos en RTX A2000 12GB.
  (base_features=32 causa VRAM overflow y pageo WDDM → backward lentísimo)
- cache_train=false si la VM de W7 está activa (consume ~16GB RAM).

---

## Hardware y entorno

- GPU: NVIDIA RTX A2000 12GB, CUDA 13.0, Driver 581.42.
- RAM: 32GB. VM W7 consume 2-4GB → cerrar antes de entrenar.
- Windows 11, PowerShell, .venv en raíz del repo.
- W&B proyecto: prostate-dose-prediction.
- Kaggle (T4 16GB, Linux) para pruebas rápidas → num_workers=4 allá.

---

## Plan de ablaciones

| Exp | Variable | Config | Estado |
|-----|----------|--------|--------|
| 001 | Baseline masks 2D | exp001_unet2d_baseline.yaml | ✓ completado |
| 002 | PSDM 2D (5ch) | exp002_unet2d_psdm.yaml | ✓ completado — ganador inputs |
| 003 | (saltado — PSDM reemplazó masks, 8ch innecesario) | — | saltado |
| 004 | 2.5D PSDM (3 ctx, 15ch) | exp004_unet25d_3ctx.yaml | ← siguiente |
| 004b | 2.5D PSDM (3 ctx, base_features=24) | — | solo si 004 no mejora |
| 005 | 2.5D PSDM (5 ctx, 25ch) | exp005_unet25d_5ctx.yaml | solo si 004 mejora |
| 006 | Loss: MAE+MomentLoss | exp006_moment_loss.yaml | pendiente |

Cada experimento hereda el mejor config anterior. Naming: configs/expNNN_descripcion.yaml.

---

## Referencias clave

- DoseDiff (Zhang et al. 2024): PSDM + multi-encoder. Referencia para exp002-003.
- HD U-Net (Nguyen et al. 2019): U-Net con conexiones densas por nivel.
- Moment Loss (Jhanwar et al. 2022): MAE + momentos M1/M2/M10.
- OpenKBP (Babier et al. 2021): dose score y DVH score como métricas estándar.
