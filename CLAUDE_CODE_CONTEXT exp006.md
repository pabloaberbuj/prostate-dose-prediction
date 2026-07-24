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
- base_features=16, depth=4, in_channels=5 (CT+BODY+masks)
- MAE body: 2.39, rectum: 5.07, bladder: 4.19, DVH score: 1.40

### exp002 — PSDM 2D (COMPLETADO — ganador de inputs)
- in_channels=5 (CT+BODY+PSDM_PTV+PSDM_Rectum+PSDM_Bladder)
- MAE body: 1.77, rectum: 3.75, bladder: 2.16, DVH score: 1.15
- Mejora: −26% rectum, −48% bladder vs exp001.
- Artefactos radiales residuales en mapa de diferencia: consecuencia estructural de predecir
  solo desde anatomía sin información de geometría de arcos. Queda para segunda etapa.

### exp003 — (saltado)
- PSDM reemplaza masks → 8ch innecesario.

### exp004 — 2.5D PSDM 3 cortes (COMPLETADO — sin mejora)
- in_channels=15 (5ch × 3 cortes), context_slices=3
- MAE body: 1.77, rectum: 3.72, bladder: 2.14, DVH score: 1.17
- Δ vs exp002: <1% en todas las métricas. Contexto axial anatómico no aporta.
- Verificado: el modelo SÍ usa los canales de contexto (normas de pesos ~1.4-1.5 para los
  tres offsets z-1/z/z+1). El problema no es capacidad sino que la anatomía vecina no informa
  sobre la geometría angular de los arcos VMAT.
- Conclusión: exp005 (5 cortes) descartado por la misma razón.

---

## Estructura del código

```
prostate-dose-prediction/
├── configs/
│   ├── exp001_unet2d_baseline.yaml
│   ├── exp002_unet2d_psdm.yaml
│   ├── exp004_unet25d_3ctx.yaml
│   └── exp006_sweep_lambda{0001,001,01,1}.yaml
├── data/
│   ├── preprocess.py
│   ├── qc_preprocess.py
│   ├── splits/splits_v1.json
│   ├── metricas_planes.csv
│   └── extract_dicom_csharp/Program.cs
├── src/
│   ├── datamodules/dose_datamodule.py
│   ├── models/
│   │   ├── unet2d.py
│   │   └── lightning_module.py          # _shift_z/_build_input (context_slices),
│   │                                     # _calcular_dvh_score → loguea val/dvh_score
│   ├── losses/losses.py                 # MAE + MomentLoss (ya implementada, sin cambios)
│   └── callbacks/logging_callbacks.py
└── scripts/
    ├── train.py
    ├── smoke_test.py
    ├── evaluate.py
    ├── analyze_errors.py
    └── profile_step.py
```

---

## Tarea inmediata — exp006: sweep de λ para MomentLoss

**Objetivo:** encontrar el valor óptimo de λ en `loss = MAE + λ * MomentLoss` que mejore
las métricas de DVH sin degradar el MAE general. La hipótesis es que penalizar directamente
los momentos estadísticos de la distribución de dosis por ROI mejora la predicción de DVH,
que es la métrica clínicamente relevante (constraints V65, V70).

**Diseño decidido en Claude.ai:**
- Base: hereda exp002 (PSDM 2D, 5ch, base_features=16). NO hereda exp004 (2.5D no aportó).
- Momentos: M1 (media), M2 (varianza), M10 (cola de dosis alta). Los tres juntos.
- ROIs para MomentLoss: PTV + Rectum + Bladder (no Body — ya bien predicho, diluiría la señal).
- Sweep de λ: 0.001, 0.01, 0.1, 1.0 (el paper Jhanwar et al. 2022 usó λ=0.01).
- Duración del sweep: 40 epochs por run (suficiente para ver tendencia de convergencia).
- Criterio de selección: mejor val/dvh_score a los 40 epochs. En caso de empate, priorizar
  el que no degrada val/mae respecto de exp002.
- Una vez elegido el λ ganador: entrenar completo con early stopping (paciencia 30) = exp006.

### Paso 1 — Verificar la implementación de MomentLoss en losses.py ✓ (COMPLETADO, sin cambios necesarios)

**Escala de los momentos:** no hay problema. `_compute_moment` implementa la fórmula de potencia-media
de Jhanwar: `M_p = mean(D^p)^(1/p)`, no el momento crudo `E[D^p]`. El exponente `1/p` devuelve M1, M2 y
M10 a la misma escala física que la dosis (~0-110% prescripción), no a escalas cuadráticas/crecientes.
No hace falta normalizar ni reponderar M1/M2/M10 por separado.

**Máscaras:** se aplican correctamente — `_compute_moment` pondera por la máscara de la estructura antes
de promediar, y `masks` en `CombinedLoss.forward` viene de `struct_masks` (`ptv`/`rectum`/`bladder`) armado
en `_shared_step`, coincide con las claves de `cfg.loss.moment_structures`. Los cortes de padding en Z
tienen máscara 0 en todas las estructuras, no contaminan el cálculo.

**M10:** implementado como potencia-media con p=10 (`(mean(D^10))^(1/10)`), consistente con Jhanwar et al.
2022 — es un proxy diferenciable de la zona de dosis alta (hotspot/D10), no un percentil literal. Está
bien así; el paper no usa percentiles porque no son diferenciables.

**Chequeo empírico de escala** (cargando el checkpoint ya entrenado de exp002, un paciente real de test,
MAE≈1.42%): el MomentLoss total sin ponderar (9 términos: 3 estructuras × [M1,M2,M10]) da ≈17.4, dominado
casi por completo por Rectum M1+M2 (17.3 de 17.4) — un solo término mal predicho arrastra toda la suma
porque no hay normalización entre términos. Esto es inherente al diseño del paper (por eso usan λ=0.01
chico), no un bug. Traducido a contribución relativa sobre el MAE:

| λ | MomentLoss ponderado | vs MAE (~1.4-1.8 según exp002) |
|---|---|---|
| 0.001 | ~0.017 | ~1% (despreciable) |
| 0.01 (paper) | ~0.17 | ~12% |
| 0.1 | ~1.74 | ~120% (mismo orden) |
| 1.0 | ~17.4 | ~1230% (domina totalmente) |

Confirma que el sweep 0.001→1.0 cubre el rango completo esperado. `gradient_clip_val: 1.0` (ya en el
config, heredado de exp002) debería contener la inestabilidad de gradiente que se ve en época 0 con
predicciones sin entrenar (M10 sin clamp superior puede explotar si `pred` está lejos del rango real
de dosis — grad_norm sin clippear llegó a 210 en el smoke test).

**Cambio adicional necesario (no en el plan original, descubierto al implementar):** el criterio de
selección del sweep es "mejor `val/dvh_score`", pero ese scalar no existía en el training loop — solo
se logueaban `mae_*_pct` y `dmean_err_*`; el `dvh_score_openkbp` (D2/D95/D99 PTV + Dmean/Dmax OARs) solo
se calculaba post-hoc en `evaluate.py`. Se agregó `DosePredictionModule._calcular_dvh_score` (mismo
criterio que `evaluate.py`) y se loguea como `{stage}/dvh_score` en `_shared_step`. Nota de implementación:
`torch.quantile` no soporta float16 — bajo AMP (`precision: 16-mixed`) hubo que castear a `.float()`
antes de calcular percentiles (bug encontrado y corregido durante el smoke test).

**Bug de performance encontrado al lanzar el sweep real (no visible en el smoke test, que solo
corre 2 batches):** con `lambda0001` corriendo de verdad, el epoch pasó de los ~5:18 min esperados
(referencia exp002/004) a ~17-18 min — el usuario lo notó por lo lento que iba. Causa: `MomentLoss.
_compute_moment` calculaba `dose.pow(p)` sobre el volumen completo `(1, 80, 256, 256)` ≈5.24M elementos
y recién después multiplicaba por la máscara para descartar todo lo que no es ROI — 9 potencias
(3 estructuras × [M1,M2,M10]) × 2 (pred y target) = 18 `pow()` sobre el volumen entero por step, cuando
PTV/Rectum/Bladder son una fracción chica de los voxeles totales. Se corrigió seleccionando la ROI
(`pred[mask>0]`) *antes* de elevar a potencia, matemáticamente idéntico (verificado numéricamente,
diff ~1e-9) pero cientos de veces menos elementos procesados. Verificado con un benchmark real de 25
steps: volvió a 0.80 it/s (era ~0.24 it/s con el bug), epoch estimado ~5.5 min — igual que exp002.
Se descartó el checkpoint parcial de `lambda0001` (3 épocas, lento pero numéricamente correcto) y se
relanzó el sweep completo desde cero con el fix. **Lección: los smoke tests de 2 batches no alcanzan
para detectar regresiones de throughput — solo se vio corriendo el training real.**

**Segundo problema (el más importante) encontrado tras el relanzamiento:** con el fix de MomentLoss
puesto, un benchmark aislado de 25 steps sin validación dio 0.80 it/s (igual que exp002) — pero la
primera época real completa (con `cache_train: true`, incluyendo el paso de validación) tardó
~12-15 min, no ~5:18 min. Causa: `cache_train: true` estimaba 19.5GB de RAM para el cache, pero el
proceso real terminó usando ~23.4GB — sobre 31.64GB de RAM total del sistema, dejando **~2GB libres**.
Con tan poco margen, cualquier pico de uso (carga de validación desde disco con `cache_val: false`,
logging de imágenes a W&B, matplotlib de los callbacks de DVH/cortes) dispara swapping a disco de
Windows, mucho más lento que leer NPZs directamente. El propio código ya loguea que el modo sin cache
tiene "<3% overhead" — es decir, `cache_train: true` no aportaba una mejora real que justifique el
riesgo. **Se revirtió `cache_train: false` en los 4 configs del sweep** (igual que exp002/004, la
config ya probada y estable). **Lección para el futuro: "VM apagada → hay RAM disponible" no es
suficiente para decidir `cache_train: true` — hay que mirar la RAM libre real del sistema (`Get-Process`
+ `Win32_OperatingSystem`), no solo si la VM de W7 está prendida o no.**

**Tercer problema (invalidó el sweep completo de 14hs que sí terminó):** con ambos fixes de
performance puestos, el sweep corrió entero y terminó, pero comparando `val/mae` en época 39 contra
exp002 en su propia época 39, los 4 runs del sweep daban 13.7-16.2 vs 6.55 de exp002 — muy
sub-entrenados. Causa: los 4 configs tenían `scheduler.max_epochs: 40` (copiado para "coincidir" con
`training.max_epochs: 40`), pero `CosineAnnealingLR` decae el LR a ~0 en `T_max` épocas — con T_max=40,
en la época 39 el LR ya había caído a ~0.15% del inicial (prácticamente congelado), mientras que exp002
(`scheduler.max_epochs: 200`) todavía tenía 91% del LR inicial en su época 39. Los 4 runs compartían el
mismo error, así que no invalida la comparación *relativa* entre λs per se, pero el régimen de
entrenamiento (LR casi apagado) no es representativo de cómo se comportaría cada λ en un training real,
así que se descartó el sweep completo. **Fix: `scheduler.max_epochs` vuelve a 200 (igual que el
entrenamiento completo eventual); `training.max_epochs: 40` se deja igual — así el Trainer corta
temprano pero el LR en la época 39 es el mismo que tendría la época 39 de un run completo real,
comparable directamente con exp002.** Verificado con diff explícito de los 4 configs contra
`exp002_unet2d_psdm.yaml` tras el fix: el único delta real (además de nombre/tags/moment_weight/
early_stopping_patience/max_epochs, todos intencionales) es que `context_slices: 3` —un campo huérfano
sin uso en exp002, ya identificado en exp004— no está presente; nada de `base_features`, `depth`,
`in_channels`, `optimizer`, `precision`, `gradient_clip_val` ni `constraints` se movió.
**Lección: los smoke tests (2 batches) tampoco alcanzan para detectar bugs de scheduling que solo se
manifiestan a lo largo de muchas épocas — hay que comparar contra una referencia conocida (exp002) en
el mismo punto de entrenamiento, no asumir que "corre sin errores" == "corre correctamente".**

Antes de crear los configs, revisar `src/losses/losses.py`:

1. **Escala de los momentos:** M2 (varianza) puede ser órdenes de magnitud mayor que M1
   si se calcula en crudo sobre % prescripción. Verificar si los momentos están normalizados
   o ponderados internamente. Si M2 domina la loss con λ=0.01 ya grande, el sweep no va a
   dar resultados interpretables.
   - Si no hay normalización: agregar normalización por estructura (dividir cada momento
     por su escala típica) O ponderar M1/M2/M10 por separado con pesos fijos (e.g., 1:0.1:1).
   - Reportar a Claude.ai si hay que cambiar algo estructural antes de continuar.

2. **Verificar que las máscaras se pasen correctamente** al calcular los momentos: la loss
   debe aplicarse solo dentro de cada ROI (mask PTV, mask Rectum, mask Bladder), no sobre
   todo el volumen.

3. **Verificar que M10 esté implementado correctamente:** M10 es la media del 10% superior
   de la distribución de dosis dentro de la ROI (equivalente a D10 en DVH). Si está
   implementado como percentil 90, es correcto.

Si algo no cuadra, consultar a Claude.ai antes de seguir.

### Paso 2 — Crear los 4 configs del sweep ✓ (COMPLETADO)

Creados a partir de `exp002_unet2d_psdm.yaml`, con `loss.moment_weight` (no `moment_lambda` — ese es
el nombre real del campo en `losses.py`/`CombinedLoss`) en 0.001 / 0.01 / 0.1 / 1.0 respectivamente:

- `configs/exp006_sweep_lambda0001.yaml`
- `configs/exp006_sweep_lambda001.yaml`
- `configs/exp006_sweep_lambda01.yaml`
- `configs/exp006_sweep_lambda1.yaml`

En los 4: `experiment.name` correspondiente, `training.max_epochs: 40`,
`training.early_stopping_patience: 40` (== max_epochs, para que no pueda dispararse antes de
terminar el sweep — más simple que tocar `train.py` para hacer el callback opcional), tags W&B
`loss:momentloss`, `arch:unet2d`, `inputs:psdm`, `stage:6`, `lambda:XXX`, y **`cache_train: true`**
(la VM de W7 está apagada — estimado 19.5 GB de RAM para el cache de train, verificado en el smoke
test). Resto igual que exp002.

### Paso 3 — Smoke test de uno de los configs ✓ (COMPLETADO)

```
python scripts/smoke_test.py --config configs/exp006_sweep_lambda001.yaml
```

OK. MomentLoss se calcula por separado del MAE y se loguea distinto (`Loss MAE: 19.22`,
`Loss Mom: 57071.28` sin ponderar — modelo sin entrenar, valores altos esperables por el término M10
sin clamp superior). `val/dvh_score` se loguea correctamente (82.68 en step 2, sin entrenar). Sin
NaN/Inf. Cache de 265 pacientes en RAM confirmado sin errores (~19.5GB, VM apagada).

### Paso 4 — Lanzar el sweep ✓ (COMPLETADO — 3 de 4 runs; λ=1.0 cancelado, ver Resumen)

Corrieron `lambda0001`, `lambda001` y `lambda01` (40 épocas c/u). `lambda1` se canceló antes de
que empezara a entrenar de verdad (se frenó apenas arrancó el proceso) porque los primeros 3 puntos
ya mostraban una tendencia clara y consistente — ver "Resumen y resultados" abajo.

Se necesitaron 3 rondas de fixes antes de tener un sweep válido (documentados arriba en el Paso 1):
optimización de `MomentLoss` (mask antes de `pow`), revertir `cache_train` a `false` (RAM insuficiente),
y corregir `scheduler.max_epochs` (quedó en 40 en el primer intento, causando que el LR se apagara
casi del todo dentro del sweep — se corrigió a 200, igual que exp002, dejando que `training.max_epochs:
40` corte el Trainer temprano).

**Nota metodológica importante:** `val/dvh_score` tiene mucho ruido época a época (std ~4-5 puntos
sobre una media ~25). Comparar solo el valor de la época 39 es engañoso — en una primera pasada eso
sugirió que λ=0.01 mejoraba mucho el DVH sobre λ=0.001 (27.1 vs 15.5), pero promediando las últimas
10 épocas (mucho más robusto) esa diferencia desaparece casi por completo. **Usar mean/mediana de una
ventana de épocas, no el punto final solo, para cualquier comparación futura de este tipo.**

### Paso 5 — Resumen y resultados (para Claude.ai)

Media ± std de `val/mae` y `val/dvh_score` sobre las últimas 10 épocas (30-39) de cada run:

| λ | val/mae | val/dvh_score | mediana dvh |
|---|---|---|---|
| 0.001 | 8.82 ± 1.25 | 25.45 ± 5.16 | 24.75 |
| 0.01 (paper) | 12.19 ± 1.26 | 24.05 ± 5.41 | 24.05 |
| 0.1 | 14.63 ± 0.70 | 26.46 ± 4.37 | 27.20 |
| 1.0 | — (no corrido) | — | — |

**Hallazgo principal:** `val/mae` empeora de forma monótona y con poco ruido a medida que aumenta λ
(8.82 → 12.19 → 14.63, tendencia clara y consistente). `val/dvh_score`, en cambio, se mantiene
**estadísticamente plano** en todo el rango probado (24-26, todas las diferencias caen dentro del
ruido de ±4-5) — no hay evidencia de que el término de MomentLoss esté mejorando la precisión de DVH,
ni siquiera al valor que usa el paper (λ=0.01). En λ=0.1 además el `std` del MAE cae a 0.70 (vs ~1.25
en los otros dos), señal de que el entrenamiento converge a un punto fijo antes — consistente con que
el gradiente ya está dominado por el término de momentos (~120% del MAE según el chequeo de escala del
Paso 1).

**Por qué se canceló λ=1.0:** con el patrón "MAE empeora monótono, DVH plano" ya establecido en 3 puntos
que cubren dos órdenes de magnitud de λ, y con λ=1.0 empujando aún más al régimen dominado por momentos
(~1230% del MAE), es muy poco probable que revierta la tendencia. Se decidió no gastar las ~5-6 horas
adicionales; los datos disponibles ya alcanzan para la decisión.

**Pregunta abierta para Claude.ai:** el MomentLoss (M1/M2/M10 vía potencia-media) no parece transferir
a mejoras en `dvh_score` (que mide D2/D95/D99 para PTV y Dmean/Dmax para OARs). Dos hipótesis a discutir:
1. 40 épocas con LR aún relativamente alto (compartido por los 3 runs) puede ser insuficiente para que
   el término de momentos aporte señal útil — el efecto podría aparecer solo cerca de convergencia,
   cuando el MAE ya está bajo y el gradiente de MomentLoss deja de ser ruido comparado con él.
2. El mismatch entre las métricas (M_p es potencia-media, dvh_score usa percentiles para PTV) podría
   significar que optimizar M1/M2/M10 simplemente no mueve la aguja en D2/D95/D99, aunque el modelo
   entrene "bien" según su propia loss.

Con esto, `exp006_moment_loss.yaml` (entrenamiento completo del λ ganador) queda en pausa hasta que
Claude.ai decida: abandonar la vía de MomentLoss, probarla solo cerca de convergencia (ej. activarla
recién después de N épocas con MAE puro), o replantear qué métrica optimizar directamente.

### exp006 — entrenamiento completo λ=0.001 (COMPLETADO)

Se decidió entrenar λ=0.001 completo igual (el que menos degrada MAE) para tener la curva de
convergencia real con `train/moment`/`val/moment` logueados por época, y como referencia de test set.
`training.max_epochs: 200`, `early_stopping_patience: 30` (config `exp006_moment_loss.yaml`, único
delta real vs exp002: `use_moment_loss`/`moment_weight=0.001`).

Cortó por early stopping en la época 172 (mejor `val/mae`=1.914 en época 142). El training se
interrumpió una vez por un reinicio de la PC (época 88) y se retomó con `--resume` — al hacerlo
apareció un bug nuevo: `torch.load` con `weights_only=True` (default desde PyTorch 2.6) no puede
deserializar checkpoints con `omegaconf.DictConfig`, rompiendo `trainer.fit(ckpt_path=...)`.
**Se agregó a `train.py` el mismo monkeypatch que ya tenía `evaluate.py`** (forzar `weights_only=False`
solo cuando se usa `--resume`). Sin este fix, ningún resume desde checkpoint va a funcionar en este
entorno (PyTorch 2.12/CUDA 13.0) — importante para futuros experimentos largos.

Evaluación en test set (60 pacientes, checkpoint época 142) vs exp002:

| Métrica | exp002 | exp006 λ=0.001 |
|---|---|---|
| MAE body | 1.77 ± 0.43% | 1.77 ± 0.42% |
| MAE PTV | 1.41 ± 0.78% | 1.16 ± 0.29% |
| MAE Rectum | 3.75 ± 1.69% | 3.96 ± 1.52% |
| MAE Bladder | 2.16 ± 1.01% | 2.18 ± 1.00% |
| Dose score | 1.77 | 1.77 |
| DVH score | 1.15 ± 0.68 | 1.17 |
| Rectum V70 cumple | 95% | 98.3% |
| Bladder V70 cumple | 98.3% | 98.3% |

**Confirma el hallazgo del sweep:** body/dose_score/dvh_score prácticamente idénticos a exp002; PTV
mejoró (~18%) y Rectum empeoró levemente (~5%), compensándose — sin cambio neto real. λ=0.001 es
efectivamente equivalente a entrenar sin MomentLoss. Resultados en `results/exp006_moment_loss_test/`.

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
- context_slices implementado en lightning_module.py via _shift_z y _build_input.
- cache_train=false si la VM de W7 está activa.

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
| 003 | (saltado) | — | — |
| 004 | 2.5D PSDM (3 ctx) | exp004_unet25d_3ctx.yaml | ✓ completado — sin mejora |
| 005 | 2.5D PSDM (5 ctx) | — | descartado |
| 006 sweep | λ sweep MomentLoss (0.001/0.01/0.1) | exp006_sweep_lambdaXXX.yaml | ✓ completado — dvh_score plano, MAE empeora con λ; λ=1.0 cancelado |
| 006 | MAE + MomentLoss (λ=0.001) | exp006_moment_loss.yaml | ✓ completado — equivalente a exp002, sin mejora neta |

---

## Referencias clave

- DoseDiff (Zhang et al. 2024): PSDM + multi-encoder.
- HD U-Net (Nguyen et al. 2019): U-Net con conexiones densas por nivel.
- Moment Loss (Jhanwar et al. 2022): MAE + momentos M1/M2/M10. λ=0.01 en el paper.
- OpenKBP (Babier et al. 2021): dose score y DVH score como métricas estándar.
