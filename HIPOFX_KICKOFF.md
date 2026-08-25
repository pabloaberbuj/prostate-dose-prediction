# Contexto de arranque — Dataset Hipofraccionado (v2, pipeline CT corregido)

## Qué es este documento

Kickoff para el nuevo chat de Claude.ai que arranca (re-arranca) la etapa hipofraccionada
sobre el **pipeline corregido** (fix del bug de CT + recorte de FOV). Reemplaza al kickoff
hipo anterior, que quedó obsoleto porque todo el trabajo hipo previo se hizo con NPZ de CT
corrupto. Leer junto con `CLAUDE_CODE_CONTEXT.md`.

---

## Por qué se re-arranca (contexto crítico)

Se descubrió que `cargar_ct()` cargaba la serie RD (dosis) en vez del CT real en TODOS los
pacientes (normo + hipo). El "canal CT" de todos los NPZ era dosis remuestreada, no anatomía.
Ver detalle completo en `CLAUDE_CODE_CONTEXT.md`. Consecuencias para el hipo:

- Todo el trabajo hipo previo (exp_hipo_001/002/003, ML clásico, splits) se hizo con CT
  corrupto → hay que rehacerlo sobre el pipeline corregido.
- El fix trajo además un cambio de FOV (el CT real es camilla-a-camilla; se agregó recorte en
  plano). FOV definitivo decidido: **caja cuadrada 34cm, centrada en centroide PTV, offset 0,
  256×256 → 1.328 mm/px**.
- En normo, el fix NO mejoró performance (el CT anatómico no aporta señal sobre el PSDM), pero
  sí aceleró la convergencia ~1.8x. La línea base normo corregida es exp002_ctfix_fov34.

---

## Configuración definitiva del pipeline (NO reabrir)

1. **FOV: caja 34cm, cuadrada, centrada en centroide PTV, offset 0, muestreo 256×256
   (1.328 mm/px).** Decidido tras comparar 34 vs 50cm: 34cm mejora predicción de OARs
   (rectum MAE 3.95→3.63, puntos DVH de recto mucho mejores) a costa del hotspot de PTV
   (D2/D01cc, que no es constraint clínico). Para el objetivo de la herramienta (predecir
   cumplimiento de OARs), 34cm es la elección correcta.
2. **cargar_ct() corregido:** selecciona serie Modality==CT, con aserciones fail-fast.
3. **CT real aporta poco a la predicción** pero acelera convergencia → se puede usar horizonte
   de entrenamiento más corto que en la serie vieja.

---

## Trabajo a rehacer sobre el pipeline corregido (todo el paquete hipo)

### 1. Re-extracción / re-preprocesado del dataset hipo desde DICOM
- Re-extraer con el C# (overlap ya corregido, no más Math.Min).
- Re-preprocesar a NPZ desde DICOM limpio con `cargar_ct()` corregido + recorte 34cm.
- Descartar TODOS los NPZ hipo viejos (CT corrupto) y los del último exp_hipo_003 (FOV sin
  recortar). Carpeta nueva, ej. `processed_hipo_ctfix34/`.
- Aplicar la limpieza de casos ya conocida:
  - `PT_a0f9d9d98bbb8c81`: plan normo colado (39fx/78Gy) → descartar.
  - `PT_04387201c8c366f3`: re-extraer (había fallado por nombre PTV).
  - `PT_92c9d5a00753519e`: verificar overlap (con C# corregido puede haber cambiado).
- QC post-preprocesado: CT sano (no saturado), OARs completos en la caja, clipping de BODY.

### 2. Rehacer el split estratificado
- Sobre el dataset re-extraído (puede haber cambiado de tamaño).
- Estratificar por: (a) cumple/no-cumple (constraint operativo, ver constraints abajo) como
  primera dimensión, (b) overlap PTV-recto REAL (ahora del C# corregido) como segunda.
- 65/15/20 aprox, asegurar suficientes positivos (no-cumplidores) en val para calibrar umbral
  (mínimo 4-5), con swaps train↔val dentro de la celda del estrato si hace falta.
- Guardar como `splits_hipo_v4.json` (o el nombre que corresponda a esta generación).

### 3. Rehacer el ML clásico (control metodológico) — OBLIGATORIO sobre el split nuevo
- **Dos razones para rehacerlo, no una:**
  a. Split nuevo → para que la comparación ML clásico vs U-Net sea sobre el MISMO test set
     (comparaciones sobre splits distintos no son interpretables — lección ya aprendida).
  b. Las features de overlap cambiaron: el overlap ahora viene del C# corregido. Como
     `overlap_rel_recto`/`overlap_rel_vejiga` eran las features DOMINANTES del modelo, el
     resultado puede cambiar de forma no trivial.
- Reentrenar regresión logística + gradient boosting sobre las 7 features escalares, mismo
  split, mismo protocolo de calibración val-frozen, mismo bootstrap. Es barato (segundos, sin GPU).
- Recién con ML clásico y U-Net sobre el MISMO split nuevo se puede sostener la comparación.

### 4. Entrenar la U-Net hipo — fine-tuning desde exp002_ctfix_fov34
- Init desde el checkpoint normo CORREGIDO (exp002_ctfix_fov34), NO desde el exp002 viejo de
  CT corrupto. Usar `--init-weights` (solo state_dict).
- LR bajo (1e-5), full fine-tuning, horizonte escalado por tamaño de train (factor =
  n_train_normo / n_train_hipo, aplicar sobre scheduler.max_epochs y early_stopping.patience).
  Aprovechar que el CT real acelera convergencia → horizonte puede ser más corto.
- Tags W&B con la generación del pipeline (ctfix, fov34).

### 5. Evaluación
- Métricas completas: MAE por estructura, AUC/sensibilidad/especificidad por constraint
  operativo (con umbral calibrado en val y CONGELADO antes de test), bootstrap IC95.
- Desglose por status (Rejected / UnApproved / TreatmentApproved) — informativo para
  generalización (fue valioso en la corrida anterior).
- Comparación U-Net vs ML clásico sobre el mismo split.

---

## Constraints del protocolo hipofraccionado

Archivo `constraints_prostata_hipo.txt`:
```
Prescripción: 70 Gy / 28 fx (2.5 Gy/fx). AAA, 6X.
PTV:     V70Gy > 98%
Rectum:  V65Gy < 15% | V55Gy < 25% | V45Gy < 45% | Dmean < 40 Gy (solo registro)
Bladder: V65Gy < 15% | V55Gy < 25% | V45Gy < 45% | Dmean < 40 Gy (solo registro)
```
- Criterio cumple/no-cumple: AND estricto sobre V65/V55/V45 por OAR (confirmar con Pablo si hay
  jerarquía/tolerancia). Dmean solo registro, no constraint operativo.
- Constraints operativos con señal suficiente (de la corrida anterior): RV65, RV55, BV65.
  Los demás (V45, V50) tenían pocos positivos.

---

## Decisiones abiertas (heredadas, resolver con datos del CSV re-extraído)

### D1. Normalización de dosis
Objetivo hipo: D98=100% (equivale a V70Gy≥98%). Pero planes rechazados tienen factores de
normalización altos (mala cobertura); normalizar agresivo puede borrar la señal de
incumplimiento. Pablo extrae factores D98 y D95 para analizar distribución. NO cerrar sin ver
esos números. Preguntas: distribución del factor entre cumplidores vs no; umbral de cobertura
mínima aceptable; ¿normalizar borra la señal?

### D2. Balance de clases
~12% no-cumplidores en el dataset viejo. Con el nuevo (superconjunto), ver prevalencia real.
Decidir oversampling en train y enriquecimiento del test con no-cumplidores.

### D3. Casos "subóptimos" que cumplen por margen estrecho — ¿cumplen o zona gris?
Ver distribución real de V65/V55/V45 en el CSV completo.

---

## Lo que NO cambia respecto al normo corregido

- Arquitectura: U-Net 2D + PSDM + MAE (exp002 sigue siendo el ganador, ahora con CT real).
- Formato NPZ, PSDM (÷15cm), FOV 34cm, base_features=16, GroupNorm, batch=1.
- Hardware, W&B, lógica de evaluate.py/analyze_errors.py.
- Fix de leak en splits (metodología: val para early stopping, test hold-out limpio).

---

## Orden de trabajo sugerido para el nuevo chat

1. Pablo re-extrae el CSV hipo con DVH metrics + factores de normalización (D95 y D98).
2. Analizar distribución de normalización y constraints → cerrar D1, D2, D3.
3. Re-preprocesar a NPZ (pipeline corregido, FOV 34cm).
4. Rehacer split estratificado (cumple/no-cumple + overlap real).
5. Rehacer ML clásico sobre el split nuevo.
6. Fine-tuning U-Net desde exp002_ctfix_fov34.
7. Evaluar: métricas completas + U-Net vs ML clásico + desglose por status.
