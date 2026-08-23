# Prompt Claude Code — Comparación homologada U-Net vs ML clásico (Proyecto 1, dataset nuevo)

## Objetivo

Producir una comparación **defendible** entre la U-Net (predicción de dosis 3D) y el ML clásico
(regresión logística) en la tarea de clasificación de cumplimiento de constraints, sobre el **dataset
nuevo (200 pacientes, splits_hipo_v3)**. El fin es justificar la elección del ML clásico para la
herramienta de tomógrafo con evidencia homologada, no con dos evaluaciones que corrieron por protocolos
distintos.

**El punto crítico:** las dos matrices de confusión deben salir del MISMO protocolo. Si la U-Net se
evalúa con un umbral distinto al del clásico, cualquier diferencia puede atribuirse al umbral y no al
modelo — y la comparación no justifica nada. Homologar es el 90% de esta tarea.

---

## Paso 0 — Verificación previa (consulta a resolver antes de evaluar)

La U-Net ya se reentrenó sobre el dataset extendido (sin indicaciones precisas, heredando config del
entrenamiento normo previo). Antes de evaluar, **verificar y reportar**:

1. **¿Existen las predicciones de dosis 3D de los 33 pacientes de test guardadas** (de una corrida
   `evaluate.py --save-pred` o equivalente)? Buscar en `results/` la carpeta de la corrida nueva de la
   U-Net. Si existen → usarlas. Si NO existen → regenerarlas con `evaluate.py --save-pred` sobre el test
   de `splits_hipo_v3.json`. **No reentrenar** — solo inferencia sobre test.
2. **Confirmar que la U-Net se evaluó sobre el MISMO test de `splits_hipo_v3.json`** (los mismos 33
   AnonID). Si la U-Net usó otro split (p.ej. uno heredado del normo o un v2), hay que re-inferir sobre
   el test v3 correcto. Reportar qué split usó la corrida existente.
3. Reportar la lista de 33 AnonID de test y confirmar que coinciden con los del clásico.

Si algo de esto no cierra, PARAR y reportar antes de seguir — no evaluar sobre poblaciones distintas.

---

## Paso 1 — De dosis predicha a clasificación (producto natural de la U-Net)

La clasificación de la U-Net debe salir de su producto natural (la dosis), NO de una cabeza de
clasificación aparte. Pipeline (reusar el del Proyecto 2 si ya existe):

- Dosis 3D predicha por paciente → `compute_pred_dvh.py` con **renormalización D95(PTV)=100%** (misma
  normalización contrafáctica que los labels del clásico — esto es esencial, los labels son D95norm).
- Del DVH renormalizado, extraer por paciente: **V65Gy(recto)%, V55Gy(recto)%, V65Gy(vejiga)%**.
- Estos valores continuos son el análogo de la "predicción" de la U-Net para cada constraint. De acá
  sale tanto la clasificación (¿supera el umbral del constraint?) como el eje de severidad del semáforo.

Guardar un CSV `results/unet_v3_eval/unet_pred_dvh_test.csv` con:
`AnonID, V65_recto_pred, V55_recto_pred, V65_vejiga_pred` + los valores GT reales (de
`gt_dvh_hipo_256.csv`) para los mismos 33, para trazabilidad.

---

## Paso 2 — Evaluar la U-Net con el MISMO protocolo del clásico

Reusar `eval_p1_cv.py` (el del clásico) o replicar su lógica exacta. Reglas idénticas:

- **Mismo test** (33, split v3).
- **Umbral calibrado por CV sobre población natural** (Approved+Rejected de train+val), **UnApproved
  fuera del pool de calibración**. Si el score de la U-Net es el V-value predicho, la calibración del
  umbral se hace sobre ese score con el mismo procedimiento CV (StratifiedKFold n=5, mediana de folds,
  seed=42) que se usó para el clásico.
- **Recto = combinación pesimista RV65+RV55** (falla si falla cualquiera).
- **Frontera de 3 zonas** (verde/naranja/rojo) con el mismo criterio de severidad.
- **Objetivo de sensibilidad:** recto 0.85, vejiga 0.90 (mismos que el clásico).
- **Bootstrap 1000, seed=42** para IC de AUC y de sens/esp.

**NO** recalibrar contra test. **NO** meter UnApproved en calibración. Mismas 4 reglas duras de siempre.

---

## Paso 3 — Tabla comparativa (el entregable)

Dos bloques, en este orden de prioridad argumentativa:

### Bloque 1 (prueba de fondo) — AUC [IC95] por constraint, independiente del umbral
Tabla: constraint × {ML clásico logreg, U-Net} con AUC + IC95. Es la evidencia principal porque esquiva
la objeción del umbral por completo. Reusar los valores del clásico de `results/proyecto1_v3_cv/`.

### Bloque 2 (ilustración clínica) — Matriz de semáforo lado a lado
Por OAR (recto, vejiga), para ambos modelos al mismo protocolo:
- Matriz binaria de la frontera verde: TP/FP/TN/FN → sens/esp/PPV/NPV con IC95.
- Matriz de 3 zonas (verde/naranja/rojo × cumple/falla) + delta_severidad_pp.
- Umbral usado + IC95 del umbral (para comparar estabilidad de calibración entre modelos).

Guardar todo en `results/comparacion_clasico_vs_unet_v3/metrics_summary.json` con estructura paralela
a la del clásico (mismas claves) para que sea diffeable.

---

## Qué NO hacer

- No reentrenar la U-Net (solo inferencia/re-evaluación).
- No evaluar los dos modelos sobre splits o normalizaciones distintas.
- No calibrar el umbral de la U-Net contra test ni con UnApproved en el pool.
- No sacar la clasificación de la U-Net de una cabeza distinta a su predicción de dosis.
- No decidir nada mirando test — solo tabular y traer el JSON a Claude.ai para el análisis.

---

## Entregable de vuelta a Claude.ai

`results/comparacion_clasico_vs_unet_v3/metrics_summary.json` + confirmación del Paso 0 (qué split y qué
predicciones usó la U-Net). Con eso armo en Claude.ai la tabla final y el argumento de por qué el clásico
es la elección para tomógrafo, con la comparación ya homologada y a prueba de objeciones.
