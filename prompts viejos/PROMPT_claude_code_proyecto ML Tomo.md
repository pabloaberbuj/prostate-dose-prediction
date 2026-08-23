# Handoff Claude Code — Proyecto 1: Herramienta ML clásico tomógrafo (próstata hipo)

## Rol

Sos especialista en ML aplicado a radioterapia. El físico médico (Pablo) tiene conocimientos
básicos-intermedios de Python. Respondé conciso y directo; al cambiar código explicá brevemente
qué y por qué. Si algo estructural no está claro, preguntá antes de cambiarlo.

---

## Objetivo

Herramienta de uso en **tomógrafo (CT-sim)** que, desde la RS del autocontour (PTV_High, Rectum,
Bladder), prediga si un paciente de próstata VMAT hipofraccionado (28 fx / 70 Gy) va a cumplir los
constraints de recto y vejiga **cuando el plan se normalice a D95(PTV)=100%** — para decidir en el
momento si la preparación es adecuada o hay que re-preparar / re-tomografiar antes de planificar.

Método: **ML clásico sobre 7 features geométricas escalares.** No deep learning (ya se verificó que
la U-Net no aporta sobre esta tarea; ver `CLAUDE_CODE_CONTEXT.md`).

**La pregunta que responde la herramienta es contrafáctica:** "si normalizás a D95=100%, ¿falla el
OAR?" — no "qué pasó en el plan real". Por eso los labels correctos son los `Flag_*_D95norm`
(recalculados a D95=100%), NO los flags a dosis entregada.

---

## Dataset

**Archivo:** `metricas_planes_hipofx_D95norm.csv` (separador `;`, 199 filas).

**Deduplicación obligatoria primero:** `df.drop_duplicates(subset='AnonID', keep='first')` →
**195 pacientes únicos** (hay 4 duplicados exactos de extracción). `AnonID` es por paciente (no por
plan), estable inter-estudio. Ya verificado: **sin solapamiento de AnonID entre grupos de Status**
→ no hay leak anatómico posible.

**Composición por `Status` (195 únicos):**

| Status | n | Qué es | Uso |
|---|---|---|---|
| `TreatmentApproved` | 133 | Planes hipo reales aprobados | train/val/test (prevalencia natural) |
| `Rejected` | 38 | Originalmente hipo, rechazados por constraints → falla real | train/val/test (prevalencia natural) |
| `UnApproved` | 24 | Normos replanificados como hipo, seleccionados por overlap alto → sintéticos enriquecedores | **SOLO train** |

**Todos los planes (incluidos UnApproved) tienen dosis real calculada en Eclipse (AAA).** Los labels
`Flag_*_D95norm` son confiables.

---

## Constraints operativos (los 3 con señal suficiente)

- `Flag_Rectum_V65Gy_lt15_D95norm` (RV65)
- `Flag_Rectum_V55Gy_lt25_D95norm` (RV55)
- `Flag_Bladder_V65Gy_lt15_D95norm` (BV65)

Flag=1 → cumple; Flag=0 → falla. Los V45/Dmean quedan como registro, no se modelan (pocos eventos).

Prevalencia natural de falla (Approved+Rejected, n=171): RV65 31%, RV55 25%, BV65 26%.

---

## Features (7, todas geométricas, conocibles en tomógrafo sin plan)

Del CSV: `VolRectum_cc`, `VolBladder_cc`, `VolPTV_cc`, `Solap_PTV_Rectum_cc`, `Solap_PTV_Bladder_cc`.
Derivadas: `overlap_rel_recto = Solap_PTV_Rectum_cc / VolRectum_cc`,
`overlap_rel_vejiga = Solap_PTV_Bladder_cc / VolBladder_cc`.

El `Solap_PTV_*` de este CSV es overlap geométrico real (el bug `Math.Min` del C# está corregido
para el dataset hipo). `overlap_rel_recto` domina RV65/RV55; `overlap_rel_vejiga` domina BV65.

Agregar también flag `tiene_VVSS` (detectable: casos sin VVSS tienen CTV_SV vacío → PTV más chico).
Para esta etapa sacarlo del CSV si hay columna que lo indique; si no, dejarlo como TODO y no bloquear.

---

## Regla de split (crítica — respeta la separación sintético/natural)

Generar `data/splits/splits_hipo_v3.json`. Reglas:

1. **Val y test:** SOLO de la población natural (Approved + Rejected, n=171). A prevalencia natural.
2. **Train:** el resto de la natural + **los 24 UnApproved enteros**.
3. **Estratificar** el reparto de la población natural por el vector de las 3 flags operativas
   (RV65, RV55, BV65) para que val/test tengan eventos de falla de cada constraint.
4. Proporciones por defecto: **test=20%, val=15%, train=65%** de la natural (cambiable). seed=42.
5. Verificar y reportar: nº de eventos de falla por constraint en cada partición (val/test no deben
   quedar sin eventos de ningún constraint; BV65 es el más justo).

**Por qué:** los UnApproved tienen prevalencia de falla 60-70% (seleccionados por overlap alto). Si
entran a val/test rompen la calibración de umbrales y las métricas dejan de reflejar el uso real.
En train, en cambio, enriquecen la clase falla — y como el error barato es el FP, un modelo algo
sesgado hacia predecir falla juega a favor.

---

## Tareas (en orden — la herramienta queda funcional al terminar la Tarea 4)

### Tarea 1 — Carga, dedup, features, split
`scripts/prep_data_p1.py`: cargar CSV (sep `;`), dedup por AnonID, computar las 7 features + derivadas,
generar `splits_hipo_v3.json` con la regla de arriba. Guardar un `data/dataset_p1.csv` con features +
3 labels + Status + partición. Imprimir tabla de prevalencias por partición.

### Tarea 2 — Modelo de clasificación (desde cero)
`scripts/train_p1_clf.py`: entrenar, para cada uno de los 3 constraints, **regresión logística**
(features estandarizadas con `StandardScaler` fiteado SOLO en train, `class_weight='balanced'`) como
**modelo a desplegar**, y `HistGradientBoostingClassifier` (`class_weight='balanced'`) como **techo de
performance / control** (no se despliega). Un modelo por constraint. Guardar scaler + modelos.

### Tarea 3 — Modelo de regresión de severidad (eje "rescatable vs imposible")
`scripts/train_p1_reg.py`: para cada constraint, entrenar una regresión sobre el **valor continuo** del
DVH (`Rectum_V65Gy_lt15_D95norm` es el valor %, la columna sin `Flag_`), mismas features, mismo scaler,
mismo split. Empezar con regresión lineal (Ridge); si el ajuste es pobre cerca del umbral, probar
`HistGradientBoostingRegressor`. El objetivo NO es predicción precisa sino **rankear "apenas pasado el
constraint" vs "muy pasado"**. Reportar MAE y correlación pred-vs-real en val/test.

### Tarea 4 — Calibración del semáforo de 3 zonas por OAR (val-frozen)
`scripts/calibrate_p1.py`. Diseño híbrido:

- **Frontera VERDE / no-verde** (crítica de seguridad, evita FN): de la **clasificación**. Umbral verde
  = punto en val donde la sensibilidad del "no-verde" alcanza el objetivo:
  - **Recto:** sens ≈ 0.85 (banda 0.80–0.90 aceptable; FP de recto es caro).
  - **Vejiga:** sens ≈ 0.90 (FP de vejiga es más tolerable).
- **Split NARANJA / ROJO** (solo dentro de no-verde): de la **regresión**. Umbral de severidad calibrado
  en val como el margen del valor predicho sobre el constraint que separa casos rescatables (fallan a
  D95=100% pero por poco) de imposibles (fallan por mucho). Arrancar con un margen simple (p.ej.
  V_pred < constraint + Δ → naranja; ≥ → rojo) y ajustar Δ mirando la distribución en val.
- **Congelar** todos los umbrales (sens objetivo, Δ severidad) y guardarlos junto al modelo. NO tocarlos
  después contra test.
- Reportar en val: % de casos en cada zona por OAR, y sens/esp de la frontera verde.

Semáforo **separado por recto y por vejiga** (acción clínica distinta: recto = gas/materia fecal;
vejiga = llenado). No colapsar a un flag único.

### Tarea 5 — Evaluación en test con bootstrap
`scripts/eval_p1.py`: sobre test (prevalencia natural), aplicar umbrales congelados. Reportar por
constraint: AUC [IC95 bootstrap 1000, seed=42], sens/esp de la frontera verde en el punto de operación,
matriz de las 3 zonas vs label real. Comparar logística vs HGB en AUC (esperado: dentro del IC uno de
otro). Guardar `results/proyecto1_v3/metrics_summary.json`.

### Tarea 6 — Serialización para inferencia
Guardar en `models/proyecto1/`: scaler, 3 modelos de clasificación, 3 de regresión, umbrales congelados,
y metadata (versión de features, orden de columnas, fecha, split usado). Un solo `joblib` con un dict, o
archivos separados + un `manifest.json`. Debe poder cargarse sin re-fitear nada.

### Tarea 7 — Extractor de features en vivo desde DICOM (parametrizado)
`scripts/extract_features_live.py`. **Necesita los nombres exactos de estructura del autocontour — dejar
como parámetros de config (`config_p1.yaml`), Pablo los completa.** Debe:
- Leer RS DICOM del autocontour, rasterizar contornos de PTV_High, Rectum, Bladder a máscara 3D en
  **grilla nativa** (sin downsample — preservar volúmenes).
- Computar volúmenes (cc) e intersección de máscaras PTV∩OAR (cc) en Python → las 7 features.
- **Ojo escala de voxel:** usar spacing nativo del DICOM, NO escalar por grilla downsampleada (lección
  del proyecto — ver `CLAUDE_CODE_CONTEXT.md`, `meta['vol_ptv_cc']`).
- Test de consistencia: correr sobre una muestra del dataset donde ya se conocen las features del CSV y
  verificar que reproducen dentro de <1%. Este es el check de exchangeabilidad real.
- Wrapper `infer_tomografo.py`: input = path RS DICOM → output = dict con {prob, zona verde/naranja/rojo,
  V_pred} por constraint.

Si los nombres de estructura no están todavía, dejar Tareas 1–6 completas y funcionales, y la 7 con la
config vacía + test de consistencia listo para correr cuando Pablo cargue los nombres.

---

## Decisiones de diseño heredadas — NO cambiar sin consultar

- Modelo a desplegar: **regresión logística** (calibrada de fábrica, interpretable, estable con n chico).
  HGB solo como techo/control.
- Umbrales **distintos por OAR** (costo FP asimétrico: recto caro, vejiga tolerable).
- FN objetivo: sens 0.80–0.90 (más honesto que 0.90 rígido con pocos positivos).
- UnApproved SOLO en train. Nunca val/test.
- Labels = `Flag_*_D95norm` (a D95=100%), nunca a dosis entregada.
- Todo reproducible: seed=42, scaler fiteado solo en train, umbrales congelados en val.

---

## Stack

Python, scikit-learn, pydicom + numpy/scipy (para rasterizar contornos en Tarea 7), pandas. CPU
suficiente (sin GPU). Repo con estructura existente (`scripts/`, `data/`, `configs/`, `results/`,
`models/`).

---

## Entregable de vuelta a Claude.ai

Al terminar, traer `results/proyecto1_v3/metrics_summary.json` + la tabla de prevalencias por partición
+ la tabla de zonas del semáforo en val/test, para análisis y decisión del punto de operación final.
