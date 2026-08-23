# Comisionamiento de PyDoseRT para el linac 6X

## Qué es y por qué es el prerrequisito de todo

PDRT no trae un beam model "genérico listo" (a diferencia de pyRadPlan). El motor de dosis
diferenciable necesita ajustarse a **nuestro** 6X antes de que la dosis recalculada tenga
sentido. Sin esto, ni el recálculo ni el mimicking son confiables.

La buena noticia: PDRT **parte de una base Varian** (`machine_config_base_varian.json`) y los
conversores aceptan formatos estándar de escaneo de agua. Se ajusta la base; no se comisiona
desde cero.

**Referencia de calidad esperada:** el paper reporta MAE ≤1.5% en perfiles de profundidad y
laterales tras comisionar. Suficiente para preplan y mejor — recordar que el objetivo NUNCA es
que PDRT compita en fidelidad con AAA; es un paso intermedio en el sandbox U-Net→Eclipse.

## Decisión de fuente de datos: phantom de agua en Eclipse, no mediciones físicas del equipo

**Definido:** el comisionamiento se hace con **perfiles, PDDs y output factors extraídos de
planes generados en Eclipse sobre un phantom de agua** (no con mediciones físicas archivadas
del equipo real). Razón de fondo: el ground truth de todo el proyecto (lo que la U-Net aprende
a predecir, el DVH contra el que se compara, "cumple/no cumple") es **AAA en Eclipse**, no el
equipo físico. Comisionar PDRT contra esa misma referencia da coherencia de sistema: PDRT se
ajusta para parecerse a lo que Eclipse va a decir cuando recalcule el preplan, que es exactamente
el uso que se le va a dar. Además es operativamente más simple: los planes de phantom se generan
según necesidad, sin depender de agenda de dosimetría física.

**Lo que esto NO resuelve, y no hace falta que resuelva:** ajustar contra AAA-en-agua no elimina
la divergencia entre PDRT y AAA en tejido heterogéneo (cada uno extrapola con su propia física
más allá de agua). Pero esto es aceptable e irrelevante para el alcance del proyecto — ver
siguiente sección.

## Por qué NO se valida contra fantomas antropomórficos ni heterogeneidades

El comisionamiento es deliberadamente acotado, por dos razones que conviene tener explícitas:

1. **Todas las mediciones de comisionamiento (las de acá y las de cualquier puesta en marcha
   real) son en agua.** No hay una versión "más correcta" del comisionamiento que use
   heterogeneidades — ni siquiera comisionando con mediciones físicas del equipo real se
   escaparía de eso.
2. **PDRT nunca sale del sandbox U-Net → Eclipse.** El comisionamiento es intencionalmente más
   chico y el algoritmo (pencil-beam simple) es peor que AAA. PDRT no se usa nunca como fuente de
   verdad ni se expone fuera de este loop — genera un preplan que Eclipse siempre recalcula y
   valida después. No hay escenario en el que la fidelidad de PDRT en heterogeneidad real importe
   por sí misma.

Por eso, el paso de comparación con pacientes reales que sigue más abajo **no es una segunda
etapa de comisionamiento buscando precisión clínica** (eso está fuera de alcance y no tiene
sentido perseguirlo). Es solo un **chequeo de que el sandbox se sostiene**: confirmar que PDRT no
diverge groseramente al pasar de agua a tejido heterogéneo, para saber si el gap observado hay
que atribuirlo a heterogeneidad (aceptable, documentar y seguir) o a un error de comisionamiento/
implementación (hay que corregir antes de seguir).

---

## Datos a generar: planes sobre phantom de agua en Eclipse

PDRT pide tres conjuntos. Verificado en `commissioning/run_commissioning_pipeline.py` y los
conversores en `commissioning/conversion/`. Se generan armando planes de campo abierto sobre un
phantom de agua en Eclipse y extrayendo la dosis calculada (AAA) en las geometrías que siguen.

### 1. Perfiles laterales (`profiles`)
Perfiles crossline/inline del haz a varias profundidades, extraídos de la dosis AAA calculada
sobre el phantom. Se usan para el modelado de penumbra y la forma lateral del haz.
- **Qué generar:** planes de campo abierto con perfiles a profundidades típicas (p. ej. dmax,
  5, 10, 20 cm) para varios tamaños de campo, extrayendo el perfil de dosis AAA en cada caso.
- **Formato de entrada al conversor:** los conversores existentes esperan `.mcc` (PTW
  BeamScan/Verisoft) o `.asc` (formato de escaneo de agua). Definir cómo se exporta el perfil
  AAA de Eclipse a uno de estos formatos (o escribir un conversor propio que tome el perfil
  extraído directamente y lo lleve al JSON esperado por el pipeline — más simple que forzar el
  formato `.mcc`/`.asc` si no hace falta esa compatibilidad).

### 2. Perfiles diagonales (`diagonals`)
Perfiles en diagonal del campo, mismo origen (dosis AAA sobre phantom). Se usan para la
**corrección off-axis** (el "horn"/aplanado fuera de eje). El pipeline ajusta una curva de
corrección radial a partir de la diagonal más grande.
- **Qué generar:** perfil diagonal AAA con campo grande (p. ej. 30×30 o 40×40, para cubrir el
  mayor rango radial).

### 3. Output factors (`output_factors`)
Factores de campo (output relativo AAA por tamaño de campo), sobre el mismo phantom.
- **Qué generar:** dosis en el eje central (a profundidad de referencia) para una serie de
  tamaños de campo. El pipeline de ejemplo usa 10×10 y 20×20 como campos de referencia para el
  matching de cola de perfil, pero conviene generar una tabla más completa.

> Los conversores existentes producen `.json` a partir de `.mcc`/`.asc`/`.csv`. Como la fuente
> ahora es Eclipse y no un escáner de agua físico, evaluar si conviene: (a) exportar desde
> Eclipse a `.csv`/`.mcc` si el formato lo permite y reusar los conversores tal cual, o (b)
> escribir un script propio que tome los perfiles/OF extraídos de Eclipse y arme directamente el
> JSON con la estructura esperada (más directo, evita el paso de formato intermedio). Definir
> esto al implementar — es un detalle de I/O, no cambia el resto del flujo.

---

## Parámetros de máquina a definir (MachineConfig)

Del comisionamiento salen los parámetros que van en `PDRT.MachineConfig(...)`:

- `tpr_20_10` — TPR 20/10 de nuestro 6X (el ejemplo trae 0.739 que es de 10MV; el de 6X es
  distinto, valor típico ~0.66–0.68 pero **usar el medido**).
- `mean_photon_energy_MeV` — energía media del espectro (parametrizada; sale del ajuste).
- `number_of_leaf_pairs = 60` — Millennium 120 (60 pares).
  **Confirmado en código:** `commissioning/machine_config_base_varian.json` trae
  `"model": "Millennium 120"` con `leaf_widths` = 10×[10mm] + 40×[5mm] + 10×[10mm] (60 leaves,
  campo 40cm) — coincide exactamente con la geometría estándar Millennium (5mm centrales,
  1cm periféricas). Preset ya correcto, no hay que definir esto a mano.
  ⚠️ **Ojo:** `MachineConfig.leaf_widths` es opcional; si se omite, `FluenceMapLayer` asume
  **anchos uniformes** (`field_size/n_pairs`) — asegurar SIEMPRE pasar el `leaf_widths` del
  preset Millennium, no dejar el default.
  Si algún linac es HD120 (leaves centrales 2.5mm), pasar su propio array — el código es
  agnóstico al ancho (`resample_fluence_map` usa el array, no asume uniformidad).
  Alternativa: `commissioning/toolkit/commissioning_toolkit.py` puede derivar `leaf_widths`
  directamente desde `mlc.leaf_boundaries` de un export de config de MLC, si se prefiere no
  copiar el array a mano.

Punto de partida: `commissioning/machine_config_base_varian.json`.

---

## Flujo de trabajo del comisionamiento

Orquestado por `commissioning/run_commissioning_pipeline.py`. Los pasos internos (según el
código): penumbra → corrección de perfil off-axis (desde diagonales) → head scatter / output
factors.

1. **Armar el phantom de agua en Eclipse** y generar los planes de campo abierto necesarios
   (tamaños de campo y profundidades a cubrir — ver checklist).
2. **Extraer** de cada plan: perfiles, diagonales, output factors calculados por AAA.
3. **Llevar a JSON** con el formato esperado por el pipeline (vía conversor existente adaptado,
   o script propio — ver nota de la sección anterior).
4. **Editar `run_commissioning_pipeline.py`**: apuntar `PROFILES_FILE`, `DIAGONALS_FILE`,
   `OUTPUT_FACTORS_FILE` a nuestros JSON; partir de `machine_config_base_varian.json`; ajustar
   parámetros del ejemplo que son específicos de energía/campo (`HS_FIELDS_CM`,
   `PROFILE_DIAGONAL_CUTOFF_DEG`, etc.).
5. **Correr el pipeline** → produce el `machine_config` ajustado a nuestro 6X.

---

## Chequeo de sandbox (NO es una segunda etapa de comisionamiento)

Antes de confiar en PDRT para mimicking, un chequeo rápido de que el ajuste hecho en agua-vía-AAA
no se rompe en tejido heterogéneo real. No busca precisión clínica (fuera de alcance, ver arriba)
— solo descarta errores groseros de comisionamiento/implementación:

1. Tomar 2-3 planes clínicos reales de próstata ya calculados con AAA en Eclipse (CT+RS+RP+RD).
2. Cargar con `load_dicom(...)` → `patient, beam_sequence`.
3. Recalcular la dosis con `engine.forward(...)` usando el `machine_config` comisionado.
4. Comparar dosis PDRT vs dosis AAA (RTDOSE de referencia) con **índice gamma** — criterio
   relajado respecto a un comisionamiento clínico real (no se busca 3%/3mm de nivel clínico,
   alcanza con confirmar que el patrón de dosis es razonable y sin divergencias groseras).
5. Si hay divergencia sistemática (p. ej. PDRT subestima en zonas de bajo scatter), documentarlo
   como bias conocido del sandbox y seguir — no es motivo para re-comisionar. Si en cambio hay
   algo roto (dosis con orden de magnitud incorrecto, geometría mal mapeada), ahí sí hay que
   revisar el comisionamiento o el mapeo de MLC/geometría antes de seguir.

Recién con esto pasado tiene sentido invertir en el mimicking. El notebook `rtplan.ipynb` es el
patrón forward/recálculo para esto.

## Cuántos perfiles/PDDs/OF hace falta generar

**No hay un número mínimo documentado como criterio de comisionamiento** (ni en README ni en
ningún doc). Es research code con configuración by-example: `run_commissioning_pipeline.py`
trae valores puntuales que corrieron sobre los datos disponibles de Umeå, no una especificación
de suficiencia como las de un protocolo clínico (tipo TG-106 para AAA). Lo único verificable en
código es el **mínimo funcional** para que el pipeline no falle, leído de los parámetros
hardcodeados del script:

- **Penumbra (paso 1):** usa **un perfil específico** — campo de referencia `10×10 cm` a
  profundidad `10 cm` (`PENUMBRA_FIELD_MM=(100,100)`, `PENUMBRA_DEPTH_MM=100`). Si no se cambian
  estos parámetros, ese punto tiene que estar en el JSON de perfiles.
- **Corrección off-axis (paso 2):** usa **la diagonal más grande disponible** en el archivo de
  diagonales — alcanza con una, aunque más diagonales dan un ajuste más robusto de la curva
  radial de corrección.
- **Output factors (paso 3):** el ejemplo usa **2 tamaños de campo** (`10×10` y `20×20`) a
  **1 profundidad** (`10 cm`) para el matching de cola de perfil (`HS_FIELDS_CM`,
  `HS_DEPTHS_MM`). El parser (`parse_output_factors_json/csv`) no parece tener un tope — se
  puede pasar una tabla más larga.

**Recomendación práctica:** dado que la fuente ahora es Eclipse (planes de phantom, no medición
física — ver más arriba), generar puntos adicionales es barato, sin costo de agenda de
dosimetría. Conviene generar más que el mínimo funcional: varios tamaños de campo (chicos tipo
próstata hasta grandes) y varias profundidades para perfiles/OF, aunque el pipeline solo use un
subconjunto fijo por defecto — da margen para ajustar `HS_FIELDS_CM`/`HS_DEPTHS_MM` si el ajuste
inicial no convence, sin tener que volver a generar datos.

---

- **Energía del ejemplo (10MV) ≠ nuestra (6X):** hay que reemplazar todos los datos de entrada
  y revisar parámetros hardcodeados por energía en el pipeline. Cambio de datos, no de lógica.
- **Cobertura de campos:** generar suficientes perfiles/OF para cubrir el rango clínico (campos
  chicos de próstata hasta campos grandes para la diagonal).
- **Formato de extracción desde Eclipse:** definir cómo se extraen perfiles/PDDs/OF de los
  planes de phantom (export de dosis + corte manual del perfil, o algún export directo) y cómo
  se llevan al JSON del pipeline (adaptar conversores existentes o script propio — ver más
  arriba). No validado aún con un caso real.
- **HD120 vs Millennium 120:** confirmar el modelo de MLC de cada linac.

---

## Checklist de arranque

- [ ] Confirmar modelo de MLC (Millennium 120 / HD120) de cada linac 6X.
- [ ] Verificar que el `MachineConfig` usado en código siempre pase `leaf_widths` explícito
      (Millennium: 10×10mm + 40×5mm + 10×10mm) — NO dejar el default uniforme de `FluenceMapLayer`.
- [ ] Definir geometría del phantom de agua y set de tamaños de campo/profundidades a cubrir.
- [ ] Generar en Eclipse los planes de campo abierto sobre el phantom (perfiles, diagonales,
      output factors) y extraer la dosis AAA calculada.
- [ ] Definir y probar el camino de conversión a JSON (adaptar conversor existente o script
      propio) con UN caso antes de generar el set completo.
- [ ] Generar el set completo de datos.
- [ ] Obtener `tpr_20_10` y demás parámetros de máquina del 6X (derivables de los datos AAA
      generados).
- [ ] Correr `run_commissioning_pipeline.py` adaptado.
- [ ] Chequeo de sandbox: recálculo vs AAA con gamma (criterio relajado) sobre 2-3 planes reales
      de próstata — no re-comisionar por divergencia esperable en heterogeneidad, solo descartar
      errores groseros.
