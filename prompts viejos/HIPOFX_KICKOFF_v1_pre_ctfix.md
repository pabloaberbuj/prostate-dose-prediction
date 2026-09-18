# Contexto de arranque — Dataset Hipofraccionado

## Qué es este documento

Complemento al `CLAUDE_CODE_CONTEXT.md` para el nuevo chat de Claude.ai que arranca la etapa
hipofraccionada. Leer ambos antes de tomar decisiones de diseño.

---

## Estado al arrancar esta etapa

La serie normofraccionada está **cerrada**. Mejor modelo: exp002 (U-Net 2D + PSDM + MAE puro).
El objetivo ahora es:
1. Construir el dataset hipofraccionado con casos cumplidores Y no-cumplidores.
2. Evaluar si el modelo puede detectar incumplimiento de constraints (la herramienta clínica real).
3. Entrenar desde cero sobre este dataset (no transfer learning — es un problema físico distinto).

---

## Lo que ya está disponible

- **179 pacientes hipofraccionados** extraídos (CSV: `metricas_planes_hipofx.csv`).
  - 28 fx × 2.5 Gy = 70 Gy total. Algoritmo AAA, 6X.
  - Incluye los 21 pacientes que fueron pasados a normofraccionado por incumplimiento de
    constraints → SÍ tienen plan hipofraccionado calculado con dosis 3D en Eclipse.
    Son los casos de incumplimiento del dataset.
  - 2 pacientes excluidos temporalmente:
    - `PT_a0f9d9d98bbb8c81`: plan normofraccionado colado (39 fx, 78 Gy) → descartar.
    - `PT_04387201c8c366f3`: fallo en extracción por nombre PTV. Pablo lo re-extrae.
  - `PT_92c9d5a00753519e`: overlap = 0.0 en ambos OARs, verificar si es válido.
  - **Re-extracción en curso por Pablo** (CSV limpio con DVH metrics).

- **Constraints del protocolo hipofraccionado** (archivo `constraints_prostata_hipo.txt`):
  ```
  Prescripción: 70 Gy / 28 fx (2.5 Gy/fx)
  PTV:     V70Gy > 98%
  Rectum:  V65Gy < 15% | V55Gy < 25% | V45Gy < 45% | Dmean < 40 Gy (solo registro)
  Bladder: V65Gy < 15% | V55Gy < 25% | V45Gy < 45% | Dmean < 40 Gy (solo registro)
  ```
  Criterio de cumplimiento: AND estricto sobre V65/V55/V45 para cada OAR (a definir con Pablo).
  Los Dmean no son constraints operativos, solo registro.

- **Pipeline de extracción** (C#/ESAPI) y **preprocesado** (Python → NPZ): reusar del normo.
  - Bug de overlap corregido: usar intersección de máscaras en Python, no el C#.
  - Bug de escala de voxel: PSDM calculado sobre grilla nativa antes del downsample → OK.
    Para cualquier otro cálculo de volumen/distancia sobre máscaras 256×256: calibrar con
    `meta['vol_ptv_cc']` nativo (ver decisiones de diseño en CLAUDE_CODE_CONTEXT.md).

- **Estructura de arcos heterogénea**: ~50% 2 arcos, ~50% 3 arcos, 1 caso de 4 arcos.
  Esto importa si en el futuro se agrega un arc prior geométrico como input.

---

## Decisiones abiertas — resolver antes de avanzar

### D1. Criterio de "cumple/no-cumple" (para cada OAR)
¿AND estricto (falla uno de los 3 V = no cumple) o hay jerarquía/tolerancia entre V65/V55/V45?
Definir con Pablo. Afecta el etiquetado de la variable objetivo y la estratificación del split.

### D2. Esquema de normalización de dosis
**Pendiente de ver datos.** En normo se usó D95(PTV)=100%. Para hipo el objetivo es D98=100%
(equivalente a V70Gy≥98%). Pero los planes rechazados tienen factores de normalización muy
altos (mala cobertura). Riesgo: normalizar agresivamente borra la señal de incumplimiento.

Pablo está extrayendo tanto el factor D98 como D95 en el CSV para analizar la distribución.
**No cerrar esta decisión sin ver esos números.** Las preguntas clave:
- ¿Cuál es la distribución del factor de normalización entre cumplidores vs no-cumplidores?
- ¿Hay un umbral clínico de "cobertura mínima aceptable" por debajo del cual el plan
  directamente no es válido (y debería excluirse o marcarse)?
- ¿Normalizar a D98=100% en planes no-cumplidores borra la señal que queremos detectar?

### D3. Estratificación del split
En normo se estratificó por terciles de overlap PTV-Recto (con bug → al final fue proxy de
tamaño de recto). Para hipo, estratificar por **cumple/no-cumple** como primera dimensión, y
overlap PTV-Recto real (calculado desde máscaras NPZ, no del C#) como segunda dimensión.
Objetivo: que train/val/test tengan proporciones similares de casos positivos (no-cumplidores).
Con solo ~21 casos no-cumplidores conocidos sobre ~177, el balance de clases es un problema
real. Ver D4.

### D4. Balance de clases en el dataset
~21/177 ≈ 12% no-cumplidores. Para entrenamiento y evaluación:
- ¿Usar oversampling de casos no-cumplidores en train?
- ¿Enriquecer el test set con todos los casos no-cumplidores disponibles?
- Hay "planes subóptimos" que cumplen por margen estrecho — ¿son cumplidores o zona gris?
Esto requiere ver la distribución real de los valores V65/V55/V45 en el CSV completo.

### D5. Qué métricas de evaluación agregar vs normo
En normo el test tenía 60/60 cumplidores (todos aprobados) → solo medías falsos negativos.
Acá por primera vez podés medir:
- Sensibilidad (detecta incumplimiento real) y especificidad (no alarma en cumplidor).
- AUC por OAR y por constraint.
- Valor clínico real de la herramienta.
Diseñar estas métricas antes de construir `evaluate.py` para el dataset hipo.

---

## Lo que NO cambia respecto al normo

- Arquitectura de arranque: exp002-equivalente (U-Net 2D + PSDM + MAE puro). Es el ganador
  de la serie normo; arrancar desde ahí para este dataset también.
- Formato NPZ, estructura de canales, normalización PSDM (÷15 cm).
- Hardware, entorno, W&B proyecto.
- Splits en JSON, misma lógica de evaluate.py/analyze_errors.py.
- base_features=16, GroupNorm, batch=1, etc. (ver decisiones de diseño en CLAUDE_CODE_CONTEXT.md).

---

## Orden de trabajo sugerido para el nuevo chat

1. Pablo trae el CSV hipofraccionado re-extraído con DVH metrics → analizar distribución
   de factores de normalización y valores de constraints → cerrar D1, D2, D3, D4.
2. Diseñar el split estratificado con las decisiones cerradas.
3. Preprocesado a NPZ (reusar pipeline, actualizar config para nueva prescripción).
4. Definir métricas de evaluación nuevas (D5) antes de tocar código.
5. Entrenar baseline hipofraccionado (exp_hipo_001).
6. Evaluar con métricas completas incluyendo sensibilidad/especificidad.
