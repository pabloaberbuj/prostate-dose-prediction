# Prostate Dose Prediction

Sistema de predicción de distribución de dosis 3D a partir de CT + contornos, para soporte de decisiones en tomógrafo en pacientes de próstata tratados con VMAT.

## Objetivo

Dado un paciente nuevo en tomógrafo, predecir si con la anatomía actual (preparación de recto y vejiga incluida) el plan VMAT cumplirá los constraints de DVH para recto y vejiga, fijada una cobertura objetivo del PTV (D95 = 100% de prescripción). Esto permite decidir si retomografiar o ajustar preparación antes de avanzar con la planificación.

El modelo predice la distribución de dosis 3D voxel-wise. Los constraints se evalúan en post-proceso a partir del DVH derivado de la predicción.

## Universo inicial

- 400 pacientes de próstata VMAT normofraccionados (39 fx × 2 Gy = 78 Gy)
- PTV único (próstata sola o próstata + vesículas)
- Algoritmo de cálculo: AAA (consistente en todo el dataset)
- Energía: 6X
- Planes generados con RapidPlan, homogéneos
- CT 512×512, espesor 3 mm, matriz de dosis 2.5 mm
- Período: 3 años aproximadamente

## Decisiones de diseño

### Output del modelo
- **Predicción**: distribución de dosis 3D voxel-wise (no clasificación, no regresión directa de métricas DVH).
- **Espacio**: dosis relativa (% de prescripción).
- **Normalización**: cada matriz escalada linealmente para llevar D95(PTV) = 100%. Se guarda el factor original como metadato para análisis posterior.

### Inputs del modelo
Configurables vía YAML. Variantes a evaluar:
1. CT + máscara BODY + máscara PTV + máscaras OARs (Bladder, Rectum)
2. CT + máscara BODY + PSDM (Physical Signed Distance Maps) de PTV/OARs
3. Multi-encoder estilo DoseDiff (CT en un encoder, PSDM en otro, fusión multi-escala)

### PSDM
- Calculado en espacio físico 3D, en centímetros.
- Normalizado dividiendo por 15 cm → rango [-1, 1] aprox.
- Signo: negativo dentro de la ROI, positivo fuera.

### Preprocesado
- CT: clip a [-1000, 1500] HU, normalización a [-1, 1].
- Dose: convertida a % de prescripción, normalizada a D95(PTV) = 100.
- Downsample in-plane a 256×256.
- Recorte axial: cortes con presencia de PTV/Bladder/Rectum + 5 cortes margen arriba/abajo.
- Formato: NPZ por paciente (CT, dose, masks, PSDM, metadata).

### Arquitecturas a comparar
1. U-Net 2D baseline (con bilinear upsample + conv 3×3 para evitar checkerboard)
2. U-Net 2.5D con 3 cortes de contexto
3. U-Net 2.5D con 5 cortes de contexto
4. U-Net 3D
5. HD U-Net (Nguyen et al. 2019)
6. Multi-encoder estilo DoseDiff (Zhang et al. 2024) sin componente de difusión

### Loss
- Baseline: MAE enmascarado por BODY.
- Combinada: MAE + λ · MomentLoss (Jhanwar et al. 2022).
  - Momentos M1, M2, M10 para PTV, Bladder, Rectum.
  - Posibilidad de momentos custom por estructura (M5/M10 para Rectum si interesa max, M1/M2 para Bladder si interesa mean).

### Training
- Framework: PyTorch + PyTorch Lightning.
- Optimizer: AdamW, lr inicial 1e-4, cosine annealing.
- Mixed precision: `precision="16-mixed"`.
- Augmentation: flip LR, rotaciones ±10°, intensity jitter ±5% HU.
- Batch size: a definir según VRAM (RTX PRO 2000 Blackwell).

### Split
- 280/60/60 (train/val/test) por paciente.
- Estratificación por terciles de % solapamiento PTV-Rectum.
- Test intocable hasta evaluación final.
- 5-fold cross-validation sobre train+val para resultados finales.

### Tracking
- Weights & Biases (W&B) — proyecto: `prostate-dose-prediction`.
- Configs YAML versionadas en `configs/`.

### Métricas de evaluación
- **Imagen**: MAE global (dentro de BODY), MAE por estructura.
- **DVH**: ΔD95 PTV, ΔV70 Rectum, ΔV65 Bladder, ΔDmean Rectum, ΔDmean Bladder.
- **OpenKBP-style**: dose score, DVH score.
- **Clínico binario**: tasa de acuerdo en "cumple/no cumple" cada constraint.
- **Visual**: DVH comparativo + cortes (axial/sagital/coronal) en pacientes fijos de validación.

### Anonimización
Extracción vía C# + ESAPI (Eclipse Scripting API). Por cada paciente se exporta:
- DICOMs (CT, RS, RD) anonimizados a carpeta.
- Fila en CSV maestro con métricas calculadas por Eclipse (D95, V70, V65, volúmenes, MUs, nº arcos, fecha del plan, etc.).

Anonimización elimina: PatientName, PatientID (reemplazado con hash), PatientBirthDate, ReferringPhysicianName, OperatorsName, StationName, InstitutionName, fechas (mantiene solo año), tags privadas. ID interno generado por sistema con tabla de mapeo encriptada local.

## Workflow nube/PC

- **Iteración rápida y debug**: Kaggle (30 h/semana de GPU T4/P100).
- **Entrenamientos completos**: PC local con RTX PRO 2000 Blackwell, overnight.
- **Tracking unificado**: W&B (los experimentos de ambas máquinas aparecen en el mismo dashboard).
- **Sincronización**: git (código) + Kaggle Datasets (preprocesado).

## Estructura del repo

```
prostate-dose-prediction/
├── configs/                # YAMLs por experimento
├── data/                   # Scripts de extracción y preprocesado
│   ├── extract_dicom_csharp/   # Proyecto C# + ESAPI (separado)
│   ├── preprocess.py
│   ├── splits.py
│   └── dvh_utils.py
├── src/
│   ├── datamodules/        # LightningDataModules
│   ├── models/             # Arquitecturas
│   ├── losses/             # MAE, moment, combinada
│   ├── transforms/         # Augmentation
│   ├── callbacks/          # Logging custom (DVH a W&B)
│   └── utils/              # Helpers
├── scripts/
│   ├── train.py            # CLI principal de entrenamiento
│   ├── evaluate.py         # Evaluación en test
│   └── predict.py          # Inferencia single-case
├── notebooks/              # EDA, análisis exploratorio
├── inference_server/       # FastAPI para deployment
├── tests/                  # Tests unitarios mínimos
├── requirements.txt
└── README.md
```

## Plan de etapas

| Etapa | Descripción | Duración estimada |
|-------|-------------|-------------------|
| 0 | Setup de infraestructura | 3-5 días |
| 1 | Extracción + anonimización + EDA | 1-2 semanas |
| 2 | Pipeline de preprocesado | 1 semana |
| 3 | Baseline 2D U-Net | 1-2 semanas |
| 4 | Ablaciones de inputs | 1-2 semanas |
| 5 | Ablaciones de arquitectura | 2-3 semanas |
| 6 | Ablaciones de loss | 1 semana |
| 7 | Análisis modelo v1 | 1 semana |
| 8 | Refinamientos (condicional) | variable |
| 9 | Deployment | 1 semana |
| 10 | Escalamiento a hipofraccionado | posterior |

## Referencias clave

- Zhang et al. 2024 — DoseDiff (PSDM + multi-encoder fusion)
- Nguyen et al. 2019 — HD U-Net
- Jhanwar et al. 2022 — Moment loss
- Babier et al. 2021 — OpenKBP challenge (métricas)
- Odena, Dumoulin & Olah 2016 — Checkerboard artifacts
