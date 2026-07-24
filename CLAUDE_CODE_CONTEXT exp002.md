# Prompt de contexto para Claude Code — Prostate Dose Prediction

## Rol

Sos un especialista en computer vision y deep learning con conocimientos de radioterapia.
Trabajamos en un proyecto de predicción de distribución de dosis 3D para pacientes de próstata tratados con VMAT.
Respondé de forma concisa y directa. Cuando hagas cambios al código explicá brevemente qué cambiaste y por qué.
Si necesitás más contexto sobre decisiones de diseño preguntá antes de cambiar algo estructural.

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

### exp001 — Baseline 2D U-Net con masks (COMPLETADO y evaluado)

Config: `configs/exp001_unet2d_baseline.yaml`
- base_features=16, depth=4, in_channels=5 (CT+BODY+PTV+Rectum+Bladder masks)
- 152 epochs, ~3 min/epoch, early stop paciencia 30

M�tricas val (epoch 122, mejor checkpoint):
- val/mae: 2.60%
- PTV MAE: 1.56%, Rectum MAE: 5.73%, Bladder MAE: 4.52%

M�tricas test set (60 pacientes):
- MAE body: 2.39 ± 0.51%
- MAE PTV: 1.45 ± 0.42%
- MAE Rectum: 5.07 ± 1.76%
- MAE Bladder: 4.19 ± 1.49%
- Dose score (OpenKBP): 2.39
- DVH score (OpenKBP): 1.40 ± 0.61
- Acuerdo constraints: Rectum V70=95%, Rectum V65=100%, Bladder V65=100%, Bladder V70=100%

Hallazgos del análisis de errores:
- Peor caso: Rectum con BAJO solapamiento PTV-Recto (MAE 5.79% vs 4.18% con alto solapamiento).
  Causa: U-Net 2D no captura geometría angular de arcos VMAT → artefactos radiales visibles.
- Bladder pequeña es el segundo caso difícil (MAE 5.25% vs 3.42% vejiga grande).
- Motivación para Etapa 4: PSDM aporta distancia física 3D que reduce ambigüedad angular.

Hallazgos de profiling:
- Training es compute-bound (I/O = 5% del step).
- base_features=32 causa VRAM overflow → pageo WDDM → backward lento (2.9s).
- base_features=16 resuelve el overflow: todo en VRAM, sin pageo.

---

## Estructura del código

```
prostate-dose-prediction/
├── configs/
│   └── exp001_unet2d_baseline.yaml      ← baseline completado
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
    └── profile_step.py                   # profiling de performance
```

---

## Tarea inmediata — Etapa 4: ablación de inputs (PSDM vs masks)

Crear exp002: misma arquitectura que exp001 pero reemplazando masks de PTV/OARs por PSDM.

### Pasos

1. Crear `configs/exp002_unet2d_psdm.yaml`:
   - Copiar exp001 y cambiar:
     - `experiment.name: exp002_unet2d_psdm`
     - `data.inputs.use_psdm: true` (reemplaza masks PTV/Rectum/Bladder por PSDM)
     - `data.inputs.use_ptv_mask: false`
     - `data.inputs.use_rectum_mask: false`
     - `data.inputs.use_bladder_mask: false`
     - `model.in_channels: 5` (CT + BODY + PSDM_PTV + PSDM_Rectum + PSDM_Bladder)

2. Verificar smoke test: `python scripts/smoke_test.py --config configs/exp002_unet2d_psdm.yaml`

3. Lanzar entrenamiento: `python scripts/train.py --config configs/exp002_unet2d_psdm.yaml`

4. Cuando termine, volver a Claude.ai con metrics.csv + summary.json del test para comparar exp001 vs exp002.

### Qué comparar al finalizar exp002
- val/mae y test/mae overall
- MAE Rectum: ¿mejora en casos de bajo solapamiento?
- MAE Bladder: ¿mejora en vejigas pequeñas?
- DVH score y tasa de acuerdo en constraints
- Figura DVH comparativa de los mismos pacientes del worst_cases de exp001

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

## Plan de ablaciones completo

| Exp | Variable | Config | Estado |
|-----|----------|--------|--------|
| 001 | Baseline masks | exp001_unet2d_baseline.yaml | ✓ completado |
| 002 | Inputs: PSDM | exp002_unet2d_psdm.yaml | ← siguiente |
| 003 | Inputs: multi-encoder | exp003_unet2d_multiencoder.yaml | pendiente |
| 004 | Arch: 2.5D (3 ctx) | exp004_unet25d_3ctx.yaml | pendiente |
| 005 | Arch: 2.5D (5 ctx) | exp005_unet25d_5ctx.yaml | pendiente |
| 006 | Loss: MAE+MomentLoss | exp006_moment_loss.yaml | pendiente |

Naming: configs/expNNN_descripcion.yaml. Cada experimento hereda el mejor config anterior.

---

## Referencias clave

- DoseDiff (Zhang et al. 2024): PSDM + multi-encoder. Referencia para exp002-003.
- HD U-Net (Nguyen et al. 2019): U-Net con conexiones densas por nivel.
- Moment Loss (Jhanwar et al. 2022): MAE + momentos M1/M2/M10.
- OpenKBP (Babier et al. 2021): dose score y DVH score como métricas estándar.
