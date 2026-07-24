# Observaciones: reducción de base_features 32 → 16

## Problema detectado

Al correr `exp001_unet2d_baseline.yaml` con `base_features: 32` el entrenamiento tardaba ~10 segundos por step, lo que proyecta ~45 minutos por epoch (objetivo: <10 min).

## Diagnóstico con `scripts/profile_step.py`

Mediciones sobre 5 pacientes, n_target=56 (menor que el real de 80):

| Componente | Tiempo |
|---|---|
| I/O disco (NPZ + pad) | 246 ms |
| GPU transfer (CPU→GPU) | 22 ms |
| Forward pass | 932 ms (promedio; alta varianza: 429→1159 ms) |
| Backward + optim | 2899 ms |
| **Step completo** | **4968 ms** |
| VRAM pico reportada | 24,956 MB |

El backward (2.9 s) representa el 58% del step. La varianza del forward (429 ms en la primera corrida vs 1159 ms en las siguientes) y el VRAM pico de ~25 GB en una GPU de 12 GB indican **overflow de activaciones a RAM del sistema (WDDM en Windows)**. El driver de Windows pagea los tensores que no entran en los 12 GB físicos, lo que penaliza fuertemente el backward.

Con n_target=80 (el real del dataset) el problema es más pronunciado, porque las activaciones escalan linealmente con el número de slices.

## Causa raíz

Un U-Net depth=4 con base_features=32 procesa un batch efectivo de (B×Z, C, H, W) = (80, 5, 256, 256). Las activaciones intermedias que backward necesita retener (skip connections de los 4 niveles del encoder + decoder, en FP16) superan los 12 GB de VRAM disponibles.

## Fix aplicado

`configs/exp001_unet2d_baseline.yaml`:
```yaml
base_features: 16   # era 32
```

Esto reduce las activaciones ~4× (los canales en cada nivel se reducen a la mitad, las convoluciones escalan con canales²). El modelo pasa de ~7.2 M a ~1.8 M parámetros.

Epoch estimada post-fix: **6–8 minutos** (todo en VRAM, sin overflow).

## Por qué base_features=16 es suficiente para el baseline

El objetivo del exp001 es confirmar que `val/mae` baja consistentemente durante las primeras epochs. No es el modelo final. Con ~1.8 M parámetros el U-Net sigue siendo capaz de aprender la distribución de dosis 2D sobre 265 pacientes.

Cuando se pase a etapas siguientes (2.5D, 3D, HD U-Net) se puede volver a base_features=32 o mayor con hardware adecuado, o reducir n_slices / resolución.

## Otros hallazgos del profiling

- **I/O no es el cuello de botella**: 246 ms por paciente (4.7% del step). Con cache_train=true el impacto es cero; sin cache el overhead es <5%. El training es compute-bound.
- **Cache_train=true**: justificado solo si hay suficiente RAM libre. Con la VM de Windows 7 activa (16 GB asignados) no cabe el cache de train (~19.5 GB). Desactivar con `cache_train: false` en el YAML.
- **El script `scripts/profile_step.py`** queda disponible para re-diagnosticar ante cualquier cambio de arquitectura.
