---
tags: [prostatedoseproject, subproyecto, ml-clasico, proyecto1]
estado: en-progreso
---

# Proyecto 1 — ML clásico (clasificación de cumplimiento)

## Qué es
Modelo de clasificación/regresión (regresión logística, gradient boosting) sobre ~7 features geométricas escalares extraídas del plan (no deep learning), que predice si un plan va a cumplir los constraints clínicos de Rectum/Bladder/PTV antes de aprobarlo. Es la línea de trabajo que se separó del pipeline U-Net cuando se descubrió que el ML clásico iguala o supera a la U-Net en esta tarea específica.

## Estado actual
Tareas 1–6 **completas**. Tarea 7 **bloqueada**.

## Realizado
- Pipeline completo: extracción de features → entrenamiento (`LogisticRegression`/`HistGradientBoostingClassifier` para clasificación, `Ridge`/`HistGradientBoostingRegressor` para regresión) → calibración → evaluación.
- **Hallazgo central del proyecto**: el ML clásico iguala o supera a la U-Net en clasificación de cumplimiento de constraints — esta es la bisagra que dividió el proyecto en dos líneas (este, tabular/clásico, vs. el pipeline U-Net/KBP de predicción de dosis completa).
- Sensibilidad de Recto recalibrada vía cross-validation: 0.50 → 0.83 en test.

## Pendientes
| Pendiente | Bloqueador |
|---|---|
| Tarea 7: extractor de features en vivo desde DICOM | Nombres de estructura del autocontour en `null` dentro de `config_p1.yaml` — falta que Pablo los defina |
| Tarea 7 (test de consistencia) | No sobrevive ninguna carpeta de DICOMs crudos hipo en disco para correr el test |

## Aprendizajes propios
- El precedente de `configs/config_p1.yaml::estructuras` (mapeo `{ptv: "PTV_High", rectum: "Rectum", bladder: "Bladder"}`) es el mejor ejemplo ya existente en el proyecto de externalizar nombres anatómicos a config — sirvió de referencia para diseñar el "anatomy schema" del pipeline U-Net (ver `CLI_INSTRUCTIVO.md`).

## Archivos clave
- `scripts/baseline_ml_clasico.py`, `train_p1_clf.py`, `train_p1_reg.py`
- `scripts/calibrate_p1*.py`, `eval_p1*.py`, `prep_data_p1*.py`, `build_manifest_p1.py`, `export_feature_importance_p1.py`
- `configs/config_p1.yaml`
- `models/proyecto1/`, `models/proyecto1_ctfix/`
- `scripts/extract_features_live.py`, `scripts/infer_tomografo.py`
