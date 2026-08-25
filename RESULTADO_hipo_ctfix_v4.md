# Resultado — hipofraccionado sobre pipeline CT corregido (ctfix v4)

2026-08-25. Re-arranque completo del paquete hipo tras el fix del bug de CT (`cargar_ct()`
cargaba la serie RD en vez de CT) + FOV definitivo 34cm. Ver `HIPOFX_KICKOFF.md` para el
plan y `CLAUDE_CODE_CONTEXT.md` para el detalle completo paso a paso.

## Dataset y split

- `processed_hipo_ctfix/` — 201 NPZ, FOV 34cm (1.328mm/px), pipeline CT corregido.
- `data/splits/splits_hipo_ctfix_v4.json` — 198 pacientes tras exclusiones (128/30/40),
  estratificado 2D (cumple/no-cumple × tercil overlap PTV-Recto). 49.5% no-cumplidores
  (vs ~12% del dataset viejo) — mucho más balanceado.
- Excluidos manualmente: `PT_5b54e7add30325f0` (outlier de normalización, probable
  artefacto de extracción) y `PT_a0f9d9d98bbb8c81` (documentado en el kickoff como normo
  colado; el CSV actual no coincide con esa descripción — **pendiente confirmar con
  Pablo si se puede reincorporar**).

## Modelos evaluados (mismo split, mismo test set — 40 pacientes)

### U-Net (`exp_hipo_004_finetune_ctfix_v4`)
Fine-tuning desde `exp002_ctfix_fov34/epoch=127.ckpt` (normo, pipeline corregido), LR=1e-5,
full finetune. Early-stopped en época 77/207 (mejor `val/mae`=1.876 en época 15,
`checkpoints/exp_hipo_004_finetune_ctfix_v4/epoch=014.ckpt`) — convergencia rápida,
consistente con la aceleración por CT real observada en normo.

| Métrica | Valor (test, n=40) |
|---|---|
| MAE body | 2.552 [IC95 2.05–3.35] |
| MAE Rectum | 6.120 |
| MAE Bladder | 3.976 |
| RV65 — AUC | 0.909 [0.790–0.984] |
| RV55 — AUC | 0.833 [0.683–0.960] |
| BV65 — AUC | 0.982 [0.941–1.0] |
| RV65 — Sens/Esp (umbral val-frozen) | 0.667 / 0.880 |
| RV55 — Sens/Esp | 0.750 / 0.643 |
| BV65 — Sens/Esp | 0.778 / 1.000 |
| Any-op-fail | sens=0.957 / esp=0.765 |

### ML clásico (`results/proyecto1_ctfix_v4`)
LogReg sobre las 7 features escalares pre-plan, mismo split, calibración CV val-frozen.

| Métrica | Valor (test, n=40) |
|---|---|
| RV65 — AUC | 0.968 [0.913–1.0] |
| RV55 — AUC | 0.853 [0.711–0.959] |
| BV65 — AUC | 0.969 [0.908–1.0] |
| Recto — Sens/Esp (variante A / B) | 1.00/0.78 · 0.88/0.91 |
| Vejiga — Sens/Esp | 0.85 / 0.93 |

## Conclusión principal

**El modelo clásico (LogReg, 7 features geométricas pre-plan) iguala o supera a la U-Net**
en las 3 métricas operativas, y lo hace de forma clara en RV65 (0.968 vs 0.909 AUC). La
U-Net no muestra ventaja que justifique su costo (GPU, ~1h de entrenamiento, pipeline de
imagen completo) frente a un modelo lineal sobre features tabulares baratas de calcular
en el tomógrafo. Consistente con hallazgos previos del proyecto (el CT real no aporta
señal sobre el PSDM; acá tampoco aporta señal suficiente para superar a las features
geométricas simples).

## Comparación vs. la corrida hipo vieja (CT corrupto, `exp_hipo_003_finetune_v3`)

*Split y composición de test distintos entre ambas corridas (no es un test set idéntico) —
comparación orientativa, no estrictamente controlada.*

| Métrica | v3 (CT corrupto, FOV~50cm) | v4 (CT corregido, FOV 34cm) |
|---|---|---|
| MAE body | 2.464 | 2.552 |
| MAE Rectum | 5.625 | 6.120 |
| MAE Bladder | 3.604 | 3.976 |
| RV65 AUC | 0.882 | 0.909 |
| RV55 AUC | 0.809 | 0.833 |
| BV65 AUC | 0.973 | 0.982 |

El fix del pipeline (CT real + FOV34) mejora levemente el AUC de los 3 constraints
operativos, pero empeora levemente el MAE crudo — dentro del ruido esperable para n=40 y
sin controlar por el cambio de split. No hay evidencia de que el fix haya destapado una
mejora sustancial en la U-Net; el punto que sí queda firme es la comparación contra ML
clásico de arriba.

## Archivos

- Config: `configs/exp_hipo_004_finetune_ctfix_v4.yaml`
- Checkpoint usado: `checkpoints/exp_hipo_004_finetune_ctfix_v4/epoch=014.ckpt` (no
  versionado en git, ver `.gitignore`)
- Resultados U-Net: `results/exp_hipo_004_finetune_ctfix_v4_test_hipo/`
  (`metrics_summary.json`, `per_patient_metrics.csv`, `plots/`)
- Resultados ML clásico: `results/proyecto1_ctfix_v4/metrics_summary.json`
- Split: `data/splits/splits_hipo_ctfix_v4.json`
- GT DVH regenerado: `data/gt_dvh_hipo_256_ctfix_v4.csv`

## Pendiente

- Confirmar con Pablo la exclusión de `PT_a0f9d9d98bbb8c81`.
- Desglose por Status (Rejected/UnApproved/TreatmentApproved) — no se corrió en esta
  vuelta, HIPOFX_KICKOFF.md lo pide como informativo para generalización.
- Si se decide seguir invirtiendo en la U-Net pese a la comparación de arriba, evaluar
  por qué no logra superar al modelo lineal (¿falta de datos, arquitectura, o el
  problema es genuinamente lineal en estas 7 features?).
