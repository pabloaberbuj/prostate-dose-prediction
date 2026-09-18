---
tags: [prostatedoseproject, subproyecto, unet, kbp, hipofraccionado]
estado: completo
---

# U-Net / KBP — Próstata hipofraccionado

## Qué es
Variante hipofraccionada (70Gy/28fx) del pipeline de predicción de dosis, fine-tuneada desde el mejor checkpoint normofraccionado. Pasó por varias generaciones de dataset/split (v1→v4), cada una motivada por un problema encontrado en la anterior.

## Estado actual
**v4 completa** (`exp_hipo_004_finetune_ctfix_v4`) — es la baseline clínica vigente hoy. Split `splits_hipo_ctfix_v4.json` (198 pacientes). Resultado documentado en `RESULTADO_hipo_ctfix_v4.md`.

## Realizado
- v1 → v2: corrección de contaminación nodal (lo que en su momento se sospechó como "techo de geometría de arcos" — ver [[../aprendizajes_transversales/hipotesis_descartadas]]).
- v2 → v3: dataset ampliado.
- v3 → v4: re-arranque completo tras el bug crítico de carga de CT (ver [[../aprendizajes_transversales/bugs_criticos_resueltos]]) y la decisión de fijar el FOV en 34cm — documentado en `HIPOFX_KICKOFF.md`. v4 usa `processed_hipo_ctfix/` (198 pac.), fine-tuning desde `exp002_unet2d_psdm_ctfix_fov34` (LR=1e-5, horizonte escalado x2.07 ≈ 207 épocas).
- Resultado v4: DVH score 1.42 (meta interna de 1.15 no cumplida), MAE similar o mejor que la generación anterior. El CT real sigue sin aportar señal (3ra confirmación independiente, ver [[../aprendizajes_transversales/hipotesis_descartadas]]).
- Serie de barrido de loss DVH (`exp_hipo_003`, `003b`, `003c`) — concluye que hace falta un λ diferenciado por estructura, no uno global.

## Pendientes
| Pendiente | Bloqueador |
|---|---|
| Confirmar si `PT_a0f9d9d98bbb8c81` se reincorpora al dataset | Decisión de Pablo |
| Desglose de resultados por Status (Rejected / UnApproved / TreatmentApproved) | No corrido todavía |
| λ diferenciado por estructura en la loss DVH | Diseñado, no implementado |
| Cerrar la brecha DVH score 1.42 vs. meta 1.15 | Sin plan de acción activo — candidato natural: retomar junto con la ponderación por banda de dosis (ver [[../aprendizajes_transversales/decisiones_diseno_modelo]]) |

## Aprendizajes propios
- `preprocess_hipo.py` reusa funciones de `preprocess.py` (carga DICOM, PSDM) pero **reimplementa** `procesar_paciente_hipo` casi entera en vez de reusar `procesar_paciente` — deuda técnica identificada durante el diseño de la CLI genérica (ver `CLI_INSTRUCTIVO.md`).
- Prescripción y FOV de hipo son deliberadamente independientes de los de normo (70Gy/28fx, crop 340mm vs. 78Gy/39fx, crop 500mm) — no se comparten constantes por diseño.

## Archivos clave
- `configs/exp_hipo_001_baseline.yaml` … `configs/exp_hipo_004_finetune_ctfix_v4.yaml`
- `data/splits/splits_hipo_ctfix_v4.json`
- `data/preprocess_hipo.py`
- `scripts/evaluate_hipo.py`
- `checkpoints/exp_hipo_004_finetune_ctfix_v4/`
- `HIPOFX_KICKOFF.md`, `RESULTADO_hipo_ctfix_v4.md`
