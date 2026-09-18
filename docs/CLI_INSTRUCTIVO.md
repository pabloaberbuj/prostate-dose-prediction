---
tags: [prostatedoseproject, cli, dosekbp, instructivo]
estado: vigente
---

# Instructivo — CLI `dosekbp` (etapa1) y anatomías nuevas

Cómo instalar y usar la CLI unificada del pipeline de predicción de dosis (KBP), y cómo entrenar una localización anatómica distinta de próstata sin tocar código. Complementa [[00_INDICE]] y los docs de subproyecto.

## 1. Instalación

Desde `repo\` (con el venv activado):

```bash
pip install -e .
```

Esto instala el paquete `dosekbp` (`pyproject.toml` + `dosekbp/cli.py`) en modo editable y agrega el comando `dosekbp` al PATH del venv. No reemplaza nada: los scripts originales (`scripts/train.py`, `scripts/evaluate.py`, `data/preprocess.py`, `scripts/predict_one.py`) siguen funcionando exactamente igual si los seguís llamando con `python <script>.py ...` — `dosekbp` es una capa que los envuelve, no los reemplaza.

## 2. Subcomandos

Cada subcomando arma `sys.argv` y corre el script original tal cual (misma lógica, mismo `argparse`) — por eso `dosekbp <subcomando> --help` muestra el `--help` real del script envuelto, no uno reescrito.

### `dosekbp preprocess` — DICOM → NPZ

```bash
# Próstata normofraccionado (default --anatomy prostate)
dosekbp preprocess --dicom-root <carpeta_dicom> --output-dir <salida> --splits data/splits/splits_v1.json --workers 4

# Próstata hipofraccionado (usa preprocess_hipo.py automáticamente al detectar --dicom-root-primario)
dosekbp preprocess --anatomy prostate_hipo --dicom-root-primario <dir1> --dicom-root-fallback <dir2> --csv <csv> --output-dir <salida> --splits <splits.json>

# Anatomía nueva
dosekbp preprocess --anatomy lung --dicom-root <carpeta_dicom> --output-dir <salida> --splits <splits.json>
```

`--anatomy` selecciona el YAML de `configs/anatomy/`. Default `"prostate"` (retrocompatible con invocaciones sin el flag).

### `dosekbp train` — entrenar un modelo

```bash
dosekbp train --config configs/exp_hipo_004_finetune_ctfix_v4.yaml
dosekbp train --config configs/exp_lung_toy_example.yaml --fast-dev-run   # smoke test rápido
```

La anatomía viaja **implícita** en `--config`, vía la clave opcional `anatomy: <nombre>` del YAML de experimento (default `"prostate"` si no la declara — los configs existentes de próstata no necesitan tocarse). No hay un `--anatomy` separado acá a propósito: evita entrenar con hipo y evaluar por error con otra anatomía.

### `dosekbp evaluate` — evaluar un checkpoint

```bash
dosekbp evaluate --dataset normo --checkpoint checkpoints/exp002_unet2d_psdm_ctfix_fov34/epoch=127.ckpt --config configs/exp002_unet2d_psdm_ctfix_fov34.yaml --output-dir results/mi_run
dosekbp evaluate --dataset hipo --exp exp_hipo_004_finetune_ctfix_v4 --checkpoint checkpoints/exp_hipo_004_finetune_ctfix_v4/last.ckpt
```

### `dosekbp predict` — inferencia de un paciente

```bash
dosekbp predict --patient-id PT_... --checkpoint <ckpt> --config <yaml>
```

## 3. Cómo declarar una anatomía nueva

1. Copiar `configs/anatomy/lung.yaml` (o `prostate.yaml`) como plantilla en `configs/anatomy/<mi_anatomia>.yaml`.
2. Completar:
   - `structures`: una entrada por estructura (`key`, `role` — `target`/`oar`/`body`—, `dicom_aliases` en orden de prioridad de autodetección, `required`, `input_channels: {mask, psdm}`). Exactamente una estructura debe tener `role: target` y coincidir con `primary_target`.
   - `primary_target`: la `key` de la estructura target (D95 de referencia, centroide de recorte).
   - `z_crop_roi_keys`: qué estructuras definen el recorte axial en Z.
   - `prescription`: `dose_gy` / `num_fractions`.
   - `preprocessing`: `ct_hu_min/max`, `inplane_size`, `z_margin_slices`, `psdm_norm_cm`, `inplane_crop_mm` — este último es el que más impacto tiene en el % de estructuras que entran completas en el recorte; calibrar sobre una muestra de pacientes reales antes de usar en serio (ver cómo se hizo para próstata: `data/preprocess.py`, comentario sobre 44 pacientes 2026-08-23).
   - `constraints`: opcional — si no se define acá, `evaluate.py` cae a este bloque solo si el experimento tampoco trae su propio `cfg.constraints`.
   - Si la anatomía es una variante de otra ya existente (ej. un protocolo distinto de la misma localización), usar `extends: <base>.yaml` y solo declarar los overrides — ver `configs/anatomy/prostate_hipo.yaml`.
3. Copiar `configs/exp_lung_toy_example.yaml` como plantilla de experimento, poner `anatomy: <mi_anatomia>` y `model.in_channels` = 1 (CT) + 1 (BODY, si `use_body_mask`) + N (una por estructura no-body con `input_channels.psdm: true`, si `use_psdm: true`).
4. `dosekbp preprocess --anatomy <mi_anatomia> ...` → `dosekbp train --config <mi_experimento>.yaml`.

Esto es exactamente lo que se validó en el Paso H de este refactor (`configs/anatomy/lung.yaml`, 7 estructuras, datos sintéticos): corrió preprocesamiento y entrenamiento completos sin tocar ningún `.py`.

## 4. Deuda técnica pendiente (no bloqueante, documentada a propósito)

| Deuda | Dónde | Por qué no se resolvió en este refactor |
|---|---|---|
| `data/preprocess_hipo.py` sigue reimplementando `procesar_paciente_hipo` en vez de reusar el `procesar_paciente()` genérico | `data/preprocess_hipo.py` | Es la parte de mayor riesgo (tiene `resolver_nombre_ptv()` por-paciente vía CSV, sin equivalente en el schema estático) — fusionar exige su propio ciclo de regresión dedicado sobre el dataset hipo completo, no solo una muestra |
| `scripts/evaluate.py` — el loop principal sigue accediendo `batch["ptv_mask"]`/`rectum_mask`/`bladder_mask` y `dvh_score_openkbp(..., ["ptv","rectum","bladder"])` hardcodeado | `scripts/evaluate.py` | Solo se generalizaron `evaluar_constraints()` y `PRESCRIPCION_GY_NORMO` (lo que pedía el plan); generalizar el resto es un cambio de mayor superficie sobre un script de ~700 líneas, deliberadamente fuera de alcance |
| `OPERATIONAL_CONSTRAINTS` de `scripts/evaluate_hipo.py` sin generalizar | `scripts/evaluate_hipo.py` | Está "congelado" (calibra el punto de operación en val) — cualquier cambio ahí necesita su propia regresión con bootstrap incluido, se decidió no tocarlo |
| `preprocess.py --help`/algunos comentarios con flechas Unicode (`→`) rompían en consola cp1252 | `data/preprocess.py` | Corregido solo en la línea que efectivamente rompía `--help` (la `description=` del parser); las flechas en comentarios/docstrings internos no se tocaron (no afectan ejecución) |

## 5. Roadmap etapa2 — interfaz (solo boceto, no implementado)

Alcance mínimo propuesto para una interfaz gráfica sobre esta CLI:

- **Streamlit** (no Flask/`tomografo_tool`, que es para otro propósito — monitoreo clínico en producción, no entrenamiento). Streamlit permite armar rápido: selector de anatomía (lee `configs/anatomy/*.yaml`), selector/editor de config de experimento, botón para lanzar `preprocess`/`train`/`evaluate` como subproceso, y un panel de progreso.
- Reusar `WandbLogger` ya existente para gráficos de entrenamiento en vivo (no reimplementar plotting) — la interfaz solo necesita embeber o linkear el dashboard de W&B del run.
- Página de resultados: leer `results/<exp>_test*/summary.json` (normo) o `metrics_summary.json` (hipo) y mostrar tablas/figuras ya generadas por `evaluate.py`/`evaluate_hipo.py`, sin recalcular nada.
- Esto es un boceto de arquitectura para decidir con Pablo cuándo se prioriza, no un compromiso de este plan.
