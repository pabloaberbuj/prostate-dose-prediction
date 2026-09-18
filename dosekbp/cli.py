"""
CLI unificada del pipeline de predicción de dosis (KBP) — etapa1 (línea de comandos)
de la generalización a otras localizaciones anatómicas. Ver docs/CLI_INSTRUCTIVO.md.

Diseño: cada subcomando ENVUELVE el script existente (data/preprocess.py,
scripts/train.py, scripts/evaluate.py, scripts/predict_one.py) sin reescribir su
lógica interna — arma sys.argv con los argumentos recibidos y corre el main() del
script original vía runpy, exactamente como si se llamara `python <script> <args>`.
Esto minimiza el riesgo de romper el pipeline vigente (hipo v4): la lógica de negocio
no cambia, solo la capa de invocación.

Selección de anatomía (ver src/config/anatomy.py):
- `dosekbp preprocess`: flag explícito --anatomy (default "prostate") — es el primer
  paso del pipeline, todavía no existe un config de experimento.
- `dosekbp train` / `evaluate` / `predict`: la anatomía viaja IMPLÍCITA en --config
  (clave opcional `anatomy:` del YAML de experimento, default "prostate" si no la
  declara) — no se pide de nuevo para evitar inconsistencias (ej. entrenar con hipo
  y evaluar pisando sin querer con otra anatomía).

Uso:
    dosekbp preprocess --anatomy prostate --dicom-root <dir> --output-dir <dir> --splits <json>
    dosekbp preprocess --anatomy prostate_hipo --dicom-root-primario <dir> --dicom-root-fallback <dir> --csv <csv> --output-dir <dir> --splits <json>
    dosekbp train --config configs/exp_hipo_004_finetune_ctfix_v4.yaml
    dosekbp evaluate --dataset hipo --exp exp_hipo_004_finetune_ctfix_v4 --checkpoint <ckpt>
    dosekbp predict --patient-id PT_... --checkpoint <ckpt> --config <yaml>

Cada subcomando pasa `--help` directo al script envuelto (ej. `dosekbp train --help`
muestra el `--help` real de scripts/train.py) porque no reimplementamos sus flags.
"""

import runpy
import sys
from pathlib import Path
from typing import List

import typer

app = typer.Typer(
    name="dosekbp",
    help="Pipeline de predicción de dosis (KBP): preprocesado, entrenamiento, "
         "evaluación e inferencia — generalizado a cualquier localización anatómica "
         "declarada en configs/anatomy/.",
    add_completion=False,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}


def _run_script(rel_path: str, argv0: str, extra_args: List[str]) -> None:
    """
    Corre <repo_root>/<rel_path> como si fuera `python <rel_path> <extra_args>`:
    mismo sys.argv, mismo sys.path (agrega el directorio del script — necesario
    para los imports relativos que ya usan estos scripts, ej. evaluate_hipo.py
    importando de evaluate.py, o preprocess_hipo.py importando de preprocess.py),
    y corre su main() bajo `if __name__ == "__main__"` sin tocar una sola línea
    del script original.
    """
    script_path = _REPO_ROOT / rel_path
    if not script_path.exists():
        typer.echo(f"No se encontró el script envuelto: {script_path}", err=True)
        raise typer.Exit(code=1)

    sys.argv = [argv0] + list(extra_args)
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    runpy.run_path(str(script_path), run_name="__main__")


@app.command(context_settings=_PASSTHROUGH, add_help_option=False,
             help="Preprocesa DICOM -> NPZ (envuelve data/preprocess.py o "
                  "data/preprocess_hipo.py según --anatomy). Pasa --help para ver "
                  "los flags reales del script envuelto.")
def preprocess(
    ctx: typer.Context,
    anatomy: str = typer.Option(
        "prostate", "--anatomy",
        help="Nombre del anatomy schema (configs/anatomy/<nombre>.yaml). "
             "'prostate'/'prostate_hipo' usan data/preprocess.py; cualquier otro "
             "nombre también usa data/preprocess.py (genérico) salvo que se pase "
             "explícitamente --dicom-root-primario (formato hipo)."),
):
    extra = list(ctx.args)
    # preprocess_hipo.py tiene su propia interfaz (--dicom-root-primario/--fallback/
    # --csv) distinta de preprocess.py (--dicom-root) — se elige el script según qué
    # flags se pasaron, no según --anatomy, para no adivinar mal el formato de datos.
    usa_formato_hipo = any(a.startswith("--dicom-root-primario") for a in extra)
    if usa_formato_hipo:
        _run_script("data/preprocess_hipo.py", "preprocess_hipo.py",
                    ["--anatomy", anatomy] + extra)
    else:
        _run_script("data/preprocess.py", "preprocess.py",
                    ["--anatomy", anatomy] + extra)


@app.command(context_settings=_PASSTHROUGH, add_help_option=False,
             help="Entrena un modelo U-Net (envuelve scripts/train.py). Pasa --help "
                  "para ver los flags reales (--config, --fast-dev-run, --resume, "
                  "--init-weights, --no-wandb, --processed-dir).")
def train(ctx: typer.Context):
    _run_script("scripts/train.py", "train.py", ctx.args)


@app.command(context_settings=_PASSTHROUGH, add_help_option=False,
             help="Evalúa un checkpoint sobre el test set (envuelve scripts/evaluate.py). "
                  "Pasa --help para ver los flags reales (--dataset {normo,hipo}, "
                  "--checkpoint, --config, --output-dir, ...).")
def evaluate(ctx: typer.Context):
    _run_script("scripts/evaluate.py", "evaluate.py", ctx.args)


@app.command(context_settings=_PASSTHROUGH, add_help_option=False,
             help="Inferencia sobre un paciente (envuelve scripts/predict_one.py). "
                  "Pasa --help para ver los flags reales (--patient-id, --checkpoint, "
                  "--config, --processed-dir, --output-dir).")
def predict(ctx: typer.Context):
    _run_script("scripts/predict_one.py", "predict_one.py", ctx.args)


if __name__ == "__main__":
    app()
