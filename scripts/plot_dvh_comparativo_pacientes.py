"""
Plots de DVH comparativo (U-Net vs. predictor RapidPlan) para un puñado de
pacientes elegidos por nivel de acuerdo -- por defecto 2 "malos" (mayor
discrepancia), 2 "medios" y 2 "buenos" (menor discrepancia), rankeados por el
promedio de mean_abs_delta_V_global_pp entre Rectum y Bladder ya calculado en
results/comparacion_rapidplan_normo_test59/comparacion_por_paciente.csv
(scripts/run_batch_comparacion_rapidplan.py).

1 figura por paciente, Rectum y Bladder lado a lado (curva U-Net vs. curva
RapidPlan superpuestas), con el valor de mean|ΔV| de cada estructura en el
titulo.

Uso (default -- 2 malos/2 medios/2 buenos automaticos):
    .venv/Scripts/python.exe scripts/plot_dvh_comparativo_pacientes.py

Uso (pacientes especificos):
    .venv/Scripts/python.exe scripts/plot_dvh_comparativo_pacientes.py \
        --anonids PT_xxx PT_yyy
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from analisis_angular import cargar_npz, inferir_dosis  # noqa: E402
from evaluate import cargar_modelo  # noqa: E402
from compute_pred_dvh import dvh_desde_dose_pred  # noqa: E402
from parse_rapidplan_csv import cargar_csv, pacientes_con_plan_unico, linea_estructura  # noqa: E402
from compare_dvh_unet_vs_rapidplan import interpolar_vol_pct  # noqa: E402
from dvh_curva_completa import D_BINS  # noqa: E402

# Defaults -- exp002 "viejo" (pre CT-fix), YA NO EXISTEN en disco al 2026-08-24
# (ver nota igual en run_batch_comparacion_rapidplan.py / memoria del proyecto,
# project_ct_channel_corrupted_wrong_series). Pasar --checkpoint/--config/
# --processed-dir explicitos apuntando a exp002_unet2d_psdm_ctfix_fov34 una vez
# que termine de entrenar.
CHECKPOINT_DEFAULT = _REPO_ROOT / "checkpoints/exp002_unet2d_psdm/epoch=191.ckpt"
CONFIG_YAML_DEFAULT = _REPO_ROOT / "configs/exp002_unet2d_psdm.yaml"
CSV_RAPIDPLAN = _REPO_ROOT / "data/objetivos_optimizacion.csv"
COMPARACION_CSV_DEFAULT = _REPO_ROOT / "results/comparacion_rapidplan_normo_test59/comparacion_por_paciente.csv"
RX_GY = 78.0
OARS = ["Rectum", "Bladder"]


def elegir_pacientes_por_defecto(comparacion_csv: Path, n_por_grupo: int = 2) -> list:
    """Devuelve [(anonid, categoria), ...] -- 2 malos/2 medios/2 buenos por
    mean_abs_delta_V_global_pp promedio (Rectum+Bladder), ya calculado.

    OJO: usar el comparacion_por_paciente.csv de la MISMA corrida (mismo
    checkpoint/config) que se va a usar despues para las curvas -- si no, el
    ranking queda calculado con un modelo distinto al que se grafica."""
    df = pd.read_csv(comparacion_csv)
    piv = df.pivot(index="AnonID", columns="estructura", values="mean_abs_delta_V_global_pp")
    piv["combinado"] = piv[OARS].mean(axis=1)
    piv = piv.sort_values("combinado")

    buenos = piv.index[:n_por_grupo].tolist()
    malos = piv.index[-n_por_grupo:].tolist()
    mid = len(piv) // 2
    medios = piv.index[mid - n_por_grupo // 2: mid - n_por_grupo // 2 + n_por_grupo].tolist()

    elegidos = ([(pid, "bueno") for pid in buenos] +
                [(pid, "medio") for pid in medios] +
                [(pid, "malo") for pid in malos])
    for pid, cat in elegidos:
        print(f"  {cat:6s} {pid}: combinado={piv.loc[pid, 'combinado']:.2f}pp "
              f"(Rectum={piv.loc[pid, 'Rectum']:.2f} Bladder={piv.loc[pid, 'Bladder']:.2f})")
    return elegidos


def main(anonids=None, output_dir=None, checkpoint=None, config_yaml=None, processed_dir=None,
         comparacion_csv=None):
    output_dir = Path(output_dir) if output_dir else _REPO_ROOT / "results/comparacion_rapidplan_normo_test59/plots_pacientes"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint) if checkpoint else CHECKPOINT_DEFAULT
    config_yaml = Path(config_yaml) if config_yaml else CONFIG_YAML_DEFAULT
    comparacion_csv = Path(comparacion_csv) if comparacion_csv else COMPARACION_CSV_DEFAULT

    if anonids:
        elegidos = [(pid, "") for pid in anonids]
    else:
        print(f"Rankeando pacientes por mean|deltaV| combinado (Rectum+Bladder), fuente: {comparacion_csv}")
        elegidos = elegir_pacientes_por_defecto(comparacion_csv)

    df_csv = cargar_csv(CSV_RAPIDPLAN)
    plan_unico, _ = pacientes_con_plan_unico(df_csv)

    cfg = OmegaConf.load(str(config_yaml))
    PROCESSED_DIR = Path(processed_dir) if processed_dir else Path(cfg.data.processed_dir)
    cfg.data.processed_dir = str(PROCESSED_DIR)
    print(f"Checkpoint: {checkpoint}")
    print(f"Config: {config_yaml}")
    print(f"Processed dir: {PROCESSED_DIR}")
    torch.set_float32_matmul_precision("high")
    model, device = cargar_modelo(str(checkpoint), cfg)

    grid_gy = D_BINS * RX_GY / 100.0

    for anonid, categoria in elegidos:
        plan_id = plan_unico.get(anonid)
        if plan_id is None:
            print(f"[!] {anonid}: sin plan unico en el CSV, salteado")
            continue

        arrays, meta = cargar_npz(PROCESSED_DIR / f"{anonid}.npz")
        dose_pred_raw = inferir_dosis(model, device, arrays, pad_z_to=None)
        unet_dvh = dvh_desde_dose_pred(
            dose_pred_raw, arrays["ptv_mask"],
            {"Rectum": arrays["rectum_mask"], "Bladder": arrays["bladder_mask"]}, RX_GY,
        )

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, oar in zip(axes, OARS):
            linea_rp = linea_estructura(df_csv, anonid, oar, plan_id=plan_id)
            if linea_rp is None:
                ax.set_title(f"{oar} -- sin datos de RapidPlan")
                continue

            v_unet_grid = interpolar_vol_pct(unet_dvh[oar]["dose_gy"], unet_dvh[oar]["vol_pct"], grid_gy)
            v_rp_grid = interpolar_vol_pct(linea_rp["dose_gy"], linea_rp["vol_pct"], grid_gy)
            mean_abs_delta = float(np.mean(np.abs(v_unet_grid - v_rp_grid)))

            ax.plot(D_BINS, v_unet_grid, color="firebrick", label="U-Net (predictor)")
            ax.plot(D_BINS, v_rp_grid, color="steelblue", linestyle="--", label="RapidPlan (predictor)")
            ax.axvspan(0, 40, color="gray", alpha=0.06)
            ax.axvspan(40, 80, color="steelblue", alpha=0.04)
            ax.axvspan(80, 110, color="darkorange", alpha=0.06)
            ax.set_xlabel("Dosis (% Rx)")
            ax.set_ylabel("Volumen (%)")
            ax.set_title(f"{oar}  --  mean|ΔV|={mean_abs_delta:.2f}pp")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        etiqueta = f"[{categoria}] " if categoria else ""
        fig.suptitle(f"{etiqueta}{anonid}", fontsize=12, fontweight="bold")
        fig.tight_layout()
        nombre_archivo = f"dvh_{categoria}_{anonid}.png" if categoria else f"dvh_{anonid}.png"
        fig.savefig(str(output_dir / nombre_archivo), dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"Guardado: {output_dir / nombre_archivo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anonids", nargs="+", default=None,
                         help="Lista de AnonID especificos (default: 2 malos/2 medios/2 buenos automaticos)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None, help=f"default: {CHECKPOINT_DEFAULT}")
    parser.add_argument("--config", default=None, help=f"default: {CONFIG_YAML_DEFAULT}")
    parser.add_argument("--processed-dir", default=None, help="default: el processed_dir del --config")
    parser.add_argument("--comparacion-csv", default=None,
                         help=f"comparacion_por_paciente.csv para el ranking automatico "
                              f"(default: {COMPARACION_CSV_DEFAULT}) -- usar el de la MISMA corrida")
    args = parser.parse_args()
    main(anonids=args.anonids, output_dir=args.output_dir, checkpoint=args.checkpoint,
         config_yaml=args.config, processed_dir=args.processed_dir,
         comparacion_csv=args.comparacion_csv)
