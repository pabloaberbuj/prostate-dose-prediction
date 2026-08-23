"""
Batch: DVH predicho por el U-Net vs. DVH-predictor de RapidPlan (el objetivo
LINEA que RapidPlan estima antes de optimizar), sobre TODO el test set normo
(n=59, splits_v1.json), usando el CSV completo de objetivos extraido via ESAPI
(insumos temporales/objetivos_optimizacion.csv, 387 pacientes).

Extiende el piloto de 1 paciente (compute_pred_dvh.py + compare_dvh_unet_vs_rapidplan.py)
a todo el cohorte, reusando exactamente el mismo codigo (dvh_desde_dose_pred,
interpolar_vol_pct) -- no hay logica nueva de calculo de DVH, solo el loop +
la lectura del CSV en vez del XML/txt de a 1 paciente.

Mismas bandas de dosis (%Rx) y grilla que dvh_curva_completa.py (D_BINS,
IDX_BAJA/MEDIA/ALTA) para que los numeros sean comparables al resto del
proyecto.

Uso:
    .venv/Scripts/python.exe scripts/run_batch_comparacion_rapidplan.py \
        --csv "../insumos temporales/objetivos_optimizacion.csv" \
        --output-dir results/comparacion_rapidplan_normo_test59
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
from dvh_curva_completa import D_BINS, IDX_BAJA, IDX_MEDIA, IDX_ALTA  # noqa: E402

# Defaults de analisis_angular.py/dvh_curva_completa.py apuntan al modelo HIPO
# (exp_hipo_002b_finetune_clean) -- este batch es sobre el test set NORMO
# (splits_v1.json, Rx=78Gy), asi que hay que forzar el checkpoint/config/
# processed_dir de exp002 explicitamente, no heredar esos defaults.
CHECKPOINT = _REPO_ROOT / "checkpoints/exp002_unet2d_psdm/epoch=191.ckpt"
CONFIG_YAML = _REPO_ROOT / "configs/exp002_unet2d_psdm.yaml"
PROCESSED_DIR = Path("C:/Pablo/ProstateDoseProject/processed")

SPLITS_PATH = _REPO_ROOT / "data/splits/splits_v1.json"
RX_GY = 78.0
OARS = ["Rectum", "Bladder"]


def main(csv_path=None, output_dir=None):
    csv_path = Path(csv_path) if csv_path else _REPO_ROOT / "../insumos temporales/objetivos_optimizacion.csv"
    output_dir = Path(output_dir) if output_dir else _REPO_ROOT / "results/comparacion_rapidplan_normo_test59"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(SPLITS_PATH) as f:
        ids_test = json.load(f)["test"]
    print(f"Pacientes de test (normo): {len(ids_test)}")

    df_csv = cargar_csv(csv_path)
    plan_unico, ambiguos = pacientes_con_plan_unico(df_csv)
    ids_ambiguos_en_test = [pid for pid in ids_test if pid in ambiguos]
    if ids_ambiguos_en_test:
        print(f"[!] {len(ids_ambiguos_en_test)} paciente(s) de test con >1 plan en el CSV "
              f"(ambiguo, salteados): {ids_ambiguos_en_test}")

    cfg = OmegaConf.load(str(CONFIG_YAML))
    cfg.data.processed_dir = str(PROCESSED_DIR)
    torch.set_float32_matmul_precision("high")
    model, device = cargar_modelo(str(CHECKPOINT), cfg)

    filas = []
    curvas_por_oar = {oar: [] for oar in OARS}  # lista de (v_unet_grid, v_rp_grid) por paciente
    salteados = []

    for anonid in ids_test:
        if anonid in ambiguos:
            continue
        if anonid not in plan_unico:
            salteados.append((anonid, "no aparece en el CSV"))
            continue
        plan_id = plan_unico[anonid]

        arrays, meta = cargar_npz(PROCESSED_DIR / f"{anonid}.npz")
        dose_pred_raw = inferir_dosis(model, device, arrays, pad_z_to=None)

        unet_dvh = dvh_desde_dose_pred(
            dose_pred_raw, arrays["ptv_mask"],
            {"Rectum": arrays["rectum_mask"], "Bladder": arrays["bladder_mask"]}, RX_GY,
        )
        factor_renorm = unet_dvh["_meta"]["factor_renorm_aplicado"]

        grid_gy = D_BINS * RX_GY / 100.0

        for oar in OARS:
            linea_rp = linea_estructura(df_csv, anonid, oar, plan_id=plan_id)
            if linea_rp is None or len(linea_rp["dose_gy"]) == 0:
                salteados.append((anonid, f"sin linea RapidPlan para {oar}"))
                continue

            v_unet_grid = interpolar_vol_pct(unet_dvh[oar]["dose_gy"], unet_dvh[oar]["vol_pct"], grid_gy)
            v_rp_grid = interpolar_vol_pct(linea_rp["dose_gy"], linea_rp["vol_pct"], grid_gy)
            delta = v_unet_grid - v_rp_grid

            curvas_por_oar[oar].append((v_unet_grid, v_rp_grid))
            filas.append({
                "AnonID": anonid, "estructura": oar, "plan_id_rapidplan": plan_id,
                "factor_renorm_unet": factor_renorm,
                "mean_abs_delta_V_global_pp": float(np.mean(np.abs(delta))),
                "mean_abs_delta_V_baja_0_40pctRx_pp": float(np.mean(np.abs(delta[IDX_BAJA]))),
                "mean_abs_delta_V_media_40_80pctRx_pp": float(np.mean(np.abs(delta[IDX_MEDIA]))),
                "mean_abs_delta_V_alta_80_110pctRx_pp": float(np.mean(np.abs(delta[IDX_ALTA]))),
                "signed_mean_delta_V_pp": float(np.mean(delta)),
            })

        print(f"  {anonid}: factor_renorm={factor_renorm:.4f}")

    if salteados:
        print(f"\n[!] {len(salteados)} caso(s) salteado(s):")
        for s in salteados:
            print(f"  {s}")

    df = pd.DataFrame(filas)
    df.to_csv(output_dir / "comparacion_por_paciente.csv", index=False)
    print(f"\nGuardado: {output_dir / 'comparacion_por_paciente.csv'} ({len(df)} filas)")

    resumen_por_estructura = {}
    for oar in OARS:
        sub = df[df["estructura"] == oar]
        if sub.empty:
            continue
        resumen_por_estructura[oar] = {
            "n_pacientes": int(len(sub)),
            "mean_abs_delta_V_global_pp":      {"mean": float(sub["mean_abs_delta_V_global_pp"].mean()),
                                                  "std": float(sub["mean_abs_delta_V_global_pp"].std())},
            "mean_abs_delta_V_baja_0_40pctRx_pp":  {"mean": float(sub["mean_abs_delta_V_baja_0_40pctRx_pp"].mean()),
                                                      "std": float(sub["mean_abs_delta_V_baja_0_40pctRx_pp"].std())},
            "mean_abs_delta_V_media_40_80pctRx_pp": {"mean": float(sub["mean_abs_delta_V_media_40_80pctRx_pp"].mean()),
                                                       "std": float(sub["mean_abs_delta_V_media_40_80pctRx_pp"].std())},
            "mean_abs_delta_V_alta_80_110pctRx_pp": {"mean": float(sub["mean_abs_delta_V_alta_80_110pctRx_pp"].mean()),
                                                       "std": float(sub["mean_abs_delta_V_alta_80_110pctRx_pp"].std())},
            "signed_mean_delta_V_pp":          {"mean": float(sub["signed_mean_delta_V_pp"].mean()),
                                                  "std": float(sub["signed_mean_delta_V_pp"].std())},
        }

    summary = {
        "n_test_total": len(ids_test),
        "csv_origen": str(csv_path),
        "rx_gy": RX_GY,
        "ambiguos_salteados": ids_ambiguos_en_test,
        "otros_salteados": salteados,
        "d_bins_pct_rx": {"min": 0, "max": 110, "paso": 1},
        "resumen_por_estructura": resumen_por_estructura,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Guardado: {output_dir / 'summary.json'}")

    # Plot: media +/- std de ambas curvas por estructura, sobre todos los pacientes.
    for oar in OARS:
        curvas = curvas_por_oar[oar]
        if not curvas:
            continue
        v_unet_all = np.array([c[0] for c in curvas])
        v_rp_all = np.array([c[1] for c in curvas])
        delta_all = v_unet_all - v_rp_all

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        ax.plot(D_BINS, v_unet_all.mean(axis=0), color="firebrick", label="U-Net (predictor, media)")
        ax.fill_between(D_BINS, v_unet_all.mean(axis=0) - v_unet_all.std(axis=0),
                         v_unet_all.mean(axis=0) + v_unet_all.std(axis=0), color="firebrick", alpha=0.15)
        ax.plot(D_BINS, v_rp_all.mean(axis=0), color="steelblue", linestyle="--",
                 label="RapidPlan (predictor, media)")
        ax.fill_between(D_BINS, v_rp_all.mean(axis=0) - v_rp_all.std(axis=0),
                         v_rp_all.mean(axis=0) + v_rp_all.std(axis=0), color="steelblue", alpha=0.15)
        ax.axvspan(0, 40, color="gray", alpha=0.08)
        ax.axvspan(40, 80, color="steelblue", alpha=0.05)
        ax.axvspan(80, 110, color="darkorange", alpha=0.08)
        ax.set_xlabel("Dosis (% Rx)")
        ax.set_ylabel("Volumen (%)")
        ax.set_title(f"{oar} -- DVH predicho (n={len(curvas)}): U-Net vs. RapidPlan")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax2 = axes[1]
        ax2.plot(D_BINS, delta_all.mean(axis=0), color="darkorange")
        ax2.fill_between(D_BINS, delta_all.mean(axis=0) - delta_all.std(axis=0),
                          delta_all.mean(axis=0) + delta_all.std(axis=0), color="darkorange", alpha=0.2)
        ax2.axhline(0, color="gray", linewidth=0.8)
        ax2.axvspan(0, 40, color="gray", alpha=0.08, label="banda baja")
        ax2.axvspan(40, 80, color="steelblue", alpha=0.05, label="banda media")
        ax2.axvspan(80, 110, color="darkorange", alpha=0.08, label="banda alta")
        ax2.set_xlabel("Dosis (% Rx)")
        ax2.set_ylabel("ΔV = V_UNet − V_RapidPlan (pp)")
        ax2.set_title(f"{oar} -- diferencia media ± std (n={len(curvas)})")
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(str(plots_dir / f"dvh_medio_{oar.lower()}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)

    print("\n=== RESUMEN mean|deltaV| por estructura (pp de volumen, U-Net vs RapidPlan predictor) ===")
    for oar, r in resumen_por_estructura.items():
        print(f"{oar:8s} (n={r['n_pacientes']}): global={r['mean_abs_delta_V_global_pp']['mean']:.2f}"
              f"±{r['mean_abs_delta_V_global_pp']['std']:.2f}  "
              f"baja={r['mean_abs_delta_V_baja_0_40pctRx_pp']['mean']:.2f}  "
              f"media={r['mean_abs_delta_V_media_40_80pctRx_pp']['mean']:.2f}  "
              f"alta={r['mean_abs_delta_V_alta_80_110pctRx_pp']['mean']:.2f}  "
              f"signed={r['signed_mean_delta_V_pp']['mean']:.2f}±{r['signed_mean_delta_V_pp']['std']:.2f}")

    return df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    main(csv_path=args.csv, output_dir=args.output_dir)
