"""Extras para el handoff del Paso 6 (exp_hipo_003_finetune_v3), que evaluate_hipo.py
no genera automaticamente:

  1. Metricas del subgrupo "normos replanificados como hipo" (Status=='UnApproved')
     dentro del test set: MAE por estructura (bootstrap CI), y confusion matrix por
     constraint operativo usando el umbral YA calibrado en val por evaluate_hipo.py
     (columnas Flag_*_pred de per_patient_metrics.csv — no se recalibra nada).
  2. Figuras DVH+slice para 2 casos no-cumplidores de test (any_op_fail_real==1),
     ademas de las mediana/mejor/peor por MAE que ya genera evaluate_hipo.py.

Uso:
    python scripts/handoff_extras_hipo_v3.py \
        --results-dir results/exp_hipo_003_finetune_v3_test_hipo_v3 \
        --config configs/exp_hipo_003_finetune_v3.yaml \
        --checkpoint checkpoints/exp_hipo_003_finetune_v3/epoch=XXX.ckpt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.datamodules.dose_datamodule import DoseDataModule  # noqa: E402
from evaluate import cargar_modelo, figura_paciente  # noqa: E402
from evaluate_hipo import bootstrap_ci, confusion_binaria, OPERATIONAL_CONSTRAINTS  # noqa: E402


def subgroup_metrics(df: pd.DataFrame, summary: dict, n_bootstrap: int, seed: int) -> dict:
    """Metricas de capa 1 (MAE) y capa 3 (constraints) para un subgrupo del test set.
    Capa 3 usa el umbral operativo YA CALIBRADO en val por evaluate_hipo.py (leido de
    metrics_summary.json) -- no se recalibra nada sobre el subgrupo, evita circularidad."""
    rng = np.random.default_rng(seed)
    out = {"n": int(len(df))}
    if len(df) == 0:
        return out

    out["mae"] = {
        struct: {
            "mean": float(df[f"mae_{struct}"].mean()),
            "std": float(df[f"mae_{struct}"].std()),
            "ci95": bootstrap_ci(df[f"mae_{struct}"].to_numpy(), n_bootstrap, rng),
        }
        for struct in ["body", "ptv", "rectum", "bladder"]
    }

    out["constraints"] = {}
    capa3 = summary.get("capa3_constraints_operativos", {})
    for tag, cfg_c in OPERATIONAL_CONSTRAINTS.items():
        struct, punto, umbral_clinico = cfg_c["struct"], cfg_c["punto"], cfg_c["umbral_clinico"]
        col_real, col_pred = f"{struct}_{punto}_real", f"{struct}_{punto}_pred"
        if col_real not in df.columns or tag not in capa3:
            continue
        umbral_operativo = capa3[tag]["umbral_operativo_val"]
        y_real = df[col_real].to_numpy()
        y_pred = df[col_pred].to_numpy()
        y_true_fail = (y_real >= umbral_clinico).astype(int)
        y_pred_fail_operativo = (y_pred >= umbral_operativo).astype(int)

        entry = {
            "n_fail_real": int(y_true_fail.sum()),
            "umbral_operativo_val_heredado": umbral_operativo,
            "confusion_umbral_operativo": confusion_binaria(y_true_fail, y_pred_fail_operativo),
        }
        if len(set(y_true_fail.tolist())) > 1:
            try:
                entry["auc"] = float(roc_auc_score(y_true_fail, y_pred))
            except Exception as e:
                entry["auc"] = None
                entry["auc_error"] = str(e)
        else:
            entry["auc"] = None
            entry["auc_note"] = "solo una clase presente en el subgrupo (todo cumple o todo falla) -- AUC no definida"
        out["constraints"][tag] = entry
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    df = pd.read_csv(results_dir / "per_patient_metrics.csv")
    summary = json.loads((results_dir / "metrics_summary.json").read_text())

    # ── 1. Subgrupo UnApproved (normos replanificados como hipo) ──
    sub = df[df["Status"] == "UnApproved"].reset_index(drop=True)
    print(f"Subgrupo UnApproved en test: n={len(sub)} / {len(df)}")
    metrics_sub = subgroup_metrics(sub, summary, args.n_bootstrap, args.bootstrap_seed)
    with open(results_dir / "subgroup_unapproved_metrics.json", "w") as f:
        json.dump(metrics_sub, f, indent=2)
    print(f"Guardado: {results_dir / 'subgroup_unapproved_metrics.json'}")

    # ── 2. Figuras DVH para 2 casos no-cumplidores ──
    no_cumple = df[df["any_op_fail_real"] == 1].sort_values("mae_body")
    if len(no_cumple) == 0:
        print("No hay casos no-cumplidores en test — se omiten figuras extra.")
        return
    elegidos = no_cumple.iloc[[0, len(no_cumple) // 2]] if len(no_cumple) > 1 else no_cumple.iloc[[0]]
    anonids_elegidos = {
        row["AnonID"]: f"nocumple{i+1}_mae{row['mae_body']:.2f}_{row['Status']}"
        for i, (_, row) in enumerate(elegidos.iterrows())
    }
    print(f"Casos no-cumplidores elegidos para figura: {anonids_elegidos}")

    cfg = OmegaConf.load(args.config)
    cfg.data.cache_train = False
    cfg.data.cache_val = False
    model, device = cargar_modelo(args.checkpoint, cfg)

    dm = DoseDataModule(cfg)
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    for batch in tqdm(test_loader, desc="Figuras no-cumplidores"):
        anonid = batch["anonid"][0] if isinstance(batch["anonid"], list) else batch["anonid"]
        if isinstance(anonid, (list, tuple)):
            anonid = anonid[0]
        if anonid not in anonids_elegidos:
            continue
        batch_gpu = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            x = model._build_input(batch_gpu)
            pred = model(x)
        figura_paciente(
            anonid,
            batch["ct"][0].numpy(),
            batch["dose"][0].numpy(),
            pred[0].cpu().numpy(),
            batch["body_mask"][0].numpy(),
            batch["ptv_mask"][0].numpy(),
            batch["rectum_mask"][0].numpy(),
            batch["bladder_mask"][0].numpy(),
            plots_dir / f"dvh_slice_{anonids_elegidos[anonid]}_{anonid}.png",
        )
    print(f"Figuras guardadas en {plots_dir}")


if __name__ == "__main__":
    main()
