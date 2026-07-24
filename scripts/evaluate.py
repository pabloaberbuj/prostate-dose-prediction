"""
Evaluación del modelo entrenado en el test set.

Genera:
  - test_metrics.csv : una fila por paciente con todas las métricas (image, DVH, constraints)
  - figures/         : una imagen por paciente con DVH comparativo + cortes axiales
  - summary.json     : estadísticas agregadas (mean ± std, percentiles, tasa de acuerdo)
  - dvh_score.json   : OpenKBP-style dose score y DVH score

Uso:
    python scripts/evaluate.py \
        --checkpoint checkpoints/exp001_unet2d_baseline/epoch=122.ckpt \
        --config     configs/exp001_unet2d_baseline.yaml \
        --output-dir results/exp001_test
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datamodules.dose_datamodule import DoseDataModule
from src.models.lightning_module import DosePredictionModule


# ──────────────────────────────────────────────────────────────────────────────
# Cálculo de métricas DVH
# ──────────────────────────────────────────────────────────────────────────────

def dvh_metrics(dose: np.ndarray, mask: np.ndarray) -> dict:
    """Calcula métricas estándar de DVH sobre la región de la máscara.

    Args:
        dose: array (Z, H, W) en % de prescripción.
        mask: array (Z, H, W) binario.

    Returns:
        dict con D95, D98, D99, D2, Dmax, Dmean, V70, V65, V60, V50, V75, V78, V95, V100.
    """
    roi = dose[mask > 0]
    if len(roi) == 0:
        return {k: float("nan") for k in [
            "D95", "D98", "D99", "D2", "Dmax", "Dmean",
            "V50", "V60", "V65", "V70", "V75", "V78", "V95", "V100",
        ]}

    return {
        # Dx (dosis que recibe el x% del volumen)
        "D95":  float(np.percentile(roi, 5)),
        "D98":  float(np.percentile(roi, 2)),
        "D99":  float(np.percentile(roi, 1)),
        "D2":   float(np.percentile(roi, 98)),
        "Dmax": float(roi.max()),
        "Dmean": float(roi.mean()),
        # Vx (% volumen que recibe >= x% prescripción)
        # 70 Gy = 89.7% de la prescripción de 78 Gy
        "V50":  float(100.0 * (roi >= 64.1).sum() / len(roi)),  # 50 Gy
        "V60":  float(100.0 * (roi >= 76.9).sum() / len(roi)),  # 60 Gy
        "V65":  float(100.0 * (roi >= 83.3).sum() / len(roi)),  # 65 Gy
        "V70":  float(100.0 * (roi >= 89.7).sum() / len(roi)),  # 70 Gy
        "V75":  float(100.0 * (roi >= 96.2).sum() / len(roi)),  # 75 Gy
        "V78":  float(100.0 * (roi >= 100.0).sum() / len(roi)), # 78 Gy
        "V95":  float(100.0 * (roi >= 95.0).sum() / len(roi)),
        "V100": float(100.0 * (roi >= 100.0).sum() / len(roi)),
    }


def evaluar_constraints(metricas_predichas: dict, constraints_cfg) -> dict:
    """Evalúa cumplimiento de constraints clínicos a partir de las métricas predichas."""
    out = {}
    # Recto
    if "rectum" in metricas_predichas:
        r = metricas_predichas["rectum"]
        if constraints_cfg.rectum.v70_max_pct is not None:
            out["rectum_V70_cumple"] = r["V70"] <= constraints_cfg.rectum.v70_max_pct
        if constraints_cfg.rectum.v65_max_pct is not None:
            out["rectum_V65_cumple"] = r["V65"] <= constraints_cfg.rectum.v65_max_pct
    # Vejiga
    if "bladder" in metricas_predichas:
        b = metricas_predichas["bladder"]
        if constraints_cfg.bladder.v70_max_pct is not None:
            out["bladder_V70_cumple"] = b["V70"] <= constraints_cfg.bladder.v70_max_pct
        if constraints_cfg.bladder.v65_max_pct is not None:
            out["bladder_V65_cumple"] = b["V65"] <= constraints_cfg.bladder.v65_max_pct
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Figura por paciente
# ──────────────────────────────────────────────────────────────────────────────

def figura_paciente(anonid: str, ct: np.ndarray, dose_real: np.ndarray,
                    dose_pred: np.ndarray, body: np.ndarray, ptv: np.ndarray,
                    rectum: np.ndarray, bladder: np.ndarray, output_path: Path):
    """Genera figura con DVH comparativo + corte central (real, pred, diferencia)."""
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ── Corte central
    slices_ptv = np.where(ptv.sum(axis=(1, 2)) > 0)[0]
    z = slices_ptv[len(slices_ptv) // 2] if len(slices_ptv) > 0 else ct.shape[0] // 2

    for col, (arr, titulo, vmin, vmax, cmap) in enumerate([
        (dose_real,                   "Real",       0, 110, "jet"),
        (dose_pred,                   "Predicha",   0, 110, "jet"),
        (dose_pred - dose_real,       "Pred − Real", -20, 20, "RdBu_r"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(ct[z], cmap="gray", vmin=-1, vmax=1, aspect="equal")
        arr_masked = np.where(body[z] > 0, arr[z], np.nan)
        im = ax.imshow(arr_masked, cmap=cmap, vmin=vmin, vmax=vmax,
                       alpha=0.6, aspect="equal")
        # Contornos
        for mask_v, color in [(ptv[z], "yellow"), (rectum[z], "red"), (bladder[z], "cyan")]:
            if mask_v.sum() > 0:
                ax.contour(mask_v, levels=[0.5], colors=[color], linewidths=0.7)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"{titulo} (z={z})", fontsize=10)
        ax.axis("off")

    # ── DVH
    ax_dvh = fig.add_subplot(gs[1, :])
    bins = np.linspace(0, 130, 200)
    for nombre, mask_v, color in [
        ("PTV",     ptv,     "blue"),
        ("Rectum",  rectum,  "red"),
        ("Bladder", bladder, "cyan"),
    ]:
        if mask_v.sum() == 0:
            continue
        roi_real = dose_real[mask_v > 0]
        roi_pred = dose_pred[mask_v > 0]
        vol_real = np.array([100.0 * (roi_real >= b).sum() / len(roi_real) for b in bins])
        vol_pred = np.array([100.0 * (roi_pred >= b).sum() / len(roi_pred) for b in bins])
        ax_dvh.plot(bins, vol_real, color=color, linewidth=1.6, label=f"{nombre} real")
        ax_dvh.plot(bins, vol_pred, color=color, linewidth=1.6, linestyle="--",
                    label=f"{nombre} pred")
    ax_dvh.axvline(95, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_dvh.axvline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_dvh.set_xlabel("Dosis (% prescripción)", fontsize=10)
    ax_dvh.set_ylabel("Volumen (%)", fontsize=10)
    ax_dvh.set_title(f"DVH comparativo — {anonid}", fontsize=11, fontweight="bold")
    ax_dvh.legend(fontsize=8, loc="upper right")
    ax_dvh.set_xlim(0, 130)
    ax_dvh.set_ylim(0, 105)
    ax_dvh.grid(alpha=0.3)

    plt.savefig(str(output_path), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# OpenKBP-style scores
# ──────────────────────────────────────────────────────────────────────────────

def dose_score_openkbp(dose_real: np.ndarray, dose_pred: np.ndarray,
                       body: np.ndarray) -> float:
    """MAE global dentro del BODY (en % de prescripción)."""
    diff = np.abs(dose_pred - dose_real) * body
    return float(diff.sum() / max(body.sum(), 1))


def dvh_score_openkbp(real_dvh: dict, pred_dvh: dict, structures: list) -> float:
    """Promedio de |Δ| sobre métricas DVH clave."""
    errs = []
    for s in structures:
        if s not in real_dvh or s not in pred_dvh:
            continue
        # PTV: D1, D95, D99
        # OARs: Dmean, D0.1cc proxy → usamos Dmax
        if s == "ptv":
            errs.append(abs(real_dvh[s]["D2"]   - pred_dvh[s]["D2"]))
            errs.append(abs(real_dvh[s]["D95"]  - pred_dvh[s]["D95"]))
            errs.append(abs(real_dvh[s]["D99"]  - pred_dvh[s]["D99"]))
        else:
            errs.append(abs(real_dvh[s]["Dmean"] - pred_dvh[s]["Dmean"]))
            errs.append(abs(real_dvh[s]["Dmax"]  - pred_dvh[s]["Dmax"]))
    return float(np.mean(errs)) if errs else float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Carga de modelo (compartida entre modo normo y modo hipo)
# ──────────────────────────────────────────────────────────────────────────────

def cargar_modelo(checkpoint_path: str, cfg):
    """Carga el checkpoint del modelo y lo mueve al dispositivo disponible.

    Patch temporal: PyTorch 2.6+ cambió weights_only=True por default, rompiendo
    checkpoints con OmegaConf. Nuestros checkpoints son de confianza.
    """
    _orig_torch_load = torch.load
    def _load_no_weights_only(*a, **kw):
        kw['weights_only'] = False
        return _orig_torch_load(*a, **kw)
    torch.load = _load_no_weights_only

    print(f"Cargando checkpoint: {checkpoint_path}")
    model = DosePredictionModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Dispositivo: {device}")
    return model, device


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["normo", "hipo"], default="normo",
                        help="normo: dataset/metricas original (default, sin cambios). "
                             "hipo: dataset hipofraccionado, ver scripts/evaluate_hipo.py")
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--config",        default=None,
                        help="Requerido en --dataset normo. En --dataset hipo, default configs/<exp>.yaml")
    parser.add_argument("--output-dir",    default=None,
                        help="Requerido en --dataset normo. En --dataset hipo, default results/<exp>_test_hipo")
    parser.add_argument("--exp",           default=None,
                        help="Nombre del experimento (modo hipo: define defaults de --config/--output-dir)")
    parser.add_argument("--processed-dir", default=None,
                        help="Override cfg.data.processed_dir (ruta a NPZs, modo normo)")
    # ---- Modo hipo (ver evaluate_hipo.py) ----
    parser.add_argument("--processed-dir-hipo", default="C:/Pablo/ProstateDoseProject/processed_hipo")
    parser.add_argument("--splits-hipo",        default="data/splits/splits_hipo_v1.json")
    parser.add_argument("--gt-dvh-csv-hipo",    default="data/gt_dvh_hipo_256.csv")
    parser.add_argument("--metricas-csv-hipo",  default="c:/Pablo/ProstateDoseProject/dicoms hipofx/metricas_planes_hipofx_D95norm.csv",
                        help="CSV con NArcos por paciente (join por AnonID) para el desglose por geometria de arcos")
    parser.add_argument("--n-bootstrap",        type=int, default=1000)
    parser.add_argument("--bootstrap-seed",     type=int, default=42)
    args = parser.parse_args()

    if args.dataset == "hipo":
        from evaluate_hipo import run_hipo_evaluation
        run_hipo_evaluation(args)
        return

    if args.config is None or args.output_dir is None:
        parser.error("--config y --output-dir son obligatorios con --dataset normo")

    cfg = OmegaConf.load(args.config)

    # Overrides para evaluación: sin cache (no necesitamos train/val en RAM)
    cfg.data.cache_train = False
    cfg.data.cache_val   = False
    if args.processed_dir is not None:
        cfg.data.processed_dir = args.processed_dir

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.experiment.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # ── Cargar modelo
    model, device = cargar_modelo(args.checkpoint, cfg)

    # ── DataModule (solo necesitamos test)
    dm = DoseDataModule(cfg)
    dm.setup(stage="test")

    # ── Iterar test set
    filas = []
    test_loader = dm.test_dataloader()
    print(f"\nEvaluando {len(dm.test_ds)} pacientes de test...")

    for batch in tqdm(test_loader, desc="Test"):
        anonid = batch["anonid"][0] if isinstance(batch["anonid"], list) else batch["anonid"]
        if isinstance(anonid, (list, tuple)):
            anonid = anonid[0]

        # Mover batch a GPU
        batch_gpu = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

        with torch.no_grad():
            x = model._build_input(batch_gpu)
            pred = model(x)

        # A CPU/numpy (sacar dimensión batch=1)
        dose_pred = pred[0].cpu().numpy()
        dose_real = batch["dose"][0].numpy()
        ct        = batch["ct"][0].numpy()
        body      = batch["body_mask"][0].numpy()
        ptv       = batch["ptv_mask"][0].numpy()
        rectum    = batch["rectum_mask"][0].numpy()
        bladder   = batch["bladder_mask"][0].numpy()

        # Métricas
        m_real = {
            "ptv":     dvh_metrics(dose_real, ptv),
            "rectum":  dvh_metrics(dose_real, rectum),
            "bladder": dvh_metrics(dose_real, bladder),
        }
        m_pred = {
            "ptv":     dvh_metrics(dose_pred, ptv),
            "rectum":  dvh_metrics(dose_pred, rectum),
            "bladder": dvh_metrics(dose_pred, bladder),
        }

        # OpenKBP scores
        ds_val  = dose_score_openkbp(dose_real, dose_pred, body)
        dvh_val = dvh_score_openkbp(m_real, m_pred, ["ptv", "rectum", "bladder"])

        # Cumplimiento de constraints (real y predicho)
        cumple_real = evaluar_constraints(m_real, cfg.constraints)
        cumple_pred = evaluar_constraints(m_pred, cfg.constraints)

        # MAEs por estructura
        mae_body    = float((np.abs(dose_pred - dose_real) * body).sum()
                            / max(body.sum(), 1))
        mae_ptv     = float((np.abs(dose_pred - dose_real) * ptv).sum()
                            / max(ptv.sum(), 1))
        mae_rectum  = float((np.abs(dose_pred - dose_real) * rectum).sum()
                            / max(rectum.sum(), 1))
        mae_bladder = float((np.abs(dose_pred - dose_real) * bladder).sum()
                            / max(bladder.sum(), 1))

        fila = {
            "anonid": anonid,
            "vol_ptv_voxels":     int(ptv.sum()),
            "vol_rectum_voxels":  int(rectum.sum()),
            "vol_bladder_voxels": int(bladder.sum()),
            "mae_body":    mae_body,
            "mae_ptv":     mae_ptv,
            "mae_rectum":  mae_rectum,
            "mae_bladder": mae_bladder,
            "dose_score_openkbp": ds_val,
            "dvh_score_openkbp":  dvh_val,
        }
        # Métricas DVH reales y predichas
        for struct in ["ptv", "rectum", "bladder"]:
            for k, v in m_real[struct].items():
                fila[f"real_{struct}_{k}"] = v
            for k, v in m_pred[struct].items():
                fila[f"pred_{struct}_{k}"] = v
        # Cumplimiento (1=cumple, 0=no)
        for k, v in cumple_real.items():
            fila[f"real_{k}"] = int(v)
        for k, v in cumple_pred.items():
            fila[f"pred_{k}"] = int(v)

        filas.append(fila)

        # Figura
        figura_paciente(
            anonid, ct, dose_real, dose_pred, body, ptv, rectum, bladder,
            figures_dir / f"{anonid}.png",
        )

    # ── Guardar CSV
    df = pd.DataFrame(filas)
    df.to_csv(output_dir / "test_metrics.csv", index=False, sep=";")
    print(f"\nMétricas guardadas en {output_dir / 'test_metrics.csv'}")

    # ── Summary
    summary = {
        "n_test": len(df),
        "mae_body":      {"mean": float(df["mae_body"].mean()),    "std": float(df["mae_body"].std())},
        "mae_ptv":       {"mean": float(df["mae_ptv"].mean()),     "std": float(df["mae_ptv"].std())},
        "mae_rectum":    {"mean": float(df["mae_rectum"].mean()),  "std": float(df["mae_rectum"].std())},
        "mae_bladder":   {"mean": float(df["mae_bladder"].mean()), "std": float(df["mae_bladder"].std())},
        "dose_score":    {"mean": float(df["dose_score_openkbp"].mean()),
                          "std":  float(df["dose_score_openkbp"].std())},
        "dvh_score":     {"mean": float(df["dvh_score_openkbp"].mean()),
                          "std":  float(df["dvh_score_openkbp"].std())},
    }

    # Tasa de acuerdo en cumplimiento de constraints
    acuerdo = {}
    for col in df.columns:
        if col.startswith("real_") and col.endswith("_cumple"):
            pred_col = col.replace("real_", "pred_")
            if pred_col in df.columns:
                acuerdo[col.replace("real_", "").replace("_cumple", "")] = {
                    "agreement_pct": float(100.0 * (df[col] == df[pred_col]).mean()),
                    "real_cumple":   int(df[col].sum()),
                    "pred_cumple":   int(df[pred_col].sum()),
                    "n":             int(len(df)),
                }
    summary["constraint_agreement"] = acuerdo

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Resumen guardado en {output_dir / 'summary.json'}")

    print("\n=== RESUMEN ===")
    print(f"Pacientes evaluados: {summary['n_test']}")
    print(f"MAE body:    {summary['mae_body']['mean']:.3f} ± {summary['mae_body']['std']:.3f} %")
    print(f"MAE PTV:     {summary['mae_ptv']['mean']:.3f} ± {summary['mae_ptv']['std']:.3f} %")
    print(f"MAE Rectum:  {summary['mae_rectum']['mean']:.3f} ± {summary['mae_rectum']['std']:.3f} %")
    print(f"MAE Bladder: {summary['mae_bladder']['mean']:.3f} ± {summary['mae_bladder']['std']:.3f} %")
    print(f"Dose score:  {summary['dose_score']['mean']:.3f}")
    print(f"DVH score:   {summary['dvh_score']['mean']:.3f}")
    print(f"\nTasa de acuerdo en constraints:")
    for k, v in acuerdo.items():
        print(f"  {k}: {v['agreement_pct']:.1f}% (real cumple: {v['real_cumple']}, "
              f"pred cumple: {v['pred_cumple']})")


if __name__ == "__main__":
    main()
