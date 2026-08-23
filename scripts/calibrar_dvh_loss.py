"""
Calibracion previa a entrenar exp_hipo_003_dvhloss — dos chequeos obligatorios
ANTES de lanzar el entrenamiento (ver tarea/CLAUDE_CODE_CONTEXT.md):

1. CHEQUEO DE SESGO DEL SIGMOIDE: comparar V_approx (sigmoide) vs V_exact (DVH real)
   sobre la dosis GT de varios pacientes de train — si el sesgo medio >0.5pp,
   hay que ajustar `dvh_sigmoid_k` en el config.
2. LAMBDA_0: medir L_MAE y L_DVH (a lambda=1) en ~5 batches de train con el
   modelo AL INIT (pesos de exp002 normo, los mismos que --init-weights usa al
   arrancar el entrenamiento real) y fijar lambda_0 = mean(L_MAE)/mean(L_DVH).

No entrena nada — solo forward passes para calibrar. Imprime los resultados y
los deja en JSON para pegar en el config y en CLAUDE_CODE_CONTEXT.md.

Uso:
    .venv/Scripts/python.exe scripts/calibrar_dvh_loss.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.datamodules.dose_datamodule import DoseDataModule  # noqa: E402
from src.models.lightning_module import DosePredictionModule  # noqa: E402
from src.losses.losses import DifferentiableDVHLoss, MaskedMAELoss  # noqa: E402

CONFIG_PATH = _REPO_ROOT / "configs/exp_hipo_003_dvhloss.yaml"
INIT_CHECKPOINT = _REPO_ROOT / "checkpoints/exp002_unet2d_psdm/epoch=191.ckpt"
OUT_PATH = _REPO_ROOT / "results/calibracion_dvh_loss.json"

N_BATCHES_LAMBDA = 5
N_PACIENTES_SESGO = 10
SESGO_TOLERANCIA_PP = 0.5  # pp = 0.005 en fraccion [0,1]


def cargar_modelo_init(cfg) -> DosePredictionModule:
    """Instancia el modelo de exp_hipo_003 (arquitectura identica a exp002/002b) y
    carga los pesos de init (exp002 normo) — EXACTAMENTE lo que --init-weights hace
    en train.py, para medir lambda_0 sobre "el modelo al init", no un modelo random."""
    model = DosePredictionModule(cfg)

    _orig_torch_load = torch.load
    def _load_no_weights_only(*a, **kw):
        kw['weights_only'] = False
        return _orig_torch_load(*a, **kw)
    torch.load = _load_no_weights_only
    print(f"Cargando pesos iniciales desde {INIT_CHECKPOINT} (init, igual que --init-weights)")
    init_ckpt = torch.load(str(INIT_CHECKPOINT), map_location='cpu')
    torch.load = _orig_torch_load
    resultado = model.load_state_dict(init_ckpt['state_dict'], strict=True)
    print(f"  state_dict cargado: {resultado}")
    return model


def construir_dvh_loss_desde_cfg(cfg) -> DifferentiableDVHLoss:
    structures_d_bins = {
        nombre: list(np.arange(cfg_s.d_min_pct, cfg_s.d_max_pct + 1e-6, cfg_s.d_step_pct))
        for nombre, cfg_s in cfg.loss.dvh_structures.items()
    }
    sigmoid_slopes = {
        nombre: cfg.loss.dvh_sigmoid_k / cfg_s.d_step_pct
        for nombre, cfg_s in cfg.loss.dvh_structures.items()
    }
    return DifferentiableDVHLoss(structures_d_bins, sigmoid_slopes), structures_d_bins, sigmoid_slopes


# ──────────────────────────────────────────────────────────────────────────────
# 1. Chequeo de sesgo del sigmoide (V_approx vs V_exact, sobre dosis GT — no
#    involucra al modelo en absoluto, es un chequeo puro de la aproximacion).
# ──────────────────────────────────────────────────────────────────────────────

def chequear_sesgo_sigmoide(dvh_loss: DifferentiableDVHLoss, structures_d_bins: dict,
                             sigmoid_slopes: dict, npz_paths: list) -> dict:
    resultados = {nombre: [] for nombre in structures_d_bins}
    for npz_path in npz_paths:
        data = np.load(str(npz_path), allow_pickle=True)
        dose = torch.from_numpy(np.array(data["dose"], dtype=np.float32))
        masks = {
            "ptv": torch.from_numpy(np.array(data["ptv_mask"], dtype=np.float32)),
            "rectum": torch.from_numpy(np.array(data["rectum_mask"], dtype=np.float32)),
            "bladder": torch.from_numpy(np.array(data["bladder_mask"], dtype=np.float32)),
        }
        data.close()
        for nombre, d_bins in structures_d_bins.items():
            roi = masks[nombre] > 0
            if roi.sum() < 1:
                continue
            vals = dose[roi]
            d_bins_t = torch.tensor(d_bins, dtype=torch.float32)
            s = sigmoid_slopes[nombre]
            v_approx = torch.sigmoid(s * (vals.unsqueeze(1) - d_bins_t.unsqueeze(0))).mean(dim=0)
            v_exact = (vals.unsqueeze(1) >= d_bins_t.unsqueeze(0)).float().mean(dim=0)
            sesgo_pp = ((v_approx - v_exact).abs() * 100.0).numpy()
            resultados[nombre].append(sesgo_pp)

    resumen = {}
    for nombre, lista in resultados.items():
        if not lista:
            continue
        arr = np.stack(lista)  # (n_pacientes, n_bins)
        resumen[nombre] = {
            "sesgo_medio_pp": float(arr.mean()),
            "sesgo_max_pp": float(arr.max()),
            "sesgo_por_bin_medio_pp": arr.mean(axis=0).tolist(),
            "n_pacientes": arr.shape[0],
            "ok": bool(arr.mean() <= SESGO_TOLERANCIA_PP),
        }
    return resumen


# ──────────────────────────────────────────────────────────────────────────────
# 2. Lambda_0 — medir L_MAE y L_DVH (lambda=1) sobre ~5 batches de train, modelo al init.
# ──────────────────────────────────────────────────────────────────────────────

def medir_lambda_0(cfg, model: DosePredictionModule, device: str, n_batches: int) -> dict:
    dm = DoseDataModule(cfg)
    dm.setup(stage="fit")
    loader = dm.train_dataloader()

    mae_fn = MaskedMAELoss()
    dvh_fn, _, _ = construir_dvh_loss_desde_cfg(cfg)
    dvh_fn = dvh_fn.to(device)

    model = model.to(device).eval()
    mae_vals, dvh_vals = [], []
    it = iter(loader)
    with torch.no_grad():
        for i in range(n_batches):
            batch = next(it)
            batch_gpu = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            x = model._build_input(batch_gpu)
            pred = model(x)
            target = batch_gpu["dose"]
            body = batch_gpu["body_mask"]
            struct_masks = {
                "ptv": batch_gpu["ptv_mask"],
                "rectum": batch_gpu["rectum_mask"],
                "bladder": batch_gpu["bladder_mask"],
            }
            mae_val = mae_fn(pred, target, body).item()
            dvh_val, _ = dvh_fn(pred, target, struct_masks)
            dvh_val = dvh_val.item()
            mae_vals.append(mae_val)
            dvh_vals.append(dvh_val)
            print(f"  batch {i+1}/{n_batches}: L_MAE={mae_val:.4f}  L_DVH={dvh_val:.6f}")

    mean_mae = float(np.mean(mae_vals))
    mean_dvh = float(np.mean(dvh_vals))
    lambda_0 = mean_mae / mean_dvh if mean_dvh > 0 else float("nan")
    return {
        "n_batches": n_batches,
        "L_MAE_por_batch": mae_vals, "L_DVH_por_batch": dvh_vals,
        "L_MAE_mean": mean_mae, "L_DVH_mean": mean_dvh,
        "lambda_0": lambda_0,
    }


def main():
    cfg = OmegaConf.load(str(CONFIG_PATH))
    torch.set_float32_matmul_precision("high")

    dvh_loss, structures_d_bins, sigmoid_slopes = construir_dvh_loss_desde_cfg(cfg)
    print("D bins por estructura:")
    for nombre, bins in structures_d_bins.items():
        print(f"  {nombre}: {[round(b,2) for b in bins]}  (pendiente_s={sigmoid_slopes[nombre]:.3f})")

    # ── 1. Sesgo del sigmoide ────────────────────────────────────────────────
    processed_dir = Path(cfg.data.processed_dir)
    with open(_REPO_ROOT / cfg.data.splits_file) as f:
        splits = json.load(f)
    npz_paths = [processed_dir / f"{a}.npz" for a in splits["train"][:N_PACIENTES_SESGO]]
    npz_paths = [p for p in npz_paths if p.exists()]
    print(f"\n=== Chequeo de sesgo sigmoide (n={len(npz_paths)} pacientes de train) ===")
    sesgo = chequear_sesgo_sigmoide(dvh_loss, structures_d_bins, sigmoid_slopes, npz_paths)
    for nombre, r in sesgo.items():
        estado = "OK" if r["ok"] else f"*** EXCEDE TOLERANCIA ({SESGO_TOLERANCIA_PP}pp) ***"
        print(f"  {nombre}: sesgo_medio={r['sesgo_medio_pp']:.3f}pp  sesgo_max={r['sesgo_max_pp']:.3f}pp  {estado}")

    # ── 2. Lambda_0 ───────────────────────────────────────────────────────────
    print(f"\n=== Lambda_0 (modelo al init = exp002 normo, {N_BATCHES_LAMBDA} batches de train) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cargar_modelo_init(cfg)
    resultado_lambda = medir_lambda_0(cfg, model, device, N_BATCHES_LAMBDA)
    print(f"  L_MAE_mean={resultado_lambda['L_MAE_mean']:.4f}  L_DVH_mean={resultado_lambda['L_DVH_mean']:.6f}")
    print(f"  lambda_0 = {resultado_lambda['lambda_0']:.4f}")

    salida = {
        "dvh_sigmoid_k_usado": cfg.loss.dvh_sigmoid_k,
        "sigmoid_slopes": sigmoid_slopes,
        "chequeo_sesgo_sigmoide": sesgo,
        "sesgo_ok_todas_estructuras": all(r["ok"] for r in sesgo.values()),
        "lambda_0": resultado_lambda,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(salida, f, indent=2)
    print(f"\nGuardado: {OUT_PATH}")

    if not salida["sesgo_ok_todas_estructuras"]:
        print("\n*** ATENCION: el sesgo del sigmoide excede la tolerancia en al menos una "
              "estructura — subir dvh_sigmoid_k (sigmoide mas empinado) y volver a correr "
              "este script antes de entrenar. ***")
    print(f"\n>>> Actualizar configs/exp_hipo_003_dvhloss.yaml: dvh_weight: {resultado_lambda['lambda_0']:.4f}")


if __name__ == "__main__":
    main()
