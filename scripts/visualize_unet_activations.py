"""
Visualización de activaciones internas de la U-Net vía PyTorch forward hooks.

Corre un forward completo sobre 1 paciente del test set limpio
(splits_hipo_v2_clean_balanced.json), captura input/enc/bottleneck/dec/output/GT
para un corte axial representativo (mayor área de PTV), y guarda 1 PNG por capa.

Capas intermedias (enc1/enc2/enc3/bottleneck/dec2/dec1, N canales > 3): PCA
(sklearn, 3 componentes) ajustada sobre los píxeles de TODO el volumen del
paciente (todos los cortes pasan por el U-Net en un único forward, ver
DosePredictionModule.forward — B*Z aplanado), para que la base de color sea
consistente entre cortes; luego se proyecta y muestra solo el corte elegido.

Uso:
    python scripts/visualize_unet_activations.py --patient-id PT_17f7da3f2f357e52
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import cargar_modelo  # noqa: E402
from predict_one import cargar_npz_paciente  # noqa: E402

DOSE_MAX_PCT = 120.0  # rango de colormap para output/GT, en % de prescripción

# (nombre_capa, ruta_al_submodulo dentro de model.model — ver src/models/unet2d.py)
LAYER_SPECS = [
    ("enc1", "inc"),
    ("enc2", "downs.0"),
    ("enc3", "downs.1"),
    ("bottleneck", "downs.3"),
    ("dec2", "ups.2"),
    ("dec1", "ups.3"),
]


def get_submodule(model: torch.nn.Module, dotted_name: str) -> torch.nn.Module:
    mod = model
    for part in dotted_name.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


def elegir_slice_representativo(ptv_mask: np.ndarray) -> int:
    """Corte axial con mayor área de PTV (centro del tumor)."""
    areas = ptv_mask.sum(axis=(1, 2))
    return int(np.argmax(areas))


def pca_a_rgb(activation: torch.Tensor, nombre: str) -> np.ndarray:
    """
    activation: (Z, C, H, W), TODOS los cortes de 1 paciente para 1 capa.
    PCA (3 comps) ajustada y aplicada sobre los Z*H*W píxeles del volumen
    completo (features = C canales). Devuelve (Z, H, W, 3) normalizado a [0,1].
    """
    Z, C, H, W = activation.shape
    arr = activation.detach().float().cpu().numpy()
    print(f"  [{nombre}] antes de PCA:   shape={arr.shape}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}")

    pixels = arr.transpose(0, 2, 3, 1).reshape(-1, C)  # (Z*H*W, C)
    n_comp = min(3, C)
    proj = PCA(n_components=n_comp).fit_transform(pixels)
    if n_comp < 3:
        proj = np.pad(proj, ((0, 0), (0, 3 - n_comp)))

    print(f"  [{nombre}] despues de PCA: shape={proj.shape}  "
          f"min={proj.min():.4f}  max={proj.max():.4f}")

    p_min = proj.min(axis=0, keepdims=True)
    p_max = proj.max(axis=0, keepdims=True)
    proj_norm = (proj - p_min) / np.clip(p_max - p_min, 1e-8, None)

    return proj_norm.reshape(Z, H, W, 3)


def upsample_a_256(rgb_slice: np.ndarray) -> np.ndarray:
    """(H, W, 3) float [0,1] -> (256, 256, 3) vía interpolación bilineal."""
    t = torch.from_numpy(rgb_slice).permute(2, 0, 1).unsqueeze(0).float()
    t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()


def guardar_frame(imagen: np.ndarray, titulo: str, out_path: Path, cmap: str = None):
    fig, ax = plt.subplots(figsize=(5, 5))
    if cmap is not None:
        ax.imshow(imagen, cmap=cmap, vmin=0, vmax=1)
    else:
        ax.imshow(imagen)
    ax.set_title(titulo, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", default="PT_17f7da3f2f357e52")
    parser.add_argument("--checkpoint", default="checkpoints/exp_hipo_002b_finetune_clean/epoch=028.ckpt")
    parser.add_argument("--config", default="configs/exp_hipo_002b_finetune_clean.yaml")
    parser.add_argument("--splits-file", default="data/splits/splits_hipo_v2_clean_balanced.json")
    parser.add_argument("--output-dir", default="results/unet_activations")
    parser.add_argument("--slice-idx", type=int, default=None,
                         help="Indice de corte axial. Default: corte con mayor area de PTV.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent

    with open(root / args.splits_file) as f:
        splits = json.load(f)
    if args.patient_id not in splits.get("test", []):
        raise RuntimeError(
            f"{args.patient_id} no esta en el test set de {args.splits_file} "
            f"(split limpio) — abortando para no visualizar un paciente fuera de test."
        )

    cfg = OmegaConf.load(root / args.config)
    torch.set_float32_matmul_precision("high")
    model, device = cargar_modelo(str(root / args.checkpoint), cfg)

    conv1 = model.model.inc.block[0]
    if conv1.in_channels != cfg.model.in_channels:
        raise RuntimeError(
            f"Mismatch: cfg.model.in_channels={cfg.model.in_channels} pero el "
            f"checkpoint espera {conv1.in_channels} canales de entrada."
        )

    npz_path = Path(cfg.data.processed_dir) / f"{args.patient_id}.npz"
    sample, meta = cargar_npz_paciente(npz_path)
    print(f"Paciente: {args.patient_id}  n_slices={meta['n_slices']}  z_range={meta['z_range']}")

    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}

    slice_idx = args.slice_idx
    if slice_idx is None:
        slice_idx = elegir_slice_representativo(sample["ptv_mask"].numpy())
    print(f"Corte axial seleccionado: {slice_idx} (mayor area de PTV)")

    activations = {}
    hooks = []
    for nombre, dotted in LAYER_SPECS:
        submod = get_submodule(model.model, dotted)

        def make_hook(nombre):
            def hook(module, inp, out):
                activations[nombre] = out.detach()
            return hook

        hooks.append(submod.register_forward_hook(make_hook(nombre)))

    with torch.no_grad():
        x = model._build_input(batch)   # (1, Z, C, H, W)
        pred = model(x)                 # (1, Z, H, W)

    for h in hooks:
        h.remove()

    out_dir = root / args.output_dir / args.patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 1

    # --- Input CT: escala de grises, [-1,1] normalizado -> [0,1] ---
    ct_slice = sample["ct"][slice_idx].numpy()
    ct_gray = np.clip((ct_slice + 1.0) / 2.0, 0.0, 1.0)
    h_orig, w_orig = ct_gray.shape
    guardar_frame(ct_gray, f"Input CT (corte {slice_idx}, resolucion {h_orig}x{w_orig})",
                  out_dir / f"{frame_idx:02d}_input_ct.png", cmap="gray")
    frame_idx += 1

    # --- Input PSDM: overlay recto=R, vejiga=G (region interior, PSDM<0), fondo negro ---
    psdm_rectum = sample["psdm_rectum"][slice_idx].numpy()
    psdm_bladder = sample["psdm_bladder"][slice_idx].numpy()
    overlay = np.zeros((*psdm_rectum.shape, 3), dtype=np.float32)
    overlay[..., 0] = (psdm_rectum < 0).astype(np.float32)
    overlay[..., 1] = (psdm_bladder < 0).astype(np.float32)
    guardar_frame(overlay, f"Input PSDM recto=R vejiga=G (corte {slice_idx}, resolucion {h_orig}x{w_orig})",
                  out_dir / f"{frame_idx:02d}_input_psdm.png")
    frame_idx += 1

    # --- Capas intermedias: PCA sobre el volumen completo, se muestra el corte elegido ---
    print("PCA por capa (ajustada sobre todos los cortes del paciente):")
    for nombre, _ in LAYER_SPECS:
        act = activations[nombre]  # (Z, C, H, W)
        _, _, h_act, w_act = act.shape
        rgb_vol = pca_a_rgb(act, nombre)
        rgb_up = upsample_a_256(rgb_vol[slice_idx])
        guardar_frame(rgb_up, f"{nombre} (corte {slice_idx}, resolucion original {h_act}x{w_act}, PCA->3ch)",
                      out_dir / f"{frame_idx:02d}_{nombre}.png")
        frame_idx += 1

    # --- Output predicho y ground truth: colormap hot, 0-120% Rx ---
    cmap_hot = plt.get_cmap("hot")

    pred_slice = pred[0, slice_idx].cpu().numpy()
    pred_rgb = cmap_hot(np.clip(pred_slice / DOSE_MAX_PCT, 0.0, 1.0))[..., :3]
    guardar_frame(pred_rgb, f"Output predicho (corte {slice_idx}, resolucion {h_orig}x{w_orig}, "
                             f"0-{DOSE_MAX_PCT:.0f}% Rx)",
                  out_dir / f"{frame_idx:02d}_output.png")
    frame_idx += 1

    gt_slice = sample["dose"][slice_idx].numpy()
    gt_rgb = cmap_hot(np.clip(gt_slice / DOSE_MAX_PCT, 0.0, 1.0))[..., :3]
    guardar_frame(gt_rgb, f"Ground truth (corte {slice_idx}, resolucion {h_orig}x{w_orig}, "
                           f"0-{DOSE_MAX_PCT:.0f}% Rx)",
                  out_dir / f"{frame_idx:02d}_gt.png")

    print(f"Frames guardados en: {out_dir}")


if __name__ == "__main__":
    main()
