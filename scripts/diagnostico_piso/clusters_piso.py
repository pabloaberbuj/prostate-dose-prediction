"""
Clusters de 5 pacientes por cercania en features (misma logica de D3 —
d3_sonda_vecinos.py: 7 features geometricas recalculadas desde mascaras NPZ,
estandarizadas, distancia euclidea), separados en 3 niveles de overlap_rel_recto
("rectum_ovp": bajo / intermedio / alto, terciles sobre el pool completo
train+val+test), tanto para normo como para hipo.

Dentro de cada nivel de overlap: se busca el "nucleo" mas denso (el paciente
cuya distancia media a sus 4 vecinos mas cercanos DENTRO DEL NIVEL es minima) y
se arma el cluster con ese paciente + sus 4 vecinos — 5 pacientes mutuamente
cercanos en el espacio completo de 7 features, no solo en overlap_rel_recto.
Asi cada cluster es una "sonda D3" concreta y visualizable: si el generador es
consistente, los 5 DVH reales deberian ser parecidos; si es idiosincratico, no.

Salida por cluster (6 en total: 2 datasets x 3 niveles):
  - results/muestra_dvh_replan/clusters/dvh_cluster_{dataset}_{nivel}.csv
    (tidy, mismo formato que export_dvh_muestra.py: DVH real y predicha del
    modelo ganador de cada serie, por paciente/estructura/bin de dosis).
  - results/muestra_dvh_replan/clusters/plot_cluster_{dataset}_{nivel}.png
    (5 filas = 5 pacientes, 3 columnas = corte central real / predicha / diff).
  - results/muestra_dvh_replan/clusters/resumen_clusters.json (metadata: que
    pacientes entraron en cada cluster, distancia intra-cluster vs. dispersion
    poblacional, features de cada paciente).

Uso:
    .venv/Scripts/python.exe scripts/diagnostico_piso/clusters_piso.py
"""

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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from analisis_angular import cargar_npz, inferir_dosis, overlap_y_volumenes  # noqa: E402
from evaluate import cargar_modelo  # noqa: E402
from dvh_curva_completa import curva_v_d, D_BINS, d95_ptv  # noqa: E402

PROCESSED_NORMO = Path(r"C:\Pablo\ProstateDoseProject\processed")
PROCESSED_HIPO = Path(r"C:\Pablo\ProstateDoseProject\processed_hipo")
CHECKPOINT_NORMO = _REPO_ROOT / "checkpoints/exp002_unet2d_psdm/epoch=191.ckpt"
CONFIG_NORMO = _REPO_ROOT / "configs/exp002_unet2d_psdm.yaml"
SPLITS_NORMO = _REPO_ROOT / "data/splits/splits_v1.json"
CHECKPOINT_HIPO = _REPO_ROOT / "checkpoints/exp_hipo_002b_finetune_clean/epoch=028.ckpt"
CONFIG_HIPO = _REPO_ROOT / "configs/exp_hipo_002b_finetune_clean.yaml"
SPLITS_HIPO = _REPO_ROOT / "data/splits/splits_hipo_v2_clean_balanced.json"

OUT_DIR = _REPO_ROOT / "results/muestra_dvh_replan/clusters"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["VolPTV_cc", "VolRectum_cc", "VolBladder_cc",
            "Solap_PTV_Rectum_cc", "Solap_PTV_Bladder_cc",
            "overlap_rel_recto", "overlap_rel_vejiga"]
NIVELES = ["bajo", "medio", "alto"]
N_CLUSTER = 5
STRUCTS = [("PTV", "ptv_mask"), ("Rectum", "rectum_mask"), ("Bladder", "bladder_mask"), ("BODY", "body_mask")]
CONTORNOS = [("ptv_mask", "yellow"), ("rectum_mask", "red"), ("bladder_mask", "cyan")]


def listar_pacientes_con_split(dataset: str) -> list:
    splits_path = SPLITS_NORMO if dataset == "normo" else SPLITS_HIPO
    with open(splits_path) as f:
        s = json.load(f)
    return [(aid, sp) for sp in ["train", "val", "test"] for aid in s[sp]]


def calcular_features(dataset: str, processed_dir: Path) -> pd.DataFrame:
    filas = []
    for aid, split_name in listar_pacientes_con_split(dataset):
        npz_path = processed_dir / f"{aid}.npz"
        if not npz_path.exists():
            continue
        arrays, meta = cargar_npz(npz_path)
        feats = overlap_y_volumenes(meta, arrays["ptv_mask"], arrays["rectum_mask"], arrays["bladder_mask"])
        filas.append({"AnonID": aid, "split": split_name, **feats})
    return pd.DataFrame(filas).dropna(subset=FEATURES)


def armar_clusters(df: pd.DataFrame) -> dict:
    """Devuelve {nivel: {'anonids': [...], 'dist_intra_cluster_mean': float,
    'dist_poblacional_mean': float}} — 1 cluster de N_CLUSTER pacientes por nivel."""
    scaler = StandardScaler()
    Xz_all = scaler.fit_transform(df[FEATURES].to_numpy(dtype=float))
    dist_poblacional_mean = float(pairwise_distances(Xz_all).mean())

    df = df.copy()
    df["nivel_ovp"] = pd.qcut(df["overlap_rel_recto"], 3, labels=NIVELES)
    df["_Xz_idx"] = np.arange(len(df))

    clusters = {}
    for nivel in NIVELES:
        sub = df[df["nivel_ovp"] == nivel]
        Xz_sub = Xz_all[sub["_Xz_idx"].to_numpy()]
        D = pairwise_distances(Xz_sub)
        np.fill_diagonal(D, np.inf)
        k = min(4, len(sub) - 1)
        dist_4nn = np.sort(D, axis=1)[:, :k].mean(axis=1)
        idx_nucleo = int(np.argmin(dist_4nn))
        vecinos = np.argsort(D[idx_nucleo])[:N_CLUSTER - 1]
        idx_cluster = [idx_nucleo] + vecinos.tolist()

        sub_cluster = sub.iloc[idx_cluster]
        Xz_cluster = Xz_sub[idx_cluster]
        dist_intra = pairwise_distances(Xz_cluster)
        dist_intra_mean = float(dist_intra[np.triu_indices(len(idx_cluster), k=1)].mean())

        clusters[nivel] = {
            "anonids": sub_cluster["AnonID"].tolist(),
            "splits": sub_cluster["split"].tolist(),
            "overlap_rel_recto": sub_cluster["overlap_rel_recto"].tolist(),
            "dist_intra_cluster_mean": dist_intra_mean,
            "dist_poblacional_mean": dist_poblacional_mean,
            "ratio_intra_vs_poblacional": dist_intra_mean / dist_poblacional_mean if dist_poblacional_mean > 0 else float("nan"),
        }
    return clusters


def dvh_cluster_csv(dataset: str, nivel: str, anonids: list, splits: list, df_feat: pd.DataFrame,
                     processed_dir: Path, model, device, out_path: Path):
    filas = []
    for aid, split_name in zip(anonids, splits):
        row = df_feat[df_feat["AnonID"] == aid].iloc[0]
        arrays, meta = cargar_npz(processed_dir / f"{aid}.npz")
        dose_pred_raw = np.clip(inferir_dosis(model, device, arrays), 0.0, None)
        d95_pred_raw = d95_ptv(dose_pred_raw, arrays["ptv_mask"])
        factor = 100.0 / d95_pred_raw if d95_pred_raw > 0 else float("nan")
        dose_pred = dose_pred_raw * factor
        dose_real = arrays["dose"]

        for struct, mask_key in STRUCTS:
            mask = arrays[mask_key]
            v_real = curva_v_d(dose_real, mask)
            v_pred = curva_v_d(dose_pred, mask)
            for d, vr, vp in zip(D_BINS, v_real, v_pred):
                filas.append({
                    "AnonID": aid, "dataset": dataset, "split": split_name,
                    "cluster_nivel_ovp": nivel,
                    "overlap_rel_recto": row["overlap_rel_recto"],
                    "VolPTV_cc": row["VolPTV_cc"], "VolRectum_cc": row["VolRectum_cc"], "VolBladder_cc": row["VolBladder_cc"],
                    "estructura": struct, "D_pct_rx": int(d),
                    "V_real_pct": float(vr), "V_pred_pct": float(vp),
                })
    df_out = pd.DataFrame(filas)
    df_out.to_csv(out_path, index=False)
    return df_out


def plot_cluster(dataset: str, nivel: str, anonids: list, processed_dir: Path, model, device, out_path: Path):
    fig, axes = plt.subplots(len(anonids), 3, figsize=(12, 3.6 * len(anonids)))
    if len(anonids) == 1:
        axes = axes[np.newaxis, :]

    for row_i, aid in enumerate(anonids):
        arrays, meta = cargar_npz(processed_dir / f"{aid}.npz")
        ct = arrays["ct"]
        ptv = arrays["ptv_mask"]
        rectum = arrays["rectum_mask"]
        bladder = arrays["bladder_mask"]
        body = arrays["body_mask"]
        dose_real = arrays["dose"]

        dose_pred_raw = np.clip(inferir_dosis(model, device, arrays), 0.0, None)
        d95_pred_raw = d95_ptv(dose_pred_raw, ptv)
        factor = 100.0 / d95_pred_raw if d95_pred_raw > 0 else float("nan")
        dose_pred = dose_pred_raw * factor

        slices_ptv = np.where(ptv.sum(axis=(1, 2)) > 0)[0]
        z = slices_ptv[len(slices_ptv) // 2] if len(slices_ptv) > 0 else ct.shape[0] // 2

        for col, (arr, titulo, vmin, vmax, cmap) in enumerate([
            (dose_real, "Real", 0, 110, "jet"),
            (dose_pred, "Predicha", 0, 110, "jet"),
            (dose_pred - dose_real, "Pred - Real", -20, 20, "RdBu_r"),
        ]):
            ax = axes[row_i, col]
            ax.imshow(ct[z], cmap="gray", vmin=-1, vmax=1, aspect="equal")
            arr_masked = np.where(body[z] > 0, arr[z], np.nan)
            im = ax.imshow(arr_masked, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.6, aspect="equal")
            for mask_key, color in CONTORNOS:
                mv = arrays[mask_key][z]
                if mv.sum() > 0:
                    ax.contour(mv, levels=[0.5], colors=[color], linewidths=0.7)
            plt.colorbar(im, ax=ax, fraction=0.046)
            if row_i == 0:
                ax.set_title(titulo, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(aid, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Cluster {dataset} — overlap_rel_recto {nivel.upper()} (n={len(anonids)}, corte central por PTV)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)


def procesar(dataset: str) -> dict:
    processed_dir = PROCESSED_NORMO if dataset == "normo" else PROCESSED_HIPO
    checkpoint = CHECKPOINT_NORMO if dataset == "normo" else CHECKPOINT_HIPO
    config_yaml = CONFIG_NORMO if dataset == "normo" else CONFIG_HIPO

    print(f"\n=== {dataset.upper()} — features + armado de clusters ===")
    df_feat = calcular_features(dataset, processed_dir)
    print(f"Pool: {len(df_feat)} pacientes")
    clusters = armar_clusters(df_feat)

    cfg = OmegaConf.load(str(config_yaml))
    cfg.data.processed_dir = str(processed_dir)
    torch.set_float32_matmul_precision("high")
    model, device = cargar_modelo(str(checkpoint), cfg)

    for nivel in NIVELES:
        c = clusters[nivel]
        print(f"  {nivel}: {c['anonids']} (dist intra={c['dist_intra_cluster_mean']:.2f} "
              f"vs. poblacional={c['dist_poblacional_mean']:.2f}, ratio={c['ratio_intra_vs_poblacional']:.2f})")

        csv_path = OUT_DIR / f"dvh_cluster_{dataset}_{nivel}.csv"
        dvh_cluster_csv(dataset, nivel, c["anonids"], c["splits"], df_feat, processed_dir, model, device, csv_path)
        print(f"    CSV: {csv_path}")

        png_path = OUT_DIR / f"plot_cluster_{dataset}_{nivel}.png"
        plot_cluster(dataset, nivel, c["anonids"], processed_dir, model, device, png_path)
        print(f"    PNG: {png_path}")

    return clusters


def main():
    resumen = {}
    for dataset in ["normo", "hipo"]:
        resumen[dataset] = procesar(dataset)
    with open(OUT_DIR / "resumen_clusters.json", "w") as f:
        json.dump(resumen, f, indent=2)
    print(f"\nGuardado: {OUT_DIR / 'resumen_clusters.json'}")


if __name__ == "__main__":
    main()
