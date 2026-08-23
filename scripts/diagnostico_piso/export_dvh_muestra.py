"""
Export ad-hoc (pedido de Pablo, no parte del diagnostico D1-D5) — 30 pacientes de
normofx + 30 de hipofx, estratificados por overlap_rel_recto ("rectum_ovp",
overlap PTV-Rectum relativo al volumen de recto — la misma feature que domina
RV65/RV55 en el baseline ML clasico), tomados del pool COMPLETO train+val+test
(no solo test), con DVH REAL y PREDICHA (modelo ganador de cada serie) para que
Pablo lo analice con su propio script.

Modelos ganadores usados:
  - normo: exp002_unet2d_psdm, checkpoint epoch=191 (referencia de toda la serie
    normofraccionada — mismo checkpoint que scripts/predict_one.py y el init-weights
    del finetune hipo).
  - hipo:  exp_hipo_002b_finetune_clean, checkpoint epoch=028 (referencia de toda
    la serie hipofraccionada, mismo checkpoint de dvh_curva_completa.py /
    analisis_angular.py).

Dosis predicha renormalizada a D95(PTV_pred)=100% antes de calcular su DVH
(mismo criterio que dvh_curva_completa.py — clip a 0 del ruido negativo del
modelo sin activacion final, despues renorm).

Estratificacion: quintiles de overlap_rel_recto (recalculado desde mascaras NPZ,
igual que analisis_angular.py/compute_overlap_real.py) sobre TODO el dataset
(train+val+test), 6 pacientes por quintil (seed=42).

Salida: results/muestra_dvh_replan/dvh_muestra_{normo,hipo}.csv — formato tidy,
una fila por (paciente, estructura, bin de dosis 0-110%Rx paso 1%).

Uso:
    .venv/Scripts/python.exe scripts/diagnostico_piso/export_dvh_muestra.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

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

OUT_DIR = _REPO_ROOT / "results/muestra_dvh_replan"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_MUESTRA = 30
N_BINS_ESTRATO = 5
SEED = 42
STRUCTS = [("PTV", "ptv_mask"), ("Rectum", "rectum_mask"), ("Bladder", "bladder_mask"), ("BODY", "body_mask")]


def listar_pacientes_con_split(dataset: str) -> list:
    splits_path = SPLITS_NORMO if dataset == "normo" else SPLITS_HIPO
    with open(splits_path) as f:
        s = json.load(f)
    filas = []
    for split_name in ["train", "val", "test"]:
        for aid in s[split_name]:
            filas.append((aid, split_name))
    return filas


def calcular_features(dataset: str, processed_dir: Path) -> pd.DataFrame:
    filas = []
    for aid, split_name in listar_pacientes_con_split(dataset):
        npz_path = processed_dir / f"{aid}.npz"
        if not npz_path.exists():
            continue
        arrays, meta = cargar_npz(npz_path)
        feats = overlap_y_volumenes(meta, arrays["ptv_mask"], arrays["rectum_mask"], arrays["bladder_mask"])
        filas.append({"AnonID": aid, "split": split_name, **feats})
    return pd.DataFrame(filas)


def muestra_estratificada(df_feat: pd.DataFrame, n_muestra: int, n_bins: int, seed: int) -> pd.DataFrame:
    df = df_feat.dropna(subset=["overlap_rel_recto"]).copy()
    df["estrato_rectum_ovp"] = pd.qcut(df["overlap_rel_recto"], n_bins, labels=False, duplicates="drop")
    estratos = sorted(df["estrato_rectum_ovp"].unique())
    n_estratos = len(estratos)
    base, resto = divmod(n_muestra, n_estratos)

    partes = []
    for i, e in enumerate(estratos):
        n_tomar = min(base + (1 if i < resto else 0), (df["estrato_rectum_ovp"] == e).sum())
        partes.append(df[df["estrato_rectum_ovp"] == e].sample(n=int(n_tomar), random_state=seed + e))
    muestra = pd.concat(partes).reset_index(drop=True)

    faltan = n_muestra - len(muestra)
    if faltan > 0:
        restantes = df[~df["AnonID"].isin(muestra["AnonID"])]
        muestra = pd.concat([muestra, restantes.sample(n=min(faltan, len(restantes)), random_state=seed + 999)]).reset_index(drop=True)
    return muestra


def exportar_dvh(dataset: str, muestra: pd.DataFrame, processed_dir: Path, checkpoint: Path, config_yaml: Path) -> pd.DataFrame:
    cfg = OmegaConf.load(str(config_yaml))
    cfg.data.processed_dir = str(processed_dir)
    torch.set_float32_matmul_precision("high")
    model, device = cargar_modelo(str(checkpoint), cfg)

    filas = []
    for _, row in muestra.iterrows():
        aid = row["AnonID"]
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
                    "AnonID": aid, "dataset": dataset, "split": row["split"],
                    "overlap_rel_recto": row["overlap_rel_recto"],
                    "estrato_rectum_ovp_quintil": int(row["estrato_rectum_ovp"]),
                    "VolPTV_cc": row["VolPTV_cc"], "VolRectum_cc": row["VolRectum_cc"], "VolBladder_cc": row["VolBladder_cc"],
                    "estructura": struct, "D_pct_rx": int(d),
                    "V_real_pct": float(vr), "V_pred_pct": float(vp),
                })
        print(f"  {aid} ({row['split']}) OK")

    return pd.DataFrame(filas)


def procesar(dataset: str):
    processed_dir = PROCESSED_NORMO if dataset == "normo" else PROCESSED_HIPO
    checkpoint = CHECKPOINT_NORMO if dataset == "normo" else CHECKPOINT_HIPO
    config_yaml = CONFIG_NORMO if dataset == "normo" else CONFIG_HIPO

    print(f"\n=== {dataset.upper()} — calculando features (overlap_rel_recto) para estratificar ===")
    df_feat = calcular_features(dataset, processed_dir)
    print(f"Pool disponible: {len(df_feat)} pacientes (train+val+test)")

    muestra = muestra_estratificada(df_feat, N_MUESTRA, N_BINS_ESTRATO, SEED)
    print(f"Muestra estratificada: {len(muestra)} pacientes "
          f"(por split: {muestra['split'].value_counts().to_dict()})")

    print(f"\n=== {dataset.upper()} — inferencia + DVH real/predicha ===")
    df_dvh = exportar_dvh(dataset, muestra, processed_dir, checkpoint, config_yaml)

    out_path = OUT_DIR / f"dvh_muestra_{dataset}.csv"
    df_dvh.to_csv(out_path, index=False)
    print(f"\nGuardado: {out_path} ({len(df_dvh)} filas = {muestra.shape[0]} pacientes x "
          f"{len(STRUCTS)} estructuras x {len(D_BINS)} bins de dosis)")
    print(f"Checkpoint usado: {checkpoint}")
    return df_dvh


def main():
    procesar("normo")
    procesar("hipo")


if __name__ == "__main__":
    main()
