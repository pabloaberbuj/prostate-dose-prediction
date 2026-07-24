"""
Ground-truth DVH sobre la grilla de prediccion (256x256) — dataset hipofraccionado.

Recalcula, directamente sobre (dose_norm, mask) ya downsampleados a 256x256 dentro
de cada NPZ de processed_hipo/ (sin interpolar de nuevo ni volver a la grilla nativa),
las metricas de DVH que va a ver el modelo:

  PTV:     D95, D98, V70Gy (=V100% de 70Gy), V95%, V98%.
  Rectum:  V65Gy, V55Gy, V45Gy, Dmean.
  Bladder: V65Gy, V55Gy, V45Gy, Dmean.

Y hace un cross-check unico contra los flags Flag_*_D95norm del CSV (ESAPI, grilla
nativa) para los 3 constraints operativos (RV65, RV55, BV65).

Salidas:
  - data/gt_dvh_hipo_256.csv
  - data/gt_dvh_hipo_256_vs_csharp_flips.csv

Uso:
    python data/compute_gt_dvh_hipo.py \
        --processed-dir "c:/Pablo/ProstateDoseProject/processed_hipo" \
        --csv           "c:/Pablo/ProstateDoseProject/dicoms hipofx/metricas_planes_hipofx_D95norm.csv" \
        --splits        data/splits/splits_hipo_v1.json \
        --out-dvh       data/gt_dvh_hipo_256.csv \
        --out-flips     data/gt_dvh_hipo_256_vs_csharp_flips.csv
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PRESCRIPCION_GY = 70.0

# Umbrales operativos (flags GT)
UMBRALES = {
    "Rectum_V65Gy_lt15":  15.0,
    "Rectum_V55Gy_lt25":  25.0,
    "Rectum_V45Gy_lt45":  45.0,
    "Bladder_V65Gy_lt15": 15.0,
    "Bladder_V55Gy_lt25": 25.0,
    "Bladder_V45Gy_lt45": 45.0,
}

# Cross-check unico: constraint GT -> (columna Vx en CSV, columna Flag en CSV, umbral)
CROSSCHECK_CONSTRAINTS = {
    "RV65": ("Rectum_V65Gy_lt15",  "Flag_Rectum_V65Gy_lt15_D95norm",  "Rectum_V65Gy_lt15_D95norm",  15.0),
    "RV55": ("Rectum_V55Gy_lt25",  "Flag_Rectum_V55Gy_lt25_D95norm",  "Rectum_V55Gy_lt25_D95norm",  25.0),
    "BV65": ("Bladder_V65Gy_lt15", "Flag_Bladder_V65Gy_lt15_D95norm", "Bladder_V65Gy_lt15_D95norm", 15.0),
}


def cargar_csv(csv_path: Path) -> dict:
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        return {row["AnonID"]: row for row in reader}


def volumen_pct_sobre_umbral(dose_gy: np.ndarray, mask: np.ndarray, umbral_gy: float) -> float:
    vals = dose_gy[mask > 0]
    if len(vals) == 0:
        return float("nan")
    return 100.0 * float(np.mean(vals >= umbral_gy))


def dosis_percentil(dose_gy: np.ndarray, mask: np.ndarray, pct_volumen: float) -> float:
    """D_x: dosis recibida por al menos x% del volumen = percentil (100-x) de la dosis."""
    vals = dose_gy[mask > 0]
    if len(vals) == 0:
        return float("nan")
    return float(np.percentile(vals, 100.0 - pct_volumen))


def calcular_dvh_paciente(npz_path: Path) -> dict:
    data = np.load(str(npz_path), allow_pickle=True)
    meta = json.loads(str(data["meta"][0]))

    dose_gy = data["dose"].astype(np.float64) * PRESCRIPCION_GY / 100.0
    ptv_mask     = data["ptv_mask"]
    rectum_mask  = data["rectum_mask"]
    bladder_mask = data["bladder_mask"]

    fila = {"AnonID": meta["anonid"], "Status": meta.get("Status", "")}

    # --- PTV ---
    fila["PTV_D95_Gy_gt256"] = dosis_percentil(dose_gy, ptv_mask, 95)
    fila["PTV_D98_Gy_gt256"] = dosis_percentil(dose_gy, ptv_mask, 98)
    fila["PTV_V70Gy_pct_gt256"] = volumen_pct_sobre_umbral(dose_gy, ptv_mask, 70.0)
    fila["PTV_V95pct_pct_gt256"] = volumen_pct_sobre_umbral(dose_gy, ptv_mask, 0.95 * PRESCRIPCION_GY)
    fila["PTV_V98pct_pct_gt256"] = volumen_pct_sobre_umbral(dose_gy, ptv_mask, 0.98 * PRESCRIPCION_GY)

    # --- Rectum / Bladder ---
    for oar_name, mask in (("Rectum", rectum_mask), ("Bladder", bladder_mask)):
        fila[f"{oar_name}_V65Gy_gt256"] = volumen_pct_sobre_umbral(dose_gy, mask, 65.0)
        fila[f"{oar_name}_V55Gy_gt256"] = volumen_pct_sobre_umbral(dose_gy, mask, 55.0)
        fila[f"{oar_name}_V45Gy_gt256"] = volumen_pct_sobre_umbral(dose_gy, mask, 45.0)
        vals = dose_gy[mask > 0]
        fila[f"{oar_name}_Dmean_gt256"] = float(np.mean(vals)) if len(vals) else float("nan")

    # --- Flags GT (1 = cumple, igual convencion que el CSV) ---
    fila["Flag_Rectum_V65Gy_lt15_gt256"]  = int(fila["Rectum_V65Gy_gt256"]  < UMBRALES["Rectum_V65Gy_lt15"])
    fila["Flag_Rectum_V55Gy_lt25_gt256"]  = int(fila["Rectum_V55Gy_gt256"]  < UMBRALES["Rectum_V55Gy_lt25"])
    fila["Flag_Rectum_V45Gy_lt45_gt256"]  = int(fila["Rectum_V45Gy_gt256"]  < UMBRALES["Rectum_V45Gy_lt45"])
    fila["Flag_Bladder_V65Gy_lt15_gt256"] = int(fila["Bladder_V65Gy_gt256"] < UMBRALES["Bladder_V65Gy_lt15"])
    fila["Flag_Bladder_V55Gy_lt25_gt256"] = int(fila["Bladder_V55Gy_gt256"] < UMBRALES["Bladder_V55Gy_lt25"])
    fila["Flag_Bladder_V45Gy_lt45_gt256"] = int(fila["Bladder_V45Gy_gt256"] < UMBRALES["Bladder_V45Gy_lt45"])

    fila["s_D95"] = meta.get("s_D95", meta.get("factor_norm"))
    fila["PTV_name_usado"] = meta.get("PTV_name_usado", "")

    return fila


def main():
    parser = argparse.ArgumentParser(description="GT DVH sobre grilla 256x256 (hipofraccionado)")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out-dvh", required=True)
    parser.add_argument("--out-flips", required=True)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    with open(args.splits) as f:
        splits = json.load(f)
    excluidos = set(splits.get("excluidos", []))
    anonids = [a for a in splits["train"] + splits["val"] + splits["test"] if a not in excluidos]

    filas_csv = cargar_csv(Path(args.csv))

    filas_gt = []
    faltantes = []
    for anonid in anonids:
        npz_path = processed_dir / f"{anonid}.npz"
        if not npz_path.exists():
            faltantes.append(anonid)
            continue
        filas_gt.append(calcular_dvh_paciente(npz_path))

    if faltantes:
        log.warning(f"{len(faltantes)} pacientes sin NPZ (no procesados aun), se omiten: {faltantes[:10]}{' ...' if len(faltantes) > 10 else ''}")

    df_gt = pd.DataFrame(filas_gt)
    out_dvh_path = Path(args.out_dvh)
    out_dvh_path.parent.mkdir(parents=True, exist_ok=True)
    df_gt.to_csv(out_dvh_path, index=False)
    log.info(f"GT DVH guardado en {out_dvh_path} ({len(df_gt)} pacientes)")

    # ─── Parte C: cross-check unico vs C# D95norm ────────────────────────────
    flips_rows = []
    resumen_flips = {}

    for tag, (gt_flag_prefix, csv_flag_col, csv_val_col, umbral) in CROSSCHECK_CONSTRAINTS.items():
        gt_flag_col = f"Flag_{gt_flag_prefix}_gt256"
        n_flips = 0
        for _, row in df_gt.iterrows():
            anonid = row["AnonID"]
            csv_row = filas_csv.get(anonid)
            if csv_row is None:
                continue
            gt_flag = int(row[gt_flag_col])
            try:
                csv_flag = int(float(csv_row[csv_flag_col]))
                csv_val = float(csv_row[csv_val_col])
            except (KeyError, ValueError):
                continue
            if gt_flag != csv_flag:
                n_flips += 1
                dist = abs(csv_val - umbral)
                flips_rows.append({
                    "AnonID": anonid,
                    "constraint": tag,
                    "GT_flag_256": gt_flag,
                    "CSharp_flag_D95norm": csv_flag,
                    "CSharp_Vx_D95norm_pct": csv_val,
                    "umbral_pct": umbral,
                    "distancia_al_umbral_pp": round(dist, 3),
                    "GT_Vx_pct_256": round(float(row[f"{gt_flag_prefix.split('_')[0]}_{gt_flag_prefix.split('_')[1]}_gt256"]), 3),
                })
        resumen_flips[tag] = n_flips

    df_flips = pd.DataFrame(flips_rows)
    if not df_flips.empty:
        df_flips = df_flips.sort_values(["constraint", "distancia_al_umbral_pp"], ascending=[True, False])
    out_flips_path = Path(args.out_flips)
    df_flips.to_csv(out_flips_path, index=False)
    log.info(f"Flips guardados en {out_flips_path} ({len(df_flips)} filas)")

    # ─── Resumen consola ──────────────────────────────────────────────────────
    log.info("\n=== Resumen procesado ===")
    log.info(f"N pacientes con GT DVH calculado: {len(df_gt)}")
    log.info(f"N pacientes sin NPZ: {len(faltantes)}")

    log.info("\n=== s_D95 usado ===")
    if len(df_gt):
        log.info(f"min={df_gt['s_D95'].min():.4f} median={df_gt['s_D95'].median():.4f} max={df_gt['s_D95'].max():.4f}")

    log.info("\n=== D95(PTV) sobre dose_norm downsampleada (deberia ser ~100) ===")
    if len(df_gt):
        # dose_norm ya esta en %, D95 en Gy: reconvertir a % para chequeo directo
        d95_pct = df_gt["PTV_D95_Gy_gt256"] / PRESCRIPCION_GY * 100.0
        log.info(f"media={d95_pct.mean():.3f} std={d95_pct.std():.3f} min={d95_pct.min():.3f} max={d95_pct.max():.3f}")

    log.info("\n=== Cross-check GT(256) vs C#(D95norm, nativo) ===")
    for tag, n in resumen_flips.items():
        log.info(f"{tag}: {n} flips")
        subset = df_flips[df_flips["constraint"] == tag] if not df_flips.empty else pd.DataFrame()
        if len(subset):
            lejos = subset[subset["distancia_al_umbral_pp"] > 3.0]
            log.info(f"    {len(subset)} flips totales, {len(lejos)} con distancia al umbral > 3 pp (revisar):")
            for _, r in lejos.iterrows():
                log.info(f"      {r['AnonID']}: dist={r['distancia_al_umbral_pp']} pp (CSV Vx={r['CSharp_Vx_D95norm_pct']}, umbral={r['umbral_pct']})")


if __name__ == "__main__":
    main()
