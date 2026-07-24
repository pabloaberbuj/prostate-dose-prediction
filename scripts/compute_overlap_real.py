"""
Auditoría del bug de solapamiento PTV-Recto (ver "Correccion overlap.md").

Recalcula el overlap PTV-Rectum real (interseccion geometrica de mascaras)
desde los NPZ preprocesados, para los 385 pacientes del dataset normofraccionado,
y lo compara contra los terciles actuales (basados en el bug Math.Min de
extract_dicom_csharp/Program.cs, guardado en Solap_PTV_Rectum_cc).

Ambos pct_solap usan el mismo denominador (VolPTV_cc de metricas_planes.csv,
ESAPI) para que la comparacion de terciles sea apples-to-apples: lo unico que
cambia es el numerador (interseccion real vs Math.Min).

Correccion de resolucion: las mascaras en el NPZ estan downsampleadas a
256x256 in-plane, pero meta['spacing_mm'] es el spacing NATIVO de la CT
(pre-downsample) -- usarlo directo sobre el array 256x256 da un volumen
equivocado por un factor ~(tamano_original/256)^2, que varia por paciente.
Se calibra el volumen por voxel con meta['vol_ptv_cc'] (volumen nativo de
PTV, calculado en preprocess.py ANTES de downsamplear) dividido por la
cantidad de voxels de PTV en la mascara ya downsampleada.

Uso:
    python scripts/compute_overlap_real.py \
        --processed-dir  ../processed \
        --splits         data/splits/splits_v1.json \
        --plan-metrics   data/metricas_planes.csv \
        --output-csv     data/overlap_real_normo.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def cargar_anonids(splits_path: Path) -> list:
    with open(splits_path) as f:
        splits = json.load(f)
    excluidos = set(splits.get("excluidos", []))
    todos = splits["train"] + splits["val"] + splits["test"]

    # ⚠️ splits_v1.json tiene al menos un AnonID repetido ENTRE splits (ver
    # "Correccion overlap.md" — hallazgo durante esta auditoría, no el bug de
    # Math.Min). Deduplicar acá evita filas duplicadas en el CSV de salida,
    # pero NO corrige el leak train/val/test en sí -- eso requiere decisión
    # aparte (no tocar splits_v1.json sin consultar).
    vistos = set()
    unicos = []
    for a in todos:
        if a in excluidos or a in vistos:
            continue
        vistos.add(a)
        unicos.append(a)
    return unicos


def overlap_real_paciente(npz_path: Path) -> dict:
    d = np.load(npz_path)
    meta = json.loads(d["meta"][0])

    ptv = d["ptv_mask"] > 0
    rectum = d["rectum_mask"] > 0

    n_ptv_vox = int(ptv.sum())
    if n_ptv_vox == 0:
        raise ValueError("ptv_mask vacia en el NPZ (post-downsample)")

    # Volumen por voxel calibrado con el volumen NATIVO de PTV (pre-downsample,
    # guardado en meta por preprocess.py), no con spacing_mm crudo.
    voxel_vol_cc = meta["vol_ptv_cc"] / n_ptv_vox

    overlap_vox = int((ptv & rectum).sum())
    overlap_cc_real = overlap_vox * voxel_vol_cc

    return {
        "anonid": meta["anonid"],
        "vol_ptv_cc_npz": meta["vol_ptv_cc"],
        "n_ptv_vox_npz": n_ptv_vox,
        "n_rectum_vox_npz": int(rectum.sum()),
        "overlap_vox_npz": overlap_vox,
        "overlap_ptv_rectum_cc_real": round(overlap_cc_real, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--plan-metrics", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    anonids = cargar_anonids(Path(args.splits))
    print(f"Pacientes en splits (excluidos ya filtrados): {len(anonids)}")

    filas = []
    errores = []
    for anonid in anonids:
        npz_path = processed_dir / f"{anonid}.npz"
        if not npz_path.exists():
            errores.append((anonid, "NPZ no encontrado"))
            continue
        try:
            filas.append(overlap_real_paciente(npz_path))
        except Exception as e:
            errores.append((anonid, str(e)))

    if errores:
        print(f"\n{len(errores)} pacientes con error:")
        for anonid, err in errores:
            print(f"  {anonid}: {err}")

    df_real = pd.DataFrame(filas)
    print(f"\nOverlap real calculado para {len(df_real)} pacientes.")

    # ── Cruzar con metricas_planes.csv (overlap bugueado + VolPTV_cc ESAPI)
    df_plan = pd.read_csv(args.plan_metrics, sep=";")
    df_plan = df_plan.drop_duplicates(subset="AnonID", keep="first")
    df = df_real.merge(
        df_plan[["AnonID", "VolPTV_cc", "VolRectum_cc", "Solap_PTV_Rectum_cc"]],
        left_on="anonid", right_on="AnonID", how="left",
    )
    faltantes = df[df["VolPTV_cc"].isna()]["anonid"].tolist()
    if faltantes:
        print(f"\n⚠️  {len(faltantes)} pacientes sin fila en metricas_planes.csv: {faltantes}")

    # pct_solap con el MISMO denominador (VolPTV_cc de ESAPI) para comparar terciles
    df["pct_solap_viejo_bug"] = df["Solap_PTV_Rectum_cc"] / df["VolPTV_cc"] * 100
    df["pct_solap_real"] = df["overlap_ptv_rectum_cc_real"] / df["VolPTV_cc"] * 100

    df_validos = df.dropna(subset=["pct_solap_viejo_bug", "pct_solap_real"]).copy()

    df_validos["tercil_viejo"] = pd.qcut(
        df_validos["pct_solap_viejo_bug"], 3, labels=["bajo", "medio", "alto"]
    )
    df_validos["tercil_nuevo"] = pd.qcut(
        df_validos["pct_solap_real"], 3, labels=["bajo", "medio", "alto"]
    )

    print(f"\nLimites terciles VIEJO (bug Math.Min): {pd.qcut(df_validos['pct_solap_viejo_bug'], 3).cat.categories.tolist()}")
    print(f"Limites terciles NUEVO (interseccion real): {pd.qcut(df_validos['pct_solap_real'], 3).cat.categories.tolist()}")

    print("\n=== TABLA DE CONFUSION (tercil viejo vs tercil nuevo) ===")
    confusion = pd.crosstab(df_validos["tercil_viejo"], df_validos["tercil_nuevo"],
                             rownames=["viejo"], colnames=["nuevo"])
    print(confusion)

    n_cambia = int((df_validos["tercil_viejo"] != df_validos["tercil_nuevo"]).sum())
    n_total = len(df_validos)
    print(f"\nPacientes que cambian de tercil: {n_cambia}/{n_total} ({100*n_cambia/n_total:.1f}%)")

    # ── Guardar CSV de salida
    out_cols = ["anonid", "overlap_ptv_rectum_cc_real", "vol_ptv_cc_npz",
                "VolPTV_cc", "VolRectum_cc", "Solap_PTV_Rectum_cc",
                "pct_solap_viejo_bug", "pct_solap_real",
                "tercil_viejo", "tercil_nuevo"]
    df_out = df_validos[out_cols].rename(columns={"anonid": "AnonID"})
    df_out.to_csv(args.output_csv, sep=";", index=False)
    print(f"\nCSV guardado en {args.output_csv}")

    confusion_path = str(Path(args.output_csv).with_name("overlap_real_confusion_terciles.csv"))
    confusion.to_csv(confusion_path, sep=";")
    print(f"Tabla de confusion guardada en {confusion_path}")


if __name__ == "__main__":
    main()
