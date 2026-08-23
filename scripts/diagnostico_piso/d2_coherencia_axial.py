"""
D2 — Coherencia axial (cráneo-caudal) del hombro de OAR (banda media 40-80%Rx).

Pregunta: el modelo es 2D puro (cada corte se procesa independiente, ver
src/models/lightning_module.py::forward — aplana B*Z y corre convs 2D, sin ningún
kernel que cruce Z). ¿Esto introduce discontinuidad en Z que la dosis real (campo
físico continuo) no tiene, o al revés, el modelo pierde estructura en Z que la
dosis real sí tiene? Se responde con 3 métricas complementarias, real vs. predicha
(renormalizada igual que dvh_curva_completa.py), sobre Rectum/Bladder, test hipo
(n=31, checkpoint exp_hipo_002b_finetune_clean):

  (a) Perfil D(z) en un punto XY FIJO (centroide 3D del órgano, mismo pixel para
      todos los cortes) — "rugosidad" = mean(|diff en z|).
  (b) Perfil de dosis MEDIA por corte dentro del órgano (más robusto que un solo
      pixel) — misma rugosidad.
  (c) Perfil de %volumen del corte en banda media [40,80]%Rx (liga esto
      directamente al hombro de banda media ya medido en dvh_curva_completa.py,
      no solo a "dosis" en general) — misma rugosidad.

Para cada métrica: ratio_pred_vs_real de la rugosidad + test de Wilcoxon pareado
(por paciente) + correlación de Spearman entre los perfiles real/pred (si el
modelo sigue la MISMA modulación en z, aunque suavizada, rho debería ser alto).

Diagnostico puro: NO entrena, NO toca configs/splits. Reusa cargar_npz/
inferir_dosis (analisis_angular.py) y d95_ptv (dvh_curva_completa.py).

Uso:
    .venv/Scripts/python.exe scripts/diagnostico_piso/d2_coherencia_axial.py
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
from scipy import stats
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from analisis_angular import (  # noqa: E402
    cargar_npz, inferir_dosis, calcular_n_slices_target,
    CHECKPOINT, CONFIG_YAML, PROCESSED_DIR,
)
from evaluate import cargar_modelo  # noqa: E402
from dvh_curva_completa import d95_ptv  # noqa: E402

SPLITS_PATH = _REPO_ROOT / "data/splits/splits_hipo_v2_clean_balanced.json"
OUT_DIR = _REPO_ROOT / "results/diagnostico_piso"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

STRUCTS = ["Rectum", "Bladder"]
MASK_KEY = {"Rectum": "rectum_mask", "Bladder": "bladder_mask"}
MIN_PIXELS_SLICE = 10   # min area del organo en un corte para usarlo en el perfil z
BANDA_MEDIA = (40.0, 80.0)  # %Rx, misma banda que dvh_curva_completa.py


def dose_pred_renorm(arrays: dict, dose_pred_raw: np.ndarray) -> np.ndarray:
    """Mismo criterio que dvh_curva_completa.analizar_paciente: clip a 0 (ruido
    negativo del modelo sin activacion final) + renorm a D95(PTV_pred)=100%."""
    dose_pred_raw = np.clip(dose_pred_raw, 0.0, None)
    d95_pred_raw = d95_ptv(dose_pred_raw, arrays["ptv_mask"])
    factor = 100.0 / d95_pred_raw if d95_pred_raw > 0 else float("nan")
    return dose_pred_raw * factor


def centroide_3d(mask: np.ndarray):
    """(row, col) promedio de TODOS los voxeles de la mascara en TODOS los cortes
    -> un unico punto XY fijo para toda la columna z (pedido explicito de la tarea)."""
    zz, rr, cc = np.nonzero(mask)
    if len(zz) == 0:
        return None
    return int(round(rr.mean())), int(round(cc.mean()))


def rugosidad(perfil: np.ndarray) -> float:
    """mean(|diff|) a lo largo de z. NaN si <2 puntos validos."""
    v = perfil[~np.isnan(perfil)]
    if len(v) < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(v))))


def perfiles_paciente(arrays: dict, dose_real: np.ndarray, dose_pred: np.ndarray, struct: str) -> dict:
    mask = arrays[MASK_KEY[struct]]
    z_validos = [z for z in range(mask.shape[0]) if mask[z].sum() >= MIN_PIXELS_SLICE]
    if len(z_validos) < 3:
        return None

    # (a) XY fijo: centroide 3D del organo, mismo pixel para todos los z_validos
    cen = centroide_3d(mask)
    if cen is None:
        return None
    row_c, col_c = cen
    perfil_xy_real = np.array([dose_real[z, row_c, col_c] for z in z_validos])
    perfil_xy_pred = np.array([dose_pred[z, row_c, col_c] for z in z_validos])

    # (b) dosis media por corte dentro del organo
    perfil_meandose_real = np.array([dose_real[z][mask[z] > 0].mean() for z in z_validos])
    perfil_meandose_pred = np.array([dose_pred[z][mask[z] > 0].mean() for z in z_validos])

    # (c) %volumen del corte en banda media [40,80]%Rx
    lo, hi = BANDA_MEDIA
    def vfrac_media(dose_vol, z):
        vals = dose_vol[z][mask[z] > 0]
        return 100.0 * float(np.mean((vals > lo) & (vals <= hi)))
    perfil_vfrac_real = np.array([vfrac_media(dose_real, z) for z in z_validos])
    perfil_vfrac_pred = np.array([vfrac_media(dose_pred, z) for z in z_validos])

    def rho_seguro(a, b):
        if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return float("nan")
        return float(stats.spearmanr(a, b).statistic)

    return {
        "z_validos": z_validos,
        "centroide_xy": [row_c, col_c],
        "n_slices": len(z_validos),
        "perfil_xy_real": perfil_xy_real, "perfil_xy_pred": perfil_xy_pred,
        "perfil_meandose_real": perfil_meandose_real, "perfil_meandose_pred": perfil_meandose_pred,
        "perfil_vfrac_real": perfil_vfrac_real, "perfil_vfrac_pred": perfil_vfrac_pred,
        "rugosidad_xy_real": rugosidad(perfil_xy_real), "rugosidad_xy_pred": rugosidad(perfil_xy_pred),
        "rugosidad_meandose_real": rugosidad(perfil_meandose_real), "rugosidad_meandose_pred": rugosidad(perfil_meandose_pred),
        "rugosidad_vfrac_real": rugosidad(perfil_vfrac_real), "rugosidad_vfrac_pred": rugosidad(perfil_vfrac_pred),
        "rho_xy": rho_seguro(perfil_xy_real, perfil_xy_pred),
        "rho_meandose": rho_seguro(perfil_meandose_real, perfil_meandose_pred),
        "rho_vfrac": rho_seguro(perfil_vfrac_real, perfil_vfrac_pred),
    }


def plot_ejemplos(df: pd.DataFrame, crudos: dict, struct: str, metrica: str, plots_dir: Path):
    """3 pacientes representativos (bajo/medio/alto ratio_pred_vs_real de rugosidad_meandose)."""
    col_ratio = f"ratio_{metrica}_{struct}"
    sub = df[["AnonID", col_ratio]].dropna().sort_values(col_ratio)
    if len(sub) < 3:
        return
    elegidos = {"bajo (modelo mas suave)": sub.iloc[0]["AnonID"],
                "mediano": sub.iloc[len(sub) // 2]["AnonID"],
                "alto (modelo mas rugoso)": sub.iloc[-1]["AnonID"]}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (etiqueta, anonid) in zip(axes, elegidos.items()):
        c = crudos[anonid][struct]
        z = c["z_validos"]
        real = c[f"perfil_{metrica}_real"]
        pred = c[f"perfil_{metrica}_pred"]
        ax.plot(z, real, "o-", color="black", ms=3, label="Real")
        ax.plot(z, pred, "o--", color="firebrick", ms=3, label="Predicha")
        ratio = df.loc[df["AnonID"] == anonid, col_ratio].values[0]
        ax.set_title(f"{etiqueta}\n{anonid} (ratio={ratio:.2f})", fontsize=9)
        ax.set_xlabel("Corte Z (indice nativo)")
        ax.set_ylabel(metrica)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f"D2 — Perfil en Z ({metrica}) real vs. predicha — {struct}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(plots_dir / f"d2_zprofile_{metrica}_{struct.lower()}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(checkpoint=None, config_yaml=None, splits_path=None, processed_dir=None, out_dir=None):
    """Parametrizable (mismo patron que dvh_curva_completa.main) para poder correr
    D2 contra cualquier checkpoint/config/split/processed_dir — no solo el hipo
    original hardcodeado en los defaults. Ver scripts/diagnostico_piso/d5_banda_media_normo.py
    para el precedente de reuso sobre normo."""
    checkpoint = Path(checkpoint) if checkpoint else CHECKPOINT
    config_yaml = Path(config_yaml) if config_yaml else CONFIG_YAML
    splits_path = Path(splits_path) if splits_path else SPLITS_PATH
    processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_path) as f:
        splits = json.load(f)
    ids_test = splits["test"]
    print(f"Pacientes test: {len(ids_test)} (splits={splits_path})")

    cfg = OmegaConf.load(str(config_yaml))
    cfg.data.processed_dir = str(processed_dir)
    torch.set_float32_matmul_precision("high")
    model, device = cargar_modelo(str(checkpoint), cfg)

    pad_z_to = None
    if cfg.model.arch == "unet3d":
        # Ver docstring de inferir_dosis / dvh_curva_completa.py -- el 3D fue
        # entrenado con Z siempre padeado a un tamano fijo, sin esto el MAE
        # (y por lo tanto cualquier perfil en Z que mida D2) queda invalido.
        pad_z_to = calcular_n_slices_target(processed_dir, splits_path)
        print(f"Modelo 3D detectado -- pad_z_to={pad_z_to}")

    filas = []
    crudos = {}
    for anonid in tqdm(ids_test, desc="D2 — coherencia axial"):
        arrays, meta = cargar_npz(processed_dir / f"{anonid}.npz")
        dose_pred_raw = inferir_dosis(model, device, arrays, pad_z_to=pad_z_to)
        dose_pred = dose_pred_renorm(arrays, dose_pred_raw)
        dose_real = arrays["dose"]

        fila = {"AnonID": anonid}
        crudos[anonid] = {}
        for struct in STRUCTS:
            r = perfiles_paciente(arrays, dose_real, dose_pred, struct)
            if r is None:
                continue
            crudos[anonid][struct] = r
            fila[f"n_slices_{struct}"] = r["n_slices"]
            for metrica in ["xy", "meandose", "vfrac"]:
                rr = r[f"rugosidad_{metrica}_real"]
                rp = r[f"rugosidad_{metrica}_pred"]
                fila[f"rugosidad_{metrica}_real_{struct}"] = rr
                fila[f"rugosidad_{metrica}_pred_{struct}"] = rp
                fila[f"ratio_{metrica}_{struct}"] = rp / rr if rr and rr > 1e-9 else float("nan")
                fila[f"rho_{metrica}_{struct}"] = r[f"rho_{metrica}"]
        filas.append(fila)

    df = pd.DataFrame(filas)
    df.to_csv(out_dir / "d2_coherencia_axial_por_paciente.csv", index=False)

    resultados = {}
    for struct in STRUCTS:
        resultados[struct] = {}
        for metrica in ["xy", "meandose", "vfrac"]:
            real = df[f"rugosidad_{metrica}_real_{struct}"].dropna()
            pred = df.loc[real.index, f"rugosidad_{metrica}_pred_{struct}"]
            rho = df[f"rho_{metrica}_{struct}"].dropna()
            if len(real) > 3:
                w = stats.wilcoxon(pred, real)
                p = float(w.pvalue)
            else:
                p = float("nan")
            ratio_mean = float((pred / real).mean())
            if p < 0.05 and ratio_mean > 1.3:
                veredicto = "modelo MAS RUGOSO que la realidad en Z (discontinuidad espuria introducida por el 2D)"
            elif p < 0.05 and ratio_mean < 0.7:
                veredicto = "modelo MAS SUAVE que la realidad en Z (pierde estructura Z real, filtro pasa-bajos, consistente con H4 del analisis angular)"
            else:
                veredicto = "sin diferencia significativa de rugosidad Z (el 2D no introduce ni pierde discontinuidad notable en esta metrica)"
            resultados[struct][metrica] = {
                "n": int(len(real)),
                "rugosidad_real_mean": float(real.mean()), "rugosidad_real_std": float(real.std()),
                "rugosidad_pred_mean": float(pred.mean()), "rugosidad_pred_std": float(pred.std()),
                "ratio_pred_vs_real_mean": ratio_mean,
                "wilcoxon_p": p,
                "spearman_rho_perfil_real_vs_pred_mean": float(rho.mean()) if len(rho) else float("nan"),
                "spearman_rho_perfil_real_vs_pred_std": float(rho.std()) if len(rho) else float("nan"),
                "veredicto": veredicto,
            }
        plot_ejemplos(df, crudos, struct, "meandose", plots_dir)
        plot_ejemplos(df, crudos, struct, "vfrac", plots_dir)

    # Veredicto global (banda media, metrica vfrac — la mas ligada al hombro clinico)
    veredictos_vfrac = [resultados[s]["vfrac"]["veredicto"] for s in STRUCTS]
    motivado_3d = any("MAS RUGOSO" in v or "MAS SUAVE" in v for v in veredictos_vfrac)

    summary = {
        "n_test": len(ids_test),
        "checkpoint": str(checkpoint),
        "config": str(config_yaml),
        "splits": str(splits_path),
        "processed_dir": str(processed_dir),
        "banda_media_pct_rx": list(BANDA_MEDIA),
        "min_pixels_slice": MIN_PIXELS_SLICE,
        "metodologia": {
            "xy": "perfil D(z) en un pixel FIJO (centroide 3D del organo, mismo (row,col) para todos los cortes)",
            "meandose": "perfil de dosis MEDIA del organo por corte (mas robusto que un solo pixel)",
            "vfrac": "perfil de %volumen del corte del organo en banda media [40,80]%Rx (liga directo al hombro clinico de dvh_curva_completa.py)",
            "rugosidad": "mean(|diff(perfil)|) a lo largo de Z",
            "veredicto_3d": "ratio_pred_vs_real >1.3 (mas rugoso) o <0.7 (mas suave) con Wilcoxon p<0.05 -> el 2D no reproduce bien la coherencia axial real -> favorece 3D U-Net para ESTE hombro. Si ratio~1 y sin p sig., el 2D ya es axial-consistente para esta banda -> NO motiva 3D por esta via.",
        },
        "resultados_por_estructura": resultados,
        "veredicto_D2_motiva_3d_unet_para_hombro_banda_media": bool(motivado_3d),
    }
    with open(out_dir / "d2_coherencia_axial.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== D2 — RESUMEN (banda media, metrica vfrac) ===")
    for struct in STRUCTS:
        r = resultados[struct]["vfrac"]
        print(f"{struct}: rugosidad_real={r['rugosidad_real_mean']:.2f}  rugosidad_pred={r['rugosidad_pred_mean']:.2f}  "
              f"ratio={r['ratio_pred_vs_real_mean']:.2f}  p={r['wilcoxon_p']:.4f}  rho={r['spearman_rho_perfil_real_vs_pred_mean']:.2f}")
        print(f"  -> {r['veredicto']}")
    print(f"\n¿Motiva 3D U-Net para este hombro?: {motivado_3d}")
    print(f"Guardado: {out_dir / 'd2_coherencia_axial.json'}")
    return df, summary


if __name__ == "__main__":
    main()
