"""
D5 — Contexto: banda media (40-80%Rx) de Rectum/Bladder en el dataset NORMO
(exp002, checkpoint ganador, test=59 con leak corregido), al lado del numero
hipo ya establecido (results/dvh_curva_completa/summary.json).

Puramente contextual/descriptivo — NO decide nada por si solo (confundido por
fraccionamiento distinto, 78Gy/39fx normo vs. 70Gy/28fx hipo, y por constraints
clinicos distintos entre series). Sirve para ver si el hombro de banda media es
un patron general del enfoque (2D + PSDM + MAE) o algo especifico del dataset
hipo/manual.

Reusa dvh_curva_completa.py sin modificarlo: se monkeypatchea su PROCESSED_DIR
(el modulo lo usa como global, resuelto en tiempo de ejecucion de main() —
funciona porque `from analisis_angular import ... PROCESSED_DIR` liga el nombre
en el namespace de dvh_curva_completa, no en el de analisis_angular) para
apuntar al processed/ normo en vez de processed_hipo/.

Uso:
    .venv/Scripts/python.exe scripts/diagnostico_piso/d5_banda_media_normo.py
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import dvh_curva_completa as dvc  # noqa: E402

PROCESSED_NORMO = Path(r"C:\Pablo\ProstateDoseProject\processed")
CHECKPOINT_NORMO = _REPO_ROOT / "checkpoints/exp002_unet2d_psdm/epoch=191.ckpt"
CONFIG_NORMO = _REPO_ROOT / "configs/exp002_unet2d_psdm.yaml"
SPLITS_NORMO_TEST59 = _REPO_ROOT / "data/splits/splits_v1.json"  # test=59, leak ya corregido

HIPO_SUMMARY_REF = _REPO_ROOT / "results/dvh_curva_completa/summary.json"

OUT_DIR = _REPO_ROOT / "results/diagnostico_piso"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Monkeypatch: dvh_curva_completa.main() usa el nombre PROCESSED_DIR de SU
    # PROPIO namespace de modulo (importado de analisis_angular en tiempo de
    # import) -> reasignarlo aca alcanza, Python resuelve globals en tiempo de
    # ejecucion, no de definicion.
    dvc.PROCESSED_DIR = PROCESSED_NORMO

    print("=== D5 — dvh_curva_completa sobre NORMO (exp002, test=59) ===")
    out_dir_normo = OUT_DIR / "d5_normo_dvh_curva_completa"
    df_normo, summary_normo = dvc.main(
        checkpoint=str(CHECKPOINT_NORMO), config_yaml=str(CONFIG_NORMO),
        out_dir=str(out_dir_normo), splits_path=str(SPLITS_NORMO_TEST59),
    )

    if not HIPO_SUMMARY_REF.exists():
        raise RuntimeError(f"No existe {HIPO_SUMMARY_REF} — correr primero scripts/dvh_curva_completa.py")
    with open(HIPO_SUMMARY_REF) as f:
        summary_hipo = json.load(f)

    comparacion = {}
    for struct in ["Rectum", "Bladder"]:
        media_normo = summary_normo["mean_abs_delta_V_por_estructura"][struct]["mean_abs_delta_V_media_40_80"]
        media_hipo = summary_hipo["mean_abs_delta_V_por_estructura"][struct]["mean_abs_delta_V_media_40_80"]
        comparacion[struct] = {
            "mean_abs_delta_V_media_40_80_normo": media_normo,
            "mean_abs_delta_V_media_40_80_hipo": media_hipo,
            "ratio_hipo_vs_normo": float(media_hipo["mean"] / media_normo["mean"]) if media_normo["mean"] > 0 else float("nan"),
        }

    summary = {
        "n_test_normo": summary_normo["n_test"],
        "n_test_hipo": summary_hipo["n_test"],
        "checkpoint_normo": str(CHECKPOINT_NORMO),
        "config_normo": str(CONFIG_NORMO),
        "splits_normo": str(SPLITS_NORMO_TEST59),
        "nota": "PURAMENTE CONTEXTUAL — normo (78Gy/39fx, RapidPlan) e hipo (70Gy/28fx, manual) "
                "difieren en fraccionamiento y en los constraints clinicos operativos que cada "
                "planificador optimiza; una banda media peor o mejor en un dataset no aisla causa "
                "por si sola (ver D3/D4 para la sonda que SI aisla el efecto de generacion vs. "
                "anatomia). Este numero es solo para ver si el hombro de banda media es un patron "
                "general del enfoque (2D+PSDM+MAE) o especifico de hipo/manual.",
        "comparacion_banda_media_normo_vs_hipo": comparacion,
    }
    with open(OUT_DIR / "d5_banda_media_normo.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== D5 — RESUMEN (banda media 40-80%Rx) ===")
    for struct, c in comparacion.items():
        print(f"{struct}: normo={c['mean_abs_delta_V_media_40_80_normo']['mean']:.2f}pp  "
              f"hipo={c['mean_abs_delta_V_media_40_80_hipo']['mean']:.2f}pp  "
              f"ratio(hipo/normo)={c['ratio_hipo_vs_normo']:.2f}")
    print(f"Guardado: {OUT_DIR / 'd5_banda_media_normo.json'}")
    return summary


if __name__ == "__main__":
    main()
