"""
D1 — Gap train vs. test del hombro de OAR (banda media 40-80%Rx).

Tarea de DIAGNOSTICO puro: NO entrena, NO toca configs/splits. Inferencia de
exp_hipo_002b_finetune_clean sobre el TRAIN hipo (98 casos, splits_hipo_v2_clean_balanced.json)
ademas del test (n=31, ya calculado en results/dvh_curva_completa/summary.json — se reusa
tal cual, mismo checkpoint/config/split, no se recalcula para no duplicar trabajo).

Reusa scripts/dvh_curva_completa.py sin modificarlo: se arma un splits.json temporal donde
"test" apunta a los 98 IDs de train (mismo formato que espera dvh_curva_completa.main()) y se
llama a esa funcion tal cual con out_dir propio.

Veredicto:
- train ~ test y ambos altos -> NO es generalizacion (underfitting o piso de generacion).
  Augmentation/mas N no ayudaria.
- train bajo, test alto -> gap de generalizacion -> N/augmentation es la palanca.

Uso:
    .venv/Scripts/python.exe scripts/diagnostico_piso/d1_gap_train_test.py
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import dvh_curva_completa as dvc  # noqa: E402
from analisis_angular import CHECKPOINT, CONFIG_YAML  # noqa: E402

SPLITS_REF = _REPO_ROOT / "data/splits/splits_hipo_v2_clean_balanced.json"
TEST_SUMMARY_REF = _REPO_ROOT / "results/dvh_curva_completa/summary.json"

OUT_DIR = _REPO_ROOT / "results/diagnostico_piso"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_SPLITS_TRAIN = OUT_DIR / "_tmp_splits_train_as_test.json"

STRUCTS_INTERES = ["Rectum", "Bladder"]
# Umbrales del veredicto: "similar" = dentro de este ratio test/train
RATIO_SIMILAR_LO, RATIO_SIMILAR_HI = 0.8, 1.25


def main():
    with open(SPLITS_REF) as f:
        splits_ref = json.load(f)
    ids_train = splits_ref["train"]
    print(f"Train hipo (splits_hipo_v2_clean_balanced.json): {len(ids_train)} pacientes")

    # splits temporal: mismo archivo, pero "test" = train, para reusar dvh_curva_completa.main()
    # sin tocar el script ni el splits real.
    splits_tmp = dict(splits_ref)
    splits_tmp["test"] = ids_train
    splits_tmp["_nota"] = "splits temporal D1 — 'test' apunta a los IDs de TRAIN real, solo para reusar main()"
    with open(TMP_SPLITS_TRAIN, "w") as f:
        json.dump(splits_tmp, f, indent=2)

    print("\n=== Inferencia sobre TRAIN hipo (98 casos), checkpoint exp_hipo_002b_finetune_clean ===")
    out_dir_train = OUT_DIR / "d1_train_inferencia"
    df_train, summary_train = dvc.main(
        checkpoint=str(CHECKPOINT), config_yaml=str(CONFIG_YAML),
        out_dir=str(out_dir_train), splits_path=str(TMP_SPLITS_TRAIN),
    )

    if not TEST_SUMMARY_REF.exists():
        raise RuntimeError(
            f"No existe {TEST_SUMMARY_REF} — correr primero scripts/dvh_curva_completa.py "
            "(vara de test ya establecida en el proyecto, se reusa aca sin recalcular)."
        )
    with open(TEST_SUMMARY_REF) as f:
        summary_test = json.load(f)

    resumen = {}
    for struct in STRUCTS_INTERES:
        media_train = summary_train["mean_abs_delta_V_por_estructura"][struct]["mean_abs_delta_V_media_40_80"]
        media_test = summary_test["mean_abs_delta_V_por_estructura"][struct]["mean_abs_delta_V_media_40_80"]
        ratio_test_train = media_test["mean"] / media_train["mean"] if media_train["mean"] > 0 else float("nan")

        if RATIO_SIMILAR_LO <= ratio_test_train <= RATIO_SIMILAR_HI:
            veredicto = "NO ES GENERALIZACION — train~test (ambos comparables): underfitting o piso de generacion. Augmentation/mas N no seria la palanca."
        elif ratio_test_train > RATIO_SIMILAR_HI:
            veredicto = "GAP DE GENERALIZACION — test bien peor que train: N/augmentation SI seria la palanca."
        else:
            veredicto = "test MEJOR que train (inusual) — revisar manualmente, no interpretar como piso ni generalizacion estandar."

        resumen[struct] = {
            "mean_abs_delta_V_media_40_80_train": media_train,
            "mean_abs_delta_V_media_40_80_test": media_test,
            "ratio_test_vs_train": float(ratio_test_train),
            "veredicto": veredicto,
        }

    summary = {
        "n_train": len(ids_train),
        "n_test": summary_test["n_test"],
        "checkpoint": str(CHECKPOINT),
        "config": str(CONFIG_YAML),
        "splits_ref": str(SPLITS_REF),
        "nota_metodologica": "El checkpoint fue seleccionado (early stopping) mirando val, no train ni test — "
                              "la comparacion train-vs-test de mean|deltaV| sigue siendo valida para diagnosticar "
                              "gap de generalizacion (el modelo NUNCA actualizo pesos viendo el error de DVH-curva-"
                              "completa directamente, solo MAE voxel-wise durante entrenamiento).",
        "banda_evaluada": "media (40-80% Rx)",
        "umbral_ratio_similar": [RATIO_SIMILAR_LO, RATIO_SIMILAR_HI],
        "resultado_por_estructura": resumen,
    }
    with open(OUT_DIR / "d1_gap_train_test.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== D1 — RESUMEN ===")
    for struct, r in resumen.items():
        print(f"{struct}: train={r['mean_abs_delta_V_media_40_80_train']['mean']:.2f}pp  "
              f"test={r['mean_abs_delta_V_media_40_80_test']['mean']:.2f}pp  "
              f"ratio={r['ratio_test_vs_train']:.2f}")
        print(f"  -> {r['veredicto']}")

    TMP_SPLITS_TRAIN.unlink(missing_ok=True)
    print(f"\nGuardado: {OUT_DIR / 'd1_gap_train_test.json'}")
    return summary


if __name__ == "__main__":
    main()
