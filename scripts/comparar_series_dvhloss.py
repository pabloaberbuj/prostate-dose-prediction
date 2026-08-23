"""
Tabla comparativa de 3 vías — 002b (referencia, sin loss DVH) vs. exp_hipo_003_dvhloss
(lambda0=50.94, resultado negativo) vs. exp_hipo_003b_dvhloss_lambda10 (lambda0/5=10.0)
— las mismas 4 varas de comparar_002b_vs_003.py, generalizadas a N corridas.

Requiere que las 3 evaluaciones de cada corrida ya hayan corrido (evaluate.py --dataset
hipo, dvh_curva_completa.py, dvh_signed.py) — ver docstrings de comparar_002b_vs_003.py
y CLAUDE_CODE_CONTEXT.md para los comandos exactos usados en cada corrida.

Uso:
    .venv/Scripts/python.exe scripts/comparar_series_dvhloss.py
"""

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

RUNS = {
    "002b_ref": {
        "dvh_summary": _REPO_ROOT / "results/dvh_curva_completa/summary.json",
        "dvh_signed": _REPO_ROOT / "results/dvh_curva_completa/summary_signed.json",
        "constraints": _REPO_ROOT / "results/exp_hipo_002b_finetune_clean_test_hipo_v2_balanced/exp002_metrics_summary.json",
    },
    "003_lambda50.94": {
        "dvh_summary": _REPO_ROOT / "results/exp_hipo_003_dvhloss_dvh_curva_completa/summary.json",
        "dvh_signed": _REPO_ROOT / "results/exp_hipo_003_dvhloss_dvh_curva_completa/summary_signed.json",
        "constraints": _REPO_ROOT / "results/exp_hipo_003_dvhloss_test_hipo_v2_balanced/metrics_summary.json",
    },
    "003b_lambda10": {
        "dvh_summary": _REPO_ROOT / "results/exp_hipo_003b_dvhloss_lambda10_dvh_curva_completa/summary.json",
        "dvh_signed": _REPO_ROOT / "results/exp_hipo_003b_dvhloss_lambda10_dvh_curva_completa/summary_signed.json",
        "constraints": _REPO_ROOT / "results/exp_hipo_003b_dvhloss_lambda10_test_hipo_v2_balanced/metrics_summary.json",
    },
    "003c_lambda5": {
        "dvh_summary": _REPO_ROOT / "results/exp_hipo_003c_dvhloss_lambda5_dvh_curva_completa/summary.json",
        "dvh_signed": _REPO_ROOT / "results/exp_hipo_003c_dvhloss_lambda5_dvh_curva_completa/summary_signed.json",
        "constraints": _REPO_ROOT / "results/exp_hipo_003c_dvhloss_lambda5_test_hipo_v2_balanced/metrics_summary.json",
    },
}

OUT_PATH = _REPO_ROOT / "results/comparacion_series_dvhloss.json"


def cargar(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}")
    with open(path) as f:
        return json.load(f)


def main():
    datos = {}
    for nombre, paths in RUNS.items():
        datos[nombre] = {k: cargar(p) for k, p in paths.items()}

    nombres = list(RUNS.keys())
    tabla = {}

    # Vara 1: mean|dV| media OAR
    tabla["vara1_mean_abs_delta_v_media_OAR"] = {
        struct: {n: datos[n]["dvh_summary"]["mean_abs_delta_V_por_estructura"][struct]["mean_abs_delta_V_media_40_80"]["mean"]
                 for n in nombres}
        for struct in ["Rectum", "Bladder"]
    }

    # Vara 2: MAE espacial + dose_score
    tabla["vara2_mae_espacial_y_dose_score"] = {
        nombre_metrica: {n: datos[n]["constraints"]["capa1_regresion_dosis"][key]["mean"] for n in nombres}
        for key, nombre_metrica in [("mae_body", "mae_body"), ("mae_ptv", "mae_ptv"),
                                     ("mae_rectum", "mae_rectum"), ("mae_bladder", "mae_bladder"),
                                     ("dose_score_openkbp", "dose_score_openkbp")]
    }

    # Vara 3: signo no-cumple RV65
    tabla["vara3_signo_no_cumple_RV65"] = {
        "signed_mean_media_no_cumple": {n: datos[n]["dvh_signed"]["Rectum"]["desglose_RV65_cumple_vs_no_cumple"]["no_cumple"]["signed_mean_media"] for n in nombres},
        "frac_optimistas_no_cumple": {n: datos[n]["dvh_signed"]["Rectum"]["desglose_RV65_cumple_vs_no_cumple"]["no_cumple"]["frac_pacientes_optimistas_media"] for n in nombres},
        "n_casos_optimistas_no_cumple": {n: round(datos[n]["dvh_signed"]["Rectum"]["desglose_RV65_cumple_vs_no_cumple"]["no_cumple"]["frac_pacientes_optimistas_media"]
                                                    * datos[n]["dvh_signed"]["Rectum"]["desglose_RV65_cumple_vs_no_cumple"]["no_cumple"]["n"])
                                          for n in nombres},
    }

    # Vara 4: constraints operativos
    tabla["vara4_constraints_operativos"] = {
        tag: {
            "auc": {n: datos[n]["constraints"]["capa3_constraints_operativos"][tag]["auc"] for n in nombres},
            "sensibilidad": {n: datos[n]["constraints"]["capa3_constraints_operativos"][tag]["sensibilidad"] for n in nombres},
            "especificidad": {n: datos[n]["constraints"]["capa3_constraints_operativos"][tag]["especificidad"] for n in nombres},
        }
        for tag in ["RV65", "RV55", "BV65"]
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(tabla, f, indent=2)
    print(f"Guardado: {OUT_PATH}\n")

    print("=== VARA 1 - mean|deltaV| banda media OAR ===")
    for struct, vals in tabla["vara1_mean_abs_delta_v_media_OAR"].items():
        print(f"  {struct}: " + "  ".join(f"{n}={vals[n]:.2f}pp" for n in nombres))

    print("\n=== VARA 2 - MAE espacial + dose_score ===")
    for metrica, vals in tabla["vara2_mae_espacial_y_dose_score"].items():
        print(f"  {metrica}: " + "  ".join(f"{n}={vals[n]:.3f}" for n in nombres))

    print("\n=== VARA 3 - signo no-cumple RV65 ===")
    for metrica, vals in tabla["vara3_signo_no_cumple_RV65"].items():
        print(f"  {metrica}: " + "  ".join(f"{n}={vals[n]:.2f}" if isinstance(vals[n], float) else f"{n}={vals[n]}" for n in nombres))

    print("\n=== VARA 4 - constraints operativos (AUC) ===")
    for tag, r in tabla["vara4_constraints_operativos"].items():
        print(f"  {tag}: " + "  ".join(f"{n}={r['auc'][n]:.3f}" for n in nombres))

    return tabla


if __name__ == "__main__":
    main()
