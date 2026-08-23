"""
Tabla comparativa 002b (referencia) vs. exp_hipo_003_dvhloss — las 4 varas pedidas:
  1. mean|ΔV| banda media Recto/Vejiga (dvh_curva_completa.py) — DEBE bajar.
  2. MAE espacial por estructura + dose_score OpenKBP (evaluate_hipo.py) — NO debe subir.
  3. Signo (dvh_signed.py): signed bias media en no-cumple RV65 + conteo de casos
     optimistas duros entre esos mismos pacientes (hoy 3/10 con 002b) — que no aumente.
  4. Constraints operativos (RV65/RV55/BV65 AUC/sens/esp, umbral val-frozen).

Requiere que las 3 evaluaciones de exp_hipo_003 ya hayan corrido:
    python scripts/evaluate.py --dataset hipo --exp exp_hipo_003_dvhloss \
        --checkpoint <ckpt> --splits-hipo data/splits/splits_hipo_v2_clean_balanced.json \
        --output-dir results/exp_hipo_003_dvhloss_test_hipo_v2_balanced
    python scripts/dvh_curva_completa.py --checkpoint <ckpt> \
        --config configs/exp_hipo_003_dvhloss.yaml \
        --out-dir results/exp_hipo_003_dvhloss_dvh_curva_completa
    python scripts/dvh_signed.py --checkpoint <ckpt> \
        --config configs/exp_hipo_003_dvhloss.yaml \
        --out-dir results/exp_hipo_003_dvhloss_dvh_curva_completa

Uso:
    .venv/Scripts/python.exe scripts/comparar_002b_vs_003.py
"""

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Referencia (002b, ya calculado en tareas anteriores) ──────────────────────
REF_DVH_SUMMARY = _REPO_ROOT / "results/dvh_curva_completa/summary.json"
REF_DVH_SIGNED = _REPO_ROOT / "results/dvh_curva_completa/summary_signed.json"
REF_CONSTRAINTS = _REPO_ROOT / "results/exp_hipo_002b_finetune_clean_test_hipo_v2_balanced/exp002_metrics_summary.json"

# ── Nuevo (exp_hipo_003_dvhloss) ───────────────────────────────────────────────
NEW_DVH_SUMMARY = _REPO_ROOT / "results/exp_hipo_003_dvhloss_dvh_curva_completa/summary.json"
NEW_DVH_SIGNED = _REPO_ROOT / "results/exp_hipo_003_dvhloss_dvh_curva_completa/summary_signed.json"
NEW_CONSTRAINTS = _REPO_ROOT / "results/exp_hipo_003_dvhloss_test_hipo_v2_balanced/metrics_summary.json"

OUT_PATH = _REPO_ROOT / "results/exp_hipo_003_dvhloss_test_hipo_v2_balanced/comparacion_002b_vs_003.json"


def cargar(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Falta {path} — correr primero las 3 evaluaciones de exp_hipo_003 "
            f"(ver docstring de este script)."
        )
    with open(path) as f:
        return json.load(f)


def vara1_mean_abs_delta_v(ref_dvh: dict, new_dvh: dict) -> dict:
    out = {}
    for struct in ["Rectum", "Bladder"]:
        r = ref_dvh["mean_abs_delta_V_por_estructura"][struct]["mean_abs_delta_V_media_40_80"]["mean"]
        n = new_dvh["mean_abs_delta_V_por_estructura"][struct]["mean_abs_delta_V_media_40_80"]["mean"]
        out[struct] = {
            "002b": r, "003_dvhloss": n,
            "delta_pp": n - r, "delta_relativo_pct": 100.0 * (n - r) / r if r else float("nan"),
            "mejoro": bool(n < r),
        }
    return out


def vara2_mae_espacial(ref_constraints: dict, new_constraints: dict) -> dict:
    out = {}
    for struct_key, nombre in [("mae_body", "body"), ("mae_ptv", "ptv"),
                                ("mae_rectum", "rectum"), ("mae_bladder", "bladder"),
                                ("dose_score_openkbp", "dose_score_openkbp")]:
        r = ref_constraints["capa1_regresion_dosis"][struct_key]["mean"]
        n = new_constraints["capa1_regresion_dosis"][struct_key]["mean"]
        out[nombre] = {
            "002b": r, "003_dvhloss": n,
            "delta": n - r, "empeoro": bool(n > r),
        }
    return out


def vara3_signo_no_cumple(ref_signed: dict, new_signed: dict) -> dict:
    ref_g = ref_signed["Rectum"]["desglose_RV65_cumple_vs_no_cumple"]["no_cumple"]
    new_g = new_signed["Rectum"]["desglose_RV65_cumple_vs_no_cumple"]["no_cumple"]
    n_ref = ref_g["n"]
    n_new = new_g["n"]
    frac_optimistas_ref = ref_g["frac_pacientes_optimistas_media"]
    frac_optimistas_new = new_g["frac_pacientes_optimistas_media"]
    return {
        "n_no_cumple_RV65": {"002b": n_ref, "003_dvhloss": n_new},
        "signed_mean_media_no_cumple": {
            "002b": ref_g["signed_mean_media"], "003_dvhloss": new_g["signed_mean_media"],
        },
        "frac_optimistas_no_cumple": {
            "002b": frac_optimistas_ref, "003_dvhloss": frac_optimistas_new,
        },
        "n_casos_optimistas_no_cumple": {
            "002b": round(frac_optimistas_ref * n_ref), "003_dvhloss": round(frac_optimistas_new * n_new),
        },
        "conteo_optimistas_no_aumento": bool(round(frac_optimistas_new * n_new) <= round(frac_optimistas_ref * n_ref)),
        "ALERTA_optimista_en_no_cumple_003": new_signed["Rectum"]["desglose_RV65_cumple_vs_no_cumple"]["ALERTA_optimista_en_no_cumple"],
    }


def vara4_constraints_operativos(ref_constraints: dict, new_constraints: dict) -> dict:
    out = {}
    for tag in ["RV65", "RV55", "BV65"]:
        r = ref_constraints["capa3_constraints_operativos"][tag]
        n = new_constraints["capa3_constraints_operativos"][tag]
        out[tag] = {
            "auc":           {"002b": r["auc"], "003_dvhloss": n["auc"]},
            "sensibilidad":  {"002b": r["sensibilidad"], "003_dvhloss": n["sensibilidad"]},
            "especificidad": {"002b": r["especificidad"], "003_dvhloss": n["especificidad"]},
            "auc_se_rompio": bool(n["auc"] < r["auc"] - 0.05),  # tolerancia, no ruido de bootstrap
        }
    return out


def main():
    ref_dvh = cargar(REF_DVH_SUMMARY)
    ref_signed = cargar(REF_DVH_SIGNED)
    ref_constraints = cargar(REF_CONSTRAINTS)
    new_dvh = cargar(NEW_DVH_SUMMARY)
    new_signed = cargar(NEW_DVH_SIGNED)
    new_constraints = cargar(NEW_CONSTRAINTS)

    comparacion = {
        "vara1_mean_abs_delta_v_media_OAR": vara1_mean_abs_delta_v(ref_dvh, new_dvh),
        "vara2_mae_espacial_y_dose_score": vara2_mae_espacial(ref_constraints, new_constraints),
        "vara3_signo_no_cumple_RV65": vara3_signo_no_cumple(ref_signed, new_signed),
        "vara4_constraints_operativos": vara4_constraints_operativos(ref_constraints, new_constraints),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(comparacion, f, indent=2)
    print(f"Guardado: {OUT_PATH}\n")

    print("=== VARA 1 — mean|ΔV| banda media OAR (DEBE bajar) ===")
    for struct, r in comparacion["vara1_mean_abs_delta_v_media_OAR"].items():
        print(f"  {struct}: 002b={r['002b']:.2f}pp -> 003={r['003_dvhloss']:.2f}pp "
              f"(delta={r['delta_pp']:+.2f}pp, {r['delta_relativo_pct']:+.1f}%)  "
              f"{'MEJORO' if r['mejoro'] else 'NO MEJORO'}")

    print("\n=== VARA 2 — MAE espacial + dose_score (NO debe subir) ===")
    for nombre, r in comparacion["vara2_mae_espacial_y_dose_score"].items():
        print(f"  {nombre}: 002b={r['002b']:.3f} -> 003={r['003_dvhloss']:.3f} (delta={r['delta']:+.3f})  "
              f"{'EMPEORO' if r['empeoro'] else 'OK'}")

    print("\n=== VARA 3 — signo en no-cumple RV65 ===")
    v3 = comparacion["vara3_signo_no_cumple_RV65"]
    print(f"  n no-cumple: 002b={v3['n_no_cumple_RV65']['002b']}  003={v3['n_no_cumple_RV65']['003_dvhloss']}")
    print(f"  signed_mean_media: 002b={v3['signed_mean_media_no_cumple']['002b']:+.2f}pp  "
          f"003={v3['signed_mean_media_no_cumple']['003_dvhloss']:+.2f}pp")
    print(f"  casos optimistas (subestima, riesgoso): 002b={v3['n_casos_optimistas_no_cumple']['002b']}  "
          f"003={v3['n_casos_optimistas_no_cumple']['003_dvhloss']}  "
          f"{'OK (no aumento)' if v3['conteo_optimistas_no_aumento'] else '*** AUMENTO ***'}")

    print("\n=== VARA 4 — constraints operativos (control, no deben romperse) ===")
    for tag, r in comparacion["vara4_constraints_operativos"].items():
        print(f"  {tag}: AUC 002b={r['auc']['002b']:.3f} -> 003={r['auc']['003_dvhloss']:.3f}  "
              f"{'*** SE ROMPIO ***' if r['auc_se_rompio'] else 'OK'}")

    return comparacion


if __name__ == "__main__":
    main()
