"""
Paso B del pipeline dosis predicha -> DVH -> ObjectiveTemplate XML
(ver HANDOFF_kbp_dvh_to_xml.md). Lee el .npz de Paso A (scripts/predict_one.py,
`dose_pred` en % Rx relativo) y calcula el DVH por OAR, en Gy, ya renormalizado
a D95(PTV)=100%.

No requiere calibracion fisica de voxel (conteo puro de voxels dentro de la
mascara, no volumen fisico) -- el bug de escala de voxel documentado en
CLAUDE_CODE_CONTEXT.md no aplica aca.

Uso:
    .venv/Scripts/python.exe scripts/compute_pred_dvh.py \
        --pred-npz results/pilot_rd/pred_PT_003fe2bb84986507.npz \
        --rx-gy 78 \
        --output-dir results/pilot_rd
"""

import argparse
import json
from pathlib import Path

import numpy as np

OARS = {"Rectum": "rectum_mask", "Bladder": "bladder_mask"}
N_PUNTOS_DVH = 200


def d95_ptv(dose_pct: np.ndarray, ptv_mask: np.ndarray) -> float:
    """D95 = dosis que cubre el 95% del volumen del PTV = percentil 5 de la dosis."""
    vals = dose_pct[ptv_mask > 0]
    return float(np.percentile(vals, 5))


def dvh_desde_dose_pred(dose_pred_raw: np.ndarray, ptv_mask: np.ndarray,
                         oar_masks: dict, rx_gy: float) -> dict:
    """Nucleo de Paso B, operando directo sobre arrays en memoria (sin pasar por
    un .npz intermedio en disco) -- para poder reusarse en un batch de muchos
    pacientes sin escribir/leer un archivo por paciente.

    oar_masks: {nombre_oar: mascara_binaria}, ej. {"Rectum": ..., "Bladder": ...}.
    """
    # Paso B.1 -- clip a 0 (el modelo no tiene activacion final, puede predecir
    # ruido negativo cerca de 0%, ver results/pilot_rd) y renormalizar a
    # D95(PTV_pred)=100%.
    dose_pred_raw = np.clip(np.asarray(dose_pred_raw, dtype=np.float32), 0.0, None)
    d95_pred_raw = d95_ptv(dose_pred_raw, ptv_mask)
    factor_renorm = 100.0 / d95_pred_raw
    dose_pred_norm = dose_pred_raw * factor_renorm

    resultado = {}
    for oar_id, mask in oar_masks.items():
        dvox = dose_pred_norm[mask > 0]  # dosis % de cada voxel del OAR
        dose_grid_pct = np.linspace(0, float(dvox.max()), N_PUNTOS_DVH)
        vol_pct = [(dvox >= d).mean() * 100.0 for d in dose_grid_pct]
        dose_grid_gy = (dose_grid_pct * rx_gy / 100.0).tolist()
        resultado[oar_id] = {"dose_gy": dose_grid_gy, "vol_pct": vol_pct}

    # PTV incluido a titulo informativo (no va como linea DVH al XML -- ahi son
    # puntos fijos segun gen_objtemplate.py -- pero sirve para comparar/graficar).
    dvox_ptv = dose_pred_norm[ptv_mask > 0]
    dose_grid_pct = np.linspace(0, float(dvox_ptv.max()), N_PUNTOS_DVH)
    vol_pct_ptv = [(dvox_ptv >= d).mean() * 100.0 for d in dose_grid_pct]
    resultado["PTV"] = {
        "dose_gy": (dose_grid_pct * rx_gy / 100.0).tolist(),
        "vol_pct": vol_pct_ptv,
    }

    resultado["_meta"] = {
        "rx_gy": rx_gy,
        "factor_renorm_aplicado": factor_renorm,
        "d95_pred_pre_renorm_pct": d95_pred_raw,
    }
    return resultado


def compute_pred_dvh(pred_npz_path: Path, rx_gy: float) -> dict:
    data = np.load(str(pred_npz_path), allow_pickle=True)
    meta = json.loads(str(data["meta"][0]))

    oar_masks = {oar_id: data[mask_key] for oar_id, mask_key in OARS.items()}
    resultado = dvh_desde_dose_pred(data["dose_pred"], data["ptv_mask"], oar_masks, rx_gy)
    resultado["_meta"]["anonid"] = meta["anonid"]
    resultado["_meta"]["pred_npz_origen"] = str(pred_npz_path)
    return resultado


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-npz", required=True)
    parser.add_argument("--rx-gy", type=float, required=True)
    parser.add_argument("--output-dir", default=None,
                         help="default: misma carpeta que --pred-npz")
    args = parser.parse_args()

    pred_npz_path = Path(args.pred_npz)
    out_dir = Path(args.output_dir) if args.output_dir else pred_npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    resultado = compute_pred_dvh(pred_npz_path, args.rx_gy)

    anonid = resultado["_meta"]["anonid"]
    out_path = out_dir / f"pred_dvh_{anonid}.json"
    with open(out_path, "w") as f:
        json.dump(resultado, f, indent=2)

    print(f"factor_renorm aplicado: {resultado['_meta']['factor_renorm_aplicado']:.4f}")
    print(f"D95(PTV_pred) antes de renormalizar: {resultado['_meta']['d95_pred_pre_renorm_pct']:.2f}%")
    for oar in OARS:
        vp = resultado[oar]["vol_pct"]
        dg = resultado[oar]["dose_gy"]
        print(f"  {oar}: dosis max muestreada={dg[-1]:.1f}Gy  V(0)={vp[0]:.1f}%  V(max)={vp[-1]:.2f}%")
    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
