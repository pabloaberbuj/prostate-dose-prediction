"""
Compara el DVH PREDICHO por el U-Net (Paso B, scripts/compute_pred_dvh.py) contra
el DVH PREDICTOR de RapidPlan (la estimacion del modelo ANTES de optimizar,
leida desde Eclipse via ObjectiveTemplate XML -- ver
scripts/parse_rapidplan_objtemplate.py -- o via el reporte de texto ESAPI --
ver scripts/parse_rapidplan_txt_report.py). Es una comparacion "predictor vs
predictor" -- ninguno de los dos paso por un motor de optimizacion todavia.

NO confundir con dvh_curva_completa.py, que compara el U-Net contra el DVH REAL
(el plan RapidPlan ya optimizado/entregado) -- son preguntas distintas.

Uso (XML):
    .venv/Scripts/python.exe scripts/compare_dvh_unet_vs_rapidplan.py \
        --pred-dvh-json results/pilot_rd/pred_dvh_PT_003fe2bb84986507.json \
        --rapidplan-xml <export_rapidplan>.xml \
        --output-dir results/pilot_rd/comparacion_rapidplan

Uso (reporte ESAPI en texto):
    .venv/Scripts/python.exe scripts/compare_dvh_unet_vs_rapidplan.py \
        --pred-dvh-json results/pilot_rd/pred_dvh_PT_003fe2bb84986507.json \
        --rapidplan-txt <reporte_esapi>.txt \
        --output-dir results/pilot_rd/comparacion_rapidplan
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_rapidplan_objtemplate import parse_template as parse_xml  # noqa: E402
from parse_rapidplan_txt_report import parse_report as parse_txt  # noqa: E402

# Nuestro nombre de OAR (el del json de Paso B) -> nombres posibles del ID de
# estructura en el XML de RapidPlan (case-insensitive). Completar si el export
# real de Pablo usa otro alias.
ALIAS_ESTRUCTURA = {
    "Rectum": ["Rectum", "Recto"],
    "Bladder": ["Bladder", "Vejiga"],
}

N_GRID = 200


def _match_struct_id(nombre_nuestro: str, ids_disponibles: list) -> str:
    alias = [a.lower() for a in ALIAS_ESTRUCTURA.get(nombre_nuestro, [nombre_nuestro])]
    for sid in ids_disponibles:
        if sid.lower() in alias:
            return sid
    return None


def _elegir_linea(lineas: list) -> dict:
    """Si hay >1 linea (ej. banda upper+lower), usar la 'upper' como estimacion
    primaria; si no hay operator=upper, usar la primera."""
    if not lineas:
        return None
    for l in lineas:
        if l["operator"] == "upper":
            return l
    return lineas[0]


def interpolar_vol_pct(dose_gy: list, vol_pct: list, grid_gy: np.ndarray) -> np.ndarray:
    """vol_pct(dose) es monotona no creciente; np.interp requiere x ascendente,
    dose_gy ya viene ordenado ascendente por el parser/Paso B."""
    return np.interp(grid_gy, dose_gy, vol_pct, left=vol_pct[0], right=vol_pct[-1])


def cargar_rapidplan(rapidplan_xml: Path = None, rapidplan_txt: Path = None) -> dict:
    if rapidplan_xml is not None:
        return parse_xml(rapidplan_xml)
    if rapidplan_txt is not None:
        return parse_txt(rapidplan_txt)
    raise ValueError("Pasar --rapidplan-xml o --rapidplan-txt")


def comparar(pred_dvh_json: Path, output_dir: Path,
             rapidplan_xml: Path = None, rapidplan_txt: Path = None):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(pred_dvh_json) as f:
        unet = json.load(f)
    rx_gy = unet["_meta"]["rx_gy"]
    anonid = unet["_meta"]["anonid"]

    rapidplan = cargar_rapidplan(rapidplan_xml, rapidplan_txt)
    ids_disponibles = list(rapidplan.keys())
    rapidplan_origen = str(rapidplan_xml) if rapidplan_xml else str(rapidplan_txt)

    resumen = {"anonid": anonid, "rx_gy": rx_gy, "rapidplan_origen": rapidplan_origen,
               "estructuras": {}}

    for oar in ["Rectum", "Bladder"]:
        struct_id = _match_struct_id(oar, ids_disponibles)
        if struct_id is None:
            print(f"[!] {oar}: no encontrado en el XML de RapidPlan "
                  f"(IDs disponibles: {ids_disponibles}) -- salteado")
            continue

        linea_rp = _elegir_linea(rapidplan[struct_id]["lineas"])
        if linea_rp is None or len(linea_rp["dose_gy"]) == 0:
            print(f"[!] {oar} ({struct_id}): sin linea Type=1 valida en el XML de RapidPlan")
            continue

        dose_unet = np.array(unet[oar]["dose_gy"])
        vol_unet = np.array(unet[oar]["vol_pct"])

        dose_max = max(dose_unet.max(), max(linea_rp["dose_gy"]))
        grid_gy = np.linspace(0, dose_max, N_GRID)

        v_unet_grid = interpolar_vol_pct(dose_unet.tolist(), vol_unet.tolist(), grid_gy)
        v_rp_grid = interpolar_vol_pct(linea_rp["dose_gy"], linea_rp["vol_pct"], grid_gy)
        delta = v_unet_grid - v_rp_grid  # + = U-Net predice mas volumen a esa dosis (pesimista)

        grid_pct_rx = grid_gy / rx_gy * 100.0
        idx_baja = grid_pct_rx <= 40
        idx_media = (grid_pct_rx > 40) & (grid_pct_rx <= 80)
        idx_alta = grid_pct_rx > 80

        resumen["estructuras"][oar] = {
            "struct_id_rapidplan": struct_id,
            "operator_linea_rapidplan": linea_rp["operator"],
            "n_puntos_rapidplan": len(linea_rp["dose_gy"]),
            "mean_abs_delta_V_global_pp": float(np.mean(np.abs(delta))),
            "mean_abs_delta_V_baja_0_40pctRx_pp": float(np.mean(np.abs(delta[idx_baja]))) if idx_baja.any() else None,
            "mean_abs_delta_V_media_40_80pctRx_pp": float(np.mean(np.abs(delta[idx_media]))) if idx_media.any() else None,
            "mean_abs_delta_V_alta_80_110pctRx_pp": float(np.mean(np.abs(delta[idx_alta]))) if idx_alta.any() else None,
            "signed_mean_delta_V_pp": float(np.mean(delta)),
        }

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        ax.plot(grid_gy, v_unet_grid, color="firebrick", label="U-Net (predictor)")
        ax.plot(grid_gy, v_rp_grid, color="steelblue", linestyle="--",
                 label=f"RapidPlan (predictor, {linea_rp['operator']})")
        ax.set_xlabel("Dosis (Gy)")
        ax.set_ylabel("Volumen (%)")
        ax.set_title(f"{oar} -- DVH predicho: U-Net vs. RapidPlan")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        ax2 = axes[1]
        ax2.plot(grid_gy, delta, color="darkorange")
        ax2.axhline(0, color="gray", linewidth=0.8)
        ax2.set_xlabel("Dosis (Gy)")
        ax2.set_ylabel("ΔV = V_UNet − V_RapidPlan (pp)")
        ax2.set_title(f"{oar} -- diferencia")
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(str(output_dir / f"dvh_comparacion_{oar.lower()}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)

    out_json = output_dir / f"comparacion_predictores_{anonid}.json"
    with open(out_json, "w") as f:
        json.dump(resumen, f, indent=2)

    print(json.dumps(resumen, indent=2))
    print(f"\nGuardado: {out_json}")
    print(f"Plots en: {output_dir}")
    return resumen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dvh-json", required=True)
    parser.add_argument("--rapidplan-xml", default=None, help="ObjectiveTemplate XML de Eclipse")
    parser.add_argument("--rapidplan-txt", default=None, help="Reporte ESAPI en texto")
    parser.add_argument("--output-dir", default=None,
                         help="default: misma carpeta que --pred-dvh-json / comparacion_rapidplan")
    args = parser.parse_args()

    if not args.rapidplan_xml and not args.rapidplan_txt:
        parser.error("Pasar --rapidplan-xml o --rapidplan-txt")

    pred_dvh_json = Path(args.pred_dvh_json)
    output_dir = Path(args.output_dir) if args.output_dir else pred_dvh_json.parent / "comparacion_rapidplan"
    comparar(
        pred_dvh_json, output_dir,
        rapidplan_xml=Path(args.rapidplan_xml) if args.rapidplan_xml else None,
        rapidplan_txt=Path(args.rapidplan_txt) if args.rapidplan_txt else None,
    )


if __name__ == "__main__":
    main()
