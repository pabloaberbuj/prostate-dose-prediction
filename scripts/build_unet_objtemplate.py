"""
Paso C del pipeline dosis predicha -> DVH -> ObjectiveTemplate XML
(ver HANDOFF_kbp_dvh_to_xml.md). Lee el .json de Paso B (scripts/compute_pred_dvh.py)
y genera el XML con gen_objtemplate.py::build_template (NO reescribir esa funcion,
solo reusarla).

Decisiones fijadas para el primer experimento (ver HANDOFF):
- Linea = DVH predicho EXACTO (sin margen).
- line_priority = 50 (constante).
- PTV: puntos fijos (lower Rx*1.01, upper Rx*1.05) -- no se usa la curva de PTV
  del json de Paso B para esto, build_template ya los pone solo.

Uso:
    .venv/Scripts/python.exe scripts/build_unet_objtemplate.py \
        --pred-dvh-json results/pilot_rd/pred_dvh_PT_003fe2bb84986507.json \
        --output-dir results/pilot_rd
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gen_objtemplate import build_template  # noqa: E402

OARS = ["Rectum", "Bladder"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dvh-json", required=True)
    parser.add_argument("--output-dir", default=None,
                         help="default: misma carpeta que --pred-dvh-json")
    parser.add_argument("--ptv-id", default="PTV_High")
    parser.add_argument("--line-priority", type=float, default=50.0)
    args = parser.parse_args()

    json_path = Path(args.pred_dvh_json)
    out_dir = Path(args.output_dir) if args.output_dir else json_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        dvh = json.load(f)

    anonid = dvh["_meta"]["anonid"]
    rx_gy = dvh["_meta"]["rx_gy"]

    oar = {k: (dvh[k]["dose_gy"], dvh[k]["vol_pct"]) for k in OARS}

    tree = build_template(
        oar, rx_gy=rx_gy, ptv_id=args.ptv_id,
        line_priority=args.line_priority, template_id=f"UNet_{anonid}",
    )
    ET.indent(tree, space="")
    out_path = out_dir / f"ObjectiveTemplate_UNet_{anonid}.xml"
    tree.write(str(out_path), encoding="UTF-8", xml_declaration=True)

    # Validacion (re-parsear y contar objetivos), igual que el demo de gen_objtemplate.py.
    r = ET.parse(str(out_path)).getroot()
    for s in r.iter("ObjectivesOneStructure"):
        objs = s.findall("StructureObjectives/Objective")
        lineas = [o for o in objs if o.find("Type").text == "1"]
        puntos = [o for o in objs if o.find("Type").text == "0"]
        print(f"{s.get('ID'):10s}  linea(Type1)={len(lineas):3d}  puntos(Type0)={len(puntos)}")
    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
