"""
Parser generico de un ObjectiveTemplate v1.7 exportado por RapidPlan (el
"predictor" de RapidPlan -- DVH estimado antes de optimizar, NO el plan ya
optimizado). Contraparte de gen_objtemplate.py::build_template (que ESCRIBE
este mismo schema para el U-Net); esto LEE un XML real generado por Eclipse.

Schema (ver HANDOFF_kbp_dvh_to_xml.md, ya decodificado):
- ObjectivesAllStructures > ObjectivesOneStructure[ID=struct] > StructureObjectives
  > Objective (repetido)
    - Type: 0=punto, 1=linea (agrupada por Group+Priority), 2=media/gEUD (a confirmar)
    - Operator: 0=upper, 1=lower, 99=mean
    - Dose (Gy), Volume (%, o xsi:nil)
    - Priority, Group

A diferencia de gen_objtemplate.py (que siempre emite 1 sola linea Type=1 por
OAR), RapidPlan puede exportar MAS de una linea por estructura (ej. banda
upper+lower de la estimacion) -- este parser NO asume una sola linea, agrupa
por (Group, Priority, Operator) y devuelve una lista de lineas por estructura.

Uso:
    .venv/Scripts/python.exe scripts/parse_rapidplan_objtemplate.py \
        --xml <archivo exportado por RapidPlan>.xml
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

OPERATOR_NOMBRE = {"0": "upper", "1": "lower", "99": "mean"}


def _texto(el, tag):
    node = el.find(tag)
    if node is None:
        return None
    nil = node.get("{http://www.w3.org/2001/XMLSchema-instance}nil")
    if nil == "true":
        return None
    return node.text


def parse_template(xml_path: Path) -> dict:
    """Devuelve {struct_id: {"lineas": [...], "puntos": [...]}}.

    Cada linea: {"group": str, "priority": float, "operator": str,
                 "dose_gy": [...], "vol_pct": [...]} -- puntos ordenados por dosis ascendente.
    Cada punto (Type=0): {"operator": str, "dose_gy": float, "vol_pct": float or None,
                           "priority": float}.
    """
    root = ET.parse(str(xml_path)).getroot()
    resultado = {}

    for struct_el in root.iter("ObjectivesOneStructure"):
        struct_id = struct_el.get("ID")
        grupos = {}  # (group, priority, operator) -> lista de (dose, vol)
        puntos = []

        for obj in struct_el.findall("StructureObjectives/Objective"):
            tipo = _texto(obj, "Type")
            operador = OPERATOR_NOMBRE.get(_texto(obj, "Operator"), _texto(obj, "Operator"))
            dosis = _texto(obj, "Dose")
            volumen = _texto(obj, "Volume")
            prioridad = _texto(obj, "Priority")
            grupo = _texto(obj, "Group")

            dosis_f = float(dosis) if dosis is not None else None
            vol_f = float(volumen) if volumen is not None else None
            prio_f = float(prioridad) if prioridad is not None else None

            if tipo == "1":  # linea
                clave = (grupo, prioridad, operador)
                grupos.setdefault(clave, []).append((dosis_f, vol_f))
            elif tipo == "0":  # punto
                puntos.append({"operator": operador, "dose_gy": dosis_f,
                                "vol_pct": vol_f, "priority": prio_f})
            else:
                print(f"  [!] {struct_id}: Type={tipo} no reconocido (ni 0 ni 1), ignorado")

        lineas = []
        for (grupo, prioridad, operador), puntos_linea in grupos.items():
            puntos_linea_ordenados = sorted(
                [(d, v) for d, v in puntos_linea if d is not None and v is not None],
                key=lambda p: p[0],
            )
            lineas.append({
                "group": grupo,
                "priority": float(prioridad) if prioridad is not None else None,
                "operator": operador,
                "dose_gy": [p[0] for p in puntos_linea_ordenados],
                "vol_pct": [p[1] for p in puntos_linea_ordenados],
            })

        resultado[struct_id] = {"lineas": lineas, "puntos": puntos}

    return resultado


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    args = parser.parse_args()

    resultado = parse_template(Path(args.xml))

    print(f"Estructuras encontradas: {list(resultado.keys())}")
    for struct_id, info in resultado.items():
        print(f"\n{struct_id}:")
        for linea in info["lineas"]:
            n = len(linea["dose_gy"])
            rango = (f"{min(linea['dose_gy']):.1f}-{max(linea['dose_gy']):.1f}Gy"
                      if n else "sin puntos validos")
            print(f"  linea group={linea['group']} priority={linea['priority']} "
                  f"operator={linea['operator']}: {n} puntos ({rango})")
        for p in info["puntos"]:
            print(f"  punto operator={p['operator']} dose={p['dose_gy']} "
                  f"vol={p['vol_pct']} priority={p['priority']}")
        if not info["lineas"] and not info["puntos"]:
            print("  (vacio)")


if __name__ == "__main__":
    main()
