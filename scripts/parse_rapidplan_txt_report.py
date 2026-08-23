"""
Parser del reporte de texto que exporta el script ESAPI de Pablo
("REPORTE DE OPTIMIZACION Y DISCRETIZACION DE CURVAS", Eclipse 13.6) --
formato alternativo al ObjectiveTemplate XML (ver parse_rapidplan_objtemplate.py)
para leer los objetivos/DVH-estimado de RapidPlan de un plan.

Devuelve el MISMO shape que parse_rapidplan_objtemplate.py::parse_template
({struct_id: {"lineas": [...], "puntos": [...]}}) para que
compare_dvh_unet_vs_rapidplan.py pueda usar cualquiera de los dos formatos sin
cambios.

Formato esperado (por linea, robusto a mojibake de acentos -- el detector no
depende de como se vean "LINEA"/"Ó" etc., solo de las claves ASCII):
    [PUNTO] Estructura: <ID> | Tipo: <Lower|Upper> | Dosis: <N> cGy | Vol: <N>% | Prioridad: <N>
    [LINEA] Estructura: <ID> | Prioridad: <N> (Iniciando volcado de puntos)
       -> Punto NNN: Dosis = <N> cGy | Vol = <N>%
       ... (mas puntos hasta la proxima cabecera)

Nota: el reporte NO indica el Operator (upper/lower) para las lineas -- se
asume "upper" (consistente con lo ya verificado en el XML real de este mismo
plan, ObjectiveTemplate_26749.xml: las lineas de OAR de RapidPlan son
Operator=0=upper). Si algun reporte futuro trae lineas lower, hay que
extender este parser (agregar esa marca al reporte ESAPI de origen).

Uso:
    .venv/Scripts/python.exe scripts/parse_rapidplan_txt_report.py \
        --txt data/Optimizacion_Plan2_20260819_143154.txt
"""

import argparse
import re
from pathlib import Path

RE_PUNTO = re.compile(
    r"Estructura:\s*(?P<struct>\S+)\s*\|\s*Tipo:\s*(?P<tipo>\w+)\s*\|\s*"
    r"Dosis:\s*(?P<dosis>[\d.]+)\s*cGy\s*\|\s*Vol:\s*(?P<vol>[\d.]+)%\s*\|\s*"
    r"Prioridad:\s*(?P<prio>[\d.]+)"
)
RE_LINEA_HEADER = re.compile(
    r"Estructura:\s*(?P<struct>\S+)\s*\|\s*Prioridad:\s*(?P<prio>[\d.]+)\s*"
    r"\(Iniciando volcado de puntos\)"
)
RE_LINEA_PUNTO = re.compile(
    r"->\s*Punto\s*\d+:\s*Dosis\s*=\s*(?P<dosis>[\d.]+)\s*cGy\s*\|\s*"
    r"Vol\s*=\s*(?P<vol>[\d.]+)%"
)

LINEA_OPERATOR_ASUMIDO = "upper"


def parse_report(txt_path: Path) -> dict:
    resultado = {}
    linea_actual = None  # (struct_id, priority) mientras se acumulan puntos de una linea

    with open(txt_path, "r", encoding="utf-8-sig", errors="replace") as f:
        for raw in f:
            linea_txt = raw.strip()
            if not linea_txt:
                linea_actual = None
                continue

            m_header = RE_LINEA_HEADER.search(linea_txt)
            if m_header:
                struct_id = m_header.group("struct")
                prio = float(m_header.group("prio"))
                resultado.setdefault(struct_id, {"lineas": [], "puntos": []})
                nueva_linea = {"group": None, "priority": prio,
                               "operator": LINEA_OPERATOR_ASUMIDO,
                               "dose_gy": [], "vol_pct": []}
                resultado[struct_id]["lineas"].append(nueva_linea)
                linea_actual = nueva_linea
                continue

            m_punto_linea = RE_LINEA_PUNTO.search(linea_txt)
            if m_punto_linea and linea_actual is not None:
                linea_actual["dose_gy"].append(float(m_punto_linea.group("dosis")) / 100.0)
                linea_actual["vol_pct"].append(float(m_punto_linea.group("vol")))
                continue

            m_punto = RE_PUNTO.search(linea_txt)
            if m_punto:
                linea_actual = None  # un [PUNTO] cierra cualquier linea en curso
                struct_id = m_punto.group("struct")
                resultado.setdefault(struct_id, {"lineas": [], "puntos": []})
                resultado[struct_id]["puntos"].append({
                    "operator": m_punto.group("tipo").lower(),
                    "dose_gy": float(m_punto.group("dosis")) / 100.0,
                    "vol_pct": float(m_punto.group("vol")),
                    "priority": float(m_punto.group("prio")),
                })
                continue

    # Las lineas ya vienen ordenadas por dosis ascendente en el reporte (verificado
    # en el export real) pero se ordena explicitamente por robustez.
    for info in resultado.values():
        for linea in info["lineas"]:
            pares = sorted(zip(linea["dose_gy"], linea["vol_pct"]), key=lambda p: p[0])
            linea["dose_gy"] = [p[0] for p in pares]
            linea["vol_pct"] = [p[1] for p in pares]

    return resultado


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt", required=True)
    args = parser.parse_args()

    resultado = parse_report(Path(args.txt))

    print(f"Estructuras encontradas: {list(resultado.keys())}")
    for struct_id, info in resultado.items():
        print(f"\n{struct_id}:")
        for linea in info["lineas"]:
            n = len(linea["dose_gy"])
            rango = (f"{min(linea['dose_gy']):.1f}-{max(linea['dose_gy']):.1f}Gy"
                      if n else "sin puntos validos")
            print(f"  linea priority={linea['priority']} operator={linea['operator']} "
                  f"(asumido, no viene en el txt): {n} puntos ({rango})")
        for p in info["puntos"]:
            print(f"  punto operator={p['operator']} dose={p['dose_gy']:.2f}Gy "
                  f"vol={p['vol_pct']}% priority={p['priority']}")


if __name__ == "__main__":
    main()
