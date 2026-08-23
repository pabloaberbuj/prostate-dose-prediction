"""
Parser del dataset completo de objetivos de optimizacion extraido via ESAPI
(CSV, 1 fila por objetivo/punto de linea, 387 pacientes). Formato:

    AnonID;HC;PlanId;TipoObjetivo;Estructura;Operador;Dosis;UnidadDosis;
    Volumen_pct;Prioridad;PtoIndex

TipoObjetivo in {"PUNTO", "LINEA"}. Para LINEA, Operador viene vacio (NaN) --
mismo supuesto ya validado (ObjectiveTemplate_26749.xml + reporte ESAPI en
texto de ese mismo paciente): las lineas de OAR de RapidPlan son upper.
PtoIndex da el orden explicito de los puntos de una linea (1..N).

Uso:
    .venv/Scripts/python.exe scripts/parse_rapidplan_csv.py \
        --csv "../insumos temporales/objetivos_optimizacion.csv" \
        --anonid PT_003fe2bb84986507
"""

import argparse
from pathlib import Path

import pandas as pd

LINEA_OPERATOR_ASUMIDO = "upper"


def cargar_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(str(csv_path), sep=";")


def pacientes_con_plan_unico(df: pd.DataFrame) -> tuple:
    """Devuelve (dict AnonID->PlanId, lista de AnonID con >1 PlanId distinto).
    Los pacientes con mas de un plan quedan afuera del dict -- ambiguo cual
    plan usar, decidir a mano cual PlanId corresponde."""
    n_planes = df.groupby("AnonID")["PlanId"].nunique()
    ambiguos = n_planes[n_planes > 1].index.tolist()
    unico = df[~df["AnonID"].isin(ambiguos)].groupby("AnonID")["PlanId"].first().to_dict()
    return unico, ambiguos


def linea_estructura(df: pd.DataFrame, anonid: str, estructura: str,
                      plan_id: str = None) -> dict:
    """Devuelve {"dose_gy": [...], "vol_pct": [...]} para la LINEA de esa
    estructura en ese paciente/plan, ordenada por DOSIS ascendente. None si no
    existe.

    NO ordenar por PtoIndex: verificado que en varios pacientes el CSV trae el
    volcado partido en 2 bloques con indexado no monotono en dosis (ej.
    PT_5b1f54b9360d9375/Bladder -- PtoIndex 1-14 es la cola de dosis alta,
    PtoIndex 15-38 es el resto de la curva en orden de dosis DECRECIENTE) --
    ordenar por PtoIndex ahi mezcla puntos de dosis muy distinta y arruina la
    monotonia de la curva. Ordenar por Dosis es robusto a esto (y coincide con
    lo que ya hacian los parsers de XML/txt)."""
    sub = df[(df["AnonID"] == anonid) & (df["Estructura"] == estructura) &
              (df["TipoObjetivo"] == "LINEA")]
    if plan_id is not None:
        sub = sub[sub["PlanId"] == plan_id]
    if sub.empty:
        return None
    sub = sub.sort_values("Dosis")
    dosis = sub["Dosis"].to_numpy(dtype=float)
    if (sub["UnidadDosis"] == "cGy").all():
        dosis = dosis / 100.0
    elif not (sub["UnidadDosis"] == "Gy").all():
        raise ValueError(f"UnidadDosis mixta/desconocida para {anonid}/{estructura}: "
                          f"{sub['UnidadDosis'].unique()}")
    return {"dose_gy": dosis.tolist(), "vol_pct": sub["Volumen_pct"].to_numpy(dtype=float).tolist(),
            "operator": LINEA_OPERATOR_ASUMIDO}


def puntos_estructura(df: pd.DataFrame, anonid: str, estructura: str,
                       plan_id: str = None) -> list:
    sub = df[(df["AnonID"] == anonid) & (df["Estructura"] == estructura) &
              (df["TipoObjetivo"] == "PUNTO")]
    if plan_id is not None:
        sub = sub[sub["PlanId"] == plan_id]
    puntos = []
    for _, row in sub.iterrows():
        dosis = row["Dosis"] / 100.0 if row["UnidadDosis"] == "cGy" else row["Dosis"]
        puntos.append({"operator": str(row["Operador"]).lower(), "dose_gy": float(dosis),
                        "vol_pct": float(row["Volumen_pct"]), "priority": float(row["Prioridad"])})
    return puntos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--anonid", required=True)
    args = parser.parse_args()

    df = cargar_csv(Path(args.csv))
    sub = df[df["AnonID"] == args.anonid]
    print(f"{args.anonid}: {sub['PlanId'].nunique()} plan(es) -- {sub['PlanId'].unique().tolist()}")
    print(f"Estructuras con objetivos: {sorted(sub['Estructura'].unique().tolist())}")

    for estructura in ["Rectum", "Bladder"]:
        linea = linea_estructura(df, args.anonid, estructura)
        if linea is None:
            print(f"  {estructura}: SIN linea LINEA en el CSV")
            continue
        n = len(linea["dose_gy"])
        print(f"  {estructura}: linea con {n} puntos "
              f"({min(linea['dose_gy']):.1f}-{max(linea['dose_gy']):.1f}Gy)")
        for p in puntos_estructura(df, args.anonid, estructura):
            print(f"    punto {p}")


if __name__ == "__main__":
    main()
