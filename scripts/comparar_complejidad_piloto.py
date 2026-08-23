"""
Piloto substructuras (HANDOFF_substructuras_dosis.md) -- tabla comparativa de
complejidad de los 3 brazos (control/U-Net, test/hot-cold, RapidPlan), reusando
`extraer_complejidad_rp.py::extraer_paciente` (NO se reescribe esa logica).

Uso:
    .venv/Scripts/python.exe scripts/comparar_complejidad_piloto.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extraer_complejidad_rp import extraer_paciente  # noqa: E402

_XML_SUBSTRUCT = Path(r"\\10.100.0.252\centro_de_datos2018\101_Cosas de\PABLO\CNN Prostata\xml substructure")

PLANES = {
    "Control (U-Net, Rectum completo)": Path(r"C:\Pablo\ProstateDoseProject\insumos temporales\planUnet.dcm"),
    "Test (U-Net, Rectum_hot+cold)": Path(r"C:\Pablo\ProstateDoseProject\insumos temporales\planTest.dcm"),
    "RapidPlan (referencia clinica)": Path(r"C:\Pablo\ProstateDoseProject\insumos temporales\PlanRapidPlan.dcm"),
    # Control_push10 RP fue sobreescrito en el share al correr push15_overlap -- ya no existe en
    # disco. Sus metricas (MU_factor=3.6251 MCSv=0.2541 SAS10=0.3284 MU_total=725.02) quedan
    # registradas en comparacion_complejidad_piloto.csv de la corrida anterior.
    "Test_push10": _XML_SUBSTRUCT / "Test" / "RP.1.2.246.352.71.5.887822707897.687201.20260823083151.dcm",
    "Control_push15_overlap": _XML_SUBSTRUCT / "Control" / "RP.1.2.246.352.71.5.887822707897.687200.20260823091201.dcm",
    "Test_push15_overlap": _XML_SUBSTRUCT / "Test" / "RP.1.2.246.352.71.5.887822707897.687201.20260823092307.dcm",
}


def main():
    filas = []
    for nombre, path in PLANES.items():
        m = extraer_paciente(path)
        m["Plan"] = nombre
        filas.append(m)

    df = pd.DataFrame(filas).set_index("Plan")
    cols = ["MU_factor", "MCSv", "SAS10", "MU_total", "NArcos"]
    print(df[cols].to_string(float_format=lambda x: f"{x:.4f}"))

    out_dir = Path(__file__).resolve().parent.parent / "results" / "substruct" / "PT_44a81316c45f64f5"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "comparacion_complejidad_piloto.csv")
    print(f"\nGuardado: {out_dir / 'comparacion_complejidad_piloto.csv'}")


if __name__ == "__main__":
    main()
