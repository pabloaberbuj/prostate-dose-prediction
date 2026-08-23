"""Convierte los exports de Eclipse en `datos comisionamiento/` al JSON que
espera `commissioning/run_commissioning_pipeline.py` (formato definido en
`toolkit/commissioning_parser.py::MeasurementParser.parse_json_profiles` /
`parse_output_factors_json`).

Fuentes (todas con posición ya centrada en 0 y dosis SIN normalizar,
confirmado por Pablo):
  - x_40,20,10_<depth>cm.txt   -> perfiles crossline (PRO, eje X), 3 campos
                                   cuadrados (40x40/20x20/10x10) por archivo.
  - y_40,20,10_<depth>cm.txt   -> ídem, in-line (PRO, eje Y).
  - diag_40x40_<depth>cm.txt   -> perfil diagonal (DIA), campo 40x40. Admite
                                   sufijo "_rev" (ej. "1.5_revcm") para una
                                   repetición corregida de una profundidad.
  - outputs.txt                -> output factors, CSV "TC,factor" con TC en
                                   formato "<Y>x<X>" en cm (convención de
                                   Pablo, Y primero) respecto de 10x10=1.0.

No usa los archivos viejos en `datos comisionamiento/viejos/` (PDD y
perfiles con sólo 2 campos abiertos, posición sin centrar) - quedan como
referencia histórica; el PDD sirve para que Pablo derive tpr_20_10 a mano,
no lo consume este script.

Uso: python commissioning/convert_eclipse_to_json.py
Salida: commissioning/data/6x/{profiles,diagonals,output_factors}_6MV.json
"""
import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "datos comisionamiento"
OUT_DIR = Path(__file__).resolve().parent / "data" / "6x"

ENERGY = "6MV"
SSD_MM_ASSUMED = 1000.0  # No quedó registrado el setup real (SSD vs SAD) del escaneo; confirmar con Pablo.


def _parse_depth_cm(filename: str) -> float:
    m = re.search(r"_([\d,\.]+)(?:_rev)?cm\.txt$", filename)
    if not m:
        raise ValueError(f"No pude extraer la profundidad del nombre: {filename}")
    return float(m.group(1).replace(",", "."))


def _read_header(lines):
    """Devuelve (start, end, header_cols, data_start_idx)."""
    start = end = None
    for i, line in enumerate(lines):
        if line.startswith("Start:"):
            nums = re.findall(r"-?\d+\.?\d*", line)
            start = tuple(float(n) for n in nums[:3])
        elif line.startswith("End:"):
            nums = re.findall(r"-?\d+\.?\d*", line)
            end = tuple(float(n) for n in nums[:3])
            j = i + 1
            while not lines[j].strip():  # línea en blanco entre End: y el header
                j += 1
            header_cols = [c.strip() for c in lines[j].split("\t")]
            return start, end, header_cols, j + 1
    raise ValueError("No encontré Start:/End: en el archivo")


def parse_crossline_profile_file(path: Path, axis: str):
    """x_/y_ 40,20,10_<depth>cm.txt -> lista de measurements PRO (uno por campo).

    axis="X" para los archivos x_... (crossline), "Y" para los y_... (in-line).
    """
    depth_cm = _parse_depth_cm(path.name)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start, end, header_cols, data_idx = _read_header(lines)

    field_cols = [c for c in header_cols if c and c.lower() != "total"]
    col_idx = {c: header_cols.index(c) for c in field_cols}

    positions_mm = []
    doses = {c: [] for c in field_cols}
    for line in lines[data_idx:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        pos_cm = float(parts[0])
        positions_mm.append(pos_cm * 10.0)
        for c in field_cols:
            doses[c].append(float(parts[col_idx[c]]))

    measurements = []
    for c in field_cols:
        fx, fy = (float(v) * 10.0 for v in c.lower().split("x"))
        measurements.append(
            {
                "id": len(measurements) + 1,
                "field_size_mm": [fx, fy],
                "depth_mm": depth_cm * 10.0,
                "ssd_mm": SSD_MM_ASSUMED,
                "energy": ENERGY,
                "scan_type": "PRO",
                "axis": axis,
                "positions": positions_mm,
                "doses": doses[c],
            }
        )
    return measurements


def parse_diag_file(path: Path):
    """diag_40x40_<depth>cm.txt -> un measurement DIA (campo 40x40)."""
    depth_cm = _parse_depth_cm(path.name)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start, end, header_cols, data_idx = _read_header(lines)

    positions_mm, doses = [], []
    for line in lines[data_idx:]:
        if not line.strip():
            continue
        pos_cm, dose = line.split("\t")
        positions_mm.append(float(pos_cm) * 10.0)
        doses.append(float(dose))

    return {
        "id": 1,
        "field_size_mm": [400.0, 400.0],
        "depth_mm": depth_cm * 10.0,
        "ssd_mm": SSD_MM_ASSUMED,
        "energy": ENERGY,
        "scan_type": "DIA",
        "axis": "D",
        "positions": positions_mm,
        "doses": doses,
    }


def parse_output_factors(path: Path):
    """outputs.txt (CSV 'TC,factor', TC='<Y>x<X>' en cm) -> lista de OF."""
    measurements = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header "TC,factor"
        for row in reader:
            if not row or not row[0].strip():
                continue
            tc, factor = row[0].strip(), float(row[1])
            y_cm, x_cm = (float(v) for v in tc.lower().split("x"))
            measurements.append(
                {
                    "field_x_mm": x_cm * 10.0,
                    "field_y_mm": y_cm * 10.0,
                    "output_factor": factor,
                }
            )
    return measurements


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    x_files = sorted(DATA_DIR.glob("x_40,20,10_*cm.txt"))
    y_files = sorted(DATA_DIR.glob("y_40,20,10_*cm.txt"))
    diag_files = sorted(DATA_DIR.glob("diag_40x40_*cm.txt"))
    of_file = DATA_DIR / "outputs.txt"

    profiles = []
    for f in x_files:
        profiles.extend(parse_crossline_profile_file(f, axis="X"))
    for f in y_files:
        profiles.extend(parse_crossline_profile_file(f, axis="Y"))
    profiles_json = {
        "metadata": {"energy": ENERGY, "ssd_mm": SSD_MM_ASSUMED, "source": "Eclipse"},
        "measurements": profiles,
    }

    diagonals = [parse_diag_file(f) for f in diag_files]
    diagonals_json = {
        "metadata": {"energy": ENERGY, "ssd_mm": SSD_MM_ASSUMED, "source": "Eclipse"},
        "measurements": diagonals,
    }

    output_factors_json = {"measurements": parse_output_factors(of_file)}

    (OUT_DIR / "profiles_6MV.json").write_text(
        json.dumps(profiles_json, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "diagonals_6MV.json").write_text(
        json.dumps(diagonals_json, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "output_factors_6MV.json").write_text(
        json.dumps(output_factors_json, indent=2), encoding="utf-8"
    )

    print(f"profiles: {len(profiles)} measurements de {len(x_files)} archivos X + {len(y_files)} archivos Y")
    print(f"diagonals: {len(diagonals)} measurements de {len(diag_files)} archivos")
    print(f"output_factors: {len(output_factors_json['measurements'])} campos")
    print(f"Escrito en {OUT_DIR}")


if __name__ == "__main__":
    main()
