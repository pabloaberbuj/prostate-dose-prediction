"""Corre recompute_check.py sobre varios pacientes DICOM y arma una tabla
resumen (gamma + diff de dosis media por estructura).

Uso: python commissioning/run_sandbox_batch.py <carpeta1> <carpeta2> ...
"""
import csv
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recompute_check import main as run_one, STRUCT_NAMES, RESULTS_DIR


def main(patient_dirs):
    all_results = []
    for d in patient_dirs:
        name = Path(d).name
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        try:
            res = run_one(d)
            all_results.append(res)
        except Exception:
            print(f"[FALLÓ {name}]")
            traceback.print_exc()
            all_results.append({"patient": name, "error": True})

    ok = [r for r in all_results if not r.get("error")]
    print(f"\n\n{'='*90}\nRESUMEN ({len(ok)}/{len(all_results)} pacientes OK)\n{'='*90}")
    header = ["patient", "n_arcs", "gamma_5pct_5mm", "gamma_3pct_3mm"] + [f"diff_{n}_pct" for n in STRUCT_NAMES]
    print(" | ".join(f"{h:>16}" for h in header))
    for r in ok:
        row = []
        for h in header:
            v = r.get(h)
            if v is None:
                row.append("n/a")
            elif isinstance(v, float):
                row.append(f"{v:+.1f}" if "diff" in h else f"{v:.1f}")
            else:
                row.append(str(v))
        print(" | ".join(f"{v:>16}" for v in row))

    if ok:
        import numpy as np
        for h in header[2:]:
            vals = [r[h] for r in ok if r.get(h) is not None]
            if vals:
                print(f"{h}: media={np.mean(vals):+.2f}  min={min(vals):+.2f}  max={max(vals):+.2f}")

    csv_path = RESULTS_DIR / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header + ["error"])
        writer.writeheader()
        for r in all_results:
            writer.writerow({h: r.get(h) for h in header + ["error"]})
    print(f"\nResumen guardado en {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
