"""Filtra metricas_planes_hipofx_D95norm.csv excluyendo los 28 pacientes con
areas ganglionares pelvicas (planes completamente distintos a prostata-sola,
contaminacion de dataset detectada 2026-07-13 — ver CLAUDE_CODE_CONTEXT.md).

No modifica el CSV original ni los NPZ en processed_hipo/ (los NPZ de los
excluidos simplemente no se referencian en el split limpio).

Uso:
    python scripts/filter_hipofx_clean.py
"""

import csv
from pathlib import Path

IN_PATH = Path(r"c:\Pablo\ProstateDoseProject\dicoms hipofx\metricas_planes_hipofx_D95norm.csv")
OUT_PATH = Path(r"c:\Pablo\ProstateDoseProject\dicoms hipofx\metricas_planes_hipofx_D95norm_clean.csv")

EXCLUDE_IDS = {
    "PT_48f5d541d1345fac", "PT_b69afd5aac626652", "PT_bb3e683b982b7e10", "PT_3567c33a62b3639b",
    "PT_186d1e95997c4d4c", "PT_4eb8a1dbe930c7d5", "PT_d0ec99de816f43a2", "PT_8ce99f6c2510ac4d",
    "PT_71418878dbbe4c8f", "PT_093bfd0af80b588e", "PT_0456deafecd33ff1", "PT_dda8bf75509cd629",
    "PT_9371aae1bc4a76ac", "PT_e18d2f582312c202", "PT_c8a9e92ed2c8657c", "PT_00dc878545e331b2",
    "PT_7f59629d324c8f85", "PT_0b1dfdffc25ab05c", "PT_e4ca9f90c174e15a", "PT_64bdc6f122da6e84",
    "PT_9303de5015932f5a", "PT_a52308a14b30db33", "PT_b9b5d34c32f39343", "PT_072f15b8e53a9cee",
    "PT_13171608b4e247ab", "PT_e12c3b41a2f2f22c", "PT_6b7e2ece0dc9ec3c", "PT_6954073bb0493b6e",
}


def main():
    with open(IN_PATH, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldnames = reader.fieldnames
        rows = list(reader)

    assert len(EXCLUDE_IDS) == 28, f"lista de exclusion deberia tener 28 IDs, tiene {len(EXCLUDE_IDS)}"

    ids_in_csv = {r["AnonID"] for r in rows}
    faltantes = EXCLUDE_IDS - ids_in_csv
    if faltantes:
        raise ValueError(f"{len(faltantes)} IDs a excluir no estan en el CSV: {sorted(faltantes)}")

    kept = [r for r in rows if r["AnonID"] not in EXCLUDE_IDS]

    print(f"Filas originales: {len(rows)}")
    print(f"Excluidas: {len(rows) - len(kept)}")
    print(f"Filas resultantes: {len(kept)}")

    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(kept)

    print(f"\nEscrito: {OUT_PATH}")


if __name__ == "__main__":
    main()
