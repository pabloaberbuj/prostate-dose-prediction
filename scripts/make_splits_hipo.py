"""Genera repo/data/splits/splits_hipo_v1.json (train/val/test = 65/15/20) para el
dataset hipofraccionado (179 pacientes), estratificado por un estrato 2D de compliance
operativo (flags D95norm):

    recto_op_fail  = Flag_Rectum_V65Gy_lt15_D95norm==0 OR Flag_Rectum_V55Gy_lt25_D95norm==0
    vejiga_op_fail = Flag_Bladder_V65Gy_lt15_D95norm==0

Ver CLAUDE_CODE_CONTEXT.md / HIPOFX_KICKOFF.md para contexto de diseño.
"""

import csv
import json
import random
from pathlib import Path

CSV_PATH = Path(r"c:\Pablo\ProstateDoseProject\dicoms hipofx\metricas_planes_hipofx_D95norm.csv")
OUT_PATH = Path(r"c:\Pablo\ProstateDoseProject\repo\data\splits\splits_hipo_v1.json")

SEED = 42
FRACTIONS = {"train": 0.65, "val": 0.15, "test": 0.20}
# Orden de siembra: celdas chicas primero (para no dejar a test sin fallo-vejiga).
CELL_ORDER = ["solo_vejiga", "solo_recto", "ambos_fallan", "ambos_cumplen"]


def flag_is_fail(row, col):
    return int(float(row[col])) == 0


def load_rows():
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        return list(reader)


def classify_cell(row):
    recto_fail = flag_is_fail(row, "Flag_Rectum_V65Gy_lt15_D95norm") or flag_is_fail(
        row, "Flag_Rectum_V55Gy_lt25_D95norm"
    )
    vejiga_fail = flag_is_fail(row, "Flag_Bladder_V65Gy_lt15_D95norm")
    if not recto_fail and not vejiga_fail:
        return "ambos_cumplen"
    if recto_fail and not vejiga_fail:
        return "solo_recto"
    if not recto_fail and vejiga_fail:
        return "solo_vejiga"
    return "ambos_fallan"


def apportion(n, fractions=FRACTIONS):
    """Reparto proporcional por mayor resto (Hamilton), exacto en n."""
    splits = list(fractions.keys())
    exact = {s: n * fractions[s] for s in splits}
    base = {s: int(exact[s]) for s in splits}
    remainder = n - sum(base.values())
    order = sorted(splits, key=lambda s: exact[s] - base[s], reverse=True)
    for s in order[:remainder]:
        base[s] += 1
    return base


def assign_cell(rows_in_cell, rng):
    ids = [r["AnonID"] for r in rows_in_cell]
    ids_shuffled = ids[:]
    rng.shuffle(ids_shuffled)
    counts = apportion(len(ids_shuffled))
    out = {}
    i = 0
    for split in ("train", "val", "test"):
        n = counts[split]
        out[split] = ids_shuffled[i : i + n]
        i += n
    return out


def main():
    rows = load_rows()
    assert len(rows) == 179, f"esperaba 179 pacientes, encontre {len(rows)}"

    by_id = {r["AnonID"]: r for r in rows}
    cells = {c: [] for c in CELL_ORDER}
    for r in rows:
        cells[classify_cell(r)].append(r)

    expected_counts = {
        "ambos_cumplen": 108,
        "solo_recto": 34,
        "solo_vejiga": 12,
        "ambos_fallan": 25,
    }
    for c in CELL_ORDER:
        assert len(cells[c]) == expected_counts[c], (
            f"celda {c}: esperaba {expected_counts[c]}, encontre {len(cells[c])}"
        )

    rng = random.Random(SEED)
    splits = {"train": [], "val": [], "test": []}
    cell_of_split = {"train": {c: [] for c in CELL_ORDER}, "val": {c: [] for c in CELL_ORDER}, "test": {c: [] for c in CELL_ORDER}}

    for cell_name in CELL_ORDER:
        assigned = assign_cell(cells[cell_name], rng)
        for split, ids in assigned.items():
            splits[split].extend(ids)
            cell_of_split[split][cell_name] = ids

    # --- Chequeos post-split ---
    def n_rejected(ids):
        return sum(1 for i in ids if by_id[i]["Status"] == "Rejected")

    # Swap dentro de la misma celda primaria si test no llega a >=4 Rejected.
    MIN_REJECTED_TEST = 4
    deficit = MIN_REJECTED_TEST - n_rejected(splits["test"])
    if deficit > 0:
        # Buscar candidatos Rejected en train/val, y no-Rejected en test, dentro de la
        # misma celda primaria, para swapear sin romper el estrato 2D ni los tamaños.
        for cell_name in CELL_ORDER:
            if deficit <= 0:
                break
            for donor_split in ("val", "train"):
                if deficit <= 0:
                    break
                donor_ids = cell_of_split[donor_split][cell_name]
                test_ids = cell_of_split["test"][cell_name]
                rejected_donors = [i for i in donor_ids if by_id[i]["Status"] == "Rejected"]
                non_rejected_test = [i for i in test_ids if by_id[i]["Status"] != "Rejected"]
                n_swap = min(deficit, len(rejected_donors), len(non_rejected_test))
                for k in range(n_swap):
                    d_id = rejected_donors[k]
                    t_id = non_rejected_test[k]
                    cell_of_split[donor_split][cell_name].remove(d_id)
                    cell_of_split[donor_split][cell_name].append(t_id)
                    cell_of_split["test"][cell_name].remove(t_id)
                    cell_of_split["test"][cell_name].append(d_id)
                    splits[donor_split].remove(d_id)
                    splits[donor_split].append(t_id)
                    splits["test"].remove(t_id)
                    splits["test"].append(d_id)
                    deficit -= 1

    # --- Verificaciones de integridad ---
    all_ids = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert set(splits["train"]) & set(splits["val"]) == set()
    assert set(splits["train"]) & set(splits["test"]) == set()
    assert set(splits["val"]) & set(splits["test"]) == set()
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 179
    assert all_ids == set(by_id.keys())

    # --- Resumen por consola ---
    def overlap_mean(ids):
        vals = [float(by_id[i]["Solap_PTV_Rectum_cc"]) for i in ids]
        return sum(vals) / len(vals) if vals else float("nan")

    def constraint_fail_counts(ids):
        rv65 = sum(1 for i in ids if flag_is_fail(by_id[i], "Flag_Rectum_V65Gy_lt15_D95norm"))
        rv55 = sum(1 for i in ids if flag_is_fail(by_id[i], "Flag_Rectum_V55Gy_lt25_D95norm"))
        bv65 = sum(1 for i in ids if flag_is_fail(by_id[i], "Flag_Bladder_V65Gy_lt15_D95norm"))
        return rv65, rv55, bv65

    print("=" * 70)
    for split in ("train", "val", "test"):
        ids = splits[split]
        cell_counts = {c: len(cell_of_split[split][c]) for c in CELL_ORDER}
        rv65, rv55, bv65 = constraint_fail_counts(ids)
        print(f"[{split}] n={len(ids)}")
        print(f"    celdas 2D: {cell_counts}")
        print(f"    n Rejected: {n_rejected(ids)}")
        print(f"    fallo operativo -> RV65:{rv65} RV55:{rv55} BV65:{bv65}")
        print(f"    media overlap Solap_PTV_Rectum_cc: {overlap_mean(ids):.3f}")
    print("=" * 70)
    print(f"Union == 179: {len(all_ids) == 179}")
    print(f"Disjuncion train/val/test: OK")

    out = {
        "train": sorted(splits["train"]),
        "val": sorted(splits["val"]),
        "test": sorted(splits["test"]),
        "metadata": {
            "estratificacion": "2D_compliance_operativo_D95norm (recto_op_fail x vejiga_op_fail)",
            "recto_op_fail": "Flag_Rectum_V65Gy_lt15_D95norm==0 OR Flag_Rectum_V55Gy_lt25_D95norm==0",
            "vejiga_op_fail": "Flag_Bladder_V65Gy_lt15_D95norm==0",
            "celdas_esperadas": expected_counts,
            "orden_siembra": CELL_ORDER,
            "fracciones": FRACTIONS,
            "random_state": SEED,
            "n_total": 179,
            "n_train": len(splits["train"]),
            "n_val": len(splits["val"]),
            "n_test": len(splits["test"]),
            "post_split_checks": {
                "min_rejected_test_target": MIN_REJECTED_TEST,
                "n_rejected_test": n_rejected(splits["test"]),
                "n_rejected_val": n_rejected(splits["val"]),
                "n_rejected_train": n_rejected(splits["train"]),
            },
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nEscrito: {OUT_PATH}")


if __name__ == "__main__":
    main()
