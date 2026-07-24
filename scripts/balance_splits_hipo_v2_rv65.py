"""Corrige el desbalance de positivos RV65 en val de splits_hipo_v2_clean.json
(2/22 = 9.1%, insuficiente para calibrar el umbral operativo — cae a fallback clinico
en los 3 experimentos) vs. train (28.6%) y test (32.3%).

Estrategia: swaps MINIMOS train<->val, DENTRO de la misma celda primaria del estrato 2D
(recto_op_fail x vejiga_op_fail, flags D95norm — misma clasificacion que
make_splits_hipo_v2_clean.py) para no romper esa estratificacion. No se toca test (su
balance de RV65 ya es adecuado, 32.3%).

RV65-fail usa el flag GT real (grilla 256, Flag_Rectum_V65Gy_lt15_gt256 de
gt_dvh_hipo_256.csv) — es el que efectivamente usa evaluate_hipo.py para capa 3, no el
flag D95norm usado solo para la estratificacion del split.

Swaps elegidos (seed=42, muestreo entre los candidatos validos de cada celda):
  - celda solo_recto: 2 swaps train(RV65-fail) <-> val(RV65-no-fail)
  - celda ambos_fallan: 1 swap train(RV65-fail) <-> val(RV65-no-fail)
Total: val gana 3 positivos RV65 (2 -> 5), train pierde 3 (sin cambiar su prevalencia
de forma apreciable, 28/98 -> 25/98).

Uso:
    python scripts/balance_splits_hipo_v2_rv65.py
"""

import csv
import json
import random
from pathlib import Path

import pandas as pd

CSV_PATH = Path(r"c:\Pablo\ProstateDoseProject\dicoms hipofx\metricas_planes_hipofx_D95norm_clean.csv")
GT_DVH_PATH = Path(r"c:\Pablo\ProstateDoseProject\repo\data\gt_dvh_hipo_256.csv")
SPLITS_IN = Path(r"c:\Pablo\ProstateDoseProject\repo\data\splits\splits_hipo_v2_clean.json")
SPLITS_OUT = Path(r"c:\Pablo\ProstateDoseProject\repo\data\splits\splits_hipo_v2_clean_balanced.json")

SEED = 42
N_SWAPS_POR_CELDA = {"solo_recto": 2, "ambos_fallan": 1}


def flag_is_fail(row, col):
    return int(float(row[col])) == 0


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


def main():
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    by_id = {r["AnonID"]: r for r in rows}
    cell_of_id = {pid: classify_cell(r) for pid, r in by_id.items()}

    gt = pd.read_csv(GT_DVH_PATH).set_index("AnonID")
    rv65_fail_real = (gt["Flag_Rectum_V65Gy_lt15_gt256"] == 0)

    splits = json.load(open(SPLITS_IN))
    splits = {k: list(v) for k, v in splits.items() if k in ("train", "val", "test")}

    def rv65_counts():
        return {s: sum(1 for i in splits[s] if bool(rv65_fail_real.get(i, False))) for s in ("train", "val", "test")}

    print("Antes de swaps:", rv65_counts())

    rng = random.Random(SEED)
    swaps_realizados = []

    for cell, n_swaps in N_SWAPS_POR_CELDA.items():
        train_fail = sorted(
            i for i in splits["train"]
            if cell_of_id[i] == cell and bool(rv65_fail_real.get(i, False))
        )
        val_nofail = sorted(
            i for i in splits["val"]
            if cell_of_id[i] == cell and not bool(rv65_fail_real.get(i, False))
        )
        assert len(train_fail) >= n_swaps and len(val_nofail) >= n_swaps, (
            f"celda {cell}: candidatos insuficientes (train_fail={len(train_fail)}, "
            f"val_nofail={len(val_nofail)}, necesito {n_swaps})"
        )
        donors_train = rng.sample(train_fail, n_swaps)
        donors_val = rng.sample(val_nofail, n_swaps)
        for t_id, v_id in zip(donors_train, donors_val):
            splits["train"].remove(t_id)
            splits["train"].append(v_id)
            splits["val"].remove(v_id)
            splits["val"].append(t_id)
            swaps_realizados.append({"celda": cell, "train_to_val": t_id, "val_to_train": v_id})

    print("\nSwaps realizados:")
    for s in swaps_realizados:
        print(f"  [{s['celda']}] train->val: {s['train_to_val']} (RV65-fail)   "
              f"val->train: {s['val_to_train']} (RV65-no-fail)")

    print("\nDespues de swaps:", rv65_counts())

    # --- Verificaciones de integridad: mismos IDs totales, mismos tamanios de celda por split
    for s in ("train", "val", "test"):
        splits[s] = sorted(splits[s])
    all_ids_before = set(json.load(open(SPLITS_IN))["train"]) | set(json.load(open(SPLITS_IN))["val"]) | set(json.load(open(SPLITS_IN))["test"])
    all_ids_after = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert all_ids_before == all_ids_after, "el conjunto total de pacientes cambio (no deberia)"
    assert set(splits["train"]) & set(splits["val"]) == set()
    assert set(splits["train"]) & set(splits["test"]) == set()
    assert set(splits["val"]) & set(splits["test"]) == set()

    old = json.load(open(SPLITS_IN))
    for s in ("train", "val", "test"):
        assert len(splits[s]) == len(old[s]), f"tamanio de {s} cambio"

    def cell_counts(ids):
        c = {}
        for i in ids:
            c[cell_of_id[i]] = c.get(cell_of_id[i], 0) + 1
        return c

    for s in ("train", "val", "test"):
        cc_old = cell_counts(old[s])
        cc_new = cell_counts(splits[s])
        assert cc_old == cc_new, f"celdas 2D de {s} cambiaron: {cc_old} -> {cc_new}"
    print("\nVerificacion OK: mismos tamanios de split y de celda 2D por split, mismo universo de pacientes.")

    meta = dict(old.get("metadata", {}))
    meta["balanceado_rv65"] = True
    meta["swaps_rv65"] = swaps_realizados
    meta["seed_balanceo"] = SEED
    meta["rv65_counts_antes"] = {"train": 28, "val": 2, "test": 10}
    meta["rv65_counts_despues"] = rv65_counts()

    out = {"train": splits["train"], "val": splits["val"], "test": splits["test"], "metadata": meta}
    with open(SPLITS_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nEscrito: {SPLITS_OUT}")


if __name__ == "__main__":
    main()
