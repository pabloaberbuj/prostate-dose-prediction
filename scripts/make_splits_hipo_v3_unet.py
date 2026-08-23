"""Genera repo/data/splits/splits_hipo_v3_unet.json (train/val/test ~= 65/15/20) para el
dataset hipofraccionado NUEVO (201 pacientes: reales hipo + pasados-a-normo "Rejected" +
normos-replanificados-como-hipo "UnApproved"), fuente
data/metricas_planes_hipofx_D95norm.csv.

NOTA DE NOMBRE: splits_hipo_v3.json YA EXISTE en este repo pero pertenece a otro proyecto
(entrenamiento ML clasico, mismos datos, distinta metodologia de split) — NO TOCAR ese
archivo. Este script escribe un archivo con sufijo _unet para no pisarlo.

Estratificacion 2D:
  - Dimension 1 (cumple/no-cumple): constraint operativo AND estricto entre
    Flag_Rectum_V65Gy_lt15_D95norm, Flag_Rectum_V55Gy_lt25_D95norm,
    Flag_Bladder_V65Gy_lt15_D95norm. Cumple = las 3 en 1. No-cumple = cualquiera en 0.
  - Dimension 2 (overlap PTV-Recto real): terciles de Solap_PTV_Rectum_cc (confirmado por
    Pablo: el bug de CalcularSolapamiento ya esta corregido en el C#, esta columna es el
    overlap geometrico real en el CSV nuevo).

Manejo de las 2 filas CSV ambiguas (mismo AnonID, 2 planes RP/RD en la misma carpeta DICOM,
CSV tiene una fila por plan): PT_70368fdeed2777a4 y PT_8b1aa3d35e1b468b. El NPZ ya existente
en processed_hipo/ viene de una carpeta con UN solo plan (extraido antes de que existiera el
segundo). Se resuelve la ambiguedad comparando meta['s_D95'] del NPZ contra FactorNorm_D95 de
cada fila candidata y quedandose con la fila mas cercana.

Correccion de balance (si val no llega a >=4-5 no-cumplidores): swaps train<->val DENTRO de
la misma celda 2D, sin tocar test (misma tecnica que balance_splits_hipo_v2_rv65.py).

Uso:
    python scripts/make_splits_hipo_v3_unet.py
"""

import csv
import json
import random
from pathlib import Path

import numpy as np

CSV_PATH = Path(r"C:\Pablo\ProstateDoseProject\repo\data\metricas_planes_hipofx_D95norm.csv")
PROCESSED_DIR = Path(r"C:\Pablo\ProstateDoseProject\processed_hipo")
OUT_PATH = Path(r"C:\Pablo\ProstateDoseProject\repo\data\splits\splits_hipo_v3_unet.json")

SEED = 42
FRACTIONS = {"train": 0.65, "val": 0.15, "test": 0.20}
MIN_NOCUMPLE_VAL = 5

AMBIGUOS = ["PT_70368fdeed2777a4", "PT_8b1aa3d35e1b468b"]


def flag_fail(row, col):
    return int(float(row[col])) == 0


def es_no_cumple(row):
    return (
        flag_fail(row, "Flag_Rectum_V65Gy_lt15_D95norm")
        or flag_fail(row, "Flag_Rectum_V55Gy_lt25_D95norm")
        or flag_fail(row, "Flag_Bladder_V65Gy_lt15_D95norm")
    )


def resolver_fila_ambigua(anonid, filas_candidatas):
    """Para pacientes con >1 fila CSV (2 planes en la misma carpeta), elige la fila cuyo
    FactorNorm_D95 mas se parece al s_D95 realmente usado en el NPZ ya preprocesado."""
    npz_path = PROCESSED_DIR / f"{anonid}.npz"
    meta = json.loads(np.load(npz_path, allow_pickle=True)["meta"][0])
    s_d95_npz = float(meta["s_D95"])
    mejor = min(filas_candidatas, key=lambda r: abs(float(r["FactorNorm_D95"]) - s_d95_npz))
    print(
        f"  {anonid}: s_D95 NPZ={s_d95_npz:.4f} -> elige fila Status={mejor['Status']} "
        f"FactorNorm_D95={mejor['FactorNorm_D95']} (candidatas: "
        f"{[(r['Status'], r['FactorNorm_D95']) for r in filas_candidatas]})"
    )
    return mejor


def load_rows():
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    por_id = {}
    for r in rows:
        por_id.setdefault(r["AnonID"], []).append(r)

    resueltas = []
    for anonid, filas in por_id.items():
        if len(filas) == 1:
            resueltas.append(filas[0])
            continue
        # Duplicado exacto (mismo Status/valores) -> se queda con una copia sin mas.
        valores_unicos = {tuple(f.items()) for f in filas}
        if len(valores_unicos) == 1:
            resueltas.append(filas[0])
            continue
        # Duplicado real (2 planes) -> resolver contra el NPZ.
        resueltas.append(resolver_fila_ambigua(anonid, filas))

    return resueltas


def apportion(n, fractions=FRACTIONS):
    splits = list(fractions.keys())
    exact = {s: n * fractions[s] for s in splits}
    base = {s: int(exact[s]) for s in splits}
    remainder = n - sum(base.values())
    order = sorted(splits, key=lambda s: exact[s] - base[s], reverse=True)
    for s in order[:remainder]:
        base[s] += 1
    return base


def assign_cell(ids, rng):
    ids_shuffled = list(ids)
    rng.shuffle(ids_shuffled)
    counts = apportion(len(ids_shuffled))
    out, i = {}, 0
    for split in ("train", "val", "test"):
        n = counts[split]
        out[split] = ids_shuffled[i : i + n]
        i += n
    return out


def main():
    rows_all = load_rows()
    print(f"Pacientes unicos tras resolver duplicados: {len(rows_all)}")

    # Exclusion: fallas de extraccion C# (campos criticos vacios, ej. RS sin
    # Rectum/Bladder encontrados -> Vol/overlap/flags vacios). Reportar antes de
    # construir el split, no dejar que rompan la estratificacion en silencio.
    campos_criticos = [
        "Solap_PTV_Rectum_cc", "Flag_Rectum_V65Gy_lt15_D95norm",
        "Flag_Rectum_V55Gy_lt25_D95norm", "Flag_Bladder_V65Gy_lt15_D95norm",
    ]
    rows, excluidos_falla = [], []
    for r in rows_all:
        if any(r[c].strip() == "" for c in campos_criticos):
            excluidos_falla.append(r["AnonID"])
        else:
            rows.append(r)
    if excluidos_falla:
        print(f"\n*** EXCLUIDOS por falla de extraccion (campos criticos vacios): "
              f"{excluidos_falla} ***")
        print("    (ver detalle: probablemente Rectum/Bladder no encontrados en el RS — "
              "revisar con Pablo si se puede re-extraer.)\n")

    by_id = {r["AnonID"]: r for r in rows}

    overlaps = np.array([float(r["Solap_PTV_Rectum_cc"]) for r in rows])
    t1, t2 = np.percentile(overlaps, [33.33, 66.67])
    print(f"Terciles overlap PTV-Recto (Solap_PTV_Rectum_cc): t1={t1:.2f} t2={t2:.2f}")

    def overlap_bin(v):
        if v <= t1:
            return "overlap_bajo"
        if v <= t2:
            return "overlap_medio"
        return "overlap_alto"

    def cell_of(row):
        cumple = "no_cumple" if es_no_cumple(row) else "cumple"
        return f"{cumple}__{overlap_bin(float(row['Solap_PTV_Rectum_cc']))}"

    cells = {}
    for r in rows:
        cells.setdefault(cell_of(r), []).append(r["AnonID"])

    print("\nCeldas 2D (cumple x tercil overlap):")
    for c in sorted(cells):
        print(f"  {c}: {len(cells[c])}")

    rng = random.Random(SEED)
    splits = {"train": [], "val": [], "test": []}
    cell_of_split = {s: {c: [] for c in cells} for s in ("train", "val", "test")}

    # Celdas chicas primero, para no dejar sin representacion en test/val.
    orden = sorted(cells, key=lambda c: len(cells[c]))
    for cell_name in orden:
        assigned = assign_cell(cells[cell_name], rng)
        for split, ids in assigned.items():
            splits[split].extend(ids)
            cell_of_split[split][cell_name] = ids

    def n_no_cumple(ids):
        return sum(1 for i in ids if es_no_cumple(by_id[i]))

    print(f"\nNo-cumplidores antes de swaps: train={n_no_cumple(splits['train'])} "
          f"val={n_no_cumple(splits['val'])} test={n_no_cumple(splits['test'])}")

    # --- Swaps train<->val dentro de la misma celda si val no llega al minimo ---
    swaps_realizados = []
    deficit = MIN_NOCUMPLE_VAL - n_no_cumple(splits["val"])
    if deficit > 0:
        no_cumple_cells = [c for c in orden if c.startswith("no_cumple__")]
        for cell_name in no_cumple_cells:
            if deficit <= 0:
                break
            train_nocumple = [i for i in cell_of_split["train"][cell_name] if es_no_cumple(by_id[i])]
            val_cumple = [i for i in cell_of_split["val"][cell_name] if not es_no_cumple(by_id[i])]
            n_swap = min(deficit, len(train_nocumple), len(val_cumple))
            for k in range(n_swap):
                t_id, v_id = train_nocumple[k], val_cumple[k]
                cell_of_split["train"][cell_name].remove(t_id)
                cell_of_split["train"][cell_name].append(v_id)
                cell_of_split["val"][cell_name].remove(v_id)
                cell_of_split["val"][cell_name].append(t_id)
                splits["train"].remove(t_id)
                splits["train"].append(v_id)
                splits["val"].remove(v_id)
                splits["val"].append(t_id)
                swaps_realizados.append({"celda": cell_name, "train_to_val": t_id, "val_to_train": v_id})
                deficit -= 1

    print(f"\nSwaps train<->val realizados: {len(swaps_realizados)}")
    for s in swaps_realizados:
        print(f"  [{s['celda']}] train->val: {s['train_to_val']}  val->train: {s['val_to_train']}")

    print(f"\nNo-cumplidores despues de swaps: train={n_no_cumple(splits['train'])} "
          f"val={n_no_cumple(splits['val'])} test={n_no_cumple(splits['test'])}")

    # --- Verificaciones de integridad ---
    all_ids = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert set(splits["train"]) & set(splits["val"]) == set()
    assert set(splits["train"]) & set(splits["test"]) == set()
    assert set(splits["val"]) & set(splits["test"]) == set()
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(rows)
    assert all_ids == set(by_id.keys())

    print("=" * 70)
    for split in ("train", "val", "test"):
        ids = splits[split]
        cc = {c: len(cell_of_split[split][c]) for c in cells}
        estados = {}
        for i in ids:
            st = by_id[i]["Status"]
            estados[st] = estados.get(st, 0) + 1
        print(f"[{split}] n={len(ids)}  no_cumple={n_no_cumple(ids)}  Status={estados}")
        print(f"    celdas: {cc}")
    print("=" * 70)

    out = {
        "train": sorted(splits["train"]),
        "val": sorted(splits["val"]),
        "test": sorted(splits["test"]),
        "excluidos": excluidos_falla,
        "metadata": {
            "excluidos_falla_extraccion": excluidos_falla,
            "proposito": "U-Net dose prediction, dataset hipo v3 (201 pacientes, incluye "
                         "Rejected + UnApproved). NO confundir con splits_hipo_v3.json "
                         "(otro proyecto, ML clasico).",
            "estratificacion": "2D: cumple/no-cumple (AND estricto RV65,RV55,BV65 D95norm) "
                                "x tercil overlap PTV-Recto (Solap_PTV_Rectum_cc, real)",
            "terciles_overlap_cc": [round(float(t1), 3), round(float(t2), 3)],
            "celdas_2d": {c: len(cells[c]) for c in cells},
            "fracciones": FRACTIONS,
            "random_state": SEED,
            "n_total": len(rows),
            "n_train": len(splits["train"]),
            "n_val": len(splits["val"]),
            "n_test": len(splits["test"]),
            "min_no_cumple_val_target": MIN_NOCUMPLE_VAL,
            "swaps_train_val": swaps_realizados,
            "ambiguos_resueltos_via_npz": AMBIGUOS,
            "dataset_source": str(CSV_PATH),
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nEscrito: {OUT_PATH}")


if __name__ == "__main__":
    main()
