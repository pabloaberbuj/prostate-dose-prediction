"""Orquestador: corre el pipeline Proyecto 1 (clasificacion + regresion + calibracion
CV + evaluacion bootstrap) sobre el dataset/split nuevo (ctfix), REUTILIZANDO el codigo
de los scripts existentes (train_p1_clf.py, train_p1_reg.py, calibrate_p1_cv.py,
eval_p1_cv.py, build_manifest_p1.py) sin modificarlos ni duplicar su logica.

Como: cada uno de esos scripts define sus rutas (DATASET_CSV, MODELS_DIR, OUT_DIR,
SPLITS_JSON, OUT_THRESHOLDS) como constantes a nivel de modulo, y sus funciones
(incluida main()) las resuelven por nombre desde el namespace del modulo en el momento
de llamarlas -- no estan "quemadas" dentro del bytecode de la funcion. Por eso alcanza
con importar cada modulo, pisarle esas constantes para que apunten a las rutas *_ctfix,
y despues invocar su main() normal. Los scripts originales (y sus outputs en
models/proyecto1/, results/proyecto1_v3/, results/proyecto1_v3_cv/) quedan intactos --
esto solo cambia el estado del modulo importado en este proceso, nunca el archivo .py.

Correspondencia de rutas nuevas <-> originales:
  data/dataset_p1_ctfix.csv        <-> data/dataset_p1.csv          (prep_data_p1_ctfix.py)
  models/proyecto1_ctfix/          <-> models/proyecto1/
  results/proyecto1_ctfix_v4/      <-> results/proyecto1_v3_cv/     (evaluacion CV, la que se usa)
  data/splits/splits_hipo_ctfix_v4.json <-> data/splits/splits_hipo_v3.json

NOTA IMPORTANTE (protocolo, ver reporte del agente que escribio esto): a diferencia de
splits_hipo_v3.json, splits_hipo_ctfix_v4.json NO enruta UnApproved aparte -- quedan
mezclados en train/val/test como cualquier otro paciente. Esto tiene dos consecuencias
que se dejan TAL CUAL vienen de los scripts originales, sin intentar "corregir":
  1. calibrate_p1_cv.pool_calibracion() sigue filtrando por Status!='UnApproved' dentro
     de train+val -- funciona igual de bien mecanicamente, solo que ahora excluye
     tambien los UnApproved que cayeron en val (antes val era 100% natural por
     construccion del split viejo).
  2. eval_p1_cv.py evalua sobre TODO el test set del split (aqui: mezcla de
     TreatmentApproved/Rejected/UnApproved), NO solo poblacion "natural" como decia el
     comentario original de eval_p1.py ("prevalencia natural"). Se elige asi
     deliberadamente porque el proposito explicito del rerun (HIPOFX_KICKOFF.md) es
     poder comparar clasico-ML vs U-Net en el MISMO test set exacto -- si aca se
     filtrara a solo-natural, el test dejaria de ser el mismo conjunto de pacientes que
     usa el lado U-Net.

No se elige variante A vs B de la frontera verde (igual que calibrate_p1_cv.py: ambas
quedan calibradas y reportadas, ninguna se descarta).

Uso:
    python scripts/prep_data_p1_ctfix.py   # (una vez, o cuando cambie el split/CSV)
    python scripts/run_p1_ctfix_pipeline.py
"""

import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

DATASET_CSV = _REPO_ROOT / "data" / "dataset_p1_ctfix.csv"
SPLITS_JSON = _REPO_ROOT / "data" / "splits" / "splits_hipo_ctfix_v4.json"
MODELS_DIR = _REPO_ROOT / "models" / "proyecto1_ctfix"
RESULTS_DIR = _REPO_ROOT / "results" / "proyecto1_ctfix_v4"


def _step(nombre, modulo_name, overrides):
    print("\n" + "#" * 78)
    print(f"# {nombre}  (scripts/{modulo_name}.py, con rutas *_ctfix)")
    print("#" * 78)
    mod = importlib.import_module(modulo_name)
    mod = importlib.reload(mod)  # aislar de estados previos si ya se importo antes
    for attr, value in overrides.items():
        assert hasattr(mod, attr), f"{modulo_name} no tiene un atributo de modulo '{attr}'"
        setattr(mod, attr, value)
    mod.main()
    return mod


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"{DATASET_CSV} no existe -- correr primero scripts/prep_data_p1_ctfix.py"
        )

    # Tarea 2: clasificacion (logreg desplegado + GB control) por constraint.
    _step("Tarea 2 -- train_p1_clf", "train_p1_clf", {
        "DATASET_CSV": DATASET_CSV,
        "MODELS_DIR": MODELS_DIR,
    })

    # Tarea 3: regresion de severidad (Ridge / HGB) por constraint.
    _step("Tarea 3 -- train_p1_reg", "train_p1_reg", {
        "DATASET_CSV": DATASET_CSV,
        "MODELS_DIR": MODELS_DIR,
    })

    # Tarea 4 (variante CV, la vigente): calibracion frontera verde + Delta.
    _step("Tarea 4 -- calibrate_p1_cv", "calibrate_p1_cv", {
        "DATASET_CSV": DATASET_CSV,
        "MODELS_DIR": MODELS_DIR,
        "OUT_THRESHOLDS": MODELS_DIR / "thresholds_cv.json",
    })

    # Tarea 5 (variante CV, la vigente): evaluacion en test con bootstrap CI95.
    _step("Tarea 5 -- eval_p1_cv", "eval_p1_cv", {
        "DATASET_CSV": DATASET_CSV,
        "MODELS_DIR": MODELS_DIR,
        "OUT_DIR": RESULTS_DIR,
    })

    # Tarea 6: manifest de despliegue. NO se reusa build_manifest_p1.main() tal cual
    # porque adentro tiene hardcodeado el nombre "thresholds.json" (no es una constante
    # de modulo pisable) -- aca solo existe thresholds_cv.json (no se corrio la
    # calibracion no-cv, superada segun la memoria del proyecto). Se reusan sus piezas
    # reutilizables (verificar_carga, FEATURE_COLS_FINAL, CONSTRAINTS) y se arma el
    # manifest con la misma forma, apuntando al archivo de thresholds que si existe.
    print("\n" + "#" * 78)
    print("# Tarea 6 -- manifest (variante propia, ver comentario arriba)")
    print("#" * 78)
    bm = importlib.import_module("build_manifest_p1")
    bm = importlib.reload(bm)
    print("Verificando que todos los artefactos cargan sin re-fitear...")
    archivos = {}
    for tag in bm.CONSTRAINTS:
        clf_path = MODELS_DIR / f"clf_{tag}.joblib"
        reg_path = MODELS_DIR / f"reg_{tag}.joblib"
        clf = bm.verificar_carga(clf_path, ["scaler", "logreg", "gb", "feature_cols"])
        reg = bm.verificar_carga(reg_path, ["scaler", "modelo", "modelo_tipo", "feature_cols", "threshold"])
        assert clf["feature_cols"] == bm.FEATURE_COLS_FINAL, f"{clf_path}: orden de features inesperado"
        assert reg["feature_cols"] == bm.FEATURE_COLS_FINAL, f"{reg_path}: orden de features inesperado"
        print(f"  OK  clf_{tag}.joblib (logreg + gb)")
        print(f"  OK  reg_{tag}.joblib (modelo={reg['modelo_tipo']})")
        archivos[tag] = {"clasificacion": clf_path.name, "regresion": reg_path.name, "modelo_regresion": reg["modelo_tipo"]}

    thresholds_path = MODELS_DIR / "thresholds_cv.json"
    with open(thresholds_path, encoding="utf-8") as f:
        json.load(f)  # solo valida que carga
    print(f"  OK  {thresholds_path.name}")

    with open(SPLITS_JSON, encoding="utf-8") as f:
        splits_meta = json.load(f)["metadata"]

    manifest = {
        "version_features": "p1_v1_7geometricas",
        "feature_cols_orden": bm.FEATURE_COLS_FINAL,
        "constraints": bm.CONSTRAINTS,
        "modelo_desplegado_clasificacion": "logreg (StandardScaler + LogisticRegression, class_weight=balanced)",
        "modelo_control_clasificacion": "HistGradientBoostingClassifier (no se despliega, solo referencia)",
        "modelo_regresion_severidad": {tag: archivos[tag]["modelo_regresion"] for tag in bm.CONSTRAINTS},
        "archivos": archivos,
        "thresholds_file": thresholds_path.name,
        "calibracion_protocolo": "CV (calibrate_p1_cv.py) -- variantes A (prob) y B (margen) reportadas, ninguna descartada",
        "dataset_csv": DATASET_CSV.name,
        "splits_file": SPLITS_JSON.name,
        "splits_metadata": splits_meta,
        "fecha_generacion": datetime.now().strftime("%Y-%m-%d"),
        "notas": [
            "Rerun de Proyecto 1 (ver scripts/prep_data_p1_ctfix.py y "
            "scripts/run_p1_ctfix_pipeline.py) sobre splits_hipo_ctfix_v4.json, para "
            "comparacion apples-to-apples con U-Net (HIPOFX_KICKOFF.md). No modifica "
            "scripts/proyecto1_v3* originales.",
            "A diferencia de splits_hipo_v3.json, este split mezcla UnApproved en las "
            "3 particiones (no lo enruta entero a train) -- ver docstring de "
            "run_p1_ctfix_pipeline.py para el detalle de como eso afecta el pool de "
            "calibracion CV y la evaluacion en test.",
        ],
    }
    manifest_path = MODELS_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado: {manifest_path}")

    print("\n" + "=" * 78)
    print(f"Pipeline ctfix completo. Artefactos en {MODELS_DIR} y {RESULTS_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
