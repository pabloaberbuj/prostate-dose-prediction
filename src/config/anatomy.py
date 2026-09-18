"""
Carga del "anatomy schema" — declara qué estructuras anatómicas existen para una
localización dada (próstata, próstata hipofraccionada, pulmón, ...), sus alias de
nombre en DICOM/RTSTRUCT, prescripción, parámetros de preprocesamiento y constraints
clínicos. Ver configs/anatomy/*.yaml para los schemas disponibles.

Diseño (ver docs/CLI_INSTRUCTIVO.md para el detalle completo):
- Un YAML de experimento (configs/exp*.yaml) puede declarar `anatomy: <nombre>` para
  elegir su schema. Si NO lo declara, el default es "prostate" — los 22 YAML de
  experimento existentes al momento de este refactor no se tocan ni se migran.
- `cfg.anatomy_schema` se agrega como namespace NUEVO y ADITIVO al config de
  experimento — nunca pisa `cfg.data`/`cfg.model`/`cfg.constraints` existentes.
- Un schema puede declarar `extends: <otro.yaml>` (mismo directorio) para heredar y
  overridear parcialmente — usado por prostate_hipo.yaml sobre prostate.yaml.
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

ANATOMY_DIR_DEFAULT = Path(__file__).resolve().parent.parent.parent / "configs" / "anatomy"


def load_anatomy(name_or_path: str, anatomy_dir: Path = ANATOMY_DIR_DEFAULT) -> DictConfig:
    """
    Carga un anatomy schema por nombre (busca `<anatomy_dir>/<name>.yaml`) o por path
    directo a un .yaml. Resuelve `extends:` recursivamente vía OmegaConf.merge (el
    hijo pisa las claves del padre, el resto se hereda).
    """
    path = Path(name_or_path)
    if not path.is_absolute():
        # Relativo a anatomy_dir tanto si es un nombre pelado ("prostate") como si
        # ya trae extensión (p.ej. el valor de `extends:`, que se escribe como
        # "prostate.yaml" y debe resolverse contra el mismo directorio del schema
        # que lo declara, no contra el cwd del proceso).
        path = anatomy_dir / (name_or_path if path.suffix else f"{name_or_path}.yaml")
    if not path.exists():
        raise FileNotFoundError(
            f"Anatomy schema no encontrado: '{name_or_path}' (buscado en {path}). "
            f"Schemas disponibles en {anatomy_dir}: "
            f"{sorted(p.stem for p in anatomy_dir.glob('*.yaml'))}"
        )

    cfg = OmegaConf.load(path)
    extends = cfg.pop("extends", None)
    if extends is not None:
        base = load_anatomy(extends, anatomy_dir=path.parent)
        base = OmegaConf.create({"anatomy": base})  # re-envolver para mergear con el mismo shape que cfg (que trae su propia clave "anatomy")
        override_structures = cfg.get("anatomy", {}).pop("structures", None)
        cfg = OmegaConf.merge(base, cfg)
        if override_structures is not None:
            cfg.anatomy.structures = _merge_structures(base.anatomy.get("structures", []),
                                                         override_structures)
    return cfg.anatomy if "anatomy" in cfg else cfg


def _merge_structures(base_structures, override_structures):
    """
    OmegaConf.merge reemplaza listas enteras en vez de mergearlas elemento a
    elemento — sin este helper, un `extends:` que solo overridea la estructura
    'ptv' (ej. prostate_hipo.yaml) perdería silenciosamente 'body'/'rectum'/
    'bladder' de la anatomía base. Mergea por el campo `key`, preservando el
    orden de `base_structures` y agregando al final cualquier `key` nuevo que
    no exista en la base.
    """
    orden = [s.key for s in base_structures]
    por_key = {s.key: OmegaConf.create(dict(s)) for s in base_structures}
    for override in override_structures:
        if override.key in por_key:
            por_key[override.key] = OmegaConf.merge(por_key[override.key], override)
        else:
            por_key[override.key] = OmegaConf.create(dict(override))
            orden.append(override.key)
    return OmegaConf.create([por_key[k] for k in orden])


def load_experiment_config(config_path: str) -> DictConfig:
    """
    Carga un YAML de experimento (configs/exp*.yaml) y le agrega `cfg.anatomy_schema`
    resuelto a partir de la clave opcional `cfg.anatomy` (nombre del schema; default
    "prostate" si el experimento no la declara — retrocompatible con los 22 YAML
    existentes al momento de este refactor, que no declaran `anatomy:`).

    IMPORTANTE — precedencia de constraints: si el experimento ya trae su propio
    bloque `cfg.constraints` (todos los existentes lo traen), ESE bloque sigue
    siendo la fuente de verdad para reproducir runs históricos exactos. El bloque
    `cfg.anatomy_schema.constraints` es el que usan los scripts ya generalizados
    (o experimentos nuevos que no definan `cfg.constraints` propio, ej. anatomías
    nuevas como "lung").
    """
    cfg = OmegaConf.load(config_path)
    anatomy_name = cfg.get("anatomy", "prostate")
    cfg.anatomy_schema = load_anatomy(anatomy_name)
    return cfg
