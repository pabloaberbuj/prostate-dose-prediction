"""
Bloque 3 (HANDOFF_substructuras_dosis.md) -- DVH de subestructuras -> ObjectiveTemplate XML.

Modo por defecto (2 vias, sin --overlap-roi-name):
  - Control: Rectum completo (linea unica) + Bladder + PTV puntos fijos.
  - Test:    Rectum_hot + Rectum_cold (dos lineas separadas) + Bladder + PTV.

Modo 3 vias (con --overlap-roi-name, ej. "Rectum!PTV"): separa el overlap
geometrico Rectum-PTV como estructura independiente, con prioridad mas baja y
SIN empuje (compite directo contra PTV -- empujarlo no tiene sentido, ver
conversacion del piloto). El resto de Rectum (hot menos overlap, cold menos
overlap) se empuja con --push-pct, a prioridad normal (--line-priority).

  - Test:    overlap (prioridad baja, sin empuje) + hot_no_overlap (empujado) +
             cold_no_overlap (empujado) + Bladder + PTV.
  - Control: Rectum completo, pero construido como el AGREGADO de los mismos
             valores de dosis por-voxel que arma el test (overlap SIN empujar +
             resto empujado), en una unica linea -- asi la unica diferencia real
             entre control y test sigue siendo la granularidad/prioridad, no la
             dosis pedida por voxel.

Particion 3 vias exacta por construccion (mutuamente excluyentes, unen a
Rectum completo): overlap = Rectum & overlap_roi; hot_no_overlap = hot & ~overlap;
cold_no_overlap = cold & ~overlap.

Bladder identica en todos los brazos (misma mascara, mismo dose_pred, sin
empujar) -- ancla, no es parte de la variable experimental.

No reescribe gen_objtemplate.build_template -- se le agrego soporte de
prioridad por-estructura (dict en vez de escalar) para poder bajarle la
prioridad solo al overlap.

Uso (2 vias):
    .venv/Scripts/python.exe scripts/build_substruct_objtemplates.py \
        --dicom-dir "//10.100.0.252/.../PT_44a81316c45f64f5" \
        --masks-npz results/substruct/PT_44a81316c45f64f5/masks.npz \
        --rx-gy 78 --push-pct 10 \
        --output-dir results/substruct/PT_44a81316c45f64f5

Uso (3 vias, overlap aparte):
    .venv/Scripts/python.exe scripts/build_substruct_objtemplates.py \
        --dicom-dir "//10.100.0.252/.../PT_44a81316c45f64f5" \
        --masks-npz results/substruct/PT_44a81316c45f64f5/masks.npz \
        --rx-gy 78 --push-pct 15 \
        --overlap-roi-name "Rectum!PTV" --overlap-priority 10 --overlap-push-pct 0 \
        --output-dir results/substruct/PT_44a81316c45f64f5
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.bridge.rtstruct_io import cargar_rtstruct_ct_only  # noqa: E402
from gen_objtemplate import build_template  # noqa: E402

N_PUNTOS_DVH = 200


def dvh_desde_mascara(dose_pct_rt: np.ndarray, mask_rt: np.ndarray, rx_gy: float):
    """DVH acumulativo (dosis Gy, vol %) de `dose_pct_rt` (convencion rt-utils,
    ya en % Rx) restringido a `mask_rt`. Mismo criterio que
    compute_pred_dvh.py::dvh_desde_dose_pred (conteo puro de voxels, sin
    calibracion fisica de voxel -- no aplica aca, ver ese script)."""
    dvox = dose_pct_rt[mask_rt]
    dose_grid_pct = np.linspace(0, float(dvox.max()), N_PUNTOS_DVH)
    vol_pct = np.array([(dvox >= d).mean() * 100.0 for d in dose_grid_pct])
    dose_grid_gy = dose_grid_pct * rx_gy / 100.0
    return dose_grid_gy, vol_pct


def empujar_dosis(dvh: tuple, push_pct: float) -> tuple:
    """Escala el eje dosis (Gy) de una curva (dose_gy, vol_pct) hacia abajo
    `push_pct`%, sin tocar el eje volumen -- pide la misma cobertura a menos
    dosis (ask mas estricto)."""
    dose_gy, vol_pct = dvh
    return dose_gy * (1.0 - push_pct / 100.0), vol_pct


def guardar_xml(tree: ET.ElementTree, path: Path):
    ET.indent(tree, space="")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(path), encoding="UTF-8", xml_declaration=True)
    # Validacion: re-parsear y contar objetivos, igual que gen_objtemplate.py.
    r = ET.parse(str(path)).getroot()
    for s in r.iter("ObjectivesOneStructure"):
        objs = s.findall("StructureObjectives/Objective")
        lineas = [o for o in objs if o.find("Type").text == "1"]
        puntos = [o for o in objs if o.find("Type").text == "0"]
        prios = sorted(set(o.find("Priority").text for o in lineas)) if lineas else []
        print(f"  {s.get('ID'):18s}  linea(Type1)={len(lineas):3d}  puntos(Type0)={len(puntos)}  "
              f"prioridad={prios}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", required=True, help="Carpeta con CT.*.dcm y RS*.dcm reales del paciente")
    parser.add_argument("--masks-npz", required=True, help="Salida de split_oar_by_dose.py")
    parser.add_argument("--rx-gy", type=float, required=True)
    parser.add_argument("--bladder-roi-name", default="Bladder")
    parser.add_argument("--ptv-id", default="PTV_High")
    parser.add_argument("--line-priority", type=float, default=50.0)
    parser.add_argument("--push-pct", type=float, default=0.0,
                         help="Empuje del eje dosis en Rectum/hot/cold (2 vias) o en "
                              "hot_no_overlap/cold_no_overlap (3 vias), en %%.")
    parser.add_argument("--overlap-roi-name", default=None,
                         help="Nombre de salida del overlap en el XML (ej. 'Rectum_overlap_PTV'). "
                              "Si se pasa, activa el modo 3 vias (overlap separado del resto del Rectum).")
    parser.add_argument("--overlap-from-ptv-intersection", action="store_true",
                         help="Calcular el overlap como Rectum & PTV_High (interseccion geometrica real) "
                              "en vez de leer un ROI clinico con ese nombre -- usar esto: el ROI clinico "
                              "'Rectum!PTV' de este paciente NO es la interseccion real (11264 vs 3727 "
                              "voxels, ni siquiera es superset de ella -- verificado, es otra cosa).")
    parser.add_argument("--overlap-priority", type=float, default=10.0)
    parser.add_argument("--overlap-push-pct", type=float, default=0.0,
                         help="Empuje especifico del overlap (default 0 -- compite "
                              "directo contra PTV, no tiene sentido empujarlo).")
    parser.add_argument("--output-dir", default=None, help="default: misma carpeta que --masks-npz")
    args = parser.parse_args()
    modo_3vias = args.overlap_roi_name is not None
    sufijo = f"_push{args.push_pct:g}" if args.push_pct else ""
    if modo_3vias:
        sufijo += "_overlap"

    dicom_dir = Path(args.dicom_dir)
    masks_npz_path = Path(args.masks_npz)
    output_dir = Path(args.output_dir) if args.output_dir else masks_npz_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando {masks_npz_path} ...")
    data = np.load(str(masks_npz_path), allow_pickle=True)
    dose_native_pct = data["dose_native_pct"]  # convencion rt-utils, ya validada
    roi_mask = data["roi_mask"]  # Rectum completo
    hot = data["hot"]
    cold = data["cold"]
    roi_name = str(data["roi_name"])
    anonid = str(data["anonid"])

    print(f"Leyendo mascara nativa de '{args.bladder_roi_name}'"
          + (f" y '{args.overlap_roi_name}'" if modo_3vias else "") + " via rt-utils ...")
    rtstruct = cargar_rtstruct_ct_only(dicom_dir)
    disponibles = rtstruct.get_roi_names()
    if args.bladder_roi_name not in disponibles:
        raise ValueError(f"'{args.bladder_roi_name}' no esta en el RS. Disponibles: {disponibles}")
    bladder_mask = rtstruct.get_roi_mask_by_name(args.bladder_roi_name)
    if bladder_mask.shape != dose_native_pct.shape:
        raise AssertionError(
            f"Shape Bladder {bladder_mask.shape} != shape dosis nativa {dose_native_pct.shape} "
            "-- masks.npz probablemente viene de otro --dicom-dir."
        )
    dvh_bladder = dvh_desde_mascara(dose_native_pct, bladder_mask, args.rx_gy)

    if not modo_3vias:
        # ---------------- Modo 2 vias (original) ----------------
        print("Calculando DVH nativo de cada estructura (desde el mismo dose_pred renormalizado) ...")
        dvh_rectum = dvh_desde_mascara(dose_native_pct, roi_mask, args.rx_gy)
        dvh_hot = dvh_desde_mascara(dose_native_pct, hot, args.rx_gy)
        dvh_cold = dvh_desde_mascara(dose_native_pct, cold, args.rx_gy)
        for nombre, (d, v) in [("Rectum", dvh_rectum), (f"{roi_name}_hot", dvh_hot),
                               (f"{roi_name}_cold", dvh_cold), ("Bladder", dvh_bladder)]:
            print(f"  {nombre:15s} dosis_max_muestreada={d[-1]:.1f}Gy  V(0)={v[0]:.1f}%  V(max)={v[-1]:.2f}%")

        if args.push_pct:
            print(f"\nEmpujando {args.push_pct:g}% el eje dosis de Rectum/hot/cold (Bladder/PTV sin cambios) ...")
            dvh_rectum = empujar_dosis(dvh_rectum, args.push_pct)
            dvh_hot = empujar_dosis(dvh_hot, args.push_pct)
            dvh_cold = empujar_dosis(dvh_cold, args.push_pct)

        oar_control = {"Rectum": dvh_rectum, "Bladder": dvh_bladder}
        oar_test = {f"{roi_name}_hot": dvh_hot, f"{roi_name}_cold": dvh_cold, "Bladder": dvh_bladder}
        prio_control = prio_test = args.line_priority

    else:
        # ---------------- Modo 3 vias (overlap separado) ----------------
        if args.overlap_from_ptv_intersection:
            if args.ptv_id not in disponibles:
                raise ValueError(f"'{args.ptv_id}' no esta en el RS. Disponibles: {disponibles}")
            ptv_mask = rtstruct.get_roi_mask_by_name(args.ptv_id)
            overlap = roi_mask & ptv_mask
            print(f"  Overlap = Rectum & {args.ptv_id} (interseccion real): {int(overlap.sum())}vox")
        else:
            if args.overlap_roi_name not in disponibles:
                raise ValueError(f"'{args.overlap_roi_name}' no esta en el RS. Disponibles: {disponibles}")
            overlap_roi = rtstruct.get_roi_mask_by_name(args.overlap_roi_name)
            if overlap_roi.shape != dose_native_pct.shape:
                raise AssertionError(f"Shape {args.overlap_roi_name} {overlap_roi.shape} != dosis nativa "
                                     f"{dose_native_pct.shape}")
            overlap = overlap_roi & roi_mask  # por si el ROI clinico se pasa un poco del Rectum
        hot_no_overlap = hot & ~overlap
        cold_no_overlap = cold & ~overlap
        assert np.array_equal(overlap | hot_no_overlap | cold_no_overlap, roi_mask), (
            "particion 3 vias rota (overlap|hot_no_overlap|cold_no_overlap != Rectum) -- no deberia pasar"
        )
        print(f"  Particion: overlap={int(overlap.sum())}vox  hot_no_overlap={int(hot_no_overlap.sum())}vox  "
              f"cold_no_overlap={int(cold_no_overlap.sum())}vox  (Rectum total={int(roi_mask.sum())}vox)")

        print("Calculando DVH nativo de cada subestructura ...")
        dvh_overlap = dvh_desde_mascara(dose_native_pct, overlap, args.rx_gy)
        dvh_hot_no = dvh_desde_mascara(dose_native_pct, hot_no_overlap, args.rx_gy)
        dvh_cold_no = dvh_desde_mascara(dose_native_pct, cold_no_overlap, args.rx_gy)
        for nombre, (d, v) in [(args.overlap_roi_name, dvh_overlap), ("hot_no_overlap", dvh_hot_no),
                               ("cold_no_overlap", dvh_cold_no), ("Bladder", dvh_bladder)]:
            print(f"  {nombre:18s} dosis_max_muestreada={d[-1]:.1f}Gy  V(0)={v[0]:.1f}%  V(max)={v[-1]:.2f}%")

        if args.overlap_push_pct:
            print(f"Empujando overlap {args.overlap_push_pct:g}% (poco usual -- confirmado explicitamente) ...")
            dvh_overlap = empujar_dosis(dvh_overlap, args.overlap_push_pct)
        if args.push_pct:
            print(f"Empujando {args.push_pct:g}% hot_no_overlap/cold_no_overlap (overlap y Bladder sin cambios) ...")
            dvh_hot_no = empujar_dosis(dvh_hot_no, args.push_pct)
            dvh_cold_no = empujar_dosis(dvh_cold_no, args.push_pct)

        # --- Test: 3 lineas, overlap a prioridad baja ---
        oar_test = {
            args.overlap_roi_name: dvh_overlap,
            "hot_no_overlap": dvh_hot_no,
            "cold_no_overlap": dvh_cold_no,
            "Bladder": dvh_bladder,
        }
        prio_test = {
            args.overlap_roi_name: args.overlap_priority,
            "hot_no_overlap": args.line_priority,
            "cold_no_overlap": args.line_priority,
            "Bladder": args.line_priority,
        }

        # --- Control: Rectum completo, agregado de los MISMOS targets por-voxel
        # que arma el test (overlap sin empujar + resto empujado), en una sola
        # linea -- asi la unica diferencia real es granularidad/prioridad, no la
        # dosis pedida por voxel. dose_ajustada es una copia del dose_pred nativo
        # con el mismo factor de empuje aplicado voxel a voxel que en el test.
        dose_ajustada = dose_native_pct.copy()
        factor_resto = 1.0 - args.push_pct / 100.0
        factor_overlap = 1.0 - args.overlap_push_pct / 100.0
        dose_ajustada[hot_no_overlap | cold_no_overlap] *= factor_resto
        dose_ajustada[overlap] *= factor_overlap
        dvh_rectum_control = dvh_desde_mascara(dose_ajustada, roi_mask, args.rx_gy)
        print(f"  Rectum (control, agregado) dosis_max_empujada={dvh_rectum_control[0][-1]:.1f}Gy")

        oar_control = {"Rectum": dvh_rectum_control, "Bladder": dvh_bladder}
        prio_control = args.line_priority

    # --- Guardar ---
    tree_control = build_template(oar_control, rx_gy=args.rx_gy, ptv_id=args.ptv_id,
                                  line_priority=prio_control, template_id=f"control{sufijo}_{anonid}")
    path_control = output_dir / f"ObjectiveTemplate_control{sufijo}_{anonid}.xml"
    print(f"\nGuardando control: {path_control}")
    guardar_xml(tree_control, path_control)

    tree_test = build_template(oar_test, rx_gy=args.rx_gy, ptv_id=args.ptv_id,
                               line_priority=prio_test, template_id=f"test{sufijo}_{anonid}")
    path_test = output_dir / f"ObjectiveTemplate_test{sufijo}_{anonid}.xml"
    print(f"\nGuardando test: {path_test}")
    guardar_xml(tree_test, path_test)

    print(f"\nListo. Control: {path_control}\nTest:    {path_test}")


if __name__ == "__main__":
    main()
