"""Corre mimicking (src/planning/mimicking.py) y escribe el RP DICOM
optimizado resultante (src/planning/write_rtplan.py) — PIPELINE_KBP_PDRT.md
paso 5, listo para importar en Eclipse y recalcular con AAA.

Uso:
    python scripts/run_write_rtplan.py <carpeta_paciente_dicom> <pred_npz> \
        [--rx-gy 78] [--n-fracciones 39] [--num-steps 1] [--deliv-weight 10.0]

IMPORTANTE: --deliv-weight NO puede ser 0.0 para un RP importable en Eclipse
-- confirmado con una prueba real: un RP escrito con deliv_weight=0.0 disparó
errores de "Leaf speed is too high" en Eclipse y bajó mucho la dosis en PTV
al recalcular con AAA (ver src/planning/mimicking.py, deliverability_loss).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.planning.build_beams import find_dicom_files, patch_rp_missing_ssd
from src.planning.mimicking import parse_struct_weights, run_pure_mimicking
from src.planning.write_rtplan import write_optimized_rtplan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("patient_dir")
    p.add_argument("pred_npz")
    p.add_argument("--rx-gy", type=float, default=78.0)
    p.add_argument("--n-fracciones", type=int, default=39)
    p.add_argument("--num-steps", type=int, default=1)
    p.add_argument("--max-iter", type=int, default=20)
    p.add_argument("--max-eval", type=int, default=25)
    p.add_argument("--history-size", type=int, default=20)
    p.add_argument("--deliv-weight", type=float, default=10.0,
                    help="0.0 produce un RP no deliverable (ver docstring del modulo)")
    p.add_argument("--dvh-weight", type=float, default=0.0,
                    help="Peso de los terminos DVH (D98 PTV, V70 Recto, V50 Vejiga contra el "
                         "DVH del target de U-Net). 0.0 = desactivado. Ver mimicking.dvh_loss")
    p.add_argument("--closed-start", action="store_true")
    p.add_argument("--ptv-conform-start", action="store_true",
                    help="Arranca conformando MLC/jaws al PTV en vez de la apertura real "
                         "(ver ptv_conforming_init.py)")
    p.add_argument("--leaf-margin-mm", type=float, default=2.0)
    p.add_argument("--jaw-margin-mm", type=float, default=2.0)
    p.add_argument("--leaf-max-step-mm", type=float, default=10.0,
                    help="Tope de cambio de una lamina entre CPs consecutivos tras el "
                         "suavizado del init conformado al PTV")
    p.add_argument("--optimize-jaws", action="store_true",
                    help="Optimiza la mordaza Y en vez de dejarla fija (default: fija, "
                         "el equipo real no tiene jaw tracking)")
    p.add_argument("--optimizer", default="lbfgs", choices=["lbfgs", "adam"],
                    help="lbfgs = paso global (mueve ~100x mas las MU que las laminas por "
                         "el desbalance de gradientes); adam = paso normalizado por parametro")
    p.add_argument("--adam-lr", type=float, default=0.01,
                    help="Learning rate para --optimizer adam (en espacio de parametro crudo)")
    p.add_argument("--project-velocity", action="store_true",
                    help="Proyecta las laminas al conjunto factible tras cada paso (restriccion "
                         "dura de velocidad). Recomendado con --optimizer adam, donde la "
                         "penalizacion pierde fuerza por la normalizacion por parametro")
    p.add_argument("--project-aperture", action="store_true",
                    help="Impide abrir laminas mas alla de la silueta del PTV (+margen). "
                         "Cerrar hacia adentro queda libre. Requiere --ptv-conform-start")
    p.add_argument("--aperture-extra-mm", type=float, default=3.0,
                    help="Cuanto puede abrir mas alla de la silueta conformada del PTV")
    p.add_argument("--project-arc-range", action="store_true",
                    help="Acota el recorrido de cada lamina en todo el arco a 150mm "
                         "(restriccion de carro del MLC)")
    p.add_argument("--max-leaf-travel-cm", type=float, default=None,
                    help="Corta la corrida cuando el recorrido total de laminas llega a este "
                         "valor (clinico ~3100). Apunta a complejidad tipo clinica en vez de mse minimo")
    p.add_argument("--beam-chunk-size", type=int, default=1,
                    help="Control points procesados juntos por el motor (c/u es un 'beam' "
                         "para PyDoseRT) -- subir si hay memoria de GPU libre")
    p.add_argument("--struct-weights", default=None,
                    help="Share del presupuesto del MSE por estructura, normalizado por "
                         "volumen. Ej: 'PTV_High=100,Rectum=30,Bladder=30,BODY=10'")
    p.add_argument("--max-wall-time-s", type=float, default=None,
                    help="Corta la corrida al terminar el paso en curso una vez superado "
                         "este tiempo acumulado, en vez de num-steps fijo")
    p.add_argument("--snapshot-every", type=int, default=5,
                    help="Escribe un RP intermedio cada N pasos (RP_snapshot.dcm), para poder "
                         "mirar el avance sin cortar la corrida. 0 = desactivado")
    args = p.parse_args()

    patient_dir = Path(args.patient_dir)
    patient_name = patient_dir.name
    out_dir = Path("commissioning/sandbox_results") / (patient_name + "_mimicking")

    # El RP template (con el parche de SSD) se resuelve ANTES de arrancar, asi
    # los snapshots intermedios pueden escribirse sin rehacer este trabajo.
    _, _, rp_template, _ = find_dicom_files(patient_dir)
    rp_template = patch_rp_missing_ssd(rp_template, out_dir / f"_RP_ssd_patched_{patient_name}.dcm")

    def on_step(step_idx, h, setup_time_s, optim_arcs):
        if not args.snapshot_every or (step_idx + 1) % args.snapshot_every:
            return
        snap = out_dir / "RP_snapshot.dcm"
        try:
            _, uid = write_optimized_rtplan(rp_template, optim_arcs, snap)
            print(f"[snapshot] paso {step_idx + 1}: RP intermedio en {snap}")
            print(f"[snapshot]   UID del plan = {uid}")
        except Exception as e:  # un snapshot fallido no debe matar la corrida
            print(f"[snapshot] paso {step_idx + 1}: FALLO al escribir el RP intermedio: {e}")

    optim_arcs, dose_pred, history, setup_time_s = run_pure_mimicking(
        str(patient_dir), args.pred_npz, args.rx_gy, args.n_fracciones,
        num_steps=args.num_steps,
        deliv_weight=args.deliv_weight,
        dvh_weight=args.dvh_weight,
        closed_start=args.closed_start,
        ptv_conform_start=args.ptv_conform_start,
        leaf_margin_mm=args.leaf_margin_mm,
        jaw_margin_mm=args.jaw_margin_mm,
        leaf_max_step_mm=args.leaf_max_step_mm,
        optimize_jaws=args.optimize_jaws,
        struct_weights=parse_struct_weights(args.struct_weights),
        optimizer_name=args.optimizer,
        adam_lr=args.adam_lr,
        project_velocity=args.project_velocity,
        project_aperture=args.project_aperture,
        aperture_extra_mm=args.aperture_extra_mm,
        project_arc_range=args.project_arc_range,
        max_leaf_travel_cm=args.max_leaf_travel_cm,
        beam_chunk_size=args.beam_chunk_size,
        max_wall_time_s=args.max_wall_time_s,
        on_step=on_step,
        lbfgs_kwargs=dict(max_iter=args.max_iter, max_eval=args.max_eval, history_size=args.history_size),
    )
    print("Historia de loss:", history)

    out_rp_path = out_dir / "RP_optimizado.dcm"
    out_rp_path, nuevo_plan_uid = write_optimized_rtplan(rp_template, optim_arcs, out_rp_path)
    print(f"RP optimizado guardado en {out_rp_path}")
    print(f"Nuevo SOPInstanceUID del plan (usar en write_rd_dicom.py --referenced-plan-uid): {nuevo_plan_uid}")


if __name__ == "__main__":
    main()
