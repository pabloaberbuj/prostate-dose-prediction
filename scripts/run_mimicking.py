"""Corre el mimicking (ver src/planning/mimicking.py) sobre 1 paciente, y
registra tiempos + historia de loss (por paso) en un log CSV acumulativo
para comparar entre corridas con distinto num_steps y distintos pesos de
regularización.

Uso:
    python scripts/run_mimicking.py <carpeta_paciente_dicom> <pred_npz> \
        [--rx-gy 78] [--n-fracciones 39] [--num-steps 5] [--deliv-weight 0.0]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.planning.mimicking import parse_struct_weights, run_pure_mimicking

TIMING_LOG_PATH = Path("commissioning/sandbox_results/mimicking_timing_log.csv")
LOG_FIELDS = [
    "run_id", "patient", "deliv_weight", "dvh_weight", "closed_start", "max_iter", "max_eval", "history_size",
    "num_steps_total", "step", "mse", "deliv", "dvh", "total",
    "step_time_s", "cumulative_time_s", "setup_time_s",
    "ptv_d98_actual_gy", "ptv_d98_target_gy",
    "rectum_v70_actual_pct", "rectum_v70_target_pct",
    "bladder_v50_actual_pct", "bladder_v50_target_pct",
]


def append_timing_log_row(run_id, patient_name, args, step_idx, h, setup_time_s):
    """Un renglon por paso, escrito en el momento (no al final) -- para no
    perder toda la corrida si el proceso se corta antes de terminar (clave
    en corridas largas de horas con `--max-wall-time-s`)."""
    TIMING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not TIMING_LOG_PATH.exists()
    with open(TIMING_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "run_id": run_id, "patient": patient_name, "deliv_weight": args.deliv_weight,
            "dvh_weight": args.dvh_weight,
            "closed_start": args.closed_start,
            "max_iter": args.max_iter, "max_eval": args.max_eval, "history_size": args.history_size,
            "num_steps_total": args.num_steps, "step": step_idx,
            "mse": h["mse"], "deliv": h["deliv"], "dvh": h["dvh"], "total": h["total"],
            "step_time_s": h["step_time_s"], "cumulative_time_s": h["cumulative_time_s"],
            "setup_time_s": setup_time_s,
            "ptv_d98_actual_gy": h["ptv_d98_actual_gy"], "ptv_d98_target_gy": h["ptv_d98_target_gy"],
            "rectum_v70_actual_pct": h["rectum_v70_actual_pct"], "rectum_v70_target_pct": h["rectum_v70_target_pct"],
            "bladder_v50_actual_pct": h["bladder_v50_actual_pct"], "bladder_v50_target_pct": h["bladder_v50_target_pct"],
        })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("patient_dir")
    p.add_argument("pred_npz")
    p.add_argument("--rx-gy", type=float, default=78.0)
    p.add_argument("--n-fracciones", type=int, default=39)
    p.add_argument("--num-steps", type=int, default=5)
    p.add_argument("--max-iter", type=int, default=20)
    p.add_argument("--max-eval", type=int, default=25)
    p.add_argument("--history-size", type=int, default=20)
    p.add_argument("--deliv-weight", type=float, default=0.0,
                    help="Peso de la regularizacion de entregabilidad (0.0 = mimicking puro)")
    p.add_argument("--dvh-weight", type=float, default=0.0,
                    help="Peso de los terminos DVH (D98 PTV, V70 Recto, V50 Vejiga contra el "
                         "DVH del target de U-Net). 0.0 = desactivado. Ver mimicking.dvh_loss")
    p.add_argument("--closed-start", action="store_true",
                    help="Arranca con MLC cerrado/jaws abiertos en vez de la apertura real "
                         "(simula el caso real sin RP del paciente) — usar con --deliv-weight 0.0")
    p.add_argument("--ptv-conform-start", action="store_true",
                    help="Arranca conformando MLC/jaws a la proyeccion BEV real del PTV "
                         "(alternativa a --closed-start, ver ptv_conforming_init.py) — "
                         "usar con --deliv-weight 0.0")
    p.add_argument("--leaf-margin-mm", type=float, default=2.0,
                    help="Margen de MLC sobre el PTV para --ptv-conform-start")
    p.add_argument("--jaw-margin-mm", type=float, default=2.0,
                    help="Margen de jaws sobre el PTV para --ptv-conform-start")
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
                         "volumen. Ej: 'PTV_High=100,Rectum=30,Bladder=30,BODY=10'. "
                         "Los no listados quedan en su default (ver STRUCT_WEIGHTS_DEFAULT)")
    p.add_argument("--max-wall-time-s", type=float, default=None,
                    help="Corta la corrida (al terminar el paso en curso) cuando el tiempo "
                         "acumulado de pasos supera este valor -- para corridas largas con "
                         "presupuesto de tiempo fijo en vez de num-steps fijo")
    args = p.parse_args()

    patient_name = Path(args.patient_dir).name
    run_id = f"{patient_name}_{int(time.time())}"

    def on_step(step_idx, h, setup_time_s, optim_arcs):
        append_timing_log_row(run_id, patient_name, args, step_idx + 1, h, setup_time_s)

    optim_arcs, dose_pred, history, setup_time_s = run_pure_mimicking(
        args.patient_dir, args.pred_npz, args.rx_gy, args.n_fracciones,
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
        lbfgs_kwargs=dict(
            max_iter=args.max_iter, max_eval=args.max_eval, history_size=args.history_size,
        ),
    )
    print("Historia de loss:", history)

    out_dir = Path("commissioning/sandbox_results") / (patient_name + "_mimicking")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "dose_pred_final.npy", dose_pred.cpu().numpy())
    print(f"Guardado en {out_dir}")
    print(f"Tiempos y loss registrados incrementalmente en {TIMING_LOG_PATH} (run_id={run_id})")


if __name__ == "__main__":
    main()
