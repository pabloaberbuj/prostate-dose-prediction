"""
Monitoreo manual de un entrenamiento en curso, cada N epochs — lee el estado
embebido en `last.ckpt` (que PyTorch Lightning reescribe al final de cada epoch
via save_last=True) en vez de parsear el log de texto (poco confiable: la barra
de progreso 'rich' no imprime una linea limpia por epoch cuando no hay mejora).

Imprime UNA linea cada vez que el numero de epoch cruza un multiplo de
`--every`, con: epoch actual, val/mae de ESE epoch (ModelCheckpoint.current_score),
mejor val/mae hasta ahora (EarlyStopping.best_score) y wait_count (cuantos
epochs sin mejora — contra el patience=30 configurado).

No decide nada por si solo (no mata el proceso) — es informativo, para que
la decision de "achicar lambda mas agresivo" se tome con datos reales en vez
de esperar a que corra el early stopping automatico (que puede tardar horas).

Uso (bloqueante, pensado para correr bajo el tool Monitor):
    .venv/Scripts/python.exe scripts/watch_lambda_run.py \
        --ckpt-dir checkpoints/exp_hipo_003b_dvhloss_lambda10 \
        --every 5 \
        --max-epochs 100
"""

import argparse
import time
from pathlib import Path

import torch


def cargar_estado(last_ckpt: Path):
    _orig = torch.load
    def _load(*a, **kw):
        kw['weights_only'] = False
        return _orig(*a, **kw)
    torch.load = _load
    try:
        ckpt = torch.load(str(last_ckpt), map_location='cpu')
    finally:
        torch.load = _orig

    epoch = ckpt.get('epoch')
    early = next((v for k, v in ckpt.get('callbacks', {}).items() if 'EarlyStopping' in k), {})
    mchk = next((v for k, v in ckpt.get('callbacks', {}).items() if 'ModelCheckpoint' in k), {})
    return {
        'epoch': epoch,
        'current_score': float(mchk.get('current_score')) if mchk.get('current_score') is not None else None,
        'best_score': float(early.get('best_score')) if early.get('best_score') is not None else None,
        'wait_count': early.get('wait_count'),
        'patience': early.get('patience'),
        'best_model_path': mchk.get('best_model_path'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--every", type=int, default=5, help="emitir una linea cada N epochs")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()

    last_ckpt = Path(args.ckpt_dir) / "last.ckpt"
    ultimo_emitido = -1

    print(f"Esperando {last_ckpt} ...", flush=True)
    while not last_ckpt.exists():
        time.sleep(args.poll_seconds)

    while True:
        try:
            estado = cargar_estado(last_ckpt)
        except Exception as e:
            time.sleep(args.poll_seconds)
            continue

        epoch = estado['epoch']
        if epoch is not None and epoch >= 0 and epoch % args.every == 0 and epoch != ultimo_emitido:
            ultimo_emitido = epoch
            print(f"EPOCH {epoch}: val/mae_epoch={estado['current_score']:.4f}  "
                  f"best_val_mae={estado['best_score']:.4f}  "
                  f"wait_count={estado['wait_count']}/{estado['patience']}  "
                  f"best_ckpt={Path(estado['best_model_path']).name if estado['best_model_path'] else None}",
                  flush=True)

        if estado['wait_count'] is not None and estado['patience'] is not None \
                and estado['wait_count'] >= estado['patience']:
            print(f"EARLY_STOPPING_TRIGGERED epoch={epoch} best_val_mae={estado['best_score']:.4f}", flush=True)
            break
        if epoch is not None and epoch >= args.max_epochs - 1:
            print(f"MAX_EPOCHS_REACHED epoch={epoch}", flush=True)
            break

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
