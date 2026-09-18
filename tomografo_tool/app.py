"""App Flask de la herramienta de tomografo. Capa de presentacion sobre el pipeline
ya construido y validado del Proyecto 1 (Tareas 6-7) -- ver pipeline.py.

Rutas:
  GET  /                     UI (semaforo + metricas + cola de pendientes)
  GET  /ultimo                JSON con el ultimo resultado procesado
  GET  /cola                  JSON con los pacientes listos, esperando seleccion manual
  GET  /estado_procesamiento  JSON con la fase actual (extrayendo features / infiriendo)
  POST /procesar_cola         dispara el procesamiento del paciente elegido (async)
  POST /abrir_carpeta         dispara el procesamiento de una carpeta manual (async)

El procesamiento en si (extraer_features + predecir_paciente, scripts/*.py del
Proyecto 1) corre en un thread aparte -- las rutas POST devuelven de inmediato un
ack y el frontend hace polling de /estado_procesamiento mientras corre, y de
/ultimo cuando termina. Esto es lo que permite mostrar en que fase esta (en vez de
un "Procesando..." fijo varios segundos sin decir nada mas).

Uso:
  python app.py
"""

import csv
import logging
import logging.handlers
import threading
from datetime import datetime
from pathlib import Path

import yaml
from flask import Flask, jsonify, render_template, request

import pipeline
from watcher import PatientFolderMonitor

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Modo desatendido (Task Scheduler, sin terminal visible): todo tiene que quedar en
# archivo, no solo en consola. RotatingFileHandler para no dejar crecer el log sin
# limite en una operacion de semanas/meses.
_file_handler = logging.handlers.RotatingFileHandler(
    LOGS_DIR / "tomografo_tool.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                     handlers=[logging.StreamHandler(), _file_handler], force=True)
# force=True: sin esto, si CUALQUIER libreria importada antes (numpy/sklearn/etc.)
# ya le agrego un handler al root logger, basicConfig() no hace nada (silencioso,
# sin error) y el RotatingFileHandler de arriba nunca queda conectado -- confirmado
# en la primera corrida real (tomografo_tool.log quedo vacio pese a que el watcher
# se conecto bien).
log = logging.getLogger("tomografo_tool.app")

_EXCLUIDOS_LOG_PATH = LOGS_DIR / "excluidos.log"
_EXCLUIDOS_LOG_HEADER = ["timestamp", "patient_id", "carpeta", "study_description", "motivo"]
_excluidos_lock = threading.Lock()

app = Flask(__name__)

# Valor de fabrica de config.yaml -- si sigue asi, todavia no se configuro la carpeta
# real del tomografo. Ver config.yaml para editarlo.
_WATCH_FOLDER_PLACEHOLDER = "C:/ruta/a/carpeta/monitoreada"

_lock = threading.Lock()
_ultimo_resultado = {"estado": "esperando"}
_monitor = None  # PatientFolderMonitor activo, o None si no hay watch_folder configurado
_estado_procesamiento = {"activo": False}


def _set_ultimo(resultado: dict):
    global _ultimo_resultado
    with _lock:
        _ultimo_resultado = resultado
    log.info("Ultimo resultado actualizado: paciente=%s estado=%s",
              resultado.get("patient_id"), resultado.get("estado"))


def _set_fase(fase: str, carpeta: str = None):
    with _lock:
        _estado_procesamiento["activo"] = True
        _estado_procesamiento["fase"] = fase
        if carpeta is not None:
            _estado_procesamiento["carpeta"] = carpeta
    log.info("Fase de procesamiento: %s (%s)", fase, carpeta)


def _limpiar_fase():
    with _lock:
        _estado_procesamiento.clear()
        _estado_procesamiento["activo"] = False


def _cargar_config():
    with open(BASE_DIR / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _registrar_excluido(carpeta, study_description: str, patient_id: str, motivo: str):
    """Log liviano (CSV) de carpetas descartadas por el filtro de localizacion --
    sin features clinicas, solo lo necesario para auditar que el filtro esta bien
    calibrado durante el piloto. csv.writer (no concatenacion de strings) porque
    StudyDescription es texto libre y puede traer comas."""
    nuevo = not _EXCLUIDOS_LOG_PATH.exists()
    with _excluidos_lock:
        with open(_EXCLUIDOS_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if nuevo:
                writer.writerow(_EXCLUIDOS_LOG_HEADER)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                patient_id, str(carpeta), study_description, motivo,
            ])
    log.info("Excluido: %s (%s) -- '%s' -- motivo=%s", carpeta, patient_id, study_description, motivo)


@app.route("/")
def index():
    return render_template("index.html", monitoreo_activo=_monitor is not None)


@app.route("/ultimo")
def ultimo():
    with _lock:
        return jsonify(_ultimo_resultado)


@app.route("/estado_procesamiento")
def estado_procesamiento():
    with _lock:
        return jsonify(dict(_estado_procesamiento))


@app.route("/cola")
def cola():
    if _monitor is None:
        return jsonify([])
    return jsonify(_monitor.listar_pendientes())


@app.route("/procesar_cola", methods=["POST"])
def procesar_cola():
    if _monitor is None:
        return jsonify({"estado": "error", "error": "El monitoreo automatico no esta activo."}), 400

    data = request.get_json(silent=True) or {}
    carpeta = data.get("carpeta")
    if not carpeta:
        return jsonify({"estado": "error", "error": "Falta el parametro 'carpeta'"}), 400

    with _lock:
        if _estado_procesamiento.get("activo"):
            return jsonify({"estado": "error",
                             "error": "Ya se esta procesando otro paciente, esperar a que termine."}), 409
        _estado_procesamiento.clear()
        _estado_procesamiento.update({"activo": True, "fase": "iniciando", "carpeta": carpeta})

    def _worker():
        try:
            resultado = _monitor.procesar_seleccionado(carpeta, on_fase=lambda f: _set_fase(f, carpeta))
            _set_ultimo(resultado)
        except ValueError as e:
            log.warning("No se pudo procesar %s: %s", carpeta, e)
            _set_ultimo({"estado": "error", "carpeta": carpeta, "patient_id": Path(carpeta).name,
                         "timestamp": "", "error": str(e)})
        finally:
            _limpiar_fase()

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"estado": "iniciado", "carpeta": carpeta}), 202


@app.route("/abrir_carpeta", methods=["POST"])
def abrir_carpeta():
    data = request.get_json(silent=True) or {}
    carpeta = data.get("carpeta") or request.form.get("carpeta")
    if not carpeta:
        return jsonify({"estado": "error", "error": "Falta el parametro 'carpeta'"}), 400

    carpeta_path = Path(carpeta)
    if not carpeta_path.is_dir():
        return jsonify({"estado": "error", "error": f"No existe la carpeta: {carpeta}"}), 400

    with _lock:
        if _estado_procesamiento.get("activo"):
            return jsonify({"estado": "error",
                             "error": "Ya se esta procesando otro paciente, esperar a que termine."}), 409
        _estado_procesamiento.clear()
        _estado_procesamiento.update({"activo": True, "fase": "iniciando", "carpeta": str(carpeta_path)})

    def _worker():
        try:
            resultado = pipeline.procesar_paciente(
                carpeta_path, on_fase=lambda f: _set_fase(f, str(carpeta_path)))
            _set_ultimo(resultado)
        finally:
            _limpiar_fase()

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"estado": "iniciado", "carpeta": str(carpeta_path)}), 202


def _iniciar_watcher(config: dict):
    watch_folder = config.get("watch_folder", "")
    if not watch_folder or watch_folder.strip() == _WATCH_FOLDER_PLACEHOLDER:
        log.warning(
            "watch_folder no configurado (sigue con el valor de ejemplo en "
            "config.yaml). NO se inicia el monitoreo automatico -- editar "
            "'watch_folder' en tomografo_tool/config.yaml con la carpeta real de "
            "exportacion del tomografo. La apertura manual (/abrir_carpeta) sigue "
            "funcionando mientras tanto."
        )
        return None

    monitor = PatientFolderMonitor(
        watch_root=Path(watch_folder),
        config=config,
        on_result=_set_ultimo,
        on_incompleto=lambda carpeta: log.warning("Caso incompleto, no procesado: %s", carpeta),
        on_excluido=_registrar_excluido,
    )
    monitor.start()
    return monitor


def _precalentar_modelos_bg():
    """Corre en background al arrancar la app: paga de una sola vez el import en
    frio de sklearn/joblib (medido ~1-3s) para que el primer paciente real del dia
    no lo pague el. No bloqueante -- si falla, solo se loguea."""
    try:
        pipeline.precalentar_modelos()
        log.info("Modelos precalentados.")
    except Exception:
        log.exception("No se pudieron precalentar los modelos (no bloqueante, "
                       "el primer paciente real pagara ese costo).")


if __name__ == "__main__":
    config = _cargar_config()
    _monitor = _iniciar_watcher(config)
    threading.Thread(target=_precalentar_modelos_bg, daemon=True).start()
    app.run(host=config.get("host", "0.0.0.0"), port=config.get("port", 5000))
