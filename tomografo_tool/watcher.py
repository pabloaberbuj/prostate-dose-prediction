"""Monitor de la carpeta del tomografo. Detecta cuando un paciente (CT + RS DICOM)
termino de llegar y lo deja LISTO para procesar -- no lo dispara solo.

Deteccion en DOS capas (ver PROMPT_herramienta_tomografo_app.md — es la parte
delicada del diseno):

  Capa 1 - Inactividad: watchdog.Observer dispara el chequeo cuando pasan
  `inactivity_timeout_sec` segundos sin archivos nuevos en una carpeta candidata.

  Capa 2 - Verificacion de estabilidad + presencia de RS: recien con inactividad NO
  alcanza para asumir que el paciente esta completo. Se verifica ademas que (a)
  todos los archivos tengan tamano estable, y (b) exista al menos un RTSTRUCT
  (identificado por Modality=='RTSTRUCT' leyendo el header DICOM, no por nombre de
  archivo). Si no hay RS todavia, se espera un timeout extendido especifico
  (`rs_extra_timeout_sec`) antes de marcar el caso como incompleto — el RS suele
  llegar despues del CT y pesa mas.

Todo el trabajo de Capa 2 corre en un loop de polling propio (`_poll_loop`), NO
dentro del callback de watchdog (que debe ser rapido y no bloquear).

Cola de pendientes, con seleccion manual: cuando llegan varios pacientes en
paralelo (varias subcarpetas listas casi al mismo tiempo, p.ej. una tanda de
exportacion), NO se procesan solos ni en orden automatico -- quedan listados en
`listar_pendientes()` y es la UI (fisico) quien elige, con `procesar_seleccionado()`,
cual de todos los que llegaron es el que corresponde procesar ahora."""

import logging
import threading
import time
from pathlib import Path

import pydicom
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

log = logging.getLogger("tomografo_tool.watcher")


def _es_rtstruct(path: Path) -> bool:
    """Modality=='RTSTRUCT' leyendo solo el header. Devuelve False (no True) ante
    cualquier error de lectura -- un archivo a mitad de escritura no debe contar
    como RS todavia, pero tampoco debe tumbar el chequeo."""
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        return str(getattr(ds, "Modality", "")).upper() == "RTSTRUCT"
    except Exception:
        return False


def _listar_archivos(carpeta: Path):
    try:
        return [p for p in carpeta.rglob("*") if p.is_file()]
    except FileNotFoundError:
        return []


class _CarpetaCandidata:
    """Estado de seguimiento de UNA carpeta-en-curso (un paciente potencial). Varias
    instancias pueden convivir para no mezclar pacientes que llegan en paralelo a
    subcarpetas distintas."""

    def __init__(self, path: Path):
        self.path = path
        self.last_event = time.monotonic()
        self.ready = False        # paso Capa 1 + Capa 2, esperando seleccion manual
        self.processing = False
        self.done = False
        self.rs_deadline = None   # se fija la primera vez que hay inactividad sin RS
        self.patient_id_hint = None
        self.listo_desde = None   # time.time() epoch, para mostrar antiguedad en la UI
        self.n_archivos = 0

    def marcar_evento(self):
        if self.done or self.ready:
            # Reaparecen archivos en una carpeta ya lista/procesada -> es un
            # paciente nuevo reusando la misma carpeta. Resetear el estado.
            self.done = False
            self.ready = False
            self.rs_deadline = None
            self.patient_id_hint = None
            self.listo_desde = None
        self.last_event = time.monotonic()


class PatientFolderMonitor:
    """Uso: crear con la carpeta raiz a monitorear + callbacks, llamar start().
    Internamente corre un Observer de watchdog (Capa 1, solo resetea timers) y un
    thread de polling propio (Capa 2, la logica de is_ready). No procesa nada por su
    cuenta: `listar_pendientes()` / `procesar_seleccionado()` son la API que usa
    app.py para mostrar la cola y disparar el procesamiento elegido a mano."""

    def __init__(self, watch_root: Path, config: dict, on_result, on_incompleto=None):
        self.watch_root = Path(watch_root)
        self.inactivity_timeout = config.get("inactivity_timeout_sec", 10)
        self.rs_extra_timeout = config.get("rs_extra_timeout_sec", 60)
        self.stability_wait = config.get("file_stability_check_sec", 1)
        self.poll_interval = config.get("poll_interval_sec", 2)
        self.on_result = on_result
        self.on_incompleto = on_incompleto or (lambda carpeta: None)

        self._candidatas = {}  # str(path) -> _CarpetaCandidata
        self._lock = threading.Lock()
        self._observer = None
        self._poll_thread = None
        self._stop = threading.Event()

    # ---- API publica ------------------------------------------------------------

    def start(self):
        self.watch_root.mkdir(parents=True, exist_ok=True)
        handler = _Handler(self)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.watch_root), recursive=True)
        self._observer.daemon = True
        self._observer.start()

        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        log.info("Monitoreando %s (inactividad=%ss, rs_extra=%ss)",
                  self.watch_root, self.inactivity_timeout, self.rs_extra_timeout)

    def stop(self):
        self._stop.set()
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)

    def listar_pendientes(self) -> list:
        """Pacientes listos (Capa1+Capa2 OK) que todavia no fueron procesados,
        ordenados por antiguedad (el que llego primero, primero) -- pero es solo
        sugerencia de orden, la UI puede elegir cualquiera."""
        with self._lock:
            candidatas = list(self._candidatas.values())
        pendientes = [
            {
                "carpeta": str(c.path),
                "patient_id_hint": c.patient_id_hint or c.path.name,
                "listo_desde": c.listo_desde,
                "n_archivos": c.n_archivos,
            }
            for c in candidatas if c.ready and not c.processing and not c.done
        ]
        pendientes.sort(key=lambda p: p["listo_desde"] or 0)
        return pendientes

    def procesar_seleccionado(self, carpeta_str: str) -> dict:
        """Dispara pipeline.procesar_paciente() para la carpeta elegida (sincrono,
        como /abrir_carpeta). Lanza ValueError si esa carpeta no esta disponible
        (no existe, ya se proceso, o ya la esta procesando otro pedido)."""
        with self._lock:
            cand = self._candidatas.get(carpeta_str)
            if cand is None or not cand.ready or cand.processing or cand.done:
                raise ValueError(f"Carpeta no disponible para procesar: {carpeta_str}")
            cand.processing = True

        import pipeline  # import tardio: evita ciclo al testear watcher solo
        try:
            resultado = pipeline.procesar_paciente(cand.path)
        finally:
            cand.done = True
            cand.processing = False
        return resultado

    # ---- Capa 1: resetear timers desde el callback de watchdog -----------------

    def _carpeta_candidata_de(self, path_evento: Path) -> Path:
        """Que carpeta se considera 'el paciente': la subcarpeta inmediata del
        watch_root si el archivo esta en una subcarpeta, o el watch_root mismo si
        el archivo llego directo ahi (caso 'un paciente a la vez sin subcarpetas')."""
        try:
            rel = path_evento.relative_to(self.watch_root)
        except ValueError:
            return self.watch_root
        return self.watch_root if len(rel.parts) <= 1 else self.watch_root / rel.parts[0]

    def _on_fs_event(self, path_str: str):
        path_evento = Path(path_str)
        carpeta = self._carpeta_candidata_de(path_evento)
        with self._lock:
            cand = self._candidatas.get(str(carpeta))
            if cand is None:
                cand = _CarpetaCandidata(carpeta)
                self._candidatas[str(carpeta)] = cand
            cand.marcar_evento()

    # ---- Capa 2: loop de polling propio -----------------------------------------

    def _poll_loop(self):
        while not self._stop.is_set():
            time.sleep(self.poll_interval)
            with self._lock:
                candidatas = list(self._candidatas.values())
            for cand in candidatas:
                if cand.ready or cand.processing or cand.done:
                    continue
                try:
                    self._evaluar(cand)
                except Exception:
                    log.exception("Error evaluando candidata %s", cand.path)

    def _evaluar(self, cand: _CarpetaCandidata):
        inactivo_desde = time.monotonic() - cand.last_event
        if inactivo_desde < self.inactivity_timeout:
            return  # Capa 1 todavia no disparo el chequeo para esta carpeta

        archivos = _listar_archivos(cand.path)
        if not archivos:
            return

        if not self._archivos_estables(archivos):
            return  # tamanos todavia cambiando, seguir esperando (no abortar)

        tiene_rs = any(_es_rtstruct(f) for f in archivos)
        if not tiene_rs:
            if cand.rs_deadline is None:
                cand.rs_deadline = time.monotonic() + self.rs_extra_timeout
                log.info("Inactividad+estabilidad en %s pero sin RS todavia; "
                          "esperando hasta %ss extra", cand.path, self.rs_extra_timeout)
                return
            if time.monotonic() < cand.rs_deadline:
                return  # todavia dentro de la ventana extendida para el RS
            # Timeout extendido vencido y sigue sin RS -> incompleto, no procesar
            log.warning("Carpeta %s marcada INCOMPLETA: sin RTSTRUCT tras timeout "
                        "extendido de %ss", cand.path, self.rs_extra_timeout)
            cand.done = True
            self.on_incompleto(cand.path)
            return

        # Inactividad + estabilidad + RS presente -> queda listo, esperando que lo
        # elijan a mano desde la UI (NO se procesa solo).
        cand.ready = True
        cand.listo_desde = time.time()
        cand.n_archivos = len(archivos)
        try:
            import pipeline
            cand.patient_id_hint = pipeline.identificar_paciente(cand.path)
        except Exception:
            cand.patient_id_hint = cand.path.name
        log.info("Paciente listo para procesar (esperando seleccion manual): %s (%s)",
                  cand.path, cand.patient_id_hint)

    def _archivos_estables(self, archivos) -> bool:
        tamanos_1 = {f: f.stat().st_size for f in archivos if f.exists()}
        time.sleep(self.stability_wait)
        for f, size_antes in tamanos_1.items():
            if not f.exists() or f.stat().st_size != size_antes:
                return False
        return True


class _Handler(FileSystemEventHandler):
    def __init__(self, monitor: PatientFolderMonitor):
        self.monitor = monitor

    def on_created(self, event):
        if not event.is_directory:
            self.monitor._on_fs_event(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self.monitor._on_fs_event(event.src_path)
