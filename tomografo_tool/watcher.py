"""Monitor de la carpeta del tomografo. Detecta cuando un paciente de PROSTATA
(CT + RS DICOM) termino de llegar y lo procesa AUTOMATICAMENTE, sin intervencion
(shadow testing sobre la carpeta compartida del centro (UNC ARIAMEVADB-SVR/va_data$/DICOM) -- recibida por
TODO el centro, todas las localizaciones/tecnicas).

Deteccion en TRES pasos (ver SPEC/plan de shadow testing -- es la parte delicada del
diseno, y la carpeta real la hace mas delicada todavia: recibe estudios de todo el
centro, muchas carpetas viejas son solo resabios "SC" de la importacion al
planificador que nunca van a tener RS):

  Paso 0 - Filtro de localizacion (TRI-STATE, corre lo antes posible): lee
  StudyDescription (0008,1030) del primer CT legible que aparezca. Si matchea
  "prostata" (configurable) y ningun termino de exclusion ("igrt"/"sbrt"/"areas",
  configurable) -> sigue. Si no matchea -> excluido, no se vuelve a evaluar. Si
  TODAVIA no hay ningun CT legible -> indeterminado, se reintenta -- salvo que la
  carpeta ya dejo de recibir archivos (estable) y sigue sin CT, en cuyo caso se
  concluye "sin_ct" (la mayoria de las carpetas SC-only del centro caen aca) en vez
  de reintentar para siempre.

  Paso 1 - Capa 1 (inactividad): dispara el chequeo de Capa 2 cuando pasan
  `inactivity_timeout_sec` segundos sin archivos nuevos en una carpeta candidata que
  ya paso el filtro de localizacion.

  Paso 2 - Capa 2 (estabilidad + presencia de RS): con inactividad NO alcanza para
  asumir que el paciente esta completo. Se verifica que (a) todos los archivos
  tengan tamano estable, y (b) exista al menos un RTSTRUCT (por Modality=='RTSTRUCT'
  del header, no por nombre de archivo). El CT suele llegar minutos antes que el RS
  (lo manda Autocontour por separado) -- mientras se espera el RS, se PRECARGA el CT
  en background (ver `_precargar_ct`) para no pagar ese I/O de nuevo despues, dentro
  de la ventana ajustada de ~3 minutos hasta el import a Eclipse. Si tras un timeout
  extendido especifico (`rs_extra_timeout_sec`) sigue sin RS, se marca "incompleto".
  Una vez que pasan inactividad+estabilidad+RS, hace falta sostenerlo por
  `min_checks_estables_consecutivos` ciclos de poll seguidos (no solo una vez) antes
  de dar por lista la carpeta -- earlier el chequeo de estabilidad interno mide un
  solo segundo, insuficiente por si solo contra un recurso de red.

Todo el trabajo de Capa 2 corre en un loop de polling propio (`_poll_loop`), NO
dentro del callback de watchdog (que debe ser rapido y no bloquear).

Procesamiento AUTOMATICO: apenas una carpeta queda lista, se encola
(`_cola_lista`) y un thread worker dedicado la procesa (secuencial, uno por vez,
evita contencion si llegan 2 pacientes de prostata casi juntos) llamando a
`pipeline.procesar_paciente()`. No hace falta seleccion manual -- las rutas
manuales de Flask (`/procesar_cola`, `/abrir_carpeta`) quedan como fallback para
casos que el filtro excluyo por error o para pruebas.

Poda periodica de `_candidatas`/`_ct_cache`: cada `pruning_check_interval_sec` se
eliminan las entradas ya resueltas (`done`) mas viejas que `pruning_age_days` --
en operacion continua de semanas/meses, sin esto la memoria crece sin limite."""

import logging
import queue
import sys
import threading
import time
import unicodedata
from pathlib import Path

import pydicom
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "data"))

from preprocess import cargar_ct  # noqa: E402

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


def _normalizar_texto(s: str) -> str:
    """Minusculas + sin acentos (unicodedata) -- asi 'PROSTATA'/'prostata'/
    'PRÓSTATA' matchean igual, y 'areas'/'áreas' tambien."""
    s = (s or "").lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def _leer_info_ct(archivos, max_muestras: int = 3) -> dict | None:
    """Busca hasta `max_muestras` archivos con Modality=='CT' entre `archivos` y lee
    StudyDescription+PatientID del primero encontrado. Devuelve None si TODAVIA no
    hay ningun CT legible (indeterminado, no excluido -- ver docstring del modulo).
    Si encuentra mas de un CT y difieren en StudyDescription, agrega
    'advertencia_estudio_mixto': True (riesgo conocido y NO resuelto de carpetas con
    mas de un estudio mezclado -- ver plan de shadow testing)."""
    encontrados = []
    for f in archivos:
        if len(encontrados) >= max_muestras:
            break
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            if str(getattr(ds, "Modality", "")).upper() == "CT":
                encontrados.append({
                    "study_description": str(getattr(ds, "StudyDescription", "") or ""),
                    "patient_id": str(getattr(ds, "PatientID", "") or ""),
                })
        except Exception:
            continue
    if not encontrados:
        return None
    descripciones = {e["study_description"] for e in encontrados}
    info = dict(encontrados[0])
    info["advertencia_estudio_mixto"] = len(descripciones) > 1
    return info


def _clasificar_localizacion(study_description: str, incluir, excluir) -> tuple:
    """(ok: bool, motivo: str). ok=True solo si matchea algun termino de `incluir` Y
    ningun termino de `excluir` (ambos normalizados, ver _normalizar_texto)."""
    d = _normalizar_texto(study_description)
    if not any(_normalizar_texto(k) in d for k in incluir):
        return False, "no_prostata"
    for k in excluir:
        if _normalizar_texto(k) in d:
            return False, f"excluido_{k}"
    return True, "prostata"


class _CarpetaCandidata:
    """Estado de seguimiento de UNA carpeta-en-curso (un paciente potencial). Varias
    instancias pueden convivir para no mezclar pacientes que llegan en paralelo a
    subcarpetas distintas."""

    def __init__(self, path: Path):
        self.path = path
        self.last_event = time.monotonic()
        self.ready = False        # paso filtro + Capa1 + Capa2, encolado para procesar
        self.processing = False
        self.done = False
        self.rs_deadline = None   # se fija la primera vez que hay inactividad sin RS
        self.patient_id_hint = None
        self.listo_desde = None   # time.time() epoch, para mostrar antiguedad en la UI
        self.n_archivos = 0
        # --- filtro de localizacion (tri-state) ---
        self.localizacion_ok = None   # None=indeterminado, True/False una vez decidido
        self.excluido = False
        # --- doble chequeo de estabilidad ---
        self.checks_estables_consecutivos = 0
        # --- precarga de CT mientras se espera el RS ---
        self.ct_precargando = False
        self.ct_precargado = False

    def marcar_evento(self):
        if self.excluido:
            # Clasificacion TERMINAL (no-prostata / tecnica excluida / sin_ct): no se
            # reevalua por mas que sigan llegando archivos a la MISMA carpeta. Un
            # estudio de RMN/PET/otra localizacion no se convierte en un CT de
            # prostata porque le sigan llegando cortes -- confirmado en el piloto:
            # una RMN puede tardar varios minutos en terminar de llegar (docenas de
            # eventos fs espaciados 15-60s), y sin este corte cada uno relanzaba la
            # clasificacion (misma conclusion, log repetido hasta 17 veces para un
            # solo estudio). Si la carpeta se reusa de verdad para un paciente
            # distinto mas adelante, se resuelve solo via la poda periodica (la
            # entrada vieja se elimina y una carpeta nueva del mismo nombre arranca
            # como candidata nueva).
            self.last_event = time.monotonic()
            return
        if self.done or self.ready:
            # Reaparecen archivos en una carpeta ya lista/procesada (NO excluida,
            # ver arriba) -> el RS puede haber llegado tarde para el MISMO paciente
            # (confirmado en el piloto: hasta 17hs de diferencia entre CT y RS), o es
            # un paciente nuevo reusando la carpeta. Resetear el estado para
            # reevaluar desde cero.
            self.done = False
            self.ready = False
            self.rs_deadline = None
            self.patient_id_hint = None
            self.listo_desde = None
            self.localizacion_ok = None
        # Cualquier evento nuevo -- incluso si la carpeta no es "reusada" -- reinicia
        # el conteo de estabilidad consecutiva: si algo cambio, hay que volver a
        # sostener la estabilidad desde cero antes de dar la carpeta por lista.
        self.checks_estables_consecutivos = 0
        self.last_event = time.monotonic()


class PatientFolderMonitor:
    """Uso: crear con la carpeta raiz a monitorear + callbacks, llamar start().
    Internamente corre un Observer de watchdog (Capa 1, solo resetea timers), un
    thread de polling propio (Capa 2 + filtro de localizacion + precarga de CT), y
    un thread worker dedicado que procesa automaticamente la cola de candidatos
    listos. `listar_pendientes()` / `procesar_seleccionado()` quedan como API de
    fallback manual (usada por app.py para abrir una carpeta a mano)."""

    def __init__(self, watch_root: Path, config: dict, on_result, on_incompleto=None, on_excluido=None):
        self.watch_root = Path(watch_root)
        self.inactivity_timeout = config.get("inactivity_timeout_sec", 15)
        self.rs_extra_timeout = config.get("rs_extra_timeout_sec", 90)
        self.stability_wait = config.get("file_stability_check_sec", 3)
        self.poll_interval = config.get("poll_interval_sec", 2)
        self.min_checks_estables = config.get("min_checks_estables_consecutivos", 2)
        self.pruning_check_interval = config.get("pruning_check_interval_sec", 3600)
        self.pruning_age_sec = config.get("pruning_age_days", 2) * 86400

        filtro = config.get("filtro_localizacion") or {}
        self.incluir = filtro.get("incluir") or ["prostata"]
        self.excluir = filtro.get("excluir") or ["igrt", "sbrt", "areas"]

        self.on_result = on_result
        self.on_incompleto = on_incompleto or (lambda carpeta: None)
        self.on_excluido = on_excluido or (lambda *a, **k: None)

        self._candidatas = {}  # str(path) -> _CarpetaCandidata
        self._lock = threading.Lock()
        self._ct_cache = {}    # str(path) -> sitk.Image precargado
        self._ct_cache_lock = threading.Lock()
        self._observer = None
        self._poll_thread = None
        self._worker_thread = None
        self._conectar_thread = None
        self._stop = threading.Event()
        self._cola_lista = queue.Queue()
        self._last_pruning = time.monotonic()

    # ---- API publica ------------------------------------------------------------

    def start(self):
        """No bloquea: la conexion a `watch_root` (puede ser un recurso de red
        todavia no disponible, p.ej. justo tras un logon) se reintenta en background
        con backoff, para no demorar el arranque de Flask."""
        self._conectar_thread = threading.Thread(target=self._conectar_con_reintentos, daemon=True)
        self._conectar_thread.start()
        self._worker_thread = threading.Thread(target=self._worker_procesamiento, daemon=True)
        self._worker_thread.start()

    def _conectar_con_reintentos(self):
        backoff = 5
        while not self._stop.is_set():
            try:
                self.watch_root.mkdir(parents=True, exist_ok=True)
                handler = _Handler(self)
                self._observer = Observer()
                self._observer.schedule(handler, str(self.watch_root), recursive=True)
                self._observer.daemon = True
                self._observer.start()

                self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
                self._poll_thread.start()
                log.info("Monitoreando %s (inactividad=%ss, rs_extra=%ss, filtro incluir=%s excluir=%s)",
                          self.watch_root, self.inactivity_timeout, self.rs_extra_timeout,
                          self.incluir, self.excluir)
                return
            except Exception:
                log.exception("No se pudo conectar a %s -- reintentando en %ss", self.watch_root, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 300)

    def stop(self):
        self._stop.set()
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)

    def listar_pendientes(self) -> list:
        """Fallback manual: candidatos listos que todavia no se encolaron para
        procesamiento automatico (en operacion normal, casi nunca hay nada aca --
        `ready` y `processing` se setean juntos apenas se decide encolar)."""
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

    def procesar_seleccionado(self, carpeta_str: str, on_fase=None) -> dict:
        """Dispara pipeline.procesar_paciente() para la carpeta elegida a mano
        (fallback manual, p.ej. una carpeta que el filtro excluyo por error). Lanza
        ValueError si esa carpeta no esta disponible (no existe, ya se proceso, o el
        auto-procesamiento ya la esta manejando)."""
        with self._lock:
            cand = self._candidatas.get(carpeta_str)
            if cand is None or not cand.ready or cand.processing or cand.done:
                raise ValueError(f"Carpeta no disponible para procesar: {carpeta_str}")
            cand.processing = True

        import pipeline  # import tardio: evita ciclo al testear watcher solo
        try:
            resultado = pipeline.procesar_paciente(cand.path, on_fase=on_fase)
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
        # Cualquier archivo nuevo invalida una precarga de CT vieja -- mejor volver a
        # leer que arriesgarse a procesar con un CT parcial cacheado.
        with self._ct_cache_lock:
            self._ct_cache.pop(str(carpeta), None)

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
            self._podar_si_corresponde()

    def _evaluar(self, cand: _CarpetaCandidata):
        archivos = _listar_archivos(cand.path)
        if not archivos:
            return

        # --- Paso 0: filtro de localizacion (tri-state) ---
        if cand.localizacion_ok is None:
            info = _leer_info_ct(archivos)
            if info is None:
                # Sin ningun CT legible todavia: indeterminado. Si la carpeta ya
                # dejo de cambiar y sigue sin CT, es basura (p.ej. resabio SC de
                # importacion al planificador) -- excluir en vez de reintentar
                # para siempre.
                if self._archivos_estables(archivos):
                    cand.excluido = True
                    cand.done = True
                    self.on_excluido(cand.path, "", "", "sin_ct")
                    log.info("Carpeta %s excluida: estable pero sin ningun CT legible (sin_ct)", cand.path)
                return
            ok, motivo = _clasificar_localizacion(info["study_description"], self.incluir, self.excluir)
            cand.localizacion_ok = ok
            cand.patient_id_hint = info["patient_id"] or cand.path.name
            if info.get("advertencia_estudio_mixto"):
                log.warning("Carpeta %s: StudyDescription distinto entre los CT muestreados -- "
                            "posible estudio mezclado (riesgo conocido, no resuelto)", cand.path)
            if not ok:
                cand.excluido = True
                cand.done = True
                self.on_excluido(cand.path, info["study_description"], info["patient_id"], motivo)
                return
            log.info("Localizacion OK (prostata) en %s (%s): '%s'",
                      cand.path, info["patient_id"], info["study_description"])

        # --- Precarga de CT mientras se espera el RS (no bloquea el poll loop) ---
        if not cand.ct_precargando and not cand.ct_precargado:
            if self._archivos_estables(archivos):
                cand.ct_precargando = True
                threading.Thread(target=self._precargar_ct, args=(cand,), daemon=True).start()

        # --- Paso 1: Capa 1 (inactividad) ---
        inactivo_desde = time.monotonic() - cand.last_event
        if inactivo_desde < self.inactivity_timeout:
            return

        # --- Paso 2: Capa 2 (estabilidad + RS) ---
        if not self._archivos_estables(archivos):
            cand.checks_estables_consecutivos = 0
            return

        tiene_rs = any(_es_rtstruct(f) for f in archivos)
        if not tiene_rs:
            if cand.rs_deadline is None:
                cand.rs_deadline = time.monotonic() + self.rs_extra_timeout
                log.info("Inactividad+estabilidad en %s pero sin RS todavia; "
                          "esperando hasta %ss extra", cand.path, self.rs_extra_timeout)
                return
            if time.monotonic() < cand.rs_deadline:
                return
            log.warning("Carpeta %s marcada INCOMPLETA: sin RTSTRUCT tras timeout "
                        "extendido de %ss", cand.path, self.rs_extra_timeout)
            cand.done = True
            self.on_incompleto(cand.path)
            return

        # --- Paso 3: sostener estabilidad por N ciclos de poll seguidos ---
        cand.checks_estables_consecutivos += 1
        if cand.checks_estables_consecutivos < self.min_checks_estables:
            return

        # --- Listo: encolar para procesamiento automatico ---
        cand.n_archivos = len(archivos)
        with self._lock:
            cand.ready = True
            cand.processing = True
            cand.listo_desde = time.time()
        self._cola_lista.put(cand)
        log.info("Paciente listo, encolado para procesamiento automatico: %s (%s)",
                  cand.path, cand.patient_id_hint)

    def _archivos_estables(self, archivos) -> bool:
        tamanos_1 = {f: f.stat().st_size for f in archivos if f.exists()}
        time.sleep(self.stability_wait)
        for f, size_antes in tamanos_1.items():
            if not f.exists() or f.stat().st_size != size_antes:
                return False
        return True

    def _precargar_ct(self, cand: _CarpetaCandidata):
        try:
            import pipeline  # import tardio: evita ciclo al testear watcher solo
            with pipeline._carpeta_plana(cand.path) as carpeta_plana:
                imagen = cargar_ct(carpeta_plana)
            with self._ct_cache_lock:
                self._ct_cache[str(cand.path)] = imagen
            cand.ct_precargado = True
            log.info("CT precargado para %s (mientras se espera el RS)", cand.path)
        except Exception:
            log.exception("No se pudo precargar el CT de %s -- no bloqueante, se "
                          "leera de la forma habitual al procesar", cand.path)
        finally:
            cand.ct_precargando = False

    # ---- Worker: procesamiento automatico, secuencial -----------------------

    def _worker_procesamiento(self):
        ultimo_heartbeat = time.monotonic()
        while not self._stop.is_set():
            try:
                cand = self._cola_lista.get(timeout=5)
            except queue.Empty:
                if time.monotonic() - ultimo_heartbeat > 600:
                    log.info("Worker de procesamiento activo, cola vacia (heartbeat)")
                    ultimo_heartbeat = time.monotonic()
                continue
            ultimo_heartbeat = time.monotonic()
            try:
                self._procesar_candidata(cand)
            except Exception:
                # Cualquier excepcion no prevista (incluyendo dentro de on_result) se
                # atrapa aca -- si el worker muriera silenciosamente, ningun paciente
                # futuro se procesaria sin que la app se caiga (Task Scheduler no lo
                # reiniciaria porque el proceso sigue vivo).
                log.exception("Error inesperado procesando %s -- se pierde este "
                              "paciente, el worker sigue vivo para el siguiente", cand.path)
                with self._lock:
                    cand.done = True
                    cand.processing = False

    def _procesar_candidata(self, cand: _CarpetaCandidata):
        # Re-verificacion final de estabilidad inmediatamente antes de procesar: si
        # algo cambio entre que quedo "lista" y que el worker la tomo de la cola, no
        # arriesgarse a leer una carpeta a medio actualizar -- devolverla a
        # "esperando" para que el poll loop la vuelva a evaluar desde cero.
        archivos = _listar_archivos(cand.path)
        if not archivos or not self._archivos_estables(archivos):
            log.warning("Carpeta %s cambio justo antes de procesar -- se pospone, "
                        "se vuelve a evaluar en el proximo poll", cand.path)
            with self._lock:
                cand.ready = False
                cand.processing = False
                cand.checks_estables_consecutivos = 0
            return

        with self._ct_cache_lock:
            imagen_ct = self._ct_cache.pop(str(cand.path), None)

        import pipeline  # import tardio: evita ciclo al testear watcher solo
        resultado = pipeline.procesar_paciente(cand.path, imagen_ct_precargada=imagen_ct)
        with self._lock:
            cand.done = True
            cand.processing = False
        self.on_result(resultado)

    # ---- Poda periodica ------------------------------------------------------

    def _podar_si_corresponde(self):
        ahora = time.monotonic()
        if ahora - self._last_pruning < self.pruning_check_interval:
            return
        self._last_pruning = ahora

        limite = ahora - self.pruning_age_sec
        with self._lock:
            a_podar = [k for k, c in self._candidatas.items() if c.done and c.last_event < limite]
            for k in a_podar:
                del self._candidatas[k]
        if a_podar:
            log.info("Poda de candidatas: %d entradas eliminadas (resueltas, > %s dias sin actividad)",
                      len(a_podar), self.pruning_age_sec / 86400)

        with self._ct_cache_lock:
            vivos = set(self._candidatas.keys())
            for k in list(self._ct_cache.keys()):
                if k not in vivos:
                    self._ct_cache.pop(k, None)


class _Handler(FileSystemEventHandler):
    def __init__(self, monitor: PatientFolderMonitor):
        self.monitor = monitor

    def on_created(self, event):
        if not event.is_directory:
            self.monitor._on_fs_event(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self.monitor._on_fs_event(event.src_path)
