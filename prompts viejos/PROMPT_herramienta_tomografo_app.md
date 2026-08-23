# Prompt Claude Code — Herramienta de tomógrafo (app web local + watchdog)

## Objetivo

Aplicación que corre en la PC del tomógrafo (o accesible por IP en la red local), que:
1. Permite abrir manualmente un paciente (carpeta con CT+RS DICOM), o
2. Monitorea automáticamente una carpeta configurada y procesa el paciente en cuanto llega completo.

Muestra: datos del paciente + semáforo grande (recto y vejiga, verde/naranja/rojo) + debajo, más chico,
las métricas extraídas (7 features + V65/V55 predichos). Guarda un registro JSON por paciente procesado
(sin necesidad de historial visual en la UI — es insumo para trabajos futuros).

**Reusa el pipeline ya construido y validado del Proyecto 1** (Tareas 6-7): `extract_features_live.py`,
`infer_tomografo.py`, modelos/umbrales en `models/proyecto1/manifest.json`. Esta app es una capa de
presentación sobre eso — no reimplementar lógica de features ni de modelo acá.

---

## Arquitectura elegida (justificación breve)

**App web local con Flask, no GUI de escritorio.** Corriendo en `localhost:5000`, es accesible desde la
misma PC y desde cualquier otra de la red por IP sin cambios de código — cubre "local o red" sin
necesidad de dos implementaciones distintas. La UI (semáforo + métricas) es HTML/CSS simple, más rápido
de iterar que un GUI de widgets dado el perfil de Python del usuario.

**Sin websockets.** La página hace polling simple (JS, cada 3-5 seg) contra un endpoint que devuelve el
último resultado. Para "un paciente cada tanto" es la opción más simple y menos propensa a romperse.

**Sin base de datos.** Un JSON por paciente en una carpeta de resultados alcanza como registro. No hace
falta SQLite ni historial en UI.

---

## Estructura de archivos

```
tomografo_tool/
├── app.py                    # Flask: rutas, estado en memoria del último resultado
├── watcher.py                 # hilo background con watchdog, detecta paciente completo
├── pipeline.py                # wrapper: llama a extract_features_live.py + infer_tomografo.py
├── config.yaml                # carpeta a monitorear, timeouts, paths a modelos
├── templates/
│   └── index.html             # UI: paciente + semáforo + métricas
├── static/
│   └── style.css              # semáforo grande, colores verde/naranja/rojo
└── registros/
    └── <AnonID_o_PatientID>_<timestamp>.json   # un JSON por paciente procesado
```

---

## Lógica del watcher (la parte delicada — leer con cuidado)

**El riesgo real no es "cuándo dejaron de llegar archivos" sino "¿terminó de escribirse cada archivo?"**
Un evento `created` de watchdog puede dispararse ANTES de que el archivo termine de copiarse (especialmente
la RS, que es pesada). Leer un DICOM a mitad de escritura causa fallos intermitentes y difíciles de
reproducir. Por eso la detección de "paciente completo" tiene DOS capas, no una:

### Capa 1 — Inactividad (dispara el chequeo)
Usar `watchdog.observers.Observer` sobre la carpeta configurada. Cada vez que llega un archivo nuevo,
resetear un timer. Si pasan **10 segundos sin archivos nuevos**, pasar a la Capa 2.

### Capa 2 — Verificación de estabilidad + presencia de RS (antes de procesar)
NO asumir que por haber pasado el timeout ya se puede procesar. Verificar:

1. **Cada archivo candidato a CT y la RS deben tener tamaño estable**: leer tamaño, esperar 1 segundo,
   releer — si cambió, seguir esperando (no abortar, resetear el chequeo de estabilidad para ese archivo).
2. **Debe existir al menos un archivo RTSTRUCT** (identificable por `Modality == 'RTSTRUCT'` al leer el
   header DICOM, no por nombre de archivo — más robusto). Si tras el timeout de inactividad no hay RS
   todavía, **NO abortar**: seguir esperando con un timeout más largo específico para la RS (p.ej. 60 seg
   adicionales), porque suele llegar después y pesar más. Solo si ese timeout extendido también vence,
   marcar el caso como "incompleto" y loguearlo (no crashear, no bloquear el watcher para el próximo
   paciente).
3. Recién cuando (a) hay inactividad general, (b) todos los archivos tienen tamaño estable, y (c) hay
   al menos una RS válida → disparar el pipeline.

Estructura sugerida: una clase `PatientFolderMonitor` con estado por carpeta-en-curso (para poder
procesar más de un paciente en secuencia sin mezclarlos), que expone `is_ready() -> bool` con la lógica
de arriba, chequeada por un loop de polling propio del watcher (no todo dentro del callback de watchdog).

### Configurable en `config.yaml`
```yaml
watch_folder: "C:/ruta/a/carpeta/monitoreada"
inactivity_timeout_sec: 10
rs_extra_timeout_sec: 60
file_stability_check_sec: 1
poll_interval_sec: 2
```

---

## Pipeline (`pipeline.py`)

Función única `procesar_paciente(carpeta: Path) -> dict`:
1. Leer RS + CT de la carpeta (usar el parser DICOM ya construido en `extract_features_live.py`).
2. Extraer las 7 features + `tiene_VVSS`.
3. Llamar a `infer_tomografo.py` → dict con `{zona, prob, V_pred}` por constraint (RV65, RV55, BV65) y
   por OAR combinado (recto = pesimista RV65+RV55, vejiga = BV65) según el manifest vigente.
4. Armar el dict de resultado completo:
```python
{
  "patient_id": ...,        # el identificador visible (AnonID o el que use el centro)
  "timestamp": ...,
  "tiene_VVSS": ...,
  "features": {  # las 7 + derivadas
    "VolRectum_cc": ..., "VolBladder_cc": ..., "VolPTV_cc": ...,
    "Solap_PTV_Rectum_cc": ..., "Solap_PTV_Bladder_cc": ...,
    "overlap_rel_recto": ..., "overlap_rel_vejiga": ...
  },
  "recto": {"zona": "verde|naranja|rojo", "V65_pred": ..., "V55_pred": ...},
  "vejiga": {"zona": "verde|naranja|rojo", "V65_pred": ...}
}
```
5. Guardar ese dict como JSON en `registros/` (nombre: `<patient_id>_<timestamp>.json`).
6. Devolver el dict a quien llamó (watcher o ruta manual de Flask).

Capturar excepciones por paciente sin tumbar el watcher — loguear el error y seguir monitoreando.

---

## App Flask (`app.py`)

Rutas mínimas:
- `GET /` → sirve `index.html`.
- `GET /ultimo` → JSON con el último resultado procesado (o `{"estado": "esperando"}` si no hay nada
  aún). La página hace polling acá.
- `POST /abrir_carpeta` → recibe un path, llama a `pipeline.procesar_paciente()` de forma síncrona,
  actualiza el "último resultado" en memoria, devuelve el dict.
- El watcher corre en un hilo separado (thread daemon) lanzado al iniciar `app.py`; cuando termina de
  procesar un paciente, actualiza la misma variable de "último resultado" en memoria (con un lock simple)
  que consume `/ultimo`.

Ejecutar con `app.run(host='0.0.0.0', port=5000)` para que sea accesible por IP en la red, no solo
localhost.

---

## UI (`templates/index.html` + `static/style.css`)

Layout, de arriba a abajo:
1. **Encabezado:** patient_id + timestamp, texto grande y claro.
2. **Semáforo, lo más notorio de la pantalla:** dos círculos grandes lado a lado, etiquetados "RECTO" y
   "VEJIGA", coloreados verde/naranja/rojo según `zona`. Tamaño de fuente grande, alto contraste — se lee
   desde lejos en la sala de tomógrafo.
3. **Debajo, chico y en gris/menor contraste:** tabla con las 7 features y los V65/V55 predichos. Info de
   soporte para el físico, no compite visualmente con el semáforo.
4. Estado "esperando paciente..." cuando no hay nada procesado, y un botón/input simple para abrir una
   carpeta manualmente.

Polling JS simple: `setInterval` cada 3-5 seg pidiendo `/ultimo`, actualiza el DOM si cambió el
`patient_id`/`timestamp` respecto al mostrado.

---

## Qué NO hacer

- No reimplementar extracción de features ni el modelo — llamar a lo ya construido y validado.
- No usar websockets ni frameworks de frontend pesados (React, etc.) — HTML+CSS+JS vanilla alcanza.
- No bloquear el watcher si un paciente falla (loguear y seguir).
- No disparar el pipeline solo por timeout de inactividad sin verificar estabilidad de tamaño + presencia
  de RS — es la causa más probable de fallos intermitentes.
- No agregar historial ni base de datos — el JSON por paciente en `registros/` es suficiente.

---

## Entregable de vuelta a Claude.ai

Confirmar: (a) que el pipeline reusa `extract_features_live.py`/`infer_tomografo.py` sin reimplementar
lógica, (b) cómo quedó resuelta la detección de RTSTRUCT (por Modality del header, no por nombre de
archivo), y (c) un ejemplo de JSON de registro generado sobre un caso de prueba, para revisar el formato
antes de darlo por cerrado.
