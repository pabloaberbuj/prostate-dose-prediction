# Monitoreo de `tomografo_tool` (shadow testing)

Guía rápida para hacer seguimiento del shadow test sin tener que releer el código.
Referencia de diseño: `pipeline.py`, `watcher.py`, `app.py` (comentarios in-line, muy
detallados) y `PROMPT_herramienta_tomografo_app.md`.

## 0. Qué es, en una frase

Un servicio Flask + watcher que monitorea `\\ARIAMEVADB-SVR\va_data$\DICOM` (la
carpeta real donde el tomógrafo exporta el CT y Autocontour manda el RS, **antes**
del import a Eclipse), detecta automáticamente pacientes de **próstata**, corre el
modelo del Proyecto 1 (Tareas 2-4/6-7) y guarda un semáforo verde/naranja/rojo por
Recto y Vejiga (riesgo de no cumplir V65/V55 recto y V65 vejiga). Es **solo lectura**
sobre el share — nunca mueve ni modifica DICOM, y no interviene en la planificación
real.

Corre desatendido vía Task Scheduler (`registrar_tarea_programada.ps1` →
`run_supervisado.ps1` → `app.py`), relanzándose solo si crashea.

## 0.1 Arrancar / detener / ver estado

**Forma normal (ya activa):** la Tarea Programada de Windows
`TomografoToolShadowTesting` (`registrar_tarea_programada.ps1`) — arranca sola al
iniciar sesión y el propio `run_supervisado.ps1` reinicia `app.py` si crashea. No
depende de tener la consola, VSCode ni el chat abiertos, y sobrevive a
reinicio/logoff de la PC. Requiere haberla registrado una vez con una consola de
administrador (ver §0.2) — ya se hizo el 2026-09-14.

```powershell
cd C:\Pablo\ProstateDoseProject\repo\tomografo_tool
Get-ScheduledTask -TaskName "TomografoToolShadowTesting" | Format-List TaskName, State   # deberia decir Running
powershell -ExecutionPolicy Bypass -File .\estado_servicio.ps1                            # PIDs actuales + ultimas lineas de log
powershell -ExecutionPolicy Bypass -File .\detener_servicio.ps1                           # detener del todo (Tarea incluida)
Start-ScheduledTask -TaskName "TomografoToolShadowTesting"                                # volver a arrancar sin esperar al proximo logon
```

`iniciar_servicio.ps1` sigue andando como fallback manual (arranca un proceso
desacoplado sin pasar por Task Scheduler) para una sesión sin permisos de admin —
no hace nada si ya hay una instancia corriendo (por la Tarea o por sí mismo), así
que no genera duplicados.

**Ante la duda:** `estado_servicio.ps1` debería listar exactamente 2 PIDs de
python (padre + reloader de Flask). Si ves 4 o más, hay una instancia vieja suelta
por fuera de la Tarea Programada — `detener_servicio.ps1` ya busca y mata también
cualquier `powershell.exe` de `run_supervisado.ps1` suelto, no solo lo que gestiona
la Tarea o su propio PID file (nos pasó una vez: un supervisor arrancado a mano en
una prueba anterior quedó relanzando `app.py` solo, y terminamos con dos `app.py`
compitiendo por el puerto 5000).

## 0.2 Si hay que volver a registrar la Tarea Programada (requiere admin)

Hace falta permisos de administrador local — `Register-ScheduledTask` da "Acceso
denegado" en una consola sin elevar. Si la sesión de Claude Code ya corre como
administrador, correrlo directo alcanza:

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Pablo\ProstateDoseProject\repo\tomografo_tool\registrar_tarea_programada.ps1"
```

Si no (consola normal, sin admin), hay que elevarla primero: botón derecho sobre
PowerShell → **"Ejecutar como administrador"** (o Windows+X → *Terminal
(Administrador)*), y ahí sí correr el comando de arriba. Es un registro por
única vez — no hace falta repetirlo salvo que se borre la Tarea o cambie la ruta
del repo.

## 1. Verificar que está corriendo

Desde una consola PowerShell **en la PC donde está registrada la tarea**:

```powershell
# 1) La tarea programada existe y está lista/corriendo
Get-ScheduledTask -TaskName "TomografoToolShadowTesting" | Format-List TaskName, State

# 2) Hay un proceso python.exe vivo corriendo app.py
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like "*tomografo_tool*app.py*" } |
  Select-Object ProcessId, CreationDate

# 3) La app responde
Invoke-WebRequest http://localhost:5000/ultimo -UseBasicParsing | Select-Object -Expand Content
```

> **Nota:** al revisar esto en la sesión que generó este documento, ni el proceso
> `python.exe` ni la tarea `TomografoToolShadowTesting` aparecieron desde esa
> sesión (que puede no tener visibilidad de tu sesión interactiva de Windows) —
> pero `logs/tomografo_tool.log` mostraba heartbeats hasta esta mañana. Correlo vos
> mismo con los comandos de arriba para confirmar el estado real.

Si `State` no es `Ready`/`Running` o el proceso no aparece, ver la sección
**Troubleshooting** más abajo.

## 2. UI web (mientras el servicio corre)

Abrir `http://localhost:5000` (o `http://<host>:5000` si corre en otra PC — puerto
configurable en `config.yaml: port`).

- **Pill arriba a la izquierda**: "Monitoreando" (watcher activo, `watch_folder`
  configurado) vs "Apertura manual" (watcher apagado — algo anda mal, ver §5).
- **Semáforo RECTO / VEJIGA**: último paciente procesado.
  - 🟢 **VERDE**: probabilidad de no cumplir el constraint por debajo del umbral
    calibrado — bajo riesgo.
  - 🟠 **NARANJA**: score por encima del umbral, pero el margen de sobrepaso
    predicho es chico.
  - 🔴 **ROJO**: score alto Y margen de sobrepaso grande (`>= delta_severidad_pp`,
    ver §4.4) — caso a mirar con más atención.
- **Métricas del caso** (sidebar): volúmenes de Recto/Vejiga/PTV, solapamiento
  PTV-OAR (absoluto y relativo), y los V_pred de cada constraint.
- **Pacientes en cola**: normalmente vacío — el procesamiento es automático. Solo
  aparece algo acá si un caso quedó "listo" pero el filtro de localización lo
  hubiera descartado por error (fallback manual, botón "Procesar este").
- **Abrir manualmente**: para forzar el procesamiento de una carpeta específica
  (típicamente para reprocesar un caso viejo de prueba, no para el flujo normal).

## 3. Dónde vive todo

```
repo/tomografo_tool/
  config.yaml          # carpeta monitoreada, filtro de localización, timeouts
  logs/
    tomografo_tool.log # log principal de la app (rotativo, 5MB x5 backups)
    supervisor.log      # arranques/caídas/reintentos del wrapper PowerShell
    excluidos.log        # CSV de carpetas descartadas por el filtro de localización
  registros/            # UN .json por paciente procesado (el "resultado" real)
```

## 4. Interpretar cada archivo

### 4.1 `logs/tomografo_tool.log`

Mensajes que vas a ver en operación normal (grepear por estas frases):

| Frase a buscar | Qué significa |
|---|---|
| `Monitoreando \\ARIAMEVADB-SVR...` | Arrancó bien, watcher conectado al share |
| `Worker de procesamiento activo, cola vacia (heartbeat)` | Cada ~10 min, "sigo vivo, no hay nada para procesar" — si dejan de aparecer, el proceso murió sin loguear el motivo (revisar `supervisor.log`) |
| `Localizacion OK (prostata) en ...` | Un CT pasó el filtro de localización, se sigue esperando el RS |
| `Paciente listo, encolado para procesamiento automatico` | Va a procesar YA (CT+RS estables) |
| `Ultimo resultado actualizado: paciente=... estado=ok` | Terminó de procesar, JSON guardado en `registros/` |
| `Excluido: <carpeta> (<id>) -- '<StudyDescription>' -- motivo=...` | Carpeta descartada por el filtro de localización (ver §4.3) |
| `Carpeta ... marcada INCOMPLETA: sin RTSTRUCT tras timeout extendido` | Llegó el CT pero nunca el RS (Autocontour no corrió o tardó demasiado) — **no se generó predicción para este paciente** |
| `Error procesando ... : ...` (con traceback) | Excepción real durante extracción de features o inferencia — revisar el traceback, y que el `registro` correspondiente tenga `"estado": "error"` |
| `No se pudo conectar a ... -- reintentando en Xs` | El share de red no está disponible (caído, VPN, permisos) |

Para ver en vivo:
```powershell
Get-Content .\logs\tomografo_tool.log -Tail 50 -Wait
```

### 4.2 `logs/supervisor.log`

Registra cada arranque/caída de `app.py` (el supervisor externo que lo relanza).
Formato: `<timestamp ISO>  <mensaje>`.

- Si ves ciclos de `Iniciando... / termino (exit=... corrio X s) / Reintentando`
  con **X chico (<60s) repetidamente** → crash-loop real, algo rompe al arrancar
  (mirar el traceback correspondiente en `tomografo_tool.log`, justo antes del
  corte). El backoff entre reintentos sube exponencialmente (10s → hasta 300s) en
  ese caso, así que puede tardar unos minutos en volver a intentar.
- Un solo `termino` con `X` grande (corrió horas/días) seguido de un reinicio
  normal es esperable — reinicio de la PC, actualización manual, etc.

### 4.3 `logs/excluidos.log` (CSV)

Columnas: `timestamp, patient_id, carpeta, study_description, motivo`.

Es el log de auditoría del **filtro de localización** (`config.yaml:
filtro_localizacion`) — qué se descartó y por qué. Dos motivos posibles:
- `no_prostata`: el `StudyDescription` no matcheó ningún término de `incluir`.
- `excluido_<término>`: matcheó `incluir` pero también algún término de `excluir`
  (`igrt`/`sbrt`/`areas`/`lecho`).

**Por qué revisarlo periódicamente:** es la forma de detectar si el filtro se está
comiendo pacientes de próstata reales por una `StudyDescription` con una
abreviatura no contemplada (ya pasó una vez en el piloto con "IMRT PROST" — ver
comentarios en `config.yaml`). Si un paciente de próstata que sabés que se planificó
no aparece en `registros/`, buscarlo acá primero por `patient_id` o carpeta.

Abrir en Excel o:
```powershell
Import-Csv .\logs\excluidos.log | Format-Table -AutoSize
# Motivos más frecuentes de exclusión (para calibrar el filtro):
Import-Csv .\logs\excluidos.log | Group-Object motivo | Sort-Object Count -Descending
```

### 4.4 `registros/<patient_id>_<timestamp>.json`

Un archivo por paciente procesado (éxito o error). Ejemplo real:

```json
{
  "estado": "ok",
  "patient_id": "1-120186-0",
  "timestamp": "20260910_104329",
  "carpeta": "\\\\ARIAMEVADB-SVR\\va_data$\\DICOM\\1-120186-0",
  "features": {
    "VolRectum_cc": 149.03, "VolBladder_cc": 317.01, "VolPTV_cc": 336.79,
    "Solap_PTV_Rectum_cc": 12.28, "Solap_PTV_Bladder_cc": 41.64,
    "overlap_rel_recto": 0.0824, "overlap_rel_vejiga": 0.1314
  },
  "recto":  { "zona": "verde", "V65_pred": 12.93, "V55_pred": 22.67 },
  "vejiga": { "zona": "rojo",  "V65_pred": 17.26 }
}
```

- `estado`: `"ok"` o `"error"` (si `"error"`, hay además `"error"` y `"traceback"` —
  cruzar con `tomografo_tool.log` para el detalle completo).
- `features`: geometría cruda extraída del CT+RS (volúmenes y solapamientos PTV-OAR).
- `recto.V65_pred` / `V55_pred`, `vejiga.V65_pred`: **valor de dosis-volumen
  predicho** (%) para cada constraint (umbrales clínicos: RV65<15%, RV55<25%,
  BV65<15% — `models/proyecto1/thresholds.json`).
- `zona`: el semáforo ya calculado —
  - `verde` si `score < umbral_verde_proba` calibrado (recto: 0.62, vejiga: 0.18),
  - si no, `rojo` si el margen de sobrepaso predicho `>= delta_severidad_pp`
    (recto: ~2.46pp, vejiga: ~2.09pp), si no `naranja`.
  - `score` = probabilidad de fallo del modelo de clasificación (recto = máximo
    entre RV65 y RV55; vejiga = BV65 directo). No viene en el JSON de `registros/`
    tal cual (solo `zona`+`V_pred`) — si hace falta el score/margen exacto, correr
    `python scripts/infer_tomografo.py --carpeta <carpeta>` a mano sobre el mismo
    caso (imprime score y margen).

**Recordatorio de alcance del modelo** (relevante al leer resultados): entrenado y
calibrado sobre próstata sin afectación nodal ni lecho post-prostatectomía — el
filtro de localización ya excluye "lecho", pero *no* filtra por técnica dentro de
"próstata" más allá de eso.

## 5. Rutina de seguimiento sugerida

### Conteo rápido de la actividad de hoy
```powershell
cd C:\Pablo\ProstateDoseProject\repo\tomografo_tool
$hoy = (Get-Date).Date
Get-ChildItem .\registros\*.json | Where-Object { $_.LastWriteTime -ge $hoy } | Measure-Object
```

### Resumen de zonas (verde/naranja/rojo) y errores, todo el histórico
```powershell
Get-ChildItem .\registros\*.json | ForEach-Object {
    $j = Get-Content $_ -Raw | ConvertFrom-Json
    [PSCustomObject]@{
        Paciente = $j.patient_id
        Fecha    = $j.timestamp
        Estado   = $j.estado
        Recto    = $j.recto.zona
        Vejiga   = $j.vejiga.zona
    }
} | Sort-Object Fecha | Format-Table -AutoSize
```

### Casos en rojo (los que más vale mirar a mano contra el plan real en Eclipse)
```powershell
Get-ChildItem .\registros\*.json | ForEach-Object {
    $j = Get-Content $_ -Raw | ConvertFrom-Json
    if ($j.recto.zona -eq "rojo" -or $j.vejiga.zona -eq "rojo") { $j.patient_id }
}
```

### Errores a revisar
```powershell
Get-ChildItem .\registros\*.json | ForEach-Object {
    $j = Get-Content $_ -Raw | ConvertFrom-Json
    if ($j.estado -eq "error") { "$($j.patient_id): $($j.error)" }
}
```

Cadencia sugerida: revisar `tomografo_tool.log` (últimas líneas) + conteo de hoy
una vez por día; `excluidos.log` una vez por semana (para pescar términos de
`StudyDescription` no contemplados); casos en rojo, a demanda cuando quieras
comparar contra el plan clínico real del mismo paciente en Eclipse.

## 6. Troubleshooting

- **No hay heartbeats nuevos en `tomografo_tool.log` desde hace horas** → el
  proceso murió. Revisar `supervisor.log`: si dice que está reintentando con
  backoff alto, esperar; si no hay entradas nuevas ahí tampoco, la tarea programada
  se cayó — reabrir sesión (la tarea corre `AtLogOn`) o re-registrarla:
  `powershell -ExecutionPolicy Bypass -File .\registrar_tarea_programada.ps1`
- **`Monitoreando` nunca aparece / "No se pudo conectar a ..."** → problema de red
  con `\\ARIAMEVADB-SVR\va_data$\DICOM` (VPN, permisos, servidor caído). El
  watcher reintenta solo con backoff creciente hasta 5 min.
- **Pill dice "Apertura manual"** → `watch_folder` en `config.yaml` quedó en el
  placeholder o vacío — revisar el archivo.
- **Un paciente de próstata que se sabe que existió no aparece en `registros/`** →
  buscarlo en `excluidos.log` (filtro lo descartó) o en el log completo por
  `INCOMPLETA` (nunca llegó el RS de Autocontour a tiempo).
- **Reiniciar manualmente**: matar el proceso `python.exe` de `app.py` (o
  `Stop-ScheduledTask -TaskName "TomografoToolShadowTesting"`) — el supervisor lo
  relanza solo si el *wrapper* PowerShell sigue corriendo; para parar del todo,
  `Unregister-ScheduledTask -TaskName "TomografoToolShadowTesting" -Confirm:$false`.
