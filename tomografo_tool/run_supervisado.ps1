# Supervisor de tomografo_tool para operacion desatendida (shadow testing).
# Lanza app.py en loop: si termina (crash o no), lo relanza con backoff exponencial
# en vez de un sleep fijo -- un sleep fijo corto haria "hot-loop" contra la carpeta
# de red si esta caida por un rato largo. El backoff sube (dobla) si el proceso
# terminó rapido (< 60s, indicio de crash-loop) y vuelve al piso si corrio sano un
# buen rato. Pensado para registrarse en Task Scheduler (ver
# registrar_tarea_programada.ps1).

$ErrorActionPreference = "Continue"

$Python = "C:\Pablo\ProstateDoseProject\repo\.venv\Scripts\python.exe"
$App = "C:\Pablo\ProstateDoseProject\repo\tomografo_tool\app.py"
$LogDir = "C:\Pablo\ProstateDoseProject\repo\tomografo_tool\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$SupervisorLog = Join-Path $LogDir "supervisor.log"

$BackoffMinSec = 10
$BackoffMaxSec = 300
$backoff = $BackoffMinSec

function Log-Supervisor($msg) {
    $linea = "$(Get-Date -Format o)  $msg"
    Add-Content -Path $SupervisorLog -Value $linea -Encoding utf8
}

Log-Supervisor "Supervisor arrancando. Python=$Python App=$App"

while ($true) {
    Log-Supervisor "Iniciando tomografo_tool (backoff actual=${backoff}s si vuelve a fallar)..."
    $inicio = Get-Date
    & $Python $App
    $exitCode = $LASTEXITCODE
    $duracionSec = ((Get-Date) - $inicio).TotalSeconds

    Log-Supervisor "tomografo_tool termino (exit=$exitCode, corrio ${duracionSec}s)."

    if ($duracionSec -lt 60) {
        # Corrio poco -- probable crash-loop, subir el backoff.
        $backoff = [Math]::Min($backoff * 2, $BackoffMaxSec)
    } else {
        # Corrio un buen rato antes de terminar -- volver al piso.
        $backoff = $BackoffMinSec
    }

    Log-Supervisor "Reintentando en ${backoff}s..."
    Start-Sleep -Seconds $backoff
}
