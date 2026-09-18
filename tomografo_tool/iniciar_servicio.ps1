# Arranca tomografo_tool en segundo plano como proceso Windows independiente --
# NO depende de que esta consola, VSCode o el chat sigan abiertos: Start-Process
# lanza un proceso totalmente desacoplado (no es un hijo de esta sesion), que sigue
# vivo hasta que se llame a detener_servicio.ps1 o se reinicie la PC.
#
# Por que esto y no un Servicio de Windows de verdad: tanto crear un Servicio
# (sc.exe / New-Service) como registrar la Tarea Programada (ver
# registrar_tarea_programada.ps1) requieren permisos de administrador local, y en
# este equipo Register-ScheduledTask ya dio "Acceso denegado" sin ser admin. Esta es
# la forma mas simple que funciona SIN esos permisos. Contras frente a un servicio
# real: no arranca solo al reiniciar la PC (hay que correr este script de nuevo
# despues de un reinicio/logoff) -- si mas adelante se consigue una sesion con
# permisos de administrador, usar registrar_tarea_programada.ps1 en su lugar para
# que ademas arranque solo en cada logon.
#
# Uso:
#   powershell -ExecutionPolicy Bypass -File iniciar_servicio.ps1
# Para detenerlo:
#   powershell -ExecutionPolicy Bypass -File detener_servicio.ps1
# Para ver el estado: ver MONITOREO.md seccion 1, o estado_servicio.ps1

$ErrorActionPreference = "Stop"

$BaseDir    = $PSScriptRoot
$Supervisor = Join-Path $BaseDir "run_supervisado.ps1"
$LogDir     = Join-Path $BaseDir "logs"
$PidFile    = Join-Path $LogDir "servicio.pid"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Evitar arrancar una segunda copia si ya hay una corriendo (identificado por
# linea de comando, no por el PID file -- mas confiable si el PID file quedo
# desactualizado de una corrida anterior que no se detuvo con este script).
$yaCorriendo = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*tomografo_tool*app.py*" }
if ($yaCorriendo) {
    Write-Output ("Ya esta corriendo (PID python: {0}). No se arranca de nuevo." -f `
        ($yaCorriendo.ProcessId -join ", "))
    Write-Output "UI: http://localhost:5000"
    exit 0
}

$proc = Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File", "`"$Supervisor`"" `
    -WindowStyle Hidden -PassThru

Set-Content -Path $PidFile -Value $proc.Id -Encoding ascii

Write-Output "Servicio iniciado (PID del supervisor: $($proc.Id))."
Write-Output "Va a tardar unos segundos en conectar y precalentar modelos -- revisar logs\tomografo_tool.log."
Write-Output "UI: http://localhost:5000"
Write-Output "Para detenerlo: powershell -ExecutionPolicy Bypass -File detener_servicio.ps1"
