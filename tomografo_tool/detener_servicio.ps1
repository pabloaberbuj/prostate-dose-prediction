# Detiene tomografo_tool sin importar como se haya arrancado: por la Tarea
# Programada "TomografoToolShadowTesting" (ver registrar_tarea_programada.ps1,
# forma normal desde que hay permiso de administrador) o "a mano" con
# iniciar_servicio.ps1 (fallback de cuando no habia permiso de admin).
#
# Hace falta cubrir ambos casos y matar el arbol completo -- no solo python.exe --
# porque run_supervisado.ps1 llama a python con "& $Python $App" (no Start-Process),
# asi que python.exe queda como hijo directo de ESE powershell.exe puntual, y ese
# powershell.exe corre en un while(true) que lo relanza solo si muere. Matar solo
# python.exe (sin el /T de taskkill, o sin tambien parar la Tarea) no alcanza.
#
# Uso:
#   powershell -ExecutionPolicy Bypass -File detener_servicio.ps1

$BaseDir  = $PSScriptRoot
$PidFile  = Join-Path $BaseDir "logs\servicio.pid"
$TaskName = "TomografoToolShadowTesting"

$detenidoAlgo = $false

# --- 1) Si esta corriendo via la Tarea Programada, pararla por ahi (Stop-ScheduledTask
#     mata la accion asociada -- el powershell.exe de run_supervisado.ps1 -- de raiz) ---
$tarea = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($tarea -and $tarea.State -eq "Running") {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    $detenidoAlgo = $true
}

# --- 2) PID file de iniciar_servicio.ps1 (arranque manual, sin Tarea Programada) ---
if (Test-Path $PidFile) {
    $procPid = Get-Content $PidFile -ErrorAction SilentlyContinue
    if ($procPid -and (Get-Process -Id $procPid -ErrorAction SilentlyContinue)) {
        taskkill /PID $procPid /T /F 2>$null | Out-Null
        $detenidoAlgo = $true
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

# --- 3) Red de seguridad final: cualquier powershell.exe de run_supervisado.ps1 o
#     python.exe de app.py que haya quedado suelto por fuera de los dos casos de
#     arriba (p.ej. arrancado a mano con Start-Process directo, como en una prueba) ---
$supervisoresSueltos = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.CommandLine -like "*run_supervisado.ps1*" }
foreach ($p in $supervisoresSueltos) {
    taskkill /PID $p.ProcessId /T /F 2>$null | Out-Null
    $detenidoAlgo = $true
}

$pythonSueltos = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*tomografo_tool*app.py*" }
foreach ($p in $pythonSueltos) {
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    $detenidoAlgo = $true
}

if ($detenidoAlgo) {
    Write-Output "Servicio detenido."
} else {
    Write-Output "No habia nada corriendo."
}
