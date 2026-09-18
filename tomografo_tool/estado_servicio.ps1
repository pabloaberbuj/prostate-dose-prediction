# Estado rapido de tomografo_tool arrancado con iniciar_servicio.ps1.
# Uso: powershell -ExecutionPolicy Bypass -File estado_servicio.ps1

$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*tomografo_tool*app.py*" }

if (-not $procs) {
    Write-Output "NO esta corriendo."
    exit 1
}

Write-Output "Corriendo -- PID(s): $($procs.ProcessId -join ', ')"
Write-Output "UI: http://localhost:5000"

$LogPath = Join-Path $PSScriptRoot "logs\tomografo_tool.log"
if (Test-Path $LogPath) {
    Write-Output "`nUltimas lineas del log:"
    Get-Content $LogPath -Tail 5
}
