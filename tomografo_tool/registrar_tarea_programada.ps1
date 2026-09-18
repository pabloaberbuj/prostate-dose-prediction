# Registra la tarea programada de Windows para correr tomografo_tool de forma
# desatendida: al iniciar sesion del usuario actual (NO "at system startup" como
# SYSTEM -- la cuenta SYSTEM tipicamente no tiene las credenciales/mapeo para acceder
# a un recurso de red \\server\share$ que si estan disponibles en la sesion
# interactiva del usuario), con un delay de 60s en el trigger (el stack de red/SMB
# puede no estar listo justo al loguearse). Corre run_supervisado.ps1, que a su vez
# relanza app.py con backoff si se cae.
#
# Uso (una sola vez, o para actualizar la tarea si cambia algo aca):
#   powershell -ExecutionPolicy Bypass -File registrar_tarea_programada.ps1

$TaskName = "TomografoToolShadowTesting"
$ScriptPath = "C:\Pablo\ProstateDoseProject\repo\tomografo_tool\run_supervisado.ps1"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$ScriptPath`""

$trigger = New-ScheduledTaskTrigger -AtLogOn
$trigger.Delay = "PT60S"   # 60s de margen para que el stack de red este listo

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0)   # 0 = sin limite de tiempo (corre indefinidamente)

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Output "Tarea '$TaskName' ya existe -- actualizando."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Shadow testing tomografo_tool -- monitorea \\ARIAMEVADB-SVR\va_data$\DICOM y corre inferencia automatica de constraints de prostata (Proyecto 1). Solo lectura del share, nunca mueve/modifica DICOM." `
    | Out-Null

Write-Output "Tarea registrada: $TaskName"
Get-ScheduledTask -TaskName $TaskName | Format-List TaskName, State
Write-Output "Para desregistrarla: Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
