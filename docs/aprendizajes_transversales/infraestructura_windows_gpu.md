---
tags: [prostatedoseproject, aprendizajes, infraestructura]
estado: vigente
---

# Infraestructura Windows / GPU

Reglas operativas aprendidas por experiencia directa en esta máquina, no documentadas en ningún lado del código — perderlas significa repetir incidentes ya resueltos.

## No leer checkpoints mientras se entrena

`torch.load` sobre un `.ckpt` que Lightning está escribiendo activamente puede romper silenciosamente el guardado del checkpoint en Windows. Evitar inspeccionar/copiar/abrir checkpoints de un training en curso.

## Cap de memoria CUDA obligatorio

Sin un cap explícito de fracción de memoria CUDA, el driver WDDM de Windows empieza a paginar a RAM del sistema en vez de fallar con OOM — esto puede **congelar la PC entera**, no solo el proceso de Python. Mantener el cap configurado (actualmente 0.88) en cualquier script de entrenamiento nuevo.

## TaskStop no mata procesos nativos huérfanos

Detener la tarea desde el harness de Claude Code no garantiza que procesos nativos (CUDA, workers de DataLoader) hijos del proceso Python terminen. Verificar con Task Manager / `nvidia-smi` y confirmar antes de asumir que la GPU quedó libre.

## VM compartida (Hyper-V) puede hacer todo 3-4x más lento

Hay una VM Hyper-V siempre activa (~16GB) en esta máquina que compite por recursos. Antes de diagnosticar una regresión de performance en el código, descartar que la causa sea la VM compartida.

## Gradient checkpointing para pacientes de Z grande

Ver [[decisiones_diseno_modelo]] — mismo aprendizaje, catalogado ahí porque es una decisión de diseño de modelo, no solo un ajuste de infraestructura.

## Machine config sin límites cinemáticos reales

El `machine_config` usado en comisionamiento no tenía límites cinemáticos reales configurados (pydosert caía en defaults silenciosos). Se corrigieron los valores reales de la máquina; el límite de MU/grado sigue sin modelarse explícitamente. Ver [[../subproyectos/06_comisionamiento_pdrt]].
