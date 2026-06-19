# Setup — Etapa 0

Guía paso a paso para dejar el entorno funcionando en PC local y en la nube. Se completa una sola vez por máquina.

## 1. Cuentas necesarias

- **GitHub** (o GitLab/Bitbucket): para alojar el repo. Crear el repo vacío `prostate-dose-prediction` antes de empezar.
- **Weights & Biases** (https://wandb.ai): cuenta gratuita. Anotar el `entity` (tu username) y guardar la API key (Settings → API keys).
- **Kaggle** (https://kaggle.com): cuenta + verificación por teléfono (necesario para acceso a GPU). En Settings → API → Create New Token, descargar `kaggle.json`.

## 2. Setup local (PC con RTX PRO 2000)

### 2.1. Clonar el repo
```bash
git clone git@github.com:<tu-usuario>/prostate-dose-prediction.git
cd prostate-dose-prediction
```

### 2.2. Crear entorno virtual
Opción A — venv (más simple):
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

Opción B — conda:
```bash
conda create -n prostate python=3.11 -y
conda activate prostate
```

### 2.3. Instalar PyTorch con soporte CUDA correcto
Antes de `pip install -r requirements.txt`, instalar PyTorch matcheando tu CUDA. Verificar versión de CUDA:
```bash
nvidia-smi
```
Mirar la línea "CUDA Version" (ej. 12.4). Ir a https://pytorch.org/get-started/locally/ y copiar el comando recomendado. Ejemplo para CUDA 12.4:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2.4. Instalar el resto
```bash
pip install -r requirements.txt
```

### 2.5. Login en W&B
```bash
wandb login
# pegar la API key cuando pida
```

### 2.6. Verificar
```bash
python scripts/hello_world.py
```
Esperado: 5 epochs corren sin error, detecta la RTX PRO 2000, loss desciende, y aparece la corrida en wandb.ai en el proyecto `prostate-dose-prediction`.

## 3. Setup en Kaggle

### 3.1. Crear notebook
- New Notebook → Settings (panel derecho):
  - Accelerator: **GPU T4 x2** (o P100 si está disponible).
  - Internet: **On**.
  - Persistence: Files only.

### 3.2. Instalar W&B y loguear
En la primera celda del notebook:
```python
!pip install -q wandb pytorch-lightning omegaconf
import wandb
wandb.login(key="<tu-api-key>")   # o usar Kaggle Secrets para no exponerla
```

Mejor práctica con secrets:
- Add-ons → Secrets → Add secret → Label: `WANDB_API_KEY`, Value: tu key.
- En código:
```python
from kaggle_secrets import UserSecretsClient
wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
```

### 3.3. Clonar repo y ejecutar
```python
!git clone https://github.com/<tu-usuario>/prostate-dose-prediction.git
%cd prostate-dose-prediction
!python scripts/hello_world.py
```
Esperado: corre en T4, loguea a W&B, la corrida aparece junto a las locales.

## 4. Setup del proyecto C# para ESAPI

Esto va en una carpeta separada del repo principal (o repo aparte si preferís), porque depende del entorno Eclipse y no se ejecuta en la PC de entrenamiento.

```
data/extract_dicom_csharp/
├── ExtractDicom.sln
├── ExtractDicom/
│   ├── Program.cs
│   ├── DicomAnonymizer.cs
│   ├── EsapiMetricsExtractor.cs
│   └── ExtractDicom.csproj
└── README.md
```

Lo armamos en la **Etapa 1**, no ahora.

## 5. Checklist final de etapa 0

- [ ] Repo creado en GitHub y clonado en PC local.
- [ ] Entorno virtual creado e instalado en PC local.
- [ ] `hello_world.py` corre en PC local con GPU detectada.
- [ ] Corrida visible en W&B desde PC local.
- [ ] Cuenta Kaggle con GPU habilitada.
- [ ] `hello_world.py` corre en Kaggle.
- [ ] Corrida de Kaggle visible en W&B (en el mismo proyecto).
- [ ] README leído y revisado.

Cuando todos los checks estén ✓, pasamos a Etapa 1.
