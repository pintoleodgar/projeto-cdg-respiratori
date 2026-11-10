import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Pasta do dataset ICBHI 2017. Noutra máquina, definir a variável de ambiente
# ICBHI_ROOT em vez de editar este ficheiro.
ICBHI_ROOT = Path(os.environ.get("ICBHI_ROOT", r"C:\Datasets\Respiratory_Sound_Database"))
AUDIO_DIR = ICBHI_ROOT / "audio_and_txt_files"
DIAGNOSIS_CSV = ICBHI_ROOT / "patient_diagnosis.csv"
DEMOGRAPHIC_TXT = ICBHI_ROOT / "demographic_info.txt"

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "lung_assistant_v3.npz"   # modelo de produção


def ensure_dirs() -> None:
    """Cria as pastas de saída. Chamar antes de escrever — importar este
    módulo já não cria nada em disco."""
    for d in (PROCESSED_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 22050
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

# Janela fixa de análise: todas as gravações são cortadas aos primeiros
# MAX_DURATION_S segundos antes da extração. No ICBHI, as gravações COPD do
# Meditron têm 32–86 s e todas as Healthy/URTI ~20 s — sem o corte, a duração
# (e tudo o que dela depende) funciona como atalho para o diagnóstico.
MAX_DURATION_S = 20.0

# Abaixo disto a gravação é recusada na inferência (a mais curta do treino tem ~7,9 s).
MIN_DURATION_S = 1.0

# Colunas de features.csv que NÃO são features do modelo.
# `duration_s` fica como metadado: é artefacto do protocolo de gravação.
NON_FEATURE_COLS = (
    "filename", "patient_id", "equipment", "chest_location",
    "label_multi", "label_binary", "duration_s",
)
