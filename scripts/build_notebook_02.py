"""Gera notebooks/02_extracao_completa.ipynb programaticamente."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "02_extracao_completa.ipynb"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


cells = [
    md(
        "# 02 — Extração completa de features (851 gravações)\n\n"
        "Engenharia de features para classificação multiclasse Healthy / COPD / URTI:\n\n"
        "- **MFCC** (13 coef.) — mean + std\n"
        "- **Delta-MFCC** (1ª derivada) — mean\n"
        "- **Delta²-MFCC** (2ª derivada) — mean\n"
        "- **Chroma STFT** (12 bins) — mean\n"
        "- **Spectral contrast** (7 bandas) — mean\n"
        "- **Features espectrais**: centroide, rolloff, bandwidth, flatness\n"
        "- **Energia**: RMS, ZCR\n\n"
        "Total: **78 features** por gravação. Persistido em `data/processed/features.csv`."
    ),
    code(
        "import sys, time\n"
        "from pathlib import Path\n"
        "PROJECT_ROOT = Path.cwd().parent\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from tqdm.auto import tqdm\n\n"
        "from src import config, metadata, features\n\n"
        "sns.set_theme(style='whitegrid')\n"
        "pd.set_option('display.max_columns', 30)"
    ),
    md("## 1. Carregar metadata filtrada (3 classes)"),
    code(
        "df = metadata.build_recordings_dataframe()\n"
        "df = metadata.filter_target_classes(df)\n"
        "df = metadata.add_target_labels(df)\n"
        "print(f'Gravações: {len(df)} | Pacientes: {df[\"patient_id\"].nunique()}')\n"
        "df['label_multi'].value_counts()"
    ),
    md(
        "## 2. Extração de features\n\n"
        "Rodamos `features.extract_from_file` em todas as 851 gravações. Tempo esperado: ~3–8 min "
        "dependendo da máquina. As features são salvas em CSV para reuso nos próximos notebooks."
    ),
    code(
        "FEATURES_CSV = config.PROCESSED_DIR / 'features.csv'\n\n"
        "if FEATURES_CSV.exists():\n"
        "    print(f'Carregando features cacheadas de {FEATURES_CSV}')\n"
        "    feats_df = pd.read_csv(FEATURES_CSV)\n"
        "else:\n"
        "    t0 = time.time()\n"
        "    rows = []\n"
        "    for _, r in tqdm(df.iterrows(), total=len(df), desc='Extraindo'):\n"
        "        f = features.extract_from_file(r['wav_path'])\n"
        "        f['filename'] = r['filename']\n"
        "        f['patient_id'] = r['patient_id']\n"
        "        f['equipment'] = r['equipment']\n"
        "        f['chest_location'] = r['chest_location']\n"
        "        f['label_multi'] = r['label_multi']\n"
        "        f['label_binary'] = r['label_binary']\n"
        "        rows.append(f)\n"
        "    feats_df = pd.DataFrame(rows)\n"
        "    feats_df.to_csv(FEATURES_CSV, index=False)\n"
        "    print(f'Extraído em {time.time()-t0:.1f}s | shape: {feats_df.shape}')\n"
        "    print(f'Salvo em: {FEATURES_CSV}')\n\n"
        "feats_df.head()"
    ),
    md("## 3. Sanity check — sem NaN / Inf"),
    code(
        "num_cols = feats_df.select_dtypes(include='number').columns\n"
        "n_nan = feats_df[num_cols].isna().sum().sum()\n"
        "n_inf = np.isinf(feats_df[num_cols]).sum().sum()\n"
        "print(f'NaN: {n_nan} | Inf: {n_inf}')\n"
        "print(f'Shape final: {feats_df.shape}')\n"
        "feats_df.describe().T.head(10)"
    ),
    md(
        "## 4. Distribuição de features por classe\n\n"
        "Visualização de algumas features-chave para confirmar separabilidade — "
        "se classes se sobrepõem totalmente em todas as dimensões, modelos clássicos vão sofrer."
    ),
    code(
        "key_feats = ['mfcc1_mean', 'mfcc2_mean', 'mfcc3_mean', 'mfcc4_mean',\n"
        "             'centroid_mean', 'rolloff_mean', 'bandwidth_mean',\n"
        "             'zcr_mean', 'rms_mean', 'flatness_mean']\n"
        "fig, axes = plt.subplots(2, 5, figsize=(18, 7))\n"
        "for ax, col in zip(axes.flat, key_feats):\n"
        "    for cls, color in zip(['Healthy', 'COPD', 'URTI'], ['#2ecc71', '#e74c3c', '#3498db']):\n"
        "        vals = feats_df[feats_df['label_multi'] == cls][col]\n"
        "        ax.hist(vals, bins=25, alpha=0.55, label=cls, color=color)\n"
        "    ax.set_title(col, fontsize=10)\n"
        "axes[0, 0].legend(fontsize=8)\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("## 5. Correlação entre features"),
    code(
        "mfcc_cols = [c for c in feats_df.columns if c.startswith('mfcc') and c.endswith('_mean')]\n"
        "corr = feats_df[mfcc_cols].corr()\n"
        "plt.figure(figsize=(10, 7))\n"
        "sns.heatmap(corr, cmap='coolwarm', center=0, annot=True, fmt='.2f',\n"
        "            cbar_kws={'label': 'corr'})\n"
        "plt.title('Correlação entre MFCCs (mean)')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## Próximo passo\n\n"
        "**Notebook 03** — treinar XGBoost, LDA, Regressão Logística e KNN sobre `features.csv` "
        "com `StratifiedGroupKFold(groups=patient_id)` e matriz de confusão a nível de paciente."
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Gerado: {OUT}")
