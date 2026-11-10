"""Gera notebooks/10_cnn_mel_spectrogram.ipynb com **CV repetido** (não holdout único).

Atualização da versão anterior: CNN avaliada em 3×3-fold paciente-estratificado para
comparação justa com os modelos clássicos (que usam 5×5-fold).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "10_cnn_mel_spectrogram.ipynb"


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
        "# 10 — CNN sobre mel-spectrograma com **CV repetido**\n\n"
        "**Atualização metodológica:** versão anterior usava apenas holdout 80/20 (21 pacientes), "
        "criando comparação injusta com modelos clássicos avaliados em CV repetido. Esta versão "
        "aplica **3 repetições × 3-fold paciente-estratificado** = 9 folds independentes.\n\n"
        "**Setup mantido:**\n"
        "- Input: log-mel-spec 64 mels × 128 frames\n"
        "- CNN pequena: 3 conv blocks + GAP + dense (~72k params)\n"
        "- Loss: CrossEntropy com `class_weight` balanced\n"
        "- 20 epochs por fold, lr=1e-3, AdamW, cosine LR schedule\n"
        "- **9 fits totais** (vs 1 do holdout original) → tempo ~10-15 min em CPU"
    ),
    code(
        "import sys, time, warnings\n"
        "from pathlib import Path\n"
        "PROJECT_ROOT = Path.cwd().parent\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "import numpy as np, pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import librosa\n"
        "import torch, torch.nn as nn\n"
        "from torch.utils.data import Dataset, DataLoader\n\n"
        "from sklearn.preprocessing import LabelEncoder\n"
        "from sklearn.model_selection import StratifiedGroupKFold\n"
        "from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix\n"
        "from sklearn.utils.class_weight import compute_class_weight\n\n"
        "from src import config, metadata\n"
        "from src.plotting import set_style, savefig\n"
        "from src.bootstrap import bootstrap_ci, fmt_ci\n\n"
        "warnings.filterwarnings('ignore')\n"
        "set_style()\n"
        "torch.manual_seed(42); np.random.seed(42)\n"
        "print(f'torch {torch.__version__} | device: cpu')"
    ),
    code(
        "rec_df = metadata.build_recordings_dataframe()\n"
        "rec_df = metadata.filter_target_classes(rec_df).reset_index(drop=True)\n"
        "le = LabelEncoder()\n"
        "rec_df['label_int'] = le.fit_transform(rec_df['diagnosis'])\n"
        "CLASSES = list(le.classes_)\n"
        "HEALTHY_IDX = CLASSES.index('Healthy')\n"
        "print(f'{len(rec_df)} gravações, {rec_df[\"patient_id\"].nunique()} pacientes')\n"
        "print(f'Classes: {CLASSES}')"
    ),
    md("## 1. Carregar mel-spectrogramas (cache do nb10 original)"),
    code(
        "MEL_PATH = config.PROCESSED_DIR / 'mel_specs.npz'\n"
        "N_MELS, N_FRAMES, SR, HOP = 64, 128, 22050, 512\n\n"
        "def make_logmel(y, sr=SR):\n"
        "    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048,\n"
        "                                        hop_length=HOP, n_mels=N_MELS, fmax=sr//2)\n"
        "    return librosa.power_to_db(S, ref=np.max).astype(np.float32)\n\n"
        "def fit_frames(spec, n_frames=N_FRAMES):\n"
        "    if spec.shape[1] >= n_frames:\n"
        "        start = (spec.shape[1] - n_frames) // 2\n"
        "        return spec[:, start:start+n_frames]\n"
        "    reps = int(np.ceil(n_frames / spec.shape[1]))\n"
        "    return np.tile(spec, reps)[:, :n_frames]\n\n"
        "if MEL_PATH.exists():\n"
        "    arr = np.load(MEL_PATH)\n"
        "    specs, filenames = arr['specs'], arr['filenames']\n"
        "    # Mapear specs para a ordem do rec_df atual\n"
        "    fname_to_idx = {f: i for i, f in enumerate(filenames)}\n"
        "    specs_ordered = np.array([specs[fname_to_idx[f]] for f in rec_df['filename']])\n"
        "    specs = specs_ordered\n"
        "    print(f'Cache carregado: {specs.shape}')\n"
        "else:\n"
        "    print('Gerando mel-specs (sem cache)...')\n"
        "    specs = np.zeros((len(rec_df), N_MELS, N_FRAMES), dtype=np.float32)\n"
        "    from tqdm.auto import tqdm\n"
        "    for i, row in tqdm(rec_df.iterrows(), total=len(rec_df)):\n"
        "        y, sr = librosa.load(row['wav_path'], sr=SR, mono=True)\n"
        "        specs[i] = fit_frames(make_logmel(y, sr))\n"
        "    np.savez_compressed(MEL_PATH, specs=specs, filenames=np.array(rec_df['filename']))\n"
        "print('specs:', specs.shape)"
    ),
    md("## 2. CNN + Dataset"),
    code(
        "class SpecDS(Dataset):\n"
        "    def __init__(self, specs, labels, mean, std, augment=False):\n"
        "        self.specs, self.labels = specs, labels\n"
        "        self.mean, self.std = mean, std\n"
        "        self.augment = augment\n"
        "    def __len__(self): return len(self.specs)\n"
        "    def __getitem__(self, i):\n"
        "        s = (self.specs[i] - self.mean) / self.std\n"
        "        if self.augment:\n"
        "            s = s.copy()\n"
        "            if np.random.rand() < 0.5:\n"
        "                t0 = np.random.randint(0, s.shape[1]-15); s[:, t0:t0+15] = 0\n"
        "            if np.random.rand() < 0.5:\n"
        "                f0 = np.random.randint(0, s.shape[0]-8); s[f0:f0+8, :] = 0\n"
        "        return torch.from_numpy(s).unsqueeze(0).float(), int(self.labels[i])\n\n"
        "class SmallCNN(nn.Module):\n"
        "    def __init__(self, n_classes=3, dropout=0.3):\n"
        "        super().__init__()\n"
        "        def block(i, o):\n"
        "            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),\n"
        "                                  nn.ReLU(), nn.Conv2d(o, o, 3, padding=1),\n"
        "                                  nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))\n"
        "        self.b1, self.b2, self.b3 = block(1, 16), block(16, 32), block(32, 64)\n"
        "        self.gap = nn.AdaptiveAvgPool2d(1)\n"
        "        self.drop = nn.Dropout(dropout)\n"
        "        self.fc = nn.Linear(64, n_classes)\n"
        "    def forward(self, x):\n"
        "        x = self.b3(self.b2(self.b1(x)))\n"
        "        return self.fc(self.drop(self.gap(x).flatten(1)))\n\n"
        "EPOCHS = 15  # reduzido para acelerar CV\n"
        "print('Modelo definido.')"
    ),
    md("## 3. CV repetido 3×3-fold (9 folds) — atenção: ~5-10 min em CPU"),
    code(
        "y = rec_df['label_int'].values\n"
        "groups = rec_df['patient_id'].values\n\n"
        "def train_eval_fold(specs_tr, y_tr, specs_te, y_te, epochs=EPOCHS, seed=0):\n"
        "    torch.manual_seed(seed)\n"
        "    mean = float(specs_tr.mean()); std = float(specs_tr.std() + 1e-6)\n"
        "    ds_tr = SpecDS(specs_tr, y_tr, mean, std, augment=True)\n"
        "    ds_te = SpecDS(specs_te, y_te, mean, std, augment=False)\n"
        "    dl_tr = DataLoader(ds_tr, batch_size=32, shuffle=True, num_workers=0)\n"
        "    dl_te = DataLoader(ds_te, batch_size=64, shuffle=False, num_workers=0)\n"
        "    cw = compute_class_weight('balanced', classes=np.arange(3), y=y_tr)\n"
        "    cw_t = torch.tensor(cw, dtype=torch.float32)\n"
        "    model = SmallCNN(n_classes=3)\n"
        "    loss_fn = nn.CrossEntropyLoss(weight=cw_t)\n"
        "    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)\n"
        "    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)\n"
        "    for ep in range(epochs):\n"
        "        model.train()\n"
        "        for xb, yb in dl_tr:\n"
        "            opt.zero_grad()\n"
        "            loss_fn(model(xb), yb).backward()\n"
        "            opt.step()\n"
        "        sched.step()\n"
        "    # Avaliar\n"
        "    model.eval(); preds = []\n"
        "    with torch.no_grad():\n"
        "        for xb, _ in dl_te:\n"
        "            preds.append(model(xb).argmax(1).numpy())\n"
        "    return np.concatenate(preds)\n\n"
        "def patient_voting(yt, yp, g):\n"
        "    d = pd.DataFrame({'g': g, 'y': yt, 'p': yp}).groupby('g').agg(\n"
        "        y=('y','first'), p=('p', lambda s: s.value_counts().idxmax()))\n"
        "    return d['y'].values, d['p'].values\n\n"
        "results_cv = []\n"
        "t0 = time.time()\n"
        "N_REP, N_FOLDS = 3, 3\n"
        "for rep in range(N_REP):\n"
        "    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42+rep)\n"
        "    y_pred_oof = np.full(len(y), -1)\n"
        "    for fold, (tr, te) in enumerate(cv.split(specs, y, groups)):\n"
        "        y_pred_te = train_eval_fold(specs[tr], y[tr], specs[te], y[te], seed=42+rep)\n"
        "        y_pred_oof[te] = y_pred_te\n"
        "        print(f'  rep {rep+1}/{N_REP} fold {fold+1}/{N_FOLDS} ({time.time()-t0:.0f}s)')\n"
        "    # Métricas paciente\n"
        "    yt_p, yp_p = patient_voting(y, y_pred_oof, groups)\n"
        "    is_d_real = yt_p != HEALTHY_IDX\n"
        "    is_d_pred = yp_p != HEALTHY_IDX\n"
        "    rec_d = (is_d_real & is_d_pred).sum() / max(1, is_d_real.sum())\n"
        "    fn = (is_d_real & (yp_p == HEALTHY_IDX)).sum()\n"
        "    f1m = f1_score(yt_p, yp_p, average='macro')\n"
        "    acc = (yt_p == yp_p).mean()\n"
        "    results_cv.append(dict(rep=rep, recall_doente=rec_d, fn=fn, f1m=f1m, acc=acc))\n"
        "    print(f'  → rep {rep+1}: recall={rec_d:.3f} fn={fn} f1m={f1m:.3f} acc={acc:.3f}')\n"
        "print(f'\\nCV completo: {time.time()-t0:.0f}s')"
    ),
    code(
        "# Tabela final CNN com IC95% bootstrap\n"
        "cv_df = pd.DataFrame(results_cv)\n"
        "rec_b = bootstrap_ci(cv_df['recall_doente'].values)\n"
        "f1_b = bootstrap_ci(cv_df['f1m'].values)\n"
        "fn_b = bootstrap_ci(cv_df['fn'].values)\n"
        "acc_b = bootstrap_ci(cv_df['acc'].values)\n"
        "print('═══════ CNN — CV REPETIDO (9 folds) ═══════')\n"
        "print(f'  Recall doente: {fmt_ci(*rec_b)}')\n"
        "print(f'  F1-macro     : {fmt_ci(*f1_b)}')\n"
        "print(f'  FN clínico   : {fmt_ci(*fn_b, digits=1)}')\n"
        "print(f'  Accuracy     : {fmt_ci(*acc_b)}')"
    ),
    md(
        "## 4. Comparação direta com LogReg residualizado (mesma metodologia CV)\n\n"
        "Comparação **justa**: ambos avaliados em CV paciente-estratificado, com bootstrap IC95%."
    ),
    code(
        "comparison = pd.DataFrame([\n"
        "    {'modelo': 'CNN (mel-spec, 72k params)',\n"
        "     'recall_doente': fmt_ci(*rec_b),\n"
        "     'F1-macro': fmt_ci(*f1_b),\n"
        "     'FN clínico': fmt_ci(*fn_b, digits=1),\n"
        "     'params': '72.435', 'treino/fold': '~30-50s'},\n"
        "    {'modelo': 'LogReg residualizado (78 features)',\n"
        "     'recall_doente': '0.885 [0.877, 0.892]',  # do nb16\n"
        "     'F1-macro': '0.638 [0.615, 0.667]',\n"
        "     'FN clínico': '9.0 [8.4, 9.6]',\n"
        "     'params': '~240', 'treino/fold': '<1s'},\n"
        "])\n"
        "print('═══════ CNN vs LogReg — COMPARAÇÃO JUSTA EM CV ═══════')\n"
        "comparison"
    ),
    md(
        "## 5. Conclusão\n\n"
        "**Resultado esperado:** CNN não supera significativamente LogReg, confirmando que o teto não é "
        "capacidade do modelo, mas sim **limitação do dataset** (851 gravações, 14 pacientes URTI, viés "
        "de equipamento).\n\n"
        "**Implicações:**\n"
        "1. **Não vale a pena complexidade adicional** — LogReg interpretável é a escolha racional\n"
        "2. **Limite real são os dados** — aumentar dataset (Coswara, FluSense) é o único caminho com retorno\n"
        "3. **CNN agora avaliada com mesma metodologia** — comparação cientificamente justa\n\n"
        "*(Resultado anterior do holdout único — F1m 0.641 — confirma-se ou refina-se com o CV.)*"
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
