"""Gera notebooks/07_caminhos_ABC.ipynb programaticamente."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "07_caminhos_ABC.ipynb"


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
        "# 07 — Três caminhos para isolar o sinal real\n\n"
        "Depois do nb06 confirmar viés grave de equipamento (probe F1m = 0.91), testamos 3 abordagens:\n\n"
        "- **A** — Repeated CV no Meditron-only (IC95% honesto do F1m sem viés).\n"
        "- **B** — Classificação por **ciclo respiratório** (crackles/wheezes) — alvo objetivo, "
        "dataset mais balanceado, anotado em todos os `.txt`.\n"
        "- **C** — **Residualização** de features sobre equipamento (remove componente linear "
        "explicada pelo dispositivo, retreina no dataset completo).\n\n"
        "Final: tabela comparativa dos resultados."
    ),
    code(
        "import sys, time, warnings\n"
        "from pathlib import Path\n"
        "PROJECT_ROOT = Path.cwd().parent\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from tqdm.auto import tqdm\n\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder\n"
        "from sklearn.linear_model import LinearRegression, LogisticRegression\n"
        "from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict\n"
        "from sklearn.neighbors import KNeighborsClassifier\n"
        "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n"
        "from sklearn.metrics import accuracy_score, f1_score, classification_report\n"
        "from xgboost import XGBClassifier\n\n"
        "from src import config, metadata, features\n"
        "import librosa\n\n"
        "warnings.filterwarnings('ignore')\n"
        "sns.set_theme(style='whitegrid')"
    ),
    code(
        "feats_df = pd.read_csv(config.PROCESSED_DIR / 'features.csv')\n"
        "FEATURE_COLS = [c for c in feats_df.columns\n"
        "                if c not in ('filename', 'patient_id', 'equipment',\n"
        "                             'chest_location', 'label_multi', 'label_binary')]\n"
        "X_full = feats_df[FEATURE_COLS].values\n"
        "le_y = LabelEncoder(); y_full = le_y.fit_transform(feats_df['label_multi'])\n"
        "g_full = feats_df['patient_id'].values\n"
        "CLASSES = list(le_y.classes_)\n"
        "print(f'Dataset: {X_full.shape}, classes={CLASSES}, pacientes={len(np.unique(g_full))}')"
    ),
    md(
        "---\n## A — Repeated 3-fold × 10 no Meditron-only\n\n"
        "Os 48 pacientes do Meditron (8 COPD + 26 Healthy + 14 URTI) são o cenário sem viés.\n"
        "3-fold porque COPD só tem 8 pacientes; 10 repetições com seeds diferentes pra IC."
    ),
    code(
        "mask = feats_df['equipment'] == 'WelchAllyn Meditron'\n"
        "feats_med = feats_df[mask].reset_index(drop=True)\n"
        "X_m = feats_med[FEATURE_COLS].values\n"
        "y_m = le_y.transform(feats_med['label_multi'])\n"
        "g_m = feats_med['patient_id'].values\n"
        "print(f'Meditron-only: {len(feats_med)} gravações, {len(np.unique(g_m))} pacientes')"
    ),
    code(
        "def make_models():\n"
        "    return {\n"
        "        'KNN':    Pipeline([('s', StandardScaler()),\n"
        "                             ('c', KNeighborsClassifier(n_neighbors=5, weights='distance'))]),\n"
        "        'LDA':    Pipeline([('s', StandardScaler()),\n"
        "                             ('c', LinearDiscriminantAnalysis())]),\n"
        "        'LogReg': Pipeline([('s', StandardScaler()),\n"
        "                             ('c', LogisticRegression(max_iter=3000, class_weight='balanced'))]),\n"
        "        'XGB':    Pipeline([('s', StandardScaler()),\n"
        "                             ('c', XGBClassifier(n_estimators=200, max_depth=3,\n"
        "                                                  learning_rate=0.05, eval_metric='mlogloss', n_jobs=1))]),\n"
        "    }\n\n"
        "def patient_metrics(y_true, y_pred, groups):\n"
        "    d = pd.DataFrame({'g': groups, 'y': y_true, 'p': y_pred})\n"
        "    a = d.groupby('g').agg(y=('y','first'), p=('p', lambda s: s.value_counts().idxmax()))\n"
        "    return a['y'].values, a['p'].values\n\n"
        "rows_A = []\n"
        "t0 = time.time()\n"
        "for rep in range(10):\n"
        "    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42+rep)\n"
        "    for name, model in make_models().items():\n"
        "        p = cross_val_predict(model, X_m, y_m, groups=g_m, cv=cv, n_jobs=1)\n"
        "        f1r = f1_score(y_m, p, average='macro')\n"
        "        yt, yp = patient_metrics(y_m, p, g_m)\n"
        "        f1p = f1_score(yt, yp, average='macro')\n"
        "        ap  = accuracy_score(yt, yp)\n"
        "        rows_A.append(dict(model=name, repeat=rep, f1m_rec=f1r, f1m_pat=f1p, acc_pat=ap))\n"
        "    if (rep+1) % 5 == 0:\n"
        "        print(f'  rep {rep+1}/10  ({time.time()-t0:.0f}s)')\n"
        "A_df = pd.DataFrame(rows_A)\n"
        "print(f'A — done ({time.time()-t0:.0f}s)')"
    ),
    code(
        "def summary_table(df, label):\n"
        "    s = df.groupby('model').agg(\n"
        "        f1m_pat_mu=('f1m_pat','mean'), f1m_pat_sd=('f1m_pat','std'),\n"
        "        acc_pat_mu=('acc_pat','mean'), acc_pat_sd=('acc_pat','std'),\n"
        "    ).round(3)\n"
        "    s['F1m paciente'] = s.apply(lambda r: f'{r.f1m_pat_mu:.3f} ± {r.f1m_pat_sd:.3f}', axis=1)\n"
        "    s['Acc paciente']  = s.apply(lambda r: f'{r.acc_pat_mu:.3f} ± {r.acc_pat_sd:.3f}', axis=1)\n"
        "    return s.sort_values('f1m_pat_mu', ascending=False)[['F1m paciente','Acc paciente']]\n\n"
        "summary_A = summary_table(A_df, 'A')\n"
        "print('A — Meditron-only (10×3-fold):')\n"
        "summary_A"
    ),
    md(
        "---\n## B — Classificação por ciclo respiratório\n\n"
        "Cada `.txt` lista intervalos `start, end, crackles, wheezes`. Cortamos cada gravação nos seus "
        "ciclos e extraímos as 78 features por **ciclo individual**. Rótulo 4-classe: "
        "`normal / crackles / wheezes / both`.\n\n"
        "Vantagens:\n"
        "- Muito mais amostras (~6.9k ciclos).\n"
        "- Classes mais balanceadas que diagnóstico.\n"
        "- Alvo objetivo (achado auscultaório), não inferido.\n\n"
        "Atenção: o **split continua por paciente**. Ciclos de um mesmo paciente não podem aparecer em treino+teste."
    ),
    code(
        "# Construir DF de ciclos com diagnóstico\n"
        "rec_df = metadata.build_recordings_dataframe()\n"
        "cycles_df = metadata.build_cycles_dataframe(rec_df)\n"
        "cycles_df['label_cycle'] = np.where(\n"
        "    (cycles_df['crackles']==0) & (cycles_df['wheezes']==0), 'normal',\n"
        "    np.where((cycles_df['crackles']==1) & (cycles_df['wheezes']==0), 'crackles',\n"
        "    np.where((cycles_df['crackles']==0) & (cycles_df['wheezes']==1), 'wheezes', 'both')))\n"
        "print(f'Ciclos totais: {len(cycles_df)}')\n"
        "print(cycles_df['label_cycle'].value_counts())"
    ),
    code(
        "# Extrair features por ciclo (1 load por wav, vários cortes) — cacheado em CSV\n"
        "CYC_CSV = config.PROCESSED_DIR / 'features_cycles.csv'\n\n"
        "if CYC_CSV.exists():\n"
        "    print(f'Cache encontrado: {CYC_CSV}')\n"
        "    cyc_feats = pd.read_csv(CYC_CSV)\n"
        "else:\n"
        "    # mapa wav -> filas do cycles_df\n"
        "    groups_per_wav = dict(list(cycles_df.groupby('filename')))\n"
        "    wav_paths = rec_df.set_index('filename')['wav_path'].to_dict()\n"
        "    rows = []\n"
        "    t0 = time.time()\n"
        "    for fname, grp in tqdm(groups_per_wav.items(), total=len(groups_per_wav), desc='áudios'):\n"
        "        try:\n"
        "            y, sr = librosa.load(wav_paths[fname], sr=config.SAMPLE_RATE, mono=True)\n"
        "        except Exception:\n"
        "            continue\n"
        "        for _, c in grp.iterrows():\n"
        "            feats = features.extract_segment(y, sr, float(c['start']), float(c['end']))\n"
        "            if feats is None:\n"
        "                continue\n"
        "            feats['filename'] = fname\n"
        "            feats['patient_id'] = int(c['patient_id'])\n"
        "            feats['cycle_idx'] = int(c['cycle_idx'])\n"
        "            feats['label_cycle'] = c['label_cycle']\n"
        "            feats['duration'] = float(c['duration'])\n"
        "            rows.append(feats)\n"
        "    cyc_feats = pd.DataFrame(rows)\n"
        "    cyc_feats.to_csv(CYC_CSV, index=False)\n"
        "    print(f'Extraído {len(cyc_feats)} ciclos em {time.time()-t0:.0f}s')\n\n"
        "print(cyc_feats.shape)\n"
        "print(cyc_feats['label_cycle'].value_counts())"
    ),
    code(
        "# Modelagem dos ciclos\n"
        "FCOLS_CYC = [c for c in cyc_feats.columns\n"
        "             if c not in ('filename','patient_id','cycle_idx','label_cycle','duration')]\n"
        "Xc = cyc_feats[FCOLS_CYC].values\n"
        "le_c = LabelEncoder(); yc = le_c.fit_transform(cyc_feats['label_cycle'])\n"
        "gc = cyc_feats['patient_id'].values\n"
        "CYC_CLASSES = list(le_c.classes_)\n"
        "print(f'Xc={Xc.shape}  classes={CYC_CLASSES}  pacientes={len(np.unique(gc))}')\n\n"
        "rows_B = []\n"
        "t0 = time.time()\n"
        "for rep in range(5):\n"
        "    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42+rep)\n"
        "    for name, model in make_models().items():\n"
        "        p = cross_val_predict(model, Xc, yc, groups=gc, cv=cv, n_jobs=1)\n"
        "        f1r = f1_score(yc, p, average='macro')\n"
        "        ar  = accuracy_score(yc, p)\n"
        "        rows_B.append(dict(model=name, repeat=rep, f1m_cyc=f1r, acc_cyc=ar))\n"
        "    print(f'  rep {rep+1}/5  ({time.time()-t0:.0f}s)')\n"
        "B_df = pd.DataFrame(rows_B)"
    ),
    code(
        "summary_B = B_df.groupby('model').agg(\n"
        "    f1_mu=('f1m_cyc','mean'), f1_sd=('f1m_cyc','std'),\n"
        "    acc_mu=('acc_cyc','mean'), acc_sd=('acc_cyc','std'),\n"
        ").round(3)\n"
        "summary_B['F1m ciclo'] = summary_B.apply(lambda r: f'{r.f1_mu:.3f} ± {r.f1_sd:.3f}', axis=1)\n"
        "summary_B['Acc ciclo']  = summary_B.apply(lambda r: f'{r.acc_mu:.3f} ± {r.acc_sd:.3f}', axis=1)\n"
        "summary_B = summary_B.sort_values('f1_mu', ascending=False)[['F1m ciclo','Acc ciclo']]\n"
        "print('B — classificação por ciclo (5×5-fold):')\n"
        "summary_B"
    ),
    md(
        "---\n## C — Residualização de features sobre equipamento\n\n"
        "Para cada feature, ajustamos uma regressão linear `feature ~ equipamento (one-hot)` "
        "e usamos os **resíduos** como novas features. Intuitivamente: removemos a parte da "
        "feature que é prevista pelo equipamento.\n\n"
        "Treinamos os 4 modelos sobre as features residualizadas no dataset completo (104 pacientes). "
        "Se F1m residualizado ≈ F1m original → o sinal não era apenas equipamento. Se cair muito → era atalho."
    ),
    code(
        "# Residualização \n"
        "ohe = OneHotEncoder(sparse_output=False, drop=None).fit(feats_df[['equipment']])\n"
        "E = ohe.transform(feats_df[['equipment']])\n"
        "lr = LinearRegression().fit(E, X_full)\n"
        "X_resid = X_full - lr.predict(E)\n"
        "print('Resíduos shape:', X_resid.shape)\n"
        "print('Std médio original :', X_full.std(0).mean().round(3))\n"
        "print('Std médio resíduos:', X_resid.std(0).mean().round(3))"
    ),
    code(
        "# Repeated CV no dataset completo COM features residualizadas\n"
        "rows_C = []\n"
        "t0 = time.time()\n"
        "for rep in range(10):\n"
        "    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42+rep)\n"
        "    for name, model in make_models().items():\n"
        "        p = cross_val_predict(model, X_resid, y_full, groups=g_full, cv=cv, n_jobs=1)\n"
        "        f1r = f1_score(y_full, p, average='macro')\n"
        "        yt, yp = patient_metrics(y_full, p, g_full)\n"
        "        f1p = f1_score(yt, yp, average='macro')\n"
        "        ap  = accuracy_score(yt, yp)\n"
        "        rows_C.append(dict(model=name, repeat=rep, f1m_rec=f1r, f1m_pat=f1p, acc_pat=ap))\n"
        "    if (rep+1) % 5 == 0:\n"
        "        print(f'  rep {rep+1}/10  ({time.time()-t0:.0f}s)')\n"
        "C_df = pd.DataFrame(rows_C)\n"
        "summary_C = summary_table(C_df, 'C')\n"
        "print('C — features residualizadas (dataset completo, 10×5-fold):')\n"
        "summary_C"
    ),
    md(
        "---\n## Comparação final"
    ),
    code(
        "# extrair média/sd numéricos para juntar tudo numa tabela\n"
        "def to_stats(df, col, suffix):\n"
        "    s = df.groupby('model')[col].agg(['mean','std']).round(3)\n"
        "    s.columns = [f'{suffix}_mu', f'{suffix}_sd']\n"
        "    return s\n\n"
        "tA = to_stats(A_df, 'f1m_pat', 'A_pat')\n"
        "tB = to_stats(B_df, 'f1m_cyc', 'B_cyc')\n"
        "tC = to_stats(C_df, 'f1m_pat', 'C_pat')\n\n"
        "# também: o nb04 (sem viés removido, sem ciclo) — recalculamos rapidamente para baseline\n"
        "rows_base = []\n"
        "for rep in range(10):\n"
        "    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42+rep)\n"
        "    for name, model in make_models().items():\n"
        "        p = cross_val_predict(model, X_full, y_full, groups=g_full, cv=cv, n_jobs=1)\n"
        "        yt, yp = patient_metrics(y_full, p, g_full)\n"
        "        rows_base.append(dict(model=name, repeat=rep,\n"
        "                               f1m_pat=f1_score(yt, yp, average='macro')))\n"
        "base_df = pd.DataFrame(rows_base)\n"
        "tBASE = to_stats(base_df, 'f1m_pat', 'BASE_pat')\n\n"
        "comp = tBASE.join(tA, how='outer').join(tC, how='outer').join(tB, how='outer')\n\n"
        "def fmt_pair(mu, sd):\n"
        "    if pd.isna(mu): return '—'\n"
        "    return f'{mu:.3f} ± {sd:.3f}'\n"
        "pretty = pd.DataFrame({\n"
        "    'BASELINE F1m pac. (full, com viés)':     [fmt_pair(comp.loc[m,'BASE_pat_mu'], comp.loc[m,'BASE_pat_sd']) for m in comp.index],\n"
        "    'A  Meditron-only F1m paciente':           [fmt_pair(comp.loc[m,'A_pat_mu'],    comp.loc[m,'A_pat_sd'])    for m in comp.index],\n"
        "    'C  Residualizado F1m paciente':           [fmt_pair(comp.loc[m,'C_pat_mu'],    comp.loc[m,'C_pat_sd'])    for m in comp.index],\n"
        "    'B  F1m por ciclo (tarefa diferente)':     [fmt_pair(comp.loc[m,'B_cyc_mu'],    comp.loc[m,'B_cyc_sd'])    for m in comp.index],\n"
        "}, index=comp.index).sort_index()\n"
        "pretty"
    ),
    code(
        "# Visualização — apenas as 3 condições comparáveis (mesmo alvo: diagnóstico)\n"
        "merge_long = []\n"
        "for cond, df, col in [('BASELINE (full)', base_df, 'f1m_pat'),\n"
        "                       ('A Meditron-only', A_df,    'f1m_pat'),\n"
        "                       ('C Residualizado', C_df,    'f1m_pat')]:\n"
        "    tmp = df[['model', col]].copy(); tmp.columns = ['model', 'F1m_pat']\n"
        "    tmp['cond'] = cond\n"
        "    merge_long.append(tmp)\n"
        "long_df = pd.concat(merge_long, ignore_index=True)\n\n"
        "fig, ax = plt.subplots(figsize=(11, 5))\n"
        "sns.boxplot(data=long_df, x='model', y='F1m_pat', hue='cond',\n"
        "             order=['KNN','LDA','LogReg','XGB'], ax=ax)\n"
        "ax.set_title('F1-macro paciente — efeito do viés de equipamento')\n"
        "ax.set_ylabel('F1-macro paciente'); ax.set_xlabel('')\n"
        "ax.axhline(0.62, color='red', linestyle='--', alpha=0.5, label='baseline trivial (sempre COPD)')\n"
        "ax.legend(loc='lower right')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## Como ler\n\n"
        "- **BASELINE → A**: se A < BASELINE, parte do número original era viés de equipamento. Se A ≥ BASELINE, o Meditron tem sinal suficiente sozinho.\n"
        "- **BASELINE → C**: se C cai com residualização → as features carregam informação de equipamento que era útil pra discriminar Healthy/URTI. Se C ≈ BASELINE → o sinal não-equipamento ainda permite separar.\n"
        "- **B**: tarefa diferente (crackles/wheezes por ciclo). Vale comparar entre modelos, **não** com diagnóstico.\n\n"
        "**Decisão prática:** o cenário mais honesto pro projeto é alguma combinação de A (Meditron-only) e C (residualizado). B é a tarefa que escala mais facilmente e pode ser o produto principal do assistente."
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
