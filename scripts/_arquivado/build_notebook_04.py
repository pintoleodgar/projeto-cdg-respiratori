"""Gera notebooks/04_repeated_cv.ipynb programaticamente."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "04_repeated_cv.ipynb"


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
        "# 04 — Repeated Stratified Group K-Fold (10×5)\n\n"
        "**Motivação:** com 14 pacientes URTI e 26 Healthy, um único 5-fold tem alta variância — "
        "trocar o `random_state` pode mudar o F1-macro em ±5–10 pontos. Reportamos média ± desvio "
        "ao longo de **10 repetições** com seeds diferentes para ter intervalos de confiança honestos.\n\n"
        "**Notas técnicas:**\n"
        "- `sklearn` não tem `RepeatedStratifiedGroupKFold` nativo — implementamos no loop.\n"
        "- Avaliação em dois níveis: **gravação** (851 amostras OOF) e **paciente** (voto majoritário, 104 pacientes).\n"
        "- Adicionamos `DummyClassifier` (sempre COPD) como baseline trivial.\n"
        "- **Wilcoxon pareado** sobre os 10 F1-macros para teste de significância entre modelos."
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
        "from scipy import stats\n\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.preprocessing import StandardScaler, LabelEncoder\n"
        "from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict\n"
        "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.neighbors import KNeighborsClassifier\n"
        "from sklearn.dummy import DummyClassifier\n"
        "from sklearn.metrics import accuracy_score, f1_score, recall_score\n"
        "from xgboost import XGBClassifier\n\n"
        "from src import config\n\n"
        "sns.set_theme(style='whitegrid')\n"
        "RNG = np.random.default_rng(42)"
    ),
    md("## 1. Dados"),
    code(
        "feats_df = pd.read_csv(config.PROCESSED_DIR / 'features.csv')\n"
        "FEATURE_COLS = [c for c in feats_df.columns\n"
        "                if c not in ('filename', 'patient_id', 'equipment',\n"
        "                             'chest_location', 'label_multi', 'label_binary')]\n"
        "X = feats_df[FEATURE_COLS].values\n"
        "le = LabelEncoder()\n"
        "y = le.fit_transform(feats_df['label_multi'])\n"
        "groups = feats_df['patient_id'].values\n"
        "CLASSES = list(le.classes_)\n"
        "print(f'X: {X.shape} | classes: {CLASSES}')\n"
        "print(f'Pacientes: {len(np.unique(groups))}')"
    ),
    md("## 2. Modelos"),
    code(
        "def make_models():\n"
        "    return {\n"
        "        'Dummy (sempre COPD)': Pipeline([\n"
        "            ('clf', DummyClassifier(strategy='most_frequent')),\n"
        "        ]),\n"
        "        'XGBoost': Pipeline([\n"
        "            ('scaler', StandardScaler()),\n"
        "            ('clf', XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,\n"
        "                                   eval_metric='mlogloss', n_jobs=1)),\n"
        "        ]),\n"
        "        'LDA': Pipeline([\n"
        "            ('scaler', StandardScaler()),\n"
        "            ('clf', LinearDiscriminantAnalysis()),\n"
        "        ]),\n"
        "        'LogReg': Pipeline([\n"
        "            ('scaler', StandardScaler()),\n"
        "            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced')),\n"
        "        ]),\n"
        "        'KNN': Pipeline([\n"
        "            ('scaler', StandardScaler()),\n"
        "            ('clf', KNeighborsClassifier(n_neighbors=5, weights='distance')),\n"
        "        ]),\n"
        "    }\n\n"
        "list(make_models().keys())"
    ),
    md(
        "## 3. Loop Repeated CV (10 × 5 folds)\n\n"
        "Para cada (modelo, repetição): roda `cross_val_predict` com `StratifiedGroupKFold` "
        "(seed = repetição). Calcula métricas em gravação E em paciente (voto majoritário)."
    ),
    code(
        "def aggregate_to_patient(y_true, y_pred, patient_ids):\n"
        "    df = pd.DataFrame({'pid': patient_ids, 'y': y_true, 'pred': y_pred})\n"
        "    agg = df.groupby('pid').agg(\n"
        "        y=('y', 'first'),\n"
        "        pred=('pred', lambda s: s.value_counts().idxmax()),\n"
        "    )\n"
        "    return agg['y'].values, agg['pred'].values\n\n"
        "N_REPEATS = 10\n"
        "N_SPLITS = 5\n\n"
        "rows = []\n"
        "t0 = time.time()\n"
        "for rep in range(N_REPEATS):\n"
        "    seed = 42 + rep\n"
        "    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)\n"
        "    for name, model in make_models().items():\n"
        "        y_pred = cross_val_predict(model, X, y, groups=groups, cv=cv, n_jobs=-1)\n"
        "        # nível gravação\n"
        "        acc_r = accuracy_score(y, y_pred)\n"
        "        f1m_r = f1_score(y, y_pred, average='macro')\n"
        "        # nível paciente\n"
        "        y_t_p, y_p_p = aggregate_to_patient(y, y_pred, groups)\n"
        "        acc_p = accuracy_score(y_t_p, y_p_p)\n"
        "        f1m_p = f1_score(y_t_p, y_p_p, average='macro')\n"
        "        rec_p = recall_score(y_t_p, y_p_p, average='macro')\n"
        "        rows.append(dict(model=name, repeat=rep, seed=seed,\n"
        "                          acc_rec=acc_r, f1m_rec=f1m_r,\n"
        "                          acc_pat=acc_p, f1m_pat=f1m_p, recall_pat=rec_p))\n"
        "    print(f'rep {rep+1}/{N_REPEATS}  ({time.time()-t0:.0f}s)')\n\n"
        "rep_df = pd.DataFrame(rows)\n"
        "print(f'\\nTotal: {time.time()-t0:.0f}s | rows: {len(rep_df)}')\n"
        "rep_df.head()"
    ),
    md("## 4. Médias ± desvios (10 repetições)"),
    code(
        "def fmt(mu, sd): return f'{mu:.3f} ± {sd:.3f}'\n\n"
        "summary = rep_df.groupby('model').agg(\n"
        "    acc_rec_mu=('acc_rec', 'mean'), acc_rec_sd=('acc_rec', 'std'),\n"
        "    f1m_rec_mu=('f1m_rec', 'mean'), f1m_rec_sd=('f1m_rec', 'std'),\n"
        "    acc_pat_mu=('acc_pat', 'mean'), acc_pat_sd=('acc_pat', 'std'),\n"
        "    f1m_pat_mu=('f1m_pat', 'mean'), f1m_pat_sd=('f1m_pat', 'std'),\n"
        "    rec_pat_mu=('recall_pat', 'mean'), rec_pat_sd=('recall_pat', 'std'),\n"
        ").sort_values('f1m_pat_mu', ascending=False)\n\n"
        "pretty = pd.DataFrame({\n"
        "    'Acc gravação': [fmt(r.acc_rec_mu, r.acc_rec_sd) for r in summary.itertuples()],\n"
        "    'F1m gravação': [fmt(r.f1m_rec_mu, r.f1m_rec_sd) for r in summary.itertuples()],\n"
        "    'Acc paciente': [fmt(r.acc_pat_mu, r.acc_pat_sd) for r in summary.itertuples()],\n"
        "    'F1m paciente': [fmt(r.f1m_pat_mu, r.f1m_pat_sd) for r in summary.itertuples()],\n"
        "    'Recall macro pac.': [fmt(r.rec_pat_mu, r.rec_pat_sd) for r in summary.itertuples()],\n"
        "}, index=summary.index)\n"
        "pretty"
    ),
    md("## 5. Boxplots da distribuição de F1-macro"),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "order = summary.index.tolist()\n\n"
        "sns.boxplot(data=rep_df, x='model', y='f1m_rec', order=order, ax=axes[0],\n"
        "            palette='Set2', width=0.5)\n"
        "sns.stripplot(data=rep_df, x='model', y='f1m_rec', order=order, ax=axes[0],\n"
        "              color='black', size=4, alpha=0.6)\n"
        "axes[0].set_title('F1-macro — nível gravação (10 repetições)')\n"
        "axes[0].set_ylabel('F1-macro'); axes[0].set_xlabel('')\n"
        "axes[0].tick_params(axis='x', rotation=15)\n\n"
        "sns.boxplot(data=rep_df, x='model', y='f1m_pat', order=order, ax=axes[1],\n"
        "            palette='Set2', width=0.5)\n"
        "sns.stripplot(data=rep_df, x='model', y='f1m_pat', order=order, ax=axes[1],\n"
        "              color='black', size=4, alpha=0.6)\n"
        "axes[1].set_title('F1-macro — nível paciente (10 repetições)')\n"
        "axes[1].set_ylabel('F1-macro'); axes[1].set_xlabel('')\n"
        "axes[1].tick_params(axis='x', rotation=15)\n\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## 6. Teste de significância — Wilcoxon pareado\n\n"
        "Comparamos pares de modelos usando os 10 F1-macros (paciente) **pareados por seed**. "
        "p-value baixo (<0.05) → diferença não-aleatória entre os modelos."
    ),
    code(
        "models_order = [m for m in summary.index if m != 'Dummy (sempre COPD)']\n"
        "pmat = pd.DataFrame(index=models_order, columns=models_order, dtype=float)\n"
        "for a in models_order:\n"
        "    for b in models_order:\n"
        "        if a == b:\n"
        "            pmat.loc[a, b] = np.nan\n"
        "            continue\n"
        "        sa = rep_df.loc[rep_df['model']==a, 'f1m_pat'].values\n"
        "        sb = rep_df.loc[rep_df['model']==b, 'f1m_pat'].values\n"
        "        # Wilcoxon precisa de diffs não-zero\n"
        "        try:\n"
        "            stat, p = stats.wilcoxon(sa, sb)\n"
        "        except ValueError:\n"
        "            p = np.nan\n"
        "        pmat.loc[a, b] = p\n\n"
        "print('Matriz de p-values (Wilcoxon pareado, F1-macro paciente):')\n"
        "print(pmat.round(4))\n\n"
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "sns.heatmap(pmat.astype(float), annot=True, fmt='.3f', cmap='RdYlGn_r',\n"
        "            vmin=0, vmax=0.2, ax=ax, cbar_kws={'label': 'p-value'})\n"
        "ax.set_title('Wilcoxon p-values (verde = diferença significativa)')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## 7. Intervalo de confiança 95% (bootstrap)\n\n"
        "Sobre os 10 F1-macros pacientes, IC95 = percentis 2.5 e 97.5 do bootstrap (n=2000)."
    ),
    code(
        "def boot_ci(arr, n=2000, ci=95, seed=0):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    arr = np.asarray(arr)\n"
        "    means = rng.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)\n"
        "    lo, hi = np.percentile(means, [(100-ci)/2, 100-(100-ci)/2])\n"
        "    return arr.mean(), lo, hi\n\n"
        "ci_rows = []\n"
        "for m in summary.index:\n"
        "    arr = rep_df.loc[rep_df['model']==m, 'f1m_pat'].values\n"
        "    mu, lo, hi = boot_ci(arr)\n"
        "    ci_rows.append(dict(model=m, mean=mu, ci_low=lo, ci_high=hi))\n"
        "ci_df = pd.DataFrame(ci_rows)\n\n"
        "fig, ax = plt.subplots(figsize=(9, 4))\n"
        "y_pos = np.arange(len(ci_df))\n"
        "ax.errorbar(ci_df['mean'], y_pos,\n"
        "            xerr=[ci_df['mean']-ci_df['ci_low'], ci_df['ci_high']-ci_df['mean']],\n"
        "            fmt='o', color='steelblue', capsize=5, markersize=8)\n"
        "ax.set_yticks(y_pos); ax.set_yticklabels(ci_df['model'])\n"
        "ax.set_xlabel('F1-macro (paciente)'); ax.set_title('Média ± IC95% (bootstrap)')\n"
        "ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3)\n"
        "plt.tight_layout(); plt.show()\n\n"
        "ci_df.round(3)"
    ),
    md(
        "## 8. Conclusões\n\n"
        "- **Repeated CV reduz drasticamente a incerteza da estimativa** — agora cada modelo tem média ± desvio claros.\n"
        "- O **Dummy baseline** mostra o piso real (acc ≈ 0.62 a nível de paciente).\n"
        "- A matriz de Wilcoxon mostra **quais diferenças entre modelos são reais e quais são ruído**.\n"
        "- Se KNN > XGB com p<0.01 nas 10 repetições, é evidência consistente; se p≈0.3, foi sorte do split.\n\n"
        "**Próximos passos sugeridos:**\n"
        "- Hyperparameter tuning (Nested CV: CV externo para avaliar + interno com `GridSearchCV` para tunar)\n"
        "- Tratar URTI com SMOTE / class_weight no XGB (`sample_weight`)\n"
        "- Modelagem a nível de **ciclo respiratório** com os intervalos anotados (mais dados, classes mais balanceadas)"
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
