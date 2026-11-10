"""Gera notebooks/16_dashboard_final.ipynb — dashboard visual único.

Comparação final de TODOS os modelos × TODAS as métricas-chave num gráfico só.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "16_dashboard_final.ipynb"


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
        "# 16 — Dashboard final\n\n"
        "**Uma única figura consolida todos os resultados do projeto** para apresentação à defesa.\n\n"
        "Inclui:\n"
        "- Recall doente (métrica principal) por modelo, com IC95% bootstrap\n"
        "- F1-macro paciente por modelo\n"
        "- Especificidade por modelo\n"
        "- FN clínico absoluto por modelo\n"
        "- Heatmap comparativo final"
    ),
    code(
        "import sys, warnings\n"
        "from pathlib import Path\n"
        "PROJECT_ROOT = Path.cwd().parent\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder\n"
        "from sklearn.linear_model import LinearRegression, LogisticRegression\n"
        "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n"
        "from sklearn.neighbors import KNeighborsClassifier\n"
        "from sklearn.model_selection import StratifiedGroupKFold\n"
        "from sklearn.dummy import DummyClassifier\n"
        "from sklearn.metrics import recall_score, f1_score, confusion_matrix\n"
        "from xgboost import XGBClassifier\n"
        "from sklearn.ensemble import RandomForestClassifier\n\n"
        "from src import config\n"
        "from src.plotting import CLASS_COLORS, MODEL_PALETTE, set_style, savefig\n"
        "from src.bootstrap import bootstrap_ci, fmt_ci\n\n"
        "warnings.filterwarnings('ignore')\n"
        "set_style()"
    ),
    code(
        "feats_df = pd.read_csv(config.PROCESSED_DIR / 'features.csv')\n"
        "FEATURE_COLS = [c for c in feats_df.columns\n"
        "                if c not in ('filename','patient_id','equipment','chest_location',\n"
        "                              'label_multi','label_binary')]\n"
        "X = feats_df[FEATURE_COLS].values\n"
        "le = LabelEncoder(); y = le.fit_transform(feats_df['label_multi'])\n"
        "groups = feats_df['patient_id'].values\n"
        "equipment = feats_df['equipment'].values\n"
        "CLASSES = list(le.classes_); HEALTHY_IDX = CLASSES.index('Healthy')\n"
        "EQUIPMENT_CATS = sorted(set(equipment))\n"
        "print(f'X={X.shape}, classes={CLASSES}')"
    ),
    code(
        "# Pipeline residualizado helper\n"
        "def fit_resid_pipe(X_tr, y_tr, eq_tr, model_cls, **kw):\n"
        "    ohe = OneHotEncoder(sparse_output=False, categories=[EQUIPMENT_CATS])\n"
        "    E_tr = ohe.fit_transform(eq_tr.reshape(-1, 1))\n"
        "    lr = LinearRegression().fit(E_tr, X_tr)\n"
        "    Xr = X_tr - lr.predict(E_tr)\n"
        "    sc = StandardScaler().fit(Xr)\n"
        "    clf = model_cls(**kw).fit(sc.transform(Xr), y_tr)\n"
        "    return ohe, lr, sc, clf\n\n"
        "def predict_resid(p, X_te, eq_te):\n"
        "    ohe, lr, sc, clf = p\n"
        "    E = ohe.transform(eq_te.reshape(-1, 1))\n"
        "    Xs = sc.transform(X_te - lr.predict(E))\n"
        "    return clf.predict(Xs)\n\n"
        "def patient_voting(y, yp, g):\n"
        "    d = pd.DataFrame({'g': g, 'y': y, 'p': yp}).groupby('g').agg(\n"
        "        y=('y','first'), p=('p', lambda s: s.value_counts().idxmax()))\n"
        "    return d['y'].values, d['p'].values"
    ),
    code(
        "# Definir TODOS os modelos a comparar\n"
        "models_to_compare = {\n"
        "    'Trivial: sempre COPD': lambda: (None, None, None, DummyClassifier(strategy='most_frequent')),\n"
        "    'LogReg (sem residualização)': lambda: ('plain', LogisticRegression(max_iter=3000, class_weight='balanced')),\n"
        "    'LogReg residualizado': ('resid', LogisticRegression, dict(max_iter=3000, class_weight='balanced')),\n"
        "    'LDA residualizado': ('resid', LinearDiscriminantAnalysis, dict()),\n"
        "    'KNN residualizado': ('resid', KNeighborsClassifier, dict(n_neighbors=5, weights='distance')),\n"
        "    'XGBoost residualizado': ('resid', XGBClassifier,\n"
        "                                 dict(n_estimators=200, max_depth=3, learning_rate=0.05,\n"
        "                                      eval_metric='mlogloss', n_jobs=1)),\n"
        "    'RandomForest residualizado (BLACK-BOX)': ('resid', RandomForestClassifier,\n"
        "                                 dict(n_estimators=300, max_depth=None, n_jobs=-1,\n"
        "                                      class_weight='balanced', random_state=42)),\n"
        "}\n\n"
        "# 5 repetições x 5-fold\n"
        "N_REP, N_FOLDS = 5, 5\n"
        "results = {name: {'recall': [], 'f1m': [], 'spec': [], 'fn': [], 'rec_URTI': []} for name in models_to_compare}\n\n"
        "import time\n"
        "t0 = time.time()\n"
        "for rep in range(N_REP):\n"
        "    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42+rep)\n"
        "    for name, cfg in models_to_compare.items():\n"
        "        y_pred = np.full(len(y), -1)\n"
        "        for tr, te in cv.split(X, y, groups):\n"
        "            if name == 'Trivial: sempre COPD':\n"
        "                clf = DummyClassifier(strategy='most_frequent').fit(X[tr], y[tr])\n"
        "                y_pred[te] = clf.predict(X[te])\n"
        "            elif name == 'LogReg (sem residualização)':\n"
        "                sc = StandardScaler().fit(X[tr])\n"
        "                clf = LogisticRegression(max_iter=3000, class_weight='balanced').fit(sc.transform(X[tr]), y[tr])\n"
        "                y_pred[te] = clf.predict(sc.transform(X[te]))\n"
        "            else:\n"
        "                _, model_cls, kw = cfg\n"
        "                pipe = fit_resid_pipe(X[tr], y[tr], equipment[tr], model_cls, **kw)\n"
        "                y_pred[te] = predict_resid(pipe, X[te], equipment[te])\n"
        "        # Métricas paciente\n"
        "        yt, yp = patient_voting(y, y_pred, groups)\n"
        "        is_d_real = yt != HEALTHY_IDX\n"
        "        is_d_pred = yp != HEALTHY_IDX\n"
        "        rec_d = (is_d_real & is_d_pred).sum() / max(1, is_d_real.sum())\n"
        "        fn = (is_d_real & (yp == HEALTHY_IDX)).sum()\n"
        "        spec = ((yt == HEALTHY_IDX) & (yp == HEALTHY_IDX)).sum() / max(1, (yt == HEALTHY_IDX).sum())\n"
        "        f1m = f1_score(yt, yp, average='macro')\n"
        "        urti_idx = CLASSES.index('URTI')\n"
        "        rec_urti = ((yt == urti_idx) & (yp == urti_idx)).sum() / max(1, (yt == urti_idx).sum())\n"
        "        results[name]['recall'].append(rec_d)\n"
        "        results[name]['f1m'].append(f1m)\n"
        "        results[name]['spec'].append(spec)\n"
        "        results[name]['fn'].append(fn)\n"
        "        results[name]['rec_URTI'].append(rec_urti)\n"
        "    print(f'rep {rep+1}/{N_REP} done ({time.time()-t0:.0f}s)')"
    ),
    code(
        "# Tabela final com IC95% bootstrap\n"
        "rows = []\n"
        "for name, m in results.items():\n"
        "    rec_b = bootstrap_ci(m['recall'])\n"
        "    f1_b = bootstrap_ci(m['f1m'])\n"
        "    spec_b = bootstrap_ci(m['spec'])\n"
        "    fn_b = bootstrap_ci(m['fn'])\n"
        "    rurti_b = bootstrap_ci(m['rec_URTI'])\n"
        "    rows.append(dict(\n"
        "        modelo=name,\n"
        "        recall_doente=fmt_ci(*rec_b),\n"
        "        FN_clinico=fmt_ci(*fn_b, digits=1),\n"
        "        rec_URTI=fmt_ci(*rurti_b),\n"
        "        especificidade=fmt_ci(*spec_b),\n"
        "        F1m=fmt_ci(*f1_b),\n"
        "    ))\n"
        "final = pd.DataFrame(rows)\n"
        "print('═════════════ DASHBOARD FINAL — IC95% bootstrap (5×5-fold, 25 amostras) ═════════════')\n"
        "final"
    ),
    md("## Dashboard visual único — 4 painéis"),
    code(
        "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n"
        "names = list(results.keys())\n"
        "x_pos = np.arange(len(names))\n\n"
        "# 1. Recall doente com IC95%\n"
        "ax = axes[0, 0]\n"
        "for i, name in enumerate(names):\n"
        "    mu, lo, hi = bootstrap_ci(results[name]['recall'])\n"
        "    ax.errorbar(i, mu, yerr=[[mu-lo],[hi-mu]], fmt='o', markersize=10,\n"
        "                 color='#e74c3c' if 'Trivial' in name else '#2ecc71', capsize=5)\n"
        "    ax.text(i, mu+0.02, f'{mu:.3f}', ha='center', fontsize=9)\n"
        "ax.set_xticks(x_pos); ax.set_xticklabels(names, rotation=20, ha='right')\n"
        "ax.set_ylabel('Recall doente'); ax.set_title('Recall doente — métrica principal')\n"
        "ax.set_ylim(0.5, 1.05); ax.grid(axis='y', alpha=0.3)\n"
        "ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)\n\n"
        "# 2. F1-macro com IC95%\n"
        "ax = axes[0, 1]\n"
        "for i, name in enumerate(names):\n"
        "    mu, lo, hi = bootstrap_ci(results[name]['f1m'])\n"
        "    ax.errorbar(i, mu, yerr=[[mu-lo],[hi-mu]], fmt='s', markersize=10,\n"
        "                 color='steelblue', capsize=5)\n"
        "    ax.text(i, mu+0.02, f'{mu:.3f}', ha='center', fontsize=9)\n"
        "ax.set_xticks(x_pos); ax.set_xticklabels(names, rotation=20, ha='right')\n"
        "ax.set_ylabel('F1-macro paciente'); ax.set_title('F1-macro paciente — métrica secundária')\n"
        "ax.set_ylim(0.0, 0.8); ax.grid(axis='y', alpha=0.3)\n\n"
        "# 3. Especificidade\n"
        "ax = axes[1, 0]\n"
        "for i, name in enumerate(names):\n"
        "    mu, lo, hi = bootstrap_ci(results[name]['spec'])\n"
        "    ax.errorbar(i, mu, yerr=[[mu-lo],[hi-mu]], fmt='^', markersize=10,\n"
        "                 color='#f39c12', capsize=5)\n"
        "    ax.text(i, mu+0.02, f'{mu:.3f}', ha='center', fontsize=9)\n"
        "ax.set_xticks(x_pos); ax.set_xticklabels(names, rotation=20, ha='right')\n"
        "ax.set_ylabel('Especificidade (recall_Healthy)')\n"
        "ax.set_title('Especificidade — quantos saudáveis são corretamente identificados')\n"
        "ax.set_ylim(-0.05, 1.05); ax.grid(axis='y', alpha=0.3)\n\n"
        "# 4. Recall URTI (calcanhar de Aquiles)\n"
        "ax = axes[1, 1]\n"
        "for i, name in enumerate(names):\n"
        "    mu, lo, hi = bootstrap_ci(results[name]['rec_URTI'])\n"
        "    ax.errorbar(i, mu, yerr=[[mu-lo],[hi-mu]], fmt='D', markersize=10,\n"
        "                 color='#9b59b6', capsize=5)\n"
        "    ax.text(i, mu+0.02, f'{mu:.3f}', ha='center', fontsize=9)\n"
        "ax.set_xticks(x_pos); ax.set_xticklabels(names, rotation=20, ha='right')\n"
        "ax.set_ylabel('Recall URTI'); ax.set_title('Recall URTI — classe minoritária')\n"
        "ax.set_ylim(-0.05, 1.05); ax.grid(axis='y', alpha=0.3)\n\n"
        "plt.suptitle('DASHBOARD FINAL — Comparação de modelos no dataset ICBHI 2017',\n"
        "              fontsize=15, y=1.00)\n"
        "savefig('nb16_dashboard_final')\n"
        "plt.show()"
    ),
    md("## Heatmap consolidado — todas as métricas, todos os modelos"),
    code(
        "# Heatmap: linhas = modelos, colunas = métricas (média)\n"
        "heatmap_data = pd.DataFrame({\n"
        "    name: {\n"
        "        'Recall doente': np.mean(m['recall']),\n"
        "        'F1-macro': np.mean(m['f1m']),\n"
        "        'Especificidade': np.mean(m['spec']),\n"
        "        'Recall URTI': np.mean(m['rec_URTI']),\n"
        "        '1 - FN/n_doentes': 1 - np.mean(m['fn'])/78,  # 78 doentes ~aprox\n"
        "    }\n"
        "    for name, m in results.items()\n"
        "}).T\n\n"
        "fig, ax = plt.subplots(figsize=(11, 6))\n"
        "sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',\n"
        "             vmin=0, vmax=1, cbar_kws={'label': 'Score'}, ax=ax,\n"
        "             linewidths=0.5, linecolor='white')\n"
        "ax.set_title('Mapa de performance — todos os modelos × todas as métricas (médias 25 folds)')\n"
        "ax.set_ylabel('Modelo')\n"
        "plt.xticks(rotation=15, ha='right')\n"
        "savefig('nb16_heatmap_final')\n"
        "plt.show()"
    ),
    md(
        "## Interpretabilidade vs Black-Box\n\n"
        "Random Forest foi incluído como **modelo black-box** para comparação:\n\n"
        "| Aspecto | LogReg (interpretável) | RandomForest (black-box) |\n"
        "|---|---|---|\n"
        "| Decomposição da decisão | **Exata:** `coef × valor` | Aproximada (feature_importances global, sem SHAP/LIME) |\n"
        "| Tamanho do modelo | ~240 parâmetros | ~300 árvores × ~100 nós ≈ 30k params |\n"
        "| Treino | < 1s | ~5s |\n"
        "| Auditável linha-a-linha | ✅ | ❌ |\n"
        "| Cumpre AI Act art. 13 (transparência) | ✅ | Requer XAI adicional |\n\n"
        "**Se RF supera LogReg em recall doente por > 5pp**, justifica complexidade adicional (com obrigação de explicabilidade via SHAP).\n"
        "**Se RF não supera**, **parsimónia favorece LogReg** — modelo simples, interpretável, auditável."
    ),
    md(
        "## Síntese para defesa\n\n"
        "Esta figura única é a **figura principal** do `RELATORIO.md` e das slides da defesa. Mostra:\n\n"
        "- **Quem tem o melhor recall doente** (métrica principal clínica)\n"
        "- **Quem é simplesmente trivial** (baseline sempre-COPD tem recall=1 mas especificidade=0)\n"
        "- **Quem tem melhor compromisso** recall vs especificidade\n"
        "- **Quem deteta URTI** (calcanhar de Aquiles)\n"
        "- **Interpretável (LogReg) vs Black-box (RF/XGB)** — trade-off compliance regulatória\n\n"
        "**Conclusão esperada:** o modelo final (LogReg residualizado) **não maximiza** recall doente em valor absoluto — o trivial 'sempre COPD' faz isso. Mas é o que tem **melhor compromisso** entre performance e interpretabilidade — essencial para AI Act art. 13."
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
