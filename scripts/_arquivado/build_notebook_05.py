"""Gera notebooks/05_nested_cv_xgb_tuning.ipynb programaticamente."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "05_nested_cv_xgb_tuning.ipynb"


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
        "# 05 — Nested CV + XGBoost tuning com `sample_weight`\n\n"
        "**Hipótese a testar:** o XGBoost foi significativamente pior que KNN/LDA/LogReg no notebook 04 "
        "(p=0.002) por **decorar a classe majoritária** (COPD = 76% dos pacientes). Vamos testar se:\n\n"
        "1. Adicionar `sample_weight` inversamente proporcional à frequência da classe corrige o viés.\n"
        "2. Tuning de hiperparâmetros (com `GridSearchCV` em CV interno) melhora ainda mais.\n\n"
        "**Por que Nested CV:** se tunamos no mesmo CV em que avaliamos, vazamos informação e o score "
        "fica otimista. Solução padrão-ouro:\n"
        "- **CV externo** (`StratifiedGroupKFold` × repetições) — só pra estimar desempenho final.\n"
        "- **CV interno** (`StratifiedGroupKFold` dentro de `GridSearchCV`) — só pra escolher hiperparâmetros.\n\n"
        "Todos os splits são por paciente (`groups=patient_id`)."
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
        "from scipy import stats\n\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.preprocessing import StandardScaler, LabelEncoder\n"
        "from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV\n"
        "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n"
        "from sklearn.neighbors import KNeighborsClassifier\n"
        "from sklearn.utils.class_weight import compute_sample_weight\n"
        "from sklearn.metrics import (accuracy_score, f1_score, recall_score,\n"
        "                              confusion_matrix, classification_report)\n"
        "from xgboost import XGBClassifier\n\n"
        "from src import config\n\n"
        "warnings.filterwarnings('ignore', category=UserWarning)\n"
        "sns.set_theme(style='whitegrid')"
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
        "sample_w = compute_sample_weight('balanced', y)\n"
        "print(f'X: {X.shape} | Classes: {CLASSES}')\n"
        "print(f'Pesos por classe: {dict(zip(CLASSES, [sample_w[y==i][0] for i in range(len(CLASSES))]))}')\n"
        "print(f'Pacientes: {len(np.unique(groups))}')"
    ),
    md(
        "## 2. Modelos a comparar\n\n"
        "- **XGB-default**: configuração do notebook 04 (perdedor).\n"
        "- **XGB-balanced**: mesmo XGB + `sample_weight` balanceado.\n"
        "- **XGB-tuned**: tuning de `max_depth`, `learning_rate`, `n_estimators`, `min_child_weight` via GridSearch + `sample_weight`.\n"
        "- **KNN**: campeão do notebook 04 (referência).\n"
        "- **LDA**: também top no notebook 04 (referência interpretável).\n\n"
        "Grade pequena (`3×3×2×2 = 36 combos`) por orçamento de tempo."
    ),
    code(
        "def xgb_pipe(**kw):\n"
        "    return Pipeline([\n"
        "        ('scaler', StandardScaler()),\n"
        "        ('clf', XGBClassifier(eval_metric='mlogloss', n_jobs=1,\n"
        "                              tree_method='hist', **kw)),\n"
        "    ])\n\n"
        "XGB_DEFAULT = xgb_pipe(n_estimators=300, max_depth=5, learning_rate=0.1)\n\n"
        "XGB_GRID = {\n"
        "    'clf__n_estimators': [200, 400],\n"
        "    'clf__max_depth':    [3, 5, 7],\n"
        "    'clf__learning_rate':[0.05, 0.1, 0.2],\n"
        "    'clf__min_child_weight': [1, 3],\n"
        "}\n"
        "print('combos no grid:', 2*3*3*2)\n\n"
        "KNN_PIPE = Pipeline([('scaler', StandardScaler()),\n"
        "                     ('clf', KNeighborsClassifier(n_neighbors=5, weights='distance'))])\n"
        "LDA_PIPE = Pipeline([('scaler', StandardScaler()),\n"
        "                     ('clf', LinearDiscriminantAnalysis())])"
    ),
    md(
        "## 3. Função de avaliação Nested CV\n\n"
        "Para cada fold externo:\n"
        "1. Separa treino/teste por paciente.\n"
        "2. Treina o modelo (com GridSearch interno se aplicável).\n"
        "3. Prediz no teste externo.\n"
        "Acumula predições out-of-fold em todo o dataset → métricas finais agregadas."
    ),
    code(
        "def patient_metrics(y_true, y_pred, patient_ids):\n"
        "    df = pd.DataFrame({'pid': patient_ids, 'y': y_true, 'pred': y_pred})\n"
        "    agg = df.groupby('pid').agg(\n"
        "        y=('y', 'first'),\n"
        "        pred=('pred', lambda s: s.value_counts().idxmax()),\n"
        "    )\n"
        "    return agg['y'].values, agg['pred'].values\n\n"
        "def eval_one_outer(model, Xtr, ytr, gtr, Xte, gtr_idx, use_weights=False,\n"
        "                    grid=None, inner_splits=3, inner_seed=0):\n"
        "    fit_params = {}\n"
        "    if use_weights:\n"
        "        wtr = compute_sample_weight('balanced', ytr)\n"
        "        fit_params['clf__sample_weight'] = wtr\n"
        "    if grid is not None:\n"
        "        inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True,\n"
        "                                      random_state=inner_seed)\n"
        "        gs = GridSearchCV(model, param_grid=grid, scoring='f1_macro',\n"
        "                          cv=inner.split(Xtr, ytr, groups=gtr),\n"
        "                          n_jobs=-1, refit=True)\n"
        "        gs.fit(Xtr, ytr, **fit_params)\n"
        "        return gs.predict(Xte), gs.best_params_\n"
        "    else:\n"
        "        model.fit(Xtr, ytr, **fit_params)\n"
        "        return model.predict(Xte), None\n\n"
        "def run_nested(model_factory, name, *, use_weights=False, grid=None,\n"
        "                 outer_repeats=3, outer_splits=5, inner_splits=3):\n"
        "    t0 = time.time()\n"
        "    fold_records = []\n"
        "    best_params_all = []\n"
        "    for rep in range(outer_repeats):\n"
        "        seed = 42 + rep\n"
        "        outer = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=seed)\n"
        "        for fold, (tr, te) in enumerate(outer.split(X, y, groups)):\n"
        "            model = model_factory()\n"
        "            y_pred_te, best = eval_one_outer(\n"
        "                model, X[tr], y[tr], groups[tr], X[te], None,\n"
        "                use_weights=use_weights, grid=grid,\n"
        "                inner_splits=inner_splits, inner_seed=seed)\n"
        "            if best is not None:\n"
        "                best_params_all.append(best)\n"
        "            # nivel gravacao no fold\n"
        "            f1m_r = f1_score(y[te], y_pred_te, average='macro')\n"
        "            acc_r = accuracy_score(y[te], y_pred_te)\n"
        "            # nivel paciente no fold\n"
        "            yt_p, yp_p = patient_metrics(y[te], y_pred_te, groups[te])\n"
        "            f1m_p = f1_score(yt_p, yp_p, average='macro') if len(set(yt_p)) > 1 else np.nan\n"
        "            acc_p = accuracy_score(yt_p, yp_p)\n"
        "            rec_p = recall_score(yt_p, yp_p, average='macro')\n"
        "            fold_records.append(dict(model=name, repeat=rep, fold=fold,\n"
        "                                      acc_rec=acc_r, f1m_rec=f1m_r,\n"
        "                                      acc_pat=acc_p, f1m_pat=f1m_p, recall_pat=rec_p))\n"
        "        print(f'  {name} rep {rep+1}/{outer_repeats} ({time.time()-t0:.0f}s)')\n"
        "    return pd.DataFrame(fold_records), best_params_all"
    ),
    md(
        "## 4. Executar a comparação\n\n"
        "**Atenção ao tempo:** o XGB-tuned roda 36 combos × 3 inner folds × 15 outer folds = 1620 fits de XGB. "
        "Tempo estimado: 5–10 min."
    ),
    code(
        "all_results = []\n"
        "best_params_log = {}\n\n"
        "configs = [\n"
        "    ('XGB-default',  lambda: xgb_pipe(n_estimators=300, max_depth=5, learning_rate=0.1), False, None),\n"
        "    ('XGB-balanced', lambda: xgb_pipe(n_estimators=300, max_depth=5, learning_rate=0.1), True,  None),\n"
        "    ('XGB-tuned',    lambda: xgb_pipe(),                                                   True,  XGB_GRID),\n"
        "    ('KNN',          lambda: Pipeline([('scaler', StandardScaler()),\n"
        "                                       ('clf', KNeighborsClassifier(n_neighbors=5, weights='distance'))]),\n"
        "                                       False, None),\n"
        "    ('LDA',          lambda: Pipeline([('scaler', StandardScaler()),\n"
        "                                       ('clf', LinearDiscriminantAnalysis())]),\n"
        "                                       False, None),\n"
        "]\n\n"
        "for name, factory, use_w, grid in configs:\n"
        "    print(f'>>> {name}')\n"
        "    df, bps = run_nested(factory, name, use_weights=use_w, grid=grid,\n"
        "                          outer_repeats=3, outer_splits=5, inner_splits=3)\n"
        "    all_results.append(df)\n"
        "    if bps:\n"
        "        best_params_log[name] = bps\n\n"
        "res_df = pd.concat(all_results, ignore_index=True)\n"
        "print('\\nTotal fold-rows:', len(res_df))"
    ),
    md("## 5. Sumário — média ± desvio por modelo (15 outer folds)"),
    code(
        "def fmt(mu, sd): return f'{mu:.3f} ± {sd:.3f}'\n\n"
        "summary = res_df.groupby('model').agg(\n"
        "    f1m_rec_mu=('f1m_rec', 'mean'), f1m_rec_sd=('f1m_rec', 'std'),\n"
        "    f1m_pat_mu=('f1m_pat', 'mean'), f1m_pat_sd=('f1m_pat', 'std'),\n"
        "    rec_pat_mu=('recall_pat', 'mean'), rec_pat_sd=('recall_pat', 'std'),\n"
        "    acc_pat_mu=('acc_pat', 'mean'), acc_pat_sd=('acc_pat', 'std'),\n"
        ").sort_values('f1m_pat_mu', ascending=False)\n\n"
        "pretty = pd.DataFrame({\n"
        "    'F1m gravação': [fmt(r.f1m_rec_mu, r.f1m_rec_sd) for r in summary.itertuples()],\n"
        "    'F1m paciente': [fmt(r.f1m_pat_mu, r.f1m_pat_sd) for r in summary.itertuples()],\n"
        "    'Recall macro pac.': [fmt(r.rec_pat_mu, r.rec_pat_sd) for r in summary.itertuples()],\n"
        "    'Acc paciente':  [fmt(r.acc_pat_mu, r.acc_pat_sd) for r in summary.itertuples()],\n"
        "}, index=summary.index)\n"
        "pretty"
    ),
    md("## 6. Distribuição (boxplot) e ICs95 bootstrap"),
    code(
        "order = summary.index.tolist()\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
        "sns.boxplot(data=res_df, x='model', y='f1m_pat', order=order, hue='model',\n"
        "            palette='Set2', legend=False, width=0.5, ax=axes[0])\n"
        "sns.stripplot(data=res_df, x='model', y='f1m_pat', order=order,\n"
        "              color='black', size=4, alpha=0.6, ax=axes[0])\n"
        "axes[0].set_title('F1-macro paciente — folds externos (n=15)')\n"
        "axes[0].set_ylabel('F1-macro'); axes[0].set_xlabel('')\n"
        "axes[0].tick_params(axis='x', rotation=15)\n\n"
        "def boot_ci(arr, n=2000, ci=95, seed=0):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    arr = np.asarray(arr); arr = arr[~np.isnan(arr)]\n"
        "    means = rng.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)\n"
        "    lo, hi = np.percentile(means, [(100-ci)/2, 100-(100-ci)/2])\n"
        "    return arr.mean(), lo, hi\n\n"
        "ci_rows = []\n"
        "for m in order:\n"
        "    arr = res_df.loc[res_df['model']==m, 'f1m_pat'].values\n"
        "    mu, lo, hi = boot_ci(arr)\n"
        "    ci_rows.append(dict(model=m, mean=mu, lo=lo, hi=hi))\n"
        "ci_df = pd.DataFrame(ci_rows)\n\n"
        "y_pos = np.arange(len(ci_df))\n"
        "axes[1].errorbar(ci_df['mean'], y_pos,\n"
        "                  xerr=[ci_df['mean']-ci_df['lo'], ci_df['hi']-ci_df['mean']],\n"
        "                  fmt='o', color='steelblue', capsize=5, markersize=8)\n"
        "axes[1].set_yticks(y_pos); axes[1].set_yticklabels(ci_df['model'])\n"
        "axes[1].set_xlabel('F1-macro paciente'); axes[1].set_title('Média ± IC95% (bootstrap)')\n"
        "axes[1].invert_yaxis(); axes[1].grid(axis='x', alpha=0.3)\n"
        "plt.tight_layout(); plt.show()\n"
        "ci_df.round(3)"
    ),
    md("## 7. Wilcoxon pareado — XGB-tuned vs cada baseline"),
    code(
        "def paired_wilcoxon(a_name, b_name, col='f1m_pat'):\n"
        "    a = res_df.loc[res_df['model']==a_name, col].values\n"
        "    b = res_df.loc[res_df['model']==b_name, col].values\n"
        "    try:\n"
        "        _, p = stats.wilcoxon(a, b)\n"
        "    except ValueError:\n"
        "        p = np.nan\n"
        "    return p, np.mean(a) - np.mean(b)\n\n"
        "pairs = [('XGB-tuned', 'XGB-default'),\n"
        "         ('XGB-tuned', 'XGB-balanced'),\n"
        "         ('XGB-tuned', 'KNN'),\n"
        "         ('XGB-tuned', 'LDA'),\n"
        "         ('XGB-balanced', 'XGB-default')]\n"
        "rows = []\n"
        "for a, b in pairs:\n"
        "    p, delta = paired_wilcoxon(a, b)\n"
        "    rows.append(dict(comparison=f'{a} vs {b}',\n"
        "                      delta_F1m=round(delta, 3),\n"
        "                      p_value=round(p, 4),\n"
        "                      significant=p<0.05))\n"
        "pd.DataFrame(rows)"
    ),
    md(
        "## 8. Best params do XGB-tuned — escolhidos pelo CV interno\n\n"
        "Qual configuração foi escolhida mais frequentemente nos 15 outer folds? Indica regime "
        "estável de hiperparâmetros."
    ),
    code(
        "if 'XGB-tuned' in best_params_log:\n"
        "    bps = pd.DataFrame(best_params_log['XGB-tuned'])\n"
        "    print('Frequência de cada hiperparâmetro escolhido:\\n')\n"
        "    for col in bps.columns:\n"
        "        print(f'-- {col} --')\n"
        "        print(bps[col].value_counts().to_string())\n"
        "        print()\n"
        "else:\n"
        "    print('Sem best_params_log.')"
    ),
    md(
        "## 9. Matriz de confusão do melhor modelo (paciente)\n\n"
        "Refita o melhor modelo no dataset todo e mostra a matriz a nível de paciente "
        "(via predições out-of-fold já calculadas, agregadas por voto)."
    ),
    code(
        "best_name = summary.index[0]\n"
        "print(f'Melhor modelo: {best_name}')\n\n"
        "# concatena predições out-of-fold já feitas (nível gravação) — para CM por paciente.\n"
        "# Pra simplicidade, recompomos via 1 outer rep (5 folds) sem repetições.\n"
        "best_factory = dict([(c[0], (c[1], c[2], c[3])) for c in configs])[best_name]\n"
        "factory, use_w, grid = best_factory\n\n"
        "outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)\n"
        "y_pred_all = np.empty_like(y)\n"
        "for tr, te in outer.split(X, y, groups):\n"
        "    model = factory()\n"
        "    y_pred_te, _ = eval_one_outer(model, X[tr], y[tr], groups[tr], X[te], None,\n"
        "                                    use_weights=use_w, grid=grid, inner_seed=42)\n"
        "    y_pred_all[te] = y_pred_te\n\n"
        "yt_p, yp_p = patient_metrics(y, y_pred_all, groups)\n"
        "yt_p_lbl = np.array([CLASSES[i] for i in yt_p])\n"
        "yp_p_lbl = np.array([CLASSES[i] for i in yp_p])\n"
        "cm = confusion_matrix(yt_p_lbl, yp_p_lbl, labels=CLASSES)\n\n"
        "fig, ax = plt.subplots(figsize=(6, 5))\n"
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
        "            xticklabels=CLASSES, yticklabels=CLASSES, cbar=False,\n"
        "            annot_kws={'size': 14}, ax=ax)\n"
        "ax.set_xlabel('Predito'); ax.set_ylabel('Verdadeiro')\n"
        "ax.set_title(f'{best_name} — matriz de confusão (104 pacientes)')\n"
        "plt.tight_layout(); plt.show()\n\n"
        "print(classification_report(yt_p_lbl, yp_p_lbl, digits=3))"
    ),
    md(
        "## Conclusões\n\n"
        "1. **`sample_weight` balanceado sozinho corrige o XGB?** O `XGB-balanced` testa exatamente isso.\n"
        "2. **Tuning ajuda além do balanceamento?** Comparar `XGB-tuned` vs `XGB-balanced`.\n"
        "3. **XGB ajustado supera o KNN?** É a pergunta final — vê o Wilcoxon `XGB-tuned vs KNN`.\n\n"
        "Próximos passos:\n"
        "- Notebook 06: classificação a nível de **ciclo respiratório** (crackles/wheezes) — dataset maior e mais balanceado\n"
        "- Notebook 07: CNN sobre mel-spectrogramas"
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
