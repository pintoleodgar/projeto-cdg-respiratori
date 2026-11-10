"""Gera notebooks/03_comparacao_modelos.ipynb programaticamente."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "03_comparacao_modelos.ipynb"


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
        "# 03 — Comparação de modelos (XGBoost vs LDA vs Reg. Logística vs KNN)\n\n"
        "**Tarefa:** classificação multiclasse `Healthy / COPD / URTI` sobre as 78 features extraídas no notebook 02.\n\n"
        "**Pontos críticos:**\n"
        "- Split **paciente-estratificado** (`StratifiedGroupKFold` com `groups=patient_id`) — gravações de um mesmo paciente nunca aparecem em treino e teste juntas.\n"
        "- Avaliação em **dois níveis**: gravação (out-of-fold) e paciente (voto majoritário).\n"
        "- Matriz de confusão **com contagem de pacientes**.\n"
        "- `StandardScaler` em pipeline para LDA, LogReg e KNN (XGBoost não precisa)."
    ),
    code(
        "import sys\n"
        "from pathlib import Path\n"
        "PROJECT_ROOT = Path.cwd().parent\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.preprocessing import StandardScaler, LabelEncoder\n"
        "from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict\n"
        "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.neighbors import KNeighborsClassifier\n"
        "from sklearn.metrics import (accuracy_score, f1_score, classification_report,\n"
        "                              confusion_matrix, ConfusionMatrixDisplay)\n"
        "from xgboost import XGBClassifier\n\n"
        "from src import config\n\n"
        "sns.set_theme(style='whitegrid')"
    ),
    md("## 1. Carregar features"),
    code(
        "feats_df = pd.read_csv(config.PROCESSED_DIR / 'features.csv')\n"
        "print(f'Shape: {feats_df.shape}')\n"
        "print(feats_df['label_multi'].value_counts())\n"
        "feats_df.head()"
    ),
    md(
        "## 2. Preparar X, y, groups\n\n"
        "`y` codificado numericamente (XGBoost exige). `groups` = `patient_id` para o split."
    ),
    code(
        "FEATURE_COLS = [c for c in feats_df.columns\n"
        "                if c not in ('filename', 'patient_id', 'equipment',\n"
        "                             'chest_location', 'label_multi', 'label_binary')]\n"
        "print(f'# features: {len(FEATURE_COLS)}')\n\n"
        "X = feats_df[FEATURE_COLS].values\n"
        "le = LabelEncoder()\n"
        "y = le.fit_transform(feats_df['label_multi'])\n"
        "groups = feats_df['patient_id'].values\n"
        "CLASSES = list(le.classes_)\n"
        "print('Classes:', CLASSES)"
    ),
    md(
        "## 3. Definir modelos\n\n"
        "Todos em `Pipeline` com `StandardScaler` (exceto XGBoost que é invariante à escala — mas mantemos para consistência)."
    ),
    code(
        "models = {\n"
        "    'XGBoost': Pipeline([\n"
        "        ('scaler', StandardScaler()),\n"
        "        ('clf', XGBClassifier(\n"
        "            n_estimators=300, max_depth=5, learning_rate=0.1,\n"
        "            eval_metric='mlogloss', random_state=42, n_jobs=1)),\n"
        "    ]),\n"
        "    'LDA': Pipeline([\n"
        "        ('scaler', StandardScaler()),\n"
        "        ('clf', LinearDiscriminantAnalysis()),\n"
        "    ]),\n"
        "    'LogReg': Pipeline([\n"
        "        ('scaler', StandardScaler()),\n"
        "        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced',\n"
        "                                   random_state=42)),\n"
        "    ]),\n"
        "    'KNN': Pipeline([\n"
        "        ('scaler', StandardScaler()),\n"
        "        ('clf', KNeighborsClassifier(n_neighbors=5, weights='distance')),\n"
        "    ]),\n"
        "}\n"
        "list(models.keys())"
    ),
    md(
        "## 4. Cross-validation paciente-estratificado (5 folds)\n\n"
        "Geramos predições **out-of-fold** para cada gravação — cada gravação é predita uma única vez por um modelo que **nunca viu seu paciente** no treino."
    ),
    code(
        "cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)\n\n"
        "results = {}\n"
        "for name, model in models.items():\n"
        "    print(f'>>> {name}')\n"
        "    y_pred = cross_val_predict(model, X, y, groups=groups, cv=cv, n_jobs=-1)\n"
        "    results[name] = {\n"
        "        'y_pred': y_pred,\n"
        "        'acc': accuracy_score(y, y_pred),\n"
        "        'f1_macro': f1_score(y, y_pred, average='macro'),\n"
        "        'f1_weighted': f1_score(y, y_pred, average='weighted'),\n"
        "    }\n"
        "print('Concluído.')"
    ),
    md("## 5. Comparativo de métricas (nível de gravação)"),
    code(
        "metrics_df = pd.DataFrame({\n"
        "    'Accuracy': {k: v['acc'] for k, v in results.items()},\n"
        "    'F1-macro': {k: v['f1_macro'] for k, v in results.items()},\n"
        "    'F1-weighted': {k: v['f1_weighted'] for k, v in results.items()},\n"
        "}).round(3)\n"
        "metrics_df = metrics_df.sort_values('F1-macro', ascending=False)\n"
        "print(metrics_df)\n\n"
        "fig, ax = plt.subplots(figsize=(9, 4))\n"
        "metrics_df.plot(kind='bar', ax=ax, colormap='Set2')\n"
        "ax.set_title('Comparação de modelos — nível de gravação')\n"
        "ax.set_ylabel('Score'); ax.set_xlabel('')\n"
        "ax.set_ylim(0, 1); ax.tick_params(axis='x', rotation=0)\n"
        "ax.legend(loc='lower right')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("## 6. Classification report detalhado"),
    code(
        "for name, r in results.items():\n"
        "    print(f'\\n===== {name} =====')\n"
        "    print(classification_report(y, r['y_pred'], target_names=CLASSES, digits=3))"
    ),
    md(
        "## 7. Agregação para nível de paciente (voto majoritário)\n\n"
        "Cada paciente tem várias gravações. Agregamos por voto majoritário das predições out-of-fold "
        "para obter **um único diagnóstico por paciente** — visão clinicamente realista do desempenho."
    ),
    code(
        "def aggregate_to_patient(y_pred, feats_df, classes):\n"
        "    pred_labels = np.array([classes[i] for i in y_pred])\n"
        "    tmp = feats_df[['patient_id', 'label_multi']].copy()\n"
        "    tmp['pred'] = pred_labels\n"
        "    agg = tmp.groupby('patient_id').agg(\n"
        "        y_true=('label_multi', 'first'),\n"
        "        y_pred=('pred', lambda s: s.value_counts().idxmax()),\n"
        "    ).reset_index()\n"
        "    return agg\n\n"
        "patient_results = {}\n"
        "for name, r in results.items():\n"
        "    agg = aggregate_to_patient(r['y_pred'], feats_df, CLASSES)\n"
        "    patient_results[name] = agg\n"
        "    acc = (agg['y_true'] == agg['y_pred']).mean()\n"
        "    f1m = f1_score(agg['y_true'], agg['y_pred'], average='macro')\n"
        "    print(f'{name:8s}  pacientes={len(agg)}  acc={acc:.3f}  F1-macro={f1m:.3f}')"
    ),
    md(
        "## 8. Matriz de confusão (pacientes)\n\n"
        "Os números nas células são **contagens de pacientes** — exatamente o que importa clinicamente."
    ),
    code(
        "fig, axes = plt.subplots(1, 4, figsize=(20, 5))\n"
        "for ax, (name, agg) in zip(axes, patient_results.items()):\n"
        "    cm = confusion_matrix(agg['y_true'], agg['y_pred'], labels=CLASSES)\n"
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
        "                xticklabels=CLASSES, yticklabels=CLASSES,\n"
        "                cbar=False, ax=ax, annot_kws={'size': 14})\n"
        "    acc = (agg['y_true'] == agg['y_pred']).mean()\n"
        "    f1m = f1_score(agg['y_true'], agg['y_pred'], average='macro')\n"
        "    ax.set_title(f'{name}\\nAcc={acc:.2f}  F1-macro={f1m:.2f}', fontsize=12)\n"
        "    ax.set_xlabel('Predito'); ax.set_ylabel('Verdadeiro')\n"
        "plt.suptitle('Matriz de confusão — pacientes (n=104)', fontsize=14, y=1.02)\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("### Matriz de confusão também a nível de gravação (referência)"),
    code(
        "fig, axes = plt.subplots(1, 4, figsize=(20, 5))\n"
        "for ax, (name, r) in zip(axes, results.items()):\n"
        "    cm = confusion_matrix(y, r['y_pred'])\n"
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',\n"
        "                xticklabels=CLASSES, yticklabels=CLASSES,\n"
        "                cbar=False, ax=ax, annot_kws={'size': 12})\n"
        "    ax.set_title(f\"{name}  (acc={r['acc']:.2f}, F1m={r['f1_macro']:.2f})\")\n"
        "    ax.set_xlabel('Predito'); ax.set_ylabel('Verdadeiro')\n"
        "plt.suptitle('Matriz de confusão — gravações (n=851)', fontsize=14, y=1.02)\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## 9. Feature importance (XGBoost)\n\n"
        "Quais features carregam mais sinal para distinguir as classes?"
    ),
    code(
        "xgb_pipe = models['XGBoost']\n"
        "xgb_pipe.fit(X, y)  # fit no dataset completo só para importance\n"
        "importances = xgb_pipe.named_steps['clf'].feature_importances_\n"
        "imp_df = (pd.DataFrame({'feature': FEATURE_COLS, 'importance': importances})\n"
        "          .sort_values('importance', ascending=False).head(20))\n\n"
        "fig, ax = plt.subplots(figsize=(9, 6))\n"
        "ax.barh(imp_df['feature'][::-1], imp_df['importance'][::-1], color='steelblue')\n"
        "ax.set_title('Top 20 features (XGBoost)')\n"
        "ax.set_xlabel('importance')\n"
        "plt.tight_layout(); plt.show()\n"
        "imp_df"
    ),
    md(
        "## Resumo\n\n"
        "1. **Métricas a nível de gravação** podem ser otimistas porque pacientes COPD têm muitas gravações.\n"
        "2. **A nível de paciente é a métrica honesta** — esperamos queda especialmente em URTI (14 pacientes só).\n"
        "3. Com COPD em ~62% dos pacientes, um classificador trivial \"sempre COPD\" já acerta 62% — qualquer modelo precisa superar essa baseline.\n\n"
        "**Próximos passos sugeridos:**\n"
        "- Hyperparameter tuning (GridSearchCV com `cv=StratifiedGroupKFold`)\n"
        "- Abordagem por **ciclo respiratório** (notebook 04)\n"
        "- CNN sobre mel-spectrogramas (notebook 05)"
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
