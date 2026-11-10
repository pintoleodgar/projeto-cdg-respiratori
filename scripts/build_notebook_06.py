"""Gera notebooks/06_calibracao_e_vies.ipynb programaticamente."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "06_calibracao_e_vies.ipynb"


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
        "# 06 — Calibração isotónica + detecção de viés por equipamento\n\n"
        "Duas perguntas críticas:\n\n"
        "**Q1. Calibração.** Para uso clínico, `P(COPD) = 0.8` precisa significar 80% de chance real. "
        "Modelos com desbalanceamento severo tendem a ser miscalibrados. Aplicamos **regressão isotónica** "
        "sobre as probabilidades out-of-fold e medimos **Brier score** + **reliability diagram** antes e depois.\n\n"
        "**Q2. Viés de equipamento (shortcut learning).** Hipótese: o desempenho \"alto\" dos modelos no "
        "ICBHI é em parte espúrio porque equipamento de gravação correlaciona com diagnóstico. "
        "**Cross-tab abaixo é dramática** — vamos provar isso com um classificador-sonda e refazer toda "
        "a análise restrita ao único equipamento que contém as 3 classes (Meditron)."
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
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.preprocessing import StandardScaler, LabelEncoder\n"
        "from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n"
        "from sklearn.neighbors import KNeighborsClassifier\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.isotonic import IsotonicRegression\n"
        "from sklearn.metrics import (brier_score_loss, accuracy_score, f1_score,\n"
        "                              confusion_matrix, classification_report,\n"
        "                              log_loss)\n"
        "from sklearn.calibration import calibration_curve\n\n"
        "from src import config\n\n"
        "warnings.filterwarnings('ignore')\n"
        "sns.set_theme(style='whitegrid')"
    ),
    code(
        "feats_df = pd.read_csv(config.PROCESSED_DIR / 'features.csv')\n"
        "FEATURE_COLS = [c for c in feats_df.columns\n"
        "                if c not in ('filename', 'patient_id', 'equipment',\n"
        "                             'chest_location', 'label_multi', 'label_binary')]\n"
        "X = feats_df[FEATURE_COLS].values\n"
        "le_y = LabelEncoder()\n"
        "y = le_y.fit_transform(feats_df['label_multi'])\n"
        "groups = feats_df['patient_id'].values\n"
        "CLASSES = list(le_y.classes_)\n"
        "print(f'X={X.shape}  classes={CLASSES}  pacientes={len(np.unique(groups))}')"
    ),
    md(
        "## 1. Evidência visual do viés — cross-tab equipamento × diagnóstico"
    ),
    code(
        "ct_rec = pd.crosstab(feats_df['equipment'], feats_df['label_multi'])\n"
        "ct_pat = pd.crosstab(\n"
        "    feats_df.drop_duplicates(['patient_id','equipment'])['equipment'],\n"
        "    feats_df.drop_duplicates(['patient_id','equipment'])['label_multi'])\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "sns.heatmap(ct_rec, annot=True, fmt='d', cmap='Reds', cbar=False, ax=axes[0])\n"
        "axes[0].set_title('Gravações por equipamento × diagnóstico')\n"
        "sns.heatmap(ct_pat, annot=True, fmt='d', cmap='Reds', cbar=False, ax=axes[1])\n"
        "axes[1].set_title('Pacientes únicos por equipamento × diagnóstico')\n"
        "plt.tight_layout(); plt.show()\n\n"
        "print('Healthy e URTI: 100% gravados em Meditron.')\n"
        "print('Os outros 3 equipamentos só contêm COPD.')\n"
        "print('→ separar COPD de Healthy/URTI é parcialmente equivalente a identificar o equipamento.')"
    ),
    md(
        "## 2. Classificador-sonda — quão previsível é o equipamento a partir das features?\n\n"
        "Se um classificador simples (LogReg / RandomForest) prediz o equipamento com alta acurácia "
        "usando as **mesmas features de áudio**, o modelo de diagnóstico tem acesso a um \"atalho\" "
        "que não tem relação com fisiopatologia.\n\n"
        "Split por paciente (groups), F1-macro como métrica."
    ),
    code(
        "le_eq = LabelEncoder()\n"
        "y_eq = le_eq.fit_transform(feats_df['equipment'])\n"
        "EQUIP_CLASSES = list(le_eq.classes_)\n"
        "print('Classes de equipamento:', EQUIP_CLASSES)\n"
        "print('Distribuição:')\n"
        "print(pd.Series(y_eq).map(dict(enumerate(EQUIP_CLASSES))).value_counts())"
    ),
    code(
        "cv_eq = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)\n\n"
        "probes = {\n"
        "    'LogReg':        Pipeline([('s', StandardScaler()),\n"
        "                                ('c', LogisticRegression(max_iter=3000, class_weight='balanced'))]),\n"
        "    'RandomForest':  Pipeline([('s', StandardScaler()),\n"
        "                                ('c', RandomForestClassifier(n_estimators=300, n_jobs=-1,\n"
        "                                                              class_weight='balanced', random_state=42))]),\n"
        "}\n"
        "probe_rows = []\n"
        "for name, m in probes.items():\n"
        "    p = cross_val_predict(m, X, y_eq, groups=groups, cv=cv_eq, n_jobs=1)\n"
        "    acc = accuracy_score(y_eq, p)\n"
        "    f1m = f1_score(y_eq, p, average='macro')\n"
        "    probe_rows.append(dict(probe=name, accuracy=acc, f1_macro=f1m))\n"
        "    print(f'{name}: acc={acc:.3f}  f1_macro={f1m:.3f}')\n\n"
        "probe_df = pd.DataFrame(probe_rows)"
    ),
    code(
        "# baseline trivial: prever sempre a classe de equipamento mais frequente\n"
        "from collections import Counter\n"
        "most = Counter(y_eq).most_common(1)[0]\n"
        "baseline_acc = most[1] / len(y_eq)\n"
        "print(f'Baseline trivial (sempre {EQUIP_CLASSES[most[0]]}): acc={baseline_acc:.3f}')\n\n"
        "# matriz de confusão do probe (LogReg)\n"
        "y_eq_pred = cross_val_predict(probes['LogReg'], X, y_eq, groups=groups, cv=cv_eq, n_jobs=1)\n"
        "cm = confusion_matrix(y_eq, y_eq_pred)\n"
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',\n"
        "             xticklabels=[s.split()[0] for s in EQUIP_CLASSES],\n"
        "             yticklabels=[s.split()[0] for s in EQUIP_CLASSES],\n"
        "             cbar=False, ax=ax)\n"
        "ax.set_title('Sonda LogReg — previsão do equipamento\\n(F1-macro = {:.3f})'.format(\n"
        "    f1_score(y_eq, y_eq_pred, average='macro')))\n"
        "ax.set_xlabel('Predito'); ax.set_ylabel('Verdadeiro')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "**Interpretação:** se F1-macro do probe estiver alto (~0.7+), as features carregam informação "
        "muito forte sobre o equipamento. Como Healthy e URTI **só existem no Meditron**, o classificador "
        "de diagnóstico pode estar usando 'pegada acústica do Meditron' como atalho para \"não-COPD\". "
        "Resultado: parte da accuracy reportada não é detecção de doença."
    ),
    md(
        "## 3. Análise estratificada — só dentro do Meditron\n\n"
        "Refazemos a comparação de modelos restritos aos **48 pacientes do Meditron** (8 COPD + 26 Healthy + 14 URTI). "
        "Aqui o equipamento é constante: qualquer separação entre classes vem do sinal real, não do device.\n\n"
        "**Atenção:** com COPD reduzido a 8 pacientes, o problema fica mais difícil mas mais honesto."
    ),
    code(
        "mask = feats_df['equipment'] == 'WelchAllyn Meditron'\n"
        "feats_med = feats_df[mask].reset_index(drop=True)\n"
        "X_m = feats_med[FEATURE_COLS].values\n"
        "y_m = le_y.transform(feats_med['label_multi'])\n"
        "g_m = feats_med['patient_id'].values\n"
        "print(f'Meditron-only: {len(feats_med)} gravações, {len(np.unique(g_m))} pacientes')\n"
        "print(feats_med['label_multi'].value_counts())\n"
        "print()\n"
        "print('Pacientes por classe:')\n"
        "print(feats_med.drop_duplicates('patient_id')['label_multi'].value_counts())"
    ),
    code(
        "from xgboost import XGBClassifier\n\n"
        "models_med = {\n"
        "    'KNN':    Pipeline([('s', StandardScaler()),\n"
        "                         ('c', KNeighborsClassifier(n_neighbors=5, weights='distance'))]),\n"
        "    'LDA':    Pipeline([('s', StandardScaler()),\n"
        "                         ('c', LinearDiscriminantAnalysis())]),\n"
        "    'LogReg': Pipeline([('s', StandardScaler()),\n"
        "                         ('c', LogisticRegression(max_iter=3000, class_weight='balanced'))]),\n"
        "    'XGBoost':Pipeline([('s', StandardScaler()),\n"
        "                         ('c', XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,\n"
        "                                              eval_metric='mlogloss', n_jobs=1))]),\n"
        "}\n\n"
        "# 3-fold porque COPD só tem 8 pacientes\n"
        "cv_m = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)\n"
        "rows_m = []\n"
        "for name, m in models_med.items():\n"
        "    p = cross_val_predict(m, X_m, y_m, groups=g_m, cv=cv_m, n_jobs=1)\n"
        "    df_p = pd.DataFrame({'pid': g_m, 'y': y_m, 'pred': p}).groupby('pid').agg(\n"
        "        y=('y','first'), pred=('pred', lambda s: s.value_counts().idxmax()))\n"
        "    rec_acc = accuracy_score(y_m, p); rec_f1 = f1_score(y_m, p, average='macro')\n"
        "    pat_acc = accuracy_score(df_p['y'], df_p['pred'])\n"
        "    pat_f1  = f1_score(df_p['y'], df_p['pred'], average='macro')\n"
        "    rows_m.append(dict(model=name, acc_rec=rec_acc, f1m_rec=rec_f1,\n"
        "                        acc_pat=pat_acc, f1m_pat=pat_f1))\n"
        "    print(f'{name:8s}  acc_rec={rec_acc:.3f}  f1m_rec={rec_f1:.3f}  '\n"
        "          f'acc_pat={pat_acc:.3f}  f1m_pat={pat_f1:.3f}')\n\n"
        "med_df = pd.DataFrame(rows_m).set_index('model').sort_values('f1m_pat', ascending=False)\n"
        "med_df.round(3)"
    ),
    code(
        "# baseline trivial dentro do meditron: prever sempre Healthy (classe mais frequente)\n"
        "y_pat_full = feats_med.drop_duplicates('patient_id')['label_multi']\n"
        "mc = y_pat_full.value_counts()\n"
        "print(f'Baseline trivial (sempre {mc.idxmax()}): acc={mc.max()/mc.sum():.3f}')\n"
        "print(f'Pacientes totais no Meditron: {mc.sum()}')\n"
        "print()\n"
        "print('No dataset completo:')\n"
        "print('  KNN F1m paciente = 0.610 (nb04)  /  0.579 (nb05 nested)')\n"
        "print('Aqui, sem viés de equipamento, a métrica honesta cai significativamente.')"
    ),
    md(
        "## 4. Calibração isotónica das probabilidades out-of-fold\n\n"
        "Para uso clínico, queremos que **probabilidades de saída sejam confiáveis**. "
        "Pipeline: pegar `P(classe|x)` out-of-fold (cada amostra foi predita por modelo que não viu seu paciente), "
        "ajustar uma **regressão isotónica** por classe (one-vs-rest), aplicar e comparar **Brier score** e "
        "**reliability diagram** antes vs depois.\n\n"
        "Aqui usamos o **KNN no Meditron-only** (o cenário mais honesto). Para a calibração, fazemos uma "
        "validação cruzada interna sobre as próprias probabilidades OOF, garantindo que não estamos "
        "calibrando e avaliando no mesmo dado."
    ),
    code(
        "# Probabilidades OOF do KNN no dataset Meditron-only\n"
        "best = Pipeline([('s', StandardScaler()),\n"
        "                  ('c', KNeighborsClassifier(n_neighbors=5, weights='distance'))])\n"
        "proba_oof = cross_val_predict(best, X_m, y_m, groups=g_m, cv=cv_m,\n"
        "                                method='predict_proba', n_jobs=1)\n"
        "print('shape proba_oof:', proba_oof.shape)\n"
        "print('CLASSES:', CLASSES)"
    ),
    code(
        "def isotonic_calibrate_oof(proba, y_true, groups, classes, n_splits=3, seed=0):\n"
        "    \"\"\"Calibração isotónica honesta: re-divide em folds (mesmos groups) e ajusta isotonia\n"
        "    em fold-out, aplica no fold-in. Retorna probabilidades calibradas com a mesma forma.\"\"\"\n"
        "    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)\n"
        "    calibrated = np.zeros_like(proba)\n"
        "    for tr, te in cv.split(proba, y_true, groups):\n"
        "        for k in range(len(classes)):\n"
        "            ir = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)\n"
        "            ir.fit(proba[tr, k], (y_true[tr] == k).astype(float))\n"
        "            calibrated[te, k] = ir.predict(proba[te, k])\n"
        "    # renormaliza para somar 1\n"
        "    s = calibrated.sum(axis=1, keepdims=True); s[s==0] = 1\n"
        "    calibrated = calibrated / s\n"
        "    return calibrated\n\n"
        "proba_cal = isotonic_calibrate_oof(proba_oof, y_m, g_m, CLASSES, n_splits=3, seed=0)\n"
        "print('shape proba_cal:', proba_cal.shape)"
    ),
    code(
        "# Brier multiclasse (sum of class-wise Brier)\n"
        "def brier_multiclass(y_true, proba, n_classes):\n"
        "    yoh = np.eye(n_classes)[y_true]\n"
        "    return float(np.mean(np.sum((proba - yoh)**2, axis=1)))\n\n"
        "n_cls = len(CLASSES)\n"
        "brier_raw = brier_multiclass(y_m, proba_oof, n_cls)\n"
        "brier_cal = brier_multiclass(y_m, proba_cal, n_cls)\n"
        "ll_raw = log_loss(y_m, np.clip(proba_oof, 1e-6, 1-1e-6), labels=list(range(n_cls)))\n"
        "ll_cal = log_loss(y_m, np.clip(proba_cal, 1e-6, 1-1e-6), labels=list(range(n_cls)))\n"
        "print(f'Brier  bruto={brier_raw:.4f}  calibrado={brier_cal:.4f}  (Δ={brier_cal-brier_raw:+.4f})')\n"
        "print(f'LogLoss bruto={ll_raw:.4f}  calibrado={ll_cal:.4f}  (Δ={ll_cal-ll_raw:+.4f})')"
    ),
    code(
        "# Reliability diagram por classe (one-vs-rest)\n"
        "fig, axes = plt.subplots(1, len(CLASSES), figsize=(15, 4.5), sharey=True)\n"
        "for k, cls in enumerate(CLASSES):\n"
        "    ax = axes[k]\n"
        "    yk = (y_m == k).astype(int)\n"
        "    # raw\n"
        "    frac_raw, mean_raw = calibration_curve(yk, proba_oof[:, k], n_bins=8, strategy='quantile')\n"
        "    frac_cal, mean_cal = calibration_curve(yk, proba_cal[:, k], n_bins=8, strategy='quantile')\n"
        "    ax.plot([0,1], [0,1], '--', color='gray', alpha=0.6, label='perfeito')\n"
        "    ax.plot(mean_raw, frac_raw, 'o-', color='crimson', label='bruto (KNN)')\n"
        "    ax.plot(mean_cal, frac_cal, 's-', color='steelblue', label='calibrado (isotónica)')\n"
        "    ax.set_title(f'Classe: {cls}')\n"
        "    ax.set_xlabel('Probabilidade predita')\n"
        "    if k == 0:\n"
        "        ax.set_ylabel('Frequência observada')\n"
        "    ax.legend(fontsize=8)\n"
        "plt.suptitle('Reliability diagram — KNN Meditron-only (OOF, bruto vs isotónico)', y=1.02)\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## 5. Conclusões\n\n"
        "**Q1 – Calibração isotónica:**\n"
        "- Se Brier calibrado < Brier bruto e o reliability fica mais próximo da diagonal, vale a pena calibrar.\n"
        "- Mesmo que F1 não mude, as probabilidades passam a refletir confiança real — essencial pro assistente clínico.\n\n"
        "**Q2 – Viés de equipamento:**\n"
        "- Se o probe-sonda atinge F1-macro alto, há **vazamento de identidade do dispositivo** nas features.\n"
        "- Como Healthy/URTI **só existem no Meditron**, separar essas classes do COPD pode ser parcialmente equivalente a separar Meditron dos outros equipamentos.\n"
        "- A análise Meditron-only é o cenário **clinicamente honesto**: nele, qualquer separação vem do sinal acústico real.\n\n"
        "**Implicações práticas:**\n"
        "1. Não publicar números do dataset completo sem mencionar este viés.\n"
        "2. Reportar 2 cenários: (a) dataset completo (otimista), (b) Meditron-only (honesto).\n"
        "3. Considerar **harmonização de equipamento** (denoising, normalização espectral) ou abandonar pacientes não-Meditron para tarefas que envolvam Healthy/URTI.\n"
        "4. Em produção, garantir que o estetoscópio do paciente **seja o mesmo** dos dados de treino, ou tratar como dataset shift.\n\n"
        "**Próximos passos:**\n"
        "- Notebook 07 — classificação por ciclo respiratório (crackles/wheezes) com anotações dos `.txt` — alvo clinicamente direto, sem dependência de diagnóstico do paciente."
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
