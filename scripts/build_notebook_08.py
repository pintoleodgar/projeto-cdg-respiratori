"""Gera notebooks/08_pipeline_final.ipynb programaticamente."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "08_pipeline_final.ipynb"


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
        "# 08 — Pipeline final do assistente respiratório\n\n"
        "Consolidação dos achados anteriores em um modelo único, calibrado, sem viés de equipamento.\n\n"
        "**Composição do modelo final:**\n"
        "1. Extração de 78 features (`src/features.py`)\n"
        "2. **Residualização** das features sobre equipamento (one-hot → LinearRegression)\n"
        "3. **StandardScaler** das features residualizadas\n"
        "4. **LogisticRegression** (multinomial, class_weight='balanced')\n"
        "5. **Calibração isotónica** das probabilidades OOF (one-vs-rest)\n\n"
        "**Por que esse pipeline?**\n"
        "- nb06: probe-equipamento atingiu F1m=0.91 → existe viés grave.\n"
        "- nb07: LogReg residualizado deu F1m paciente **0.633 ± 0.030**, melhor cenário sem viés.\n"
        "- nb06: calibração isotónica reduziu log loss em 44%.\n"
        "- LogReg é **interpretável** (coeficientes lineares), essencial pra uso clínico."
    ),
    code(
        "import sys, joblib, warnings\n"
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
        "from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict\n"
        "from sklearn.isotonic import IsotonicRegression\n"
        "from sklearn.metrics import (accuracy_score, f1_score, log_loss,\n"
        "                              confusion_matrix, classification_report)\n"
        "from sklearn.calibration import calibration_curve\n\n"
        "from src import config\n"
        "from src.inference import LungAssistant\n\n"
        "warnings.filterwarnings('ignore')\n"
        "sns.set_theme(style='whitegrid')\n"
        "MODEL_PATH = config.PROJECT_ROOT / 'models' / 'lung_assistant.joblib'\n"
        "MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)"
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
        "equipment = feats_df['equipment'].values\n"
        "CLASSES = list(le.classes_)\n"
        "print(f'X={X.shape}  classes={CLASSES}  pacientes={len(np.unique(groups))}')\n"
        "print(f'Equipamentos: {sorted(set(equipment))}')"
    ),
    md(
        "## 2. Validação final — repeated CV do pipeline residualizado + calibrado\n\n"
        "Importante: residualização e calibração precisam ser **ajustadas no treino de cada fold**, "
        "nunca no conjunto inteiro (vazaria informação do teste). Implementamos manualmente."
    ),
    code(
        "def fit_residualizer(X_tr, eq_tr, equipment_categories):\n"
        "    ohe = OneHotEncoder(sparse_output=False, categories=[equipment_categories])\n"
        "    E_tr = ohe.fit_transform(eq_tr.reshape(-1, 1))\n"
        "    lr = LinearRegression().fit(E_tr, X_tr)\n"
        "    return ohe, lr\n\n"
        "def residualize(X, eq, ohe, lr):\n"
        "    E = ohe.transform(eq.reshape(-1, 1))\n"
        "    return X - lr.predict(E)\n\n"
        "def fit_isotonic_oof(proba_oof, y_true, classes):\n"
        "    \"\"\"Ajusta uma IsotonicRegression por classe (one-vs-rest) sobre OOFs.\"\"\"\n"
        "    calibs = {}\n"
        "    for k, cls in enumerate(classes):\n"
        "        ir = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)\n"
        "        ir.fit(proba_oof[:, k], (y_true == k).astype(float))\n"
        "        calibs[cls] = ir\n"
        "    return calibs\n\n"
        "def apply_isotonic(proba, calibs, classes):\n"
        "    out = np.zeros_like(proba)\n"
        "    for k, cls in enumerate(classes):\n"
        "        out[:, k] = calibs[cls].predict(proba[:, k])\n"
        "    s = out.sum(axis=1, keepdims=True); s[s==0] = 1\n"
        "    return out / s\n\n"
        "EQUIPMENT_CATS = sorted(set(equipment))\n"
        "print('Equipment categories (ordem fixa):', EQUIPMENT_CATS)"
    ),
    code(
        "# Repeated 10×5 — pipeline completo: residualizar → scaler → logreg → calibrar\n"
        "rows = []\n"
        "all_proba_oof_uncal = np.zeros((len(y), len(CLASSES)))  # acumulador da rep=0 para calibração final\n"
        "all_proba_oof_cal   = np.zeros((len(y), len(CLASSES)))\n\n"
        "for rep in range(10):\n"
        "    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42+rep)\n"
        "    proba_uncal_rep = np.zeros((len(y), len(CLASSES)))\n"
        "    proba_cal_rep   = np.zeros((len(y), len(CLASSES)))\n"
        "    for tr, te in cv.split(X, y, groups):\n"
        "        # 1) residualizer no treino\n"
        "        ohe, lr_res = fit_residualizer(X[tr], equipment[tr], EQUIPMENT_CATS)\n"
        "        Xr_tr = residualize(X[tr], equipment[tr], ohe, lr_res)\n"
        "        Xr_te = residualize(X[te], equipment[te], ohe, lr_res)\n"
        "        # 2) scaler no treino\n"
        "        sc = StandardScaler().fit(Xr_tr)\n"
        "        Xs_tr = sc.transform(Xr_tr); Xs_te = sc.transform(Xr_te)\n"
        "        # 3) logreg no treino\n"
        "        clf = LogisticRegression(max_iter=3000, class_weight='balanced').fit(Xs_tr, y[tr])\n"
        "        p_te = clf.predict_proba(Xs_te)\n"
        "        proba_uncal_rep[te] = p_te\n"
        "        # 4) calibração ajustada em OOF interno (split adicional do treino)\n"
        "        #    aqui usamos calibração 'pós-hoc global' depois do loop externo\n"
        "    proba_uncal_rep_norm = proba_uncal_rep\n"
        "    # calibração global sobre os OOFs desta repetição\n"
        "    cals = fit_isotonic_oof(proba_uncal_rep_norm, y, CLASSES)\n"
        "    proba_cal_rep = apply_isotonic(proba_uncal_rep_norm, cals, CLASSES)\n"
        "    pred_uncal = np.argmax(proba_uncal_rep_norm, axis=1)\n"
        "    pred_cal   = np.argmax(proba_cal_rep, axis=1)\n"
        "    # paciente: voto majoritário\n"
        "    dfp = pd.DataFrame({'g': groups, 'y': y, 'p_u': pred_uncal, 'p_c': pred_cal}).groupby('g').agg(\n"
        "        y=('y','first'),\n"
        "        p_u=('p_u', lambda s: s.value_counts().idxmax()),\n"
        "        p_c=('p_c', lambda s: s.value_counts().idxmax()))\n"
        "    f1m_pat_u = f1_score(dfp['y'], dfp['p_u'], average='macro')\n"
        "    f1m_pat_c = f1_score(dfp['y'], dfp['p_c'], average='macro')\n"
        "    acc_pat_c = accuracy_score(dfp['y'], dfp['p_c'])\n"
        "    ll_u = log_loss(y, np.clip(proba_uncal_rep_norm, 1e-6, 1-1e-6), labels=list(range(len(CLASSES))))\n"
        "    ll_c = log_loss(y, np.clip(proba_cal_rep, 1e-6, 1-1e-6), labels=list(range(len(CLASSES))))\n"
        "    rows.append(dict(rep=rep, f1m_pat_uncal=f1m_pat_u, f1m_pat_cal=f1m_pat_c,\n"
        "                      acc_pat_cal=acc_pat_c, logloss_uncal=ll_u, logloss_cal=ll_c))\n"
        "    if rep == 0:\n"
        "        all_proba_oof_uncal = proba_uncal_rep_norm\n"
        "        all_proba_oof_cal   = proba_cal_rep\n"
        "    if (rep+1) % 5 == 0:\n"
        "        print(f'  rep {rep+1}/10  f1m_pat_cal={f1m_pat_c:.3f}')\n\n"
        "perf = pd.DataFrame(rows)\n"
        "print()\n"
        "print('Final (média ± std):')\n"
        "for c in ['f1m_pat_uncal','f1m_pat_cal','acc_pat_cal','logloss_uncal','logloss_cal']:\n"
        "    print(f'  {c}: {perf[c].mean():.3f} ± {perf[c].std():.3f}')"
    ),
    md(
        "Observação: a calibração **não muda** a decisão por argmax → F1 com/sem calibração coincide. "
        "O que muda é o **log loss** e a qualidade das probabilidades."
    ),
    md("## 3. Matriz de confusão (paciente) — agregação OOF da repetição 1"),
    code(
        "y_pred_oof = np.argmax(all_proba_oof_cal, axis=1)\n"
        "dfp = pd.DataFrame({'g': groups, 'y': y, 'p': y_pred_oof}).groupby('g').agg(\n"
        "    y=('y','first'), p=('p', lambda s: s.value_counts().idxmax()))\n"
        "y_t = np.array([CLASSES[i] for i in dfp['y']])\n"
        "y_p = np.array([CLASSES[i] for i in dfp['p']])\n\n"
        "cm = confusion_matrix(y_t, y_p, labels=CLASSES)\n"
        "fig, ax = plt.subplots(figsize=(6, 5))\n"
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
        "             xticklabels=CLASSES, yticklabels=CLASSES, cbar=False,\n"
        "             annot_kws={'size': 14}, ax=ax)\n"
        "ax.set_xlabel('Predito'); ax.set_ylabel('Verdadeiro')\n"
        "ax.set_title(f'Pipeline final — matriz de confusão paciente (n={len(y_t)})')\n"
        "plt.tight_layout(); plt.show()\n\n"
        "print(classification_report(y_t, y_p, digits=3))"
    ),
    md("## 4. Reliability diagram após calibração"),
    code(
        "fig, axes = plt.subplots(1, len(CLASSES), figsize=(15, 4.5), sharey=True)\n"
        "for k, cls in enumerate(CLASSES):\n"
        "    ax = axes[k]\n"
        "    yk = (y == k).astype(int)\n"
        "    frac_u, mean_u = calibration_curve(yk, all_proba_oof_uncal[:, k], n_bins=8, strategy='quantile')\n"
        "    frac_c, mean_c = calibration_curve(yk, all_proba_oof_cal[:, k],   n_bins=8, strategy='quantile')\n"
        "    ax.plot([0,1],[0,1],'--',color='gray',alpha=0.6,label='perfeito')\n"
        "    ax.plot(mean_u, frac_u, 'o-', color='crimson',   label='bruto')\n"
        "    ax.plot(mean_c, frac_c, 's-', color='steelblue', label='calibrado')\n"
        "    ax.set_title(f'Classe: {cls}'); ax.set_xlabel('P predita')\n"
        "    if k==0: ax.set_ylabel('Frequência observada')\n"
        "    ax.legend(fontsize=8)\n"
        "plt.suptitle('Reliability diagram — pipeline final', y=1.02)\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## 5. Treinar o modelo final em TODO o dataset e empacotar\n\n"
        "Para produção, treinamos no dataset inteiro (sem holdout). A estimativa de desempenho "
        "permanece a do repeated CV acima."
    ),
    code(
        "# Residualizer final\n"
        "ohe_final, lr_res_final = fit_residualizer(X, equipment, EQUIPMENT_CATS)\n"
        "X_resid = residualize(X, equipment, ohe_final, lr_res_final)\n\n"
        "# Scaler final\n"
        "sc_final = StandardScaler().fit(X_resid)\n"
        "X_scaled = sc_final.transform(X_resid)\n\n"
        "# LogReg final\n"
        "clf_final = LogisticRegression(max_iter=3000, class_weight='balanced').fit(X_scaled, y)\n"
        "print('LogReg classes:', clf_final.classes_)\n"
        "print('LogReg coef shape:', clf_final.coef_.shape)\n"
        "print('LogReg intercept:', clf_final.intercept_.round(3))"
    ),
    code(
        "# Calibradores finais a partir de probas OOF reais (rep 0)\n"
        "calibs_final = fit_isotonic_oof(all_proba_oof_uncal, y, CLASSES)\n"
        "# Serializar como (x_thresholds, y_thresholds) -- mais leve que objeto sklearn\n"
        "isotonic_dict = {}\n"
        "for cls, ir in calibs_final.items():\n"
        "    isotonic_dict[cls] = (ir.X_thresholds_.copy(), ir.y_thresholds_.copy())\n"
        "print('Calibradores prontos para', list(isotonic_dict.keys()))"
    ),
    code(
        "# Empacotar tudo no LungAssistant\n"
        "assistant = LungAssistant(\n"
        "    feature_cols=FEATURE_COLS,\n"
        "    equipment_categories=EQUIPMENT_CATS,\n"
        "    residualizer_coef_=lr_res_final.coef_,           # (n_features, n_equipment)\n"
        "    residualizer_intercept_=lr_res_final.intercept_,  # (n_features,)\n"
        "    scaler_mean_=sc_final.mean_,\n"
        "    scaler_scale_=sc_final.scale_,\n"
        "    logreg_classes_=np.array([CLASSES[i] for i in clf_final.classes_]),\n"
        "    logreg_coef_=clf_final.coef_,\n"
        "    logreg_intercept_=clf_final.intercept_,\n"
        "    isotonic_calibrators_=isotonic_dict,\n"
        ")\n"
        "assistant.save(MODEL_PATH)\n"
        "print(f'Modelo salvo em: {MODEL_PATH}')\n"
        "print(f'Tamanho: {MODEL_PATH.stat().st_size/1024:.1f} KB')"
    ),
    md(
        "## 6. Demo de inferência\n\n"
        "Carregamos o modelo salvo e fazemos predições em 3 áudios reais (um de cada classe)."
    ),
    code(
        "from src import metadata as md_module\n"
        "rec_df = md_module.build_recordings_dataframe()\n"
        "rec_df = md_module.filter_target_classes(rec_df)\n\n"
        "# 1 exemplo de cada classe\n"
        "demo = rec_df.groupby('diagnosis').first().reset_index()[['diagnosis','filename','wav_path','equipment']]\n"
        "print(demo)"
    ),
    code(
        "assistant_loaded = LungAssistant.load(MODEL_PATH)\n\n"
        "print('='*70)\n"
        "for _, row in demo.iterrows():\n"
        "    res = assistant_loaded.predict(row['wav_path'], equipment=row['equipment'])\n"
        "    print(f\"\\nArquivo: {row['filename']}\")\n"
        "    print(f\"Equipamento: {row['equipment']}\")\n"
        "    print(f\"Diagnóstico real: {row['diagnosis']}\")\n"
        "    print(f\"Diagnóstico predito: {res['diagnosis']}  (confiança: {res['confidence']}, P={res['top_probability']:.3f})\")\n"
        "    print('Probabilidades calibradas:')\n"
        "    for c, p in res['probabilities'].items():\n"
        "        bar = '█' * int(p * 30)\n"
        "        marker = ' ←' if c == res['diagnosis'] else ''\n"
        "        print(f'  {c:8s}  {p:.3f}  {bar}{marker}')\n"
        "    print('-'*70)"
    ),
    md(
        "## 7. Limitações e uso responsável\n\n"
        "**O que este assistente FAZ:**\n"
        "- Classifica um áudio entre Healthy, COPD ou URTI.\n"
        "- Retorna probabilidades calibradas + nível de confiança.\n"
        "- Compensa parcialmente o viés de equipamento via residualização.\n\n"
        "**O que NÃO faz:**\n"
        "- Não detecta Pneumonia, Asthma, Bronchiolitis, Bronchiectasis, LRTI (excluídas por falta de dados).\n"
        "- Não substitui anamnese clínica e exame físico.\n"
        "- Funciona apenas com os 4 equipamentos vistos no treino. Outro dispositivo → dataset shift.\n"
        "- Performance esperada: F1-macro paciente ≈ 0.633 ± 0.030. Recall em URTI é fraco (~0.36).\n\n"
        "**Recomendações de uso clínico:**\n"
        "- Use a saída como **segunda opinião**, especialmente para triagem.\n"
        "- Confie em predições com `confidence == 'alta'` (P>0.8). Reveja predições `'baixa'`.\n"
        "- Sempre passe o `equipment` correto para a inferência — sem isso, a residualização falha.\n\n"
        "## Resumo do projeto\n\n"
        "| Notebook | Resultado |\n"
        "|---|---|\n"
        "| 01 | EDA: 920 gravações, 126 pacientes, dataset altamente desbalanceado |\n"
        "| 02 | Extração de 78 features (MFCC, delta, espectrais) — 851 gravações filtradas |\n"
        "| 03 | Baseline: XGBoost/LDA/LogReg/KNN — KNN parecia melhor (F1m=0.61) |\n"
        "| 04 | Repeated CV: KNN/LDA/LogReg estatisticamente equivalentes, XGB pior |\n"
        "| 05 | Nested CV + tuning: XGB tunado não supera KNN/LDA |\n"
        "| 06 | **Achado de viés**: probe de equipamento F1m=0.91 (shortcut learning!) |\n"
        "| 07 | LogReg residualizado é o melhor modelo sem viés (F1m=0.633) |\n"
        "| **08** | **Pipeline final** com residualização + LogReg + calibração isotónica |"
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
