"""Gera notebooks/15_validacao_critica.ipynb programaticamente.

Endereça os pontos fracos identificados na análise crítica:
- Probe pós-residualização (verifica se viés foi mitigado)
- Análise por subgrupo demográfico (age, sex)
- Análise dos 4 pacientes URTI sempre errados
- Ensemble bagging (LogReg + LDA + KNN voto majoritário)
- Comparação com baseline trivial "sempre doente"
- Bootstrap IC95% para métricas principais
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "15_validacao_critica.ipynb"


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
        "# 15 — Validação crítica + análises avançadas\n\n"
        "Este notebook endereça **6 pontos fracos** identificados na análise crítica do projeto:\n\n"
        "1. **Probe pós-residualização** — verifica se a residualização linear realmente mitigou o viés de equipamento\n"
        "2. **Análise por subgrupo demográfico** — fairness por idade e sexo\n"
        "3. **Análise individual dos 4 pacientes URTI sempre errados** — limite das features ou caso específico?\n"
        "4. **Ensemble bagging** — voto majoritário entre LogReg/LDA/KNN melhora recall?\n"
        "5. **Comparação com baseline trivial** — quanto o modelo realmente adiciona sobre 'sempre doente'?\n"
        "6. **Bootstrap IC95%** — intervalos de confiança robustos para métricas finais\n\n"
        "**Princípio:** identificar e quantificar honestamente os limites do projeto. Cada análise pode invalidar ou validar uma afirmação central do `RELATORIO.md`."
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
        "from sklearn.ensemble import VotingClassifier\n"
        "from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict\n"
        "from sklearn.dummy import DummyClassifier\n"
        "from sklearn.metrics import (accuracy_score, recall_score, precision_score,\n"
        "                              f1_score, confusion_matrix, classification_report)\n\n"
        "from src import config, metadata\n"
        "from src.plotting import CLASS_COLORS, MODEL_PALETTE, set_style, savefig\n"
        "from src.bootstrap import bootstrap_ci, fmt_ci\n\n"
        "warnings.filterwarnings('ignore')\n"
        "set_style()"
    ),
    code(
        "# Carregar features + metadata demográfica\n"
        "feats_df = pd.read_csv(config.PROCESSED_DIR / 'features.csv')\n"
        "rec_df = metadata.build_recordings_dataframe()\n"
        "rec_df = metadata.filter_target_classes(rec_df)\n"
        "rec_df = metadata.add_target_labels(rec_df)\n"
        "# merge demografia\n"
        "feats_df = feats_df.merge(\n"
        "    rec_df[['filename', 'age', 'sex', 'adult_bmi', 'child_weight', 'child_height']],\n"
        "    on='filename', how='left')\n"
        "FEATURE_COLS = [c for c in feats_df.columns\n"
        "                if c not in ('filename','patient_id','equipment','chest_location',\n"
        "                              'label_multi','label_binary','age','sex','adult_bmi',\n"
        "                              'child_weight','child_height')]\n"
        "X = feats_df[FEATURE_COLS].values\n"
        "le = LabelEncoder()\n"
        "y = le.fit_transform(feats_df['label_multi'])\n"
        "groups = feats_df['patient_id'].values\n"
        "equipment = feats_df['equipment'].values\n"
        "CLASSES = list(le.classes_)\n"
        "HEALTHY_IDX = CLASSES.index('Healthy')\n"
        "EQUIPMENT_CATS = sorted(set(equipment))\n"
        "print(f'X={X.shape}, classes={CLASSES}')\n"
        "print(f'Demografia disponível: idade {feats_df[\"age\"].notna().sum()}/{len(feats_df)}, sexo {feats_df[\"sex\"].notna().sum()}/{len(feats_df)}')"
    ),
    md(
        "## 1. Probe pós-residualização — a residualização funcionou?\n\n"
        "**Hipótese a testar:** se a residualização linear remove viés de equipamento, um classificador-sonda treinado nas features **residualizadas** deve ter F1 muito menor que o probe original (F1=0,914 no nb06).\n\n"
        "**Procedimento:**\n"
        "1. Aplicar residualização global (apenas para esta análise — não é o pipeline de produção)\n"
        "2. Treinar LogReg para prever **equipamento** a partir das features residualizadas\n"
        "3. Comparar F1m com o probe original\n\n"
        "**Resultado esperado se residualização funciona:** F1 do probe pós-residualização deve cair para próximo do baseline trivial (~0,76 sempre-AKG)."
    ),
    code(
        "le_eq = LabelEncoder()\n"
        "y_eq = le_eq.fit_transform(equipment)\n"
        "EQUIP_CLASSES = list(le_eq.classes_)\n\n"
        "# Residualização global (apenas para análise — em produção é por fold)\n"
        "ohe = OneHotEncoder(sparse_output=False, categories=[EQUIPMENT_CATS])\n"
        "E = ohe.fit_transform(equipment.reshape(-1, 1))\n"
        "lr_res = LinearRegression().fit(E, X)\n"
        "X_resid = X - lr_res.predict(E)\n\n"
        "# Probe 1: LogReg sobre features ORIGINAIS (replica nb06)\n"
        "cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)\n"
        "probe_orig = LogisticRegression(max_iter=3000, class_weight='balanced')\n"
        "y_pred_orig = cross_val_predict(probe_orig, X, y_eq, groups=groups, cv=cv, n_jobs=1)\n"
        "f1m_orig = f1_score(y_eq, y_pred_orig, average='macro')\n"
        "acc_orig = accuracy_score(y_eq, y_pred_orig)\n\n"
        "# Probe 2: LogReg sobre features RESIDUALIZADAS\n"
        "probe_resid = LogisticRegression(max_iter=3000, class_weight='balanced')\n"
        "y_pred_resid = cross_val_predict(probe_resid, X_resid, y_eq, groups=groups, cv=cv, n_jobs=1)\n"
        "f1m_resid = f1_score(y_eq, y_pred_resid, average='macro')\n"
        "acc_resid = accuracy_score(y_eq, y_pred_resid)\n\n"
        "# Baseline trivial\n"
        "from collections import Counter\n"
        "most_common = Counter(y_eq).most_common(1)[0]\n"
        "baseline_acc = most_common[1] / len(y_eq)\n\n"
        "print('═══════ Probe de equipamento — antes vs depois da residualização ═══════')\n"
        "print(f'  Baseline trivial (sempre {EQUIP_CLASSES[most_common[0]]}): acc = {baseline_acc:.3f}')\n"
        "print(f'  Probe ORIGINAL  (features cruas)     : acc = {acc_orig:.3f}  F1-macro = {f1m_orig:.3f}')\n"
        "print(f'  Probe RESIDUALIZADO                  : acc = {acc_resid:.3f}  F1-macro = {f1m_resid:.3f}')\n"
        "print()\n"
        "redução = (f1m_orig - f1m_resid) / f1m_orig * 100\n"
        "print(f'→ Redução de F1 do probe: {redução:.1f}%')\n"
        "if f1m_resid > 0.5:\n"
        "    print('⚠️  Probe residualizado ainda tem F1 > 0,5 → leakage residual significativo (não-linear)')\n"
        "    print('    Caminho recomendado: Domain-Adversarial Neural Network (DANN)')\n"
        "else:\n"
        "    print('Probe LINEAR residualizado < 0,5 → a componente linear do viés foi removida.')\n"
        "    print('Isto é esperado por construção; ver a sonda não-linear na célula seguinte.')"
    ),
    md(
        "### 1.1 Sonda não-linear — o teste que realmente importa\n\n"
        "A residualização linear põe a zero a média de cada feature por equipamento. Uma sonda **linear** "
        "falha, portanto, quase por construção — o resultado acima não prova que o viés desapareceu.\n\n"
        "Repetimos a sonda com um **RandomForest**, com a residualização ajustada **dentro de cada fold** "
        "(como no pipeline de produção). Se o F1 continuar alto, a assinatura do equipamento ainda está "
        "nas features (variâncias, interações, forma das distribuições)."
    ),
    code(
        "from sklearn.ensemble import RandomForestClassifier\n\n"
        "f1_rf_cru, f1_rf_res = [], []\n"
        "for tr, te in cv.split(X, y_eq, groups):\n"
        "    lr_f = LinearRegression().fit(E[tr], X[tr])\n"
        "    Xr_tr, Xr_te = X[tr] - lr_f.predict(E[tr]), X[te] - lr_f.predict(E[te])\n"
        "    for store, a, b in ((f1_rf_cru, X[tr], X[te]), (f1_rf_res, Xr_tr, Xr_te)):\n"
        "        rf = RandomForestClassifier(300, random_state=0, class_weight='balanced').fit(a, y_eq[tr])\n"
        "        store.append(f1_score(y_eq[te], rf.predict(b), average='macro'))\n"
        "f1m_rf_cru, f1m_rf_res = np.mean(f1_rf_cru), np.mean(f1_rf_res)\n\n"
        "print('═══════ Sonda de equipamento — linear vs não-linear (F1-macro) ═══════')\n"
        "print(f'  LogReg       cru = {f1m_orig:.3f}   residualizado = {f1m_resid:.3f}')\n"
        "print(f'  RandomForest cru = {f1m_rf_cru:.3f}   residualizado = {f1m_rf_res:.3f}')\n"
        "print()\n"
        "if f1m_rf_res > 0.5:\n"
        "    print('⚠️  A sonda não-linear ainda identifica o equipamento → o viés foi MITIGADO')\n"
        "    print('    (componente linear), não eliminado. O LogReg final é linear e por isso não')\n"
        "    print('    explora diretamente o que resta, mas modelos não-lineares (XGB, RF, CNN) podem.')\n"
        "else:\n"
        "    print('Sonda não-linear também falha → viés de equipamento removido.')"
    ),
    code(
        "# Matriz de confusão dos 2 probes lado-a-lado\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "for ax, y_p, title in [\n"
        "    (axes[0], y_pred_orig,  f'Probe ORIGINAL\\nF1-macro = {f1m_orig:.3f}'),\n"
        "    (axes[1], y_pred_resid, f'Probe RESIDUALIZADO\\nF1-macro = {f1m_resid:.3f}'),\n"
        "]:\n"
        "    cm = confusion_matrix(y_eq, y_p, labels=range(len(EQUIP_CLASSES)))\n"
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',\n"
        "                 xticklabels=[s.split()[0] for s in EQUIP_CLASSES],\n"
        "                 yticklabels=[s.split()[0] for s in EQUIP_CLASSES],\n"
        "                 cbar=False, ax=ax)\n"
        "    ax.set_title(title)\n"
        "    ax.set_xlabel('Predito'); ax.set_ylabel('Verdadeiro')\n"
        "savefig('nb15_probe_pre_pos_residualizacao')\n"
        "plt.show()"
    ),
    md(
        "## 2. Análise por subgrupo demográfico (fairness)\n\n"
        "**Pergunta:** o modelo funciona igualmente bem em todos os subgrupos? Diferenças de performance por idade/sexo são evidência de **viés clínico** (não só de equipamento)."
    ),
    code(
        "# Gerar predições OOF com pipeline completo (residualização + LogReg)\n"
        "def fit_full_pipeline(X_tr, y_tr, eq_tr):\n"
        "    ohe = OneHotEncoder(sparse_output=False, categories=[EQUIPMENT_CATS])\n"
        "    E_tr = ohe.fit_transform(eq_tr.reshape(-1, 1))\n"
        "    lr = LinearRegression().fit(E_tr, X_tr)\n"
        "    Xr = X_tr - lr.predict(E_tr)\n"
        "    sc = StandardScaler().fit(Xr)\n"
        "    Xs = sc.transform(Xr)\n"
        "    clf = LogisticRegression(max_iter=3000, class_weight='balanced').fit(Xs, y_tr)\n"
        "    return ohe, lr, sc, clf\n\n"
        "y_pred_oof = np.full(len(y), -1)\n"
        "for tr, te in cv.split(X, y, groups):\n"
        "    ohe, lr, sc, clf = fit_full_pipeline(X[tr], y[tr], equipment[tr])\n"
        "    E_te = ohe.transform(equipment[te].reshape(-1, 1))\n"
        "    Xr_te = X[te] - lr.predict(E_te)\n"
        "    Xs_te = sc.transform(Xr_te)\n"
        "    y_pred_oof[te] = clf.predict(Xs_te)\n\n"
        "feats_df['y_pred'] = y_pred_oof\n"
        "feats_df['y_true'] = y\n"
        "feats_df['correct'] = (y_pred_oof == y)\n"
        "feats_df['is_disease'] = (y != HEALTHY_IDX)\n"
        "feats_df['detected_disease'] = (y_pred_oof != HEALTHY_IDX)\n"
        "print('Predições OOF prontas. Avaliando por subgrupo...')"
    ),
    code(
        "# Subgrupo: por sexo\n"
        "by_sex = feats_df.groupby('sex').agg(\n"
        "    n=('correct', 'size'),\n"
        "    acc=('correct', 'mean'),\n"
        "    recall_disease=('detected_disease',\n"
        "                     lambda s: s[feats_df.loc[s.index, 'is_disease']].mean() if feats_df.loc[s.index, 'is_disease'].any() else np.nan),\n"
        ").round(3)\n"
        "print('Por sexo:')\n"
        "print(by_sex)"
    ),
    code(
        "# Subgrupo: por faixa etária (crianças vs adultos)\n"
        "feats_df['faixa_idade'] = pd.cut(\n"
        "    feats_df['age'],\n"
        "    bins=[-1, 5, 18, 50, 80, 200],\n"
        "    labels=['<6 anos', '6–18', '18–50', '50–80', '>80'],\n"
        ")\n"
        "by_age = feats_df.groupby('faixa_idade', observed=True).agg(\n"
        "    n=('correct', 'size'),\n"
        "    acc=('correct', 'mean'),\n"
        ").round(3)\n"
        "print('Por faixa etária:')\n"
        "print(by_age)"
    ),
    code(
        "# Visualização — recall de doença por subgrupo\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))\n\n"
        "# Por sexo\n"
        "for sex in feats_df['sex'].dropna().unique():\n"
        "    sub = feats_df[(feats_df['sex'] == sex) & (feats_df['is_disease'])]\n"
        "    if len(sub) > 0:\n"
        "        rec = sub['detected_disease'].mean()\n"
        "        axes[0].bar(sex, rec, color='#3498db', edgecolor='black')\n"
        "        axes[0].text(sex, rec + 0.02, f'{rec:.2f}\\n(n={len(sub)})', ha='center', fontsize=10)\n"
        "axes[0].set_ylim(0, 1.05); axes[0].set_ylabel('Recall doença')\n"
        "axes[0].set_title('Recall de doença por SEXO')\n"
        "axes[0].set_xlabel('')\n\n"
        "# Por idade\n"
        "for faixa in feats_df['faixa_idade'].cat.categories:\n"
        "    sub = feats_df[(feats_df['faixa_idade'] == faixa) & (feats_df['is_disease'])]\n"
        "    if len(sub) > 0:\n"
        "        rec = sub['detected_disease'].mean()\n"
        "        axes[1].bar(str(faixa), rec, color='#e67e22', edgecolor='black')\n"
        "        axes[1].text(str(faixa), rec + 0.02, f'{rec:.2f}\\n(n={len(sub)})', ha='center', fontsize=10)\n"
        "axes[1].set_ylim(0, 1.05); axes[1].set_ylabel('Recall doença')\n"
        "axes[1].set_title('Recall de doença por FAIXA ETÁRIA')\n"
        "axes[1].set_xlabel('')\n"
        "savefig('nb15_fairness_subgrupos')\n"
        "plt.show()"
    ),
    md(
        "## 3. Análise dos pacientes URTI sempre errados\n\n"
        "**Hipótese do nb12:** existem ~4 pacientes URTI cuja P(doente) é tão baixa que nenhum threshold razoável os captura. **Quem são eles?** Demografia, equipamento, etc."
    ),
    code(
        "# Identificar pacientes URTI sempre classificados como Healthy (a nível de paciente)\n"
        "urti_idx = CLASSES.index('URTI')\n"
        "df_urti = feats_df[feats_df['y_true'] == urti_idx].copy()\n"
        "patient_summary = df_urti.groupby('patient_id').agg(\n"
        "    n_grav=('y_pred', 'size'),\n"
        "    pred_healthy=('y_pred', lambda s: (s == HEALTHY_IDX).sum()),\n"
        "    pred_copd=('y_pred', lambda s: (s == CLASSES.index('COPD')).sum()),\n"
        "    pred_urti=('y_pred', lambda s: (s == urti_idx).sum()),\n"
        "    age=('age', 'first'),\n"
        "    sex=('sex', 'first'),\n"
        "    equipment=('equipment', 'first'),\n"
        ")\n"
        "patient_summary['frac_healthy'] = patient_summary['pred_healthy'] / patient_summary['n_grav']\n"
        "patient_summary = patient_summary.sort_values('frac_healthy', ascending=False)\n"
        "print('Pacientes URTI ordenados por % de classificações como Healthy:')\n"
        "print(patient_summary)\n\n"
        "n_perdidos = (patient_summary['frac_healthy'] > 0.5).sum()\n"
        "print(f'\\n→ {n_perdidos} pacientes URTI são MAIORITARIAMENTE classificados como Healthy')"
    ),
    md(
        "## 4. Ensemble bagging — LogReg + LDA + KNN voto majoritário\n\n"
        "**Hipótese:** se LogReg, LDA e KNN são estatisticamente equivalentes (nb04), combiná-los via voto majoritário pode reduzir variância e aumentar recall sem aumentar viés."
    ),
    code(
        "def fit_model_on_residual(model_class, X_tr, y_tr, eq_tr, **kwargs):\n"
        "    ohe = OneHotEncoder(sparse_output=False, categories=[EQUIPMENT_CATS])\n"
        "    E_tr = ohe.fit_transform(eq_tr.reshape(-1, 1))\n"
        "    lr = LinearRegression().fit(E_tr, X_tr)\n"
        "    Xr = X_tr - lr.predict(E_tr)\n"
        "    sc = StandardScaler().fit(Xr)\n"
        "    Xs = sc.transform(Xr)\n"
        "    clf = model_class(**kwargs).fit(Xs, y_tr)\n"
        "    return {'ohe': ohe, 'lr': lr, 'sc': sc, 'clf': clf}\n\n"
        "def predict_residual(p, X_te, eq_te):\n"
        "    E_te = p['ohe'].transform(eq_te.reshape(-1, 1))\n"
        "    Xr = X_te - p['lr'].predict(E_te)\n"
        "    Xs = p['sc'].transform(Xr)\n"
        "    return p['clf'].predict_proba(Xs)\n\n"
        "# Avaliar em CV repetido\n"
        "model_results = {'LogReg': [], 'LDA': [], 'KNN': [], 'Ensemble': []}\n"
        "for rep in range(5):\n"
        "    cv_r = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42+rep)\n"
        "    pred_lr = np.full(len(y), -1)\n"
        "    pred_lda = np.full(len(y), -1)\n"
        "    pred_knn = np.full(len(y), -1)\n"
        "    pred_ens = np.full(len(y), -1)\n"
        "    for tr, te in cv_r.split(X, y, groups):\n"
        "        p_lr = fit_model_on_residual(LogisticRegression, X[tr], y[tr], equipment[tr],\n"
        "                                       max_iter=3000, class_weight='balanced')\n"
        "        p_lda = fit_model_on_residual(LinearDiscriminantAnalysis, X[tr], y[tr], equipment[tr])\n"
        "        p_knn = fit_model_on_residual(KNeighborsClassifier, X[tr], y[tr], equipment[tr],\n"
        "                                       n_neighbors=5, weights='distance')\n"
        "        proba_lr = predict_residual(p_lr, X[te], equipment[te])\n"
        "        proba_lda = predict_residual(p_lda, X[te], equipment[te])\n"
        "        proba_knn = predict_residual(p_knn, X[te], equipment[te])\n"
        "        # Predição individual\n"
        "        pred_lr[te] = proba_lr.argmax(axis=1)\n"
        "        pred_lda[te] = proba_lda.argmax(axis=1)\n"
        "        pred_knn[te] = proba_knn.argmax(axis=1)\n"
        "        # Ensemble: média das probas\n"
        "        proba_avg = (proba_lr + proba_lda + proba_knn) / 3\n"
        "        pred_ens[te] = proba_avg.argmax(axis=1)\n"
        "    # Métricas (nível paciente, voto majoritário)\n"
        "    def patient_metrics_local(yp):\n"
        "        d = pd.DataFrame({'g': groups, 'y': y, 'p': yp}).groupby('g').agg(\n"
        "            y=('y','first'), p=('p', lambda s: s.value_counts().idxmax()))\n"
        "        is_d = d['y'] != HEALTHY_IDX\n"
        "        is_p = d['p'] != HEALTHY_IDX\n"
        "        rec_d = (is_d & is_p).sum() / max(1, is_d.sum())\n"
        "        f1m = f1_score(d['y'], d['p'], average='macro')\n"
        "        return rec_d, f1m, d\n"
        "    for name, ypred in [('LogReg', pred_lr), ('LDA', pred_lda),\n"
        "                          ('KNN', pred_knn), ('Ensemble', pred_ens)]:\n"
        "        rec, f1m, _ = patient_metrics_local(ypred)\n"
        "        model_results[name].append((rec, f1m))\n"
        "print('Ensemble bagging — 5 repetições × 5 folds concluído.')"
    ),
    code(
        "# Tabela com Bootstrap IC95%\n"
        "rows = []\n"
        "for name, vals in model_results.items():\n"
        "    recalls = [v[0] for v in vals]\n"
        "    f1ms = [v[1] for v in vals]\n"
        "    mu_r, lo_r, hi_r = bootstrap_ci(recalls)\n"
        "    mu_f, lo_f, hi_f = bootstrap_ci(f1ms)\n"
        "    rows.append(dict(\n"
        "        modelo=name,\n"
        "        recall_doente=fmt_ci(mu_r, lo_r, hi_r),\n"
        "        f1_macro=fmt_ci(mu_f, lo_f, hi_f),\n"
        "    ))\n"
        "ens_df = pd.DataFrame(rows)\n"
        "print('=== Ensemble vs modelos individuais (n=5 repetições, IC95% bootstrap) ===')\n"
        "ens_df"
    ),
    md(
        "## 5. Comparação com baseline trivial 'sempre doente'\n\n"
        "Recall doente do nosso modelo deve ser comparado contra **DummyClassifier strategy='constant', value=Disease** — chamar tudo de 'doente' tem recall doente = 1,000 trivialmente."
    ),
    code(
        "# Baseline trivial 1: sempre 'COPD' (classe majoritária)\n"
        "dummy_const = DummyClassifier(strategy='most_frequent').fit(X, y)\n"
        "y_pred_dummy_const = dummy_const.predict(X)\n"
        "rec_const = ((y != HEALTHY_IDX) & (y_pred_dummy_const != HEALTHY_IDX)).sum() / (y != HEALTHY_IDX).sum()\n"
        "spec_const = ((y == HEALTHY_IDX) & (y_pred_dummy_const == HEALTHY_IDX)).sum() / max(1,(y == HEALTHY_IDX).sum())\n\n"
        "# Baseline trivial 2: sempre 'doente' (qualquer classe não-Healthy)\n"
        "# Como predição multiclasse, 'sempre COPD' já dá recall doente = 1\n"
        "print('=== Comparação com baselines triviais ===')\n"
        "print(f'  Baseline \"sempre COPD\" : recall doente = {rec_const:.3f}, especificidade = {spec_const:.3f}')\n"
        "print(f'    (acerta todos os doentes porque \"COPD ≠ Healthy\")')\n"
        "print(f'    (perde 100% dos saudáveis)')\n"
        "print()\n"
        "# Nosso modelo final\n"
        "ens_recall = bootstrap_ci([v[0] for v in model_results['Ensemble']])\n"
        "print(f'  Ensemble residualizado: recall doente = {ens_recall[0]:.3f} [IC95: {ens_recall[1]:.3f}, {ens_recall[2]:.3f}]')\n"
        "print()\n"
        "print('→ Recall doente alto SOZINHO não é virtude — baseline trivial atinge 1.000')\n"
        "print('→ A métrica honesta é o trade-off recall × especificidade.')"
    ),
    md(
        "## 6. Síntese — tabela final com bootstrap IC95% para todos os modelos\n\n"
        "Métrica governante: **Recall doente** com IC95% bootstrap."
    ),
    code(
        "# Tabela final consolidada\n"
        "summary_rows = []\n"
        "# Baseline trivial\n"
        "summary_rows.append(dict(\n"
        "    modelo='Trivial: sempre COPD',\n"
        "    recall_doente=f'{rec_const:.3f} [1.000, 1.000]',\n"
        "    f1_macro='(N/A)',\n"
        "    especificidade=f'{spec_const:.3f}',\n"
        "    nota='Inútil clinicamente',\n"
        "))\n"
        "# Modelos\n"
        "for name in ['LogReg', 'LDA', 'KNN', 'Ensemble']:\n"
        "    vals = model_results[name]\n"
        "    rec_b = bootstrap_ci([v[0] for v in vals])\n"
        "    f1_b = bootstrap_ci([v[1] for v in vals])\n"
        "    # Especificidade: usar última rep como proxy (não pareada)\n"
        "    summary_rows.append(dict(\n"
        "        modelo=name + ' residualizado',\n"
        "        recall_doente=fmt_ci(*rec_b),\n"
        "        f1_macro=fmt_ci(*f1_b),\n"
        "        especificidade='—',\n"
        "        nota='Pipeline residualizado',\n"
        "    ))\n"
        "summary = pd.DataFrame(summary_rows)\n"
        "print('═══════ TABELA FINAL — IC95% bootstrap ═══════')\n"
        "summary"
    ),
    md(
        "## 7. Conclusões da validação crítica\n\n"
        "Cada secção responde a um ponto fraco identificado:\n\n"
        "1. **Probe pós-residualização:** a sonda linear cai (por construção), mas a sonda não-linear (secção 1.1) continua a identificar o equipamento. A residualização mitiga a componente linear do viés; o viés residual persiste → trabalho futuro com DANN ou recolha de dados equilibrada por equipamento.\n\n"
        "2. **Fairness por subgrupo:** se recall variar muito por sexo/idade, o modelo tem viés clínico não-quantificado. Documentar.\n\n"
        "3. **Pacientes URTI errados:** identificámos individualmente os pacientes problemáticos. Análise demografia/equipamento pode revelar padrão acionável.\n\n"
        "4. **Ensemble:** se vencer LogReg sozinho, integrar como `mode='ensemble'` no `LungAssistant`. Caso contrário, parsimónia favorece LogReg.\n\n"
        "5. **Baseline trivial:** confirma que recall = 1,000 sozinho **não é virtude** — métrica honesta é trade-off.\n\n"
        "6. **Bootstrap IC95%:** intervalos de confiança robustos substituem desvio padrão dos folds."
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
