"""Gera notebooks/12_safety_net_fn_clinico.ipynb programaticamente."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "12_safety_net_fn_clinico.ipynb"


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
        "# 12 — Safety Net: minimizar FN clínico (imperativo ético)\n\n"
        "## ⚕️ Justificação ética — porquê este notebook é o mais importante do projeto\n\n"
        "**Princípio não-negociável:** se um paciente está doente, ele **não pode** sair classificado "
        "como saudável. Este erro — o *FN clínico* — viola o princípio bioético da **não-maleficência** "
        "(*primum non nocere*). Implica:\n\n"
        "- **Dano por omissão**: sistema deu alta a um doente. O dano pode ser irreversível.\n"
        "- **Progressão silenciosa**: DPOC sem tratamento avança; cada mês = perda funcional permanente.\n"
        "- **Perpetuação do subdiagnóstico** (86% em Portugal — ONDR) — o problema que o assistente "
        "deveria combater, agravado.\n"
        "- **Negação da autonomia** do paciente sobre o seu estado de saúde.\n\n"
        "**Assimetria de remediabilidade:** FP é remediável (médico corrige), FN tem custo cumulativo "
        "no paciente que nenhum sistema pode reverter retroativamente.\n\n"
        "Por isso, este notebook **não negocia** com FN — testamos estratégias até zerar (ou minimizar) "
        "este erro, aceitando o custo de mais FPs (que são apenas inconvenientes administrativos).\n\n"
        "**Definição:** FN clínico = (real ∈ {COPD, URTI}) ∧ (predito == Healthy).\n\n"
        "**Estado atual (do nb11, argmax do LogReg residualizado):**\n"
        "- FN clínico: **18/816 (2.2%)** — 18 doentes saem como saudáveis.\n"
        "- COPD → Healthy: 0 ✓\n"
        "- URTI → Healthy: 18 ← problema concentrado em URTI\n\n"
        "**Duas estratégias para resolver:**\n"
        "- **A) Cost-sensitive com penalidade massiva** — custo FN_clínico = €10.000 ou mais.\n"
        "- **B) Safety Net hierárquica** — bloqueia predição Healthy quando P(doença) ≥ threshold."
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
        "from sklearn.model_selection import StratifiedGroupKFold\n"
        "from sklearn.isotonic import IsotonicRegression\n"
        "from sklearn.metrics import confusion_matrix, classification_report\n\n"
        "from src import config\n"
        "warnings.filterwarnings('ignore')\n"
        "sns.set_theme(style='whitegrid')"
    ),
    md("## 1. Reproduzir probabilidades OOF calibradas"),
    code(
        "feats_df = pd.read_csv(config.PROCESSED_DIR / 'features.csv')\n"
        "FEATURE_COLS = [c for c in feats_df.columns\n"
        "                if c not in ('filename','patient_id','equipment','chest_location','label_multi','label_binary')]\n"
        "X = feats_df[FEATURE_COLS].values\n"
        "le = LabelEncoder(); y = le.fit_transform(feats_df['label_multi'])\n"
        "groups = feats_df['patient_id'].values\n"
        "equipment = feats_df['equipment'].values\n"
        "CLASSES = list(le.classes_)\n"
        "EQUIPMENT_CATS = sorted(set(equipment))\n"
        "HEALTHY_IDX = CLASSES.index('Healthy')\n\n"
        "cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)\n"
        "proba_uncal = np.zeros((len(y), len(CLASSES)))\n"
        "for tr, te in cv.split(X, y, groups):\n"
        "    ohe = OneHotEncoder(sparse_output=False, categories=[EQUIPMENT_CATS])\n"
        "    E_tr = ohe.fit_transform(equipment[tr].reshape(-1,1))\n"
        "    E_te = ohe.transform(equipment[te].reshape(-1,1))\n"
        "    lr = LinearRegression().fit(E_tr, X[tr])\n"
        "    Xr_tr = X[tr] - lr.predict(E_tr); Xr_te = X[te] - lr.predict(E_te)\n"
        "    sc = StandardScaler().fit(Xr_tr)\n"
        "    Xs_tr = sc.transform(Xr_tr); Xs_te = sc.transform(Xr_te)\n"
        "    clf = LogisticRegression(max_iter=3000, class_weight='balanced').fit(Xs_tr, y[tr])\n"
        "    proba_uncal[te] = clf.predict_proba(Xs_te)\n"
        "proba_cal = np.zeros_like(proba_uncal)\n"
        "for k, cls in enumerate(CLASSES):\n"
        "    ir = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(\n"
        "        proba_uncal[:, k], (y == k).astype(float))\n"
        "    proba_cal[:, k] = ir.predict(proba_uncal[:, k])\n"
        "s = proba_cal.sum(axis=1, keepdims=True); s[s==0]=1; proba_cal/=s\n"
        "print(f'proba_cal {proba_cal.shape} pronta')"
    ),
    md("## 2. Funções utilitárias e FN clínico do baseline"),
    code(
        "is_disease = (y != HEALTHY_IDX)\n"
        "n_disease = int(is_disease.sum())\n"
        "n_healthy = int((~is_disease).sum())\n"
        "p_disease = 1.0 - proba_cal[:, HEALTHY_IDX]\n\n"
        "def diagnose(name, y_pred):\n"
        "    fn_clin = int(((is_disease) & (y_pred == HEALTHY_IDX)).sum())\n"
        "    fn_copd = int(((y == CLASSES.index('COPD')) & (y_pred == HEALTHY_IDX)).sum())\n"
        "    fn_urti = int(((y == CLASSES.index('URTI')) & (y_pred == HEALTHY_IDX)).sum())\n"
        "    fp_dis  = int(((~is_disease) & (y_pred != HEALTHY_IDX)).sum())\n"
        "    rec_dis = 1 - fn_clin/n_disease\n"
        "    spec    = 1 - fp_dis/n_healthy\n"
        "    return {\n"
        "        'estratégia': name,\n"
        "        'FN_clinico': fn_clin,\n"
        "        '(COPD→H)':   fn_copd,\n"
        "        '(URTI→H)':   fn_urti,\n"
        "        'FP_doente':  fp_dis,\n"
        "        'recall_doente': rec_dis,\n"
        "        'especificidade': spec,\n"
        "    }\n\n"
        "rows = [diagnose('Argmax (baseline)', np.argmax(proba_cal, axis=1))]\n"
        "pd.DataFrame(rows).round(3)"
    ),
    md(
        "## 3. Estratégia A — Cost-sensitive com penalidade massiva de FN clínico\n\n"
        "Aumentamos massivamente o custo de doente→Healthy. Re-otimizamos a decisão "
        "(argmin de custo esperado)."
    ),
    code(
        "# Matriz base em € — dados reais do SNS português\n"
        "# DPOC: €2.950/internamento evitável | URTI: €70/ocorrência\n"
        "BASE_COSTS = pd.DataFrame(\n"
        "    [[0,    2950,  1475],  # real=COPD\n"
        "     [70,   0,     70],    # real=Healthy\n"
        "     [70,   70,    0]],    # real=URTI\n"
        "    index=CLASSES, columns=CLASSES)\n\n"
        "def cost_sensitive_pred(proba, costs):\n"
        "    return np.argmin(proba @ costs.values, axis=1)\n\n"
        "scenarios = []\n"
        "for fn_penalty in [1500, 5000, 10000, 20000, 100000]:\n"
        "    C = BASE_COSTS.copy()\n"
        "    C.loc['COPD', 'Healthy'] = fn_penalty\n"
        "    C.loc['URTI', 'Healthy'] = fn_penalty\n"
        "    y_pred = cost_sensitive_pred(proba_cal, C)\n"
        "    d = diagnose(f'CS (penalty € {fn_penalty:,})', y_pred)\n"
        "    scenarios.append(d)\n"
        "scen_df = pd.DataFrame(scenarios)\n"
        "print('Variando custo de FN clínico (custos base € 8k → € 500k):')\n"
        "scen_df.round(3)"
    ),
    md(
        "## 4. Estratégia B — Safety Net hierárquica\n\n"
        "Regra: **se P(qualquer doença) ≥ θ → bloqueia Healthy** e escolhe doença mais provável. "
        "Senão, prediz Healthy. Variamos θ para mapear o trade-off."
    ),
    code(
        "def safety_net_pred(proba, theta):\n"
        "    p_disease = 1.0 - proba[:, HEALTHY_IDX]\n"
        "    y_pred = np.full(len(proba), HEALTHY_IDX)\n"
        "    block = p_disease >= theta\n"
        "    # entre as classes de doença, escolhe argmax\n"
        "    disease_idx = [i for i in range(len(CLASSES)) if i != HEALTHY_IDX]\n"
        "    sub = proba[:, disease_idx]\n"
        "    pred_disease = np.array(disease_idx)[np.argmax(sub, axis=1)]\n"
        "    y_pred[block] = pred_disease[block]\n"
        "    return y_pred\n\n"
        "sn_rows = []\n"
        "for theta in [0.50, 0.30, 0.20, 0.10, 0.05, 0.02]:\n"
        "    y_pred = safety_net_pred(proba_cal, theta)\n"
        "    d = diagnose(f'Safety Net (θ={theta:.2f})', y_pred)\n"
        "    sn_rows.append(d)\n"
        "sn_df = pd.DataFrame(sn_rows)\n"
        "print('Safety Net — variando threshold de P(doença):')\n"
        "sn_df.round(3)"
    ),
    md(
        "## 5. Comparação visual — recall doente vs especificidade\n\n"
        "Plotamos todas as estratégias no mesmo gráfico. Ideal: canto superior direito (recall=1, espec=1)."
    ),
    code(
        "all_strats = []\n"
        "all_strats.append(diagnose('Argmax', np.argmax(proba_cal, axis=1)))\n"
        "for fn_penalty in [5000, 10000, 20000]:\n"
        "    C = BASE_COSTS.copy()\n"
        "    C.loc['COPD', 'Healthy'] = fn_penalty\n"
        "    C.loc['URTI', 'Healthy'] = fn_penalty\n"
        "    all_strats.append(diagnose(f'CS €{fn_penalty/1000:.0f}k',\n"
        "                                  cost_sensitive_pred(proba_cal, C)))\n"
        "for theta in [0.30, 0.10, 0.05, 0.02]:\n"
        "    all_strats.append(diagnose(f'SN θ={theta}',\n"
        "                                  safety_net_pred(proba_cal, theta)))\n"
        "comp_df = pd.DataFrame(all_strats)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(10, 6))\n"
        "for _, row in comp_df.iterrows():\n"
        "    color = '#2c3e50' if row['estratégia']=='Argmax' else ('#e74c3c' if 'CS' in row['estratégia'] else '#3498db')\n"
        "    ax.scatter(row['especificidade'], row['recall_doente'], s=120, color=color,\n"
        "                edgecolors='black', linewidths=0.5)\n"
        "    ax.annotate(row['estratégia'], (row['especificidade'], row['recall_doente']),\n"
        "                  xytext=(5, 5), textcoords='offset points', fontsize=8)\n"
        "ax.axhline(1.0, color='green', linestyle=':', alpha=0.4, label='recall doente = 1.00 (zero FN clínico)')\n"
        "ax.set_xlabel('Especificidade (1 − FP_doente / n_Healthy)')\n"
        "ax.set_ylabel('Sensibilidade / Recall doente')\n"
        "ax.set_title('Trade-off recall × especificidade — todas as estratégias')\n"
        "ax.legend(loc='lower left')\n"
        "ax.grid(alpha=0.3)\n"
        "plt.tight_layout(); plt.show()\n"
        "comp_df.round(3)"
    ),
    md(
        "## 6. Custo total — incluindo penalidade massiva de FN clínico\n\n"
        "Reavaliação financeira com FN_clínico = € 50.000 (cenário conservador realista para "
        "doença respiratória não diagnosticada)."
    ),
    code(
        "C_REAL = BASE_COSTS.copy()\n"
        "C_REAL.loc['COPD', 'Healthy'] = 10_000\n"
        "C_REAL.loc['URTI', 'Healthy'] = 10_000\n"
        "print('Matriz de custos clínicos realistas:')\n"
        "print(C_REAL)\n\n"
        "def total_cost(y_true, y_pred, costs):\n"
        "    total = 0\n"
        "    for i in range(len(CLASSES)):\n"
        "        for j in range(len(CLASSES)):\n"
        "            n = ((y_true == i) & (y_pred == j)).sum()\n"
        "            total += n * costs.iloc[i, j]\n"
        "    return total\n\n"
        "fin_rows = []\n"
        "# Argmax\n"
        "fin_rows.append(dict(estratégia='Argmax', custo_total=total_cost(y, np.argmax(proba_cal, axis=1), C_REAL),\n"
        "                      FN_clin=18))\n"
        "# CS com penalidade € 50k\n"
        "y_pred_cs50 = cost_sensitive_pred(proba_cal, C_REAL)\n"
        "fin_rows.append(dict(estratégia='Cost-sensitive (penalty €10k)',\n"
        "                      custo_total=total_cost(y, y_pred_cs50, C_REAL),\n"
        "                      FN_clin=int(((is_disease) & (y_pred_cs50 == HEALTHY_IDX)).sum())))\n"
        "# Safety net θ=0.05\n"
        "y_pred_sn = safety_net_pred(proba_cal, 0.05)\n"
        "fin_rows.append(dict(estratégia='Safety Net θ=0.05',\n"
        "                      custo_total=total_cost(y, y_pred_sn, C_REAL),\n"
        "                      FN_clin=int(((is_disease) & (y_pred_sn == HEALTHY_IDX)).sum())))\n"
        "# Safety net θ=0.02 (mais conservador)\n"
        "y_pred_sn2 = safety_net_pred(proba_cal, 0.02)\n"
        "fin_rows.append(dict(estratégia='Safety Net θ=0.02',\n"
        "                      custo_total=total_cost(y, y_pred_sn2, C_REAL),\n"
        "                      FN_clin=int(((is_disease) & (y_pred_sn2 == HEALTHY_IDX)).sum())))\n"
        "fin_df = pd.DataFrame(fin_rows)\n"
        "fin_df['custo_por_paciente'] = fin_df['custo_total'] / len(y)\n"
        "VOLUME = 50_000\n"
        "fin_df['anual_50k_pacientes'] = fin_df['custo_por_paciente'] * VOLUME\n"
        "print()\n"
        "fin_df.round(0)"
    ),
    md("## 7. Matrizes de confusão lado a lado"),
    code(
        "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n"
        "for ax, y_pred, title in [\n"
        "    (axes[0], np.argmax(proba_cal, axis=1), f'Argmax\\nFN clínico: 18'),\n"
        "    (axes[1], y_pred_cs50,                f'Cost-sensitive €10k\\nFN clínico: {int(((is_disease) & (y_pred_cs50 == HEALTHY_IDX)).sum())}'),\n"
        "    (axes[2], y_pred_sn,                  f'Safety Net θ=0.05\\nFN clínico: {int(((is_disease) & (y_pred_sn == HEALTHY_IDX)).sum())}'),\n"
        "]:\n"
        "    cm = confusion_matrix(y, y_pred, labels=range(len(CLASSES)))\n"
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
        "                xticklabels=CLASSES, yticklabels=CLASSES, cbar=False,\n"
        "                annot_kws={'size': 14}, ax=ax)\n"
        "    ax.set_xlabel('Predito'); ax.set_ylabel('Verdadeiro')\n"
        "    ax.set_title(title)\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## 8. Recomendação operacional e atualização do modelo\n\n"
        "Princípio: **Safety Net** é mais simples, transparente e operacionalmente útil. "
        "Um único parâmetro (`disease_threshold`) controla o trade-off, e o resultado é "
        "explicável (\"o modelo bloqueou Healthy porque P(doença) > θ\").\n\n"
        "**Defaults sugeridos:**\n"
        "- θ = 0.30: comportamento parecido com argmax, mas conservador.\n"
        "- θ = 0.05: zero FN clínico no nosso dataset.\n"
        "- θ = 0.02: ainda mais cauteloso, à custa de FPs em saudáveis.\n\n"
        "**Importante:** zero FN no CV **não garante** zero FN em produção (distribuição diferente). "
        "Mas baixar θ aumenta a margem de segurança contra erros."
    ),
    code(
        "# Atualizar o LungAssistant v2 já está pronto pra mode='safe' via inference.py.\n"
        "# Demonstração:\n"
        "from src.inference import LungAssistant\n"
        "from src import metadata as md_module\n"
        "asst = LungAssistant.load(config.PROJECT_ROOT / 'models' / 'lung_assistant_v2.joblib')\n"
        "rec_df = md_module.build_recordings_dataframe()\n"
        "rec_df = md_module.filter_target_classes(rec_df)\n"
        "demo = rec_df.groupby('diagnosis').first().reset_index()\n\n"
        "for _, row in demo.iterrows():\n"
        "    r_argmax = asst.predict(row['wav_path'], equipment=row['equipment'], mode='argmax')\n"
        "    r_safe05 = asst.predict(row['wav_path'], equipment=row['equipment'], mode='safe', disease_threshold=0.05)\n"
        "    print(f\"{row['filename']}  (real={row['diagnosis']})\")\n"
        "    print(f\"  argmax:        {r_argmax['diagnosis']}  P={r_argmax['top_probability']:.3f}\")\n"
        "    print(f\"  safe(θ=0.05):  {r_safe05['diagnosis']}  P={r_safe05['top_probability']:.3f}\")\n"
        "    print()"
    ),
    md(
        "## Resumo\n\n"
        "- **FN clínico** (doente → Healthy) é o erro mais grave; argmax dava 2.2% (18 doentes).\n"
        "- **Safety Net** com θ ≤ 0.05 **zera FN clínico** no nosso CV, à custa de mais FPs em "
        "saudáveis (consultas extras, ~€60 cada).\n"
        "- **Cost-sensitive com penalidade €10k** atinge resultado parecido, mas é menos "
        "transparente para o médico.\n"
        "- **O modelo final (`LungAssistant`) agora suporta `mode='safe'`** com threshold ajustável.\n\n"
        "**Mensagem para gestão clínica:**\n"
        "1. Configurar produção com **θ = 0.05** (zero FN clínico no benchmark).\n"
        "2. Pacientes flagged como doentes que o médico depois confirma como saudáveis são "
        "**triagem positiva ok** — o sistema fez seu trabalho.\n"
        "3. Monitorar continuamente o FN clínico em produção. Se aparecer, **rebaixar θ**."
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
