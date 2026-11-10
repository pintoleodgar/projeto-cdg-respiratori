"""Gera notebooks/11_custo_clinico_threshold.ipynb programaticamente."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "11_custo_clinico_threshold.ipynb"


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
        "# 11 — Análise económica como dimensão **secundária** da decisão clínica\n\n"
        "## ⚕️ Princípio orientador (não-financeiro)\n\n"
        "**A justificação primária para minimizar Falsos Negativos é ética, não económica.** "
        "Em saúde, FN clínico (doente classificado como saudável) é eticamente inaceitável sob "
        "o princípio da **não-maleficência** (*primum non nocere*). Cada FN representa:\n\n"
        "- **Dano por omissão**: paciente vai para casa doente; doença progride.\n"
        "- **Perda funcional irreversível** em DPOC (tecido pulmonar não regenera).\n"
        "- **Negação do direito do paciente** a saber o seu diagnóstico (princípio da autonomia).\n"
        "- **Perpetuação do subdiagnóstico** (86% em Portugal — ONDR) que o sistema deveria combater.\n\n"
        "Por outro lado, FP é **remediável** (médico corrige na consulta).\n\n"
        "## A análise económica é consequência mensurável\n\n"
        "Quantificamos abaixo o impacto financeiro como **forma de comunicar a gestão hospitalar** "
        "a relevância de minimizar FN — não como argumento primário. Os custos abaixo são "
        "**proxies imperfeitos do dano clínico real**:\n\n"
        "- **FN em COPD/URTI**: ~€1.500–4.000 (internação tardia) — mas o custo verdadeiro é "
        "qualidade de vida + sobrevida perdida, não dinheiro.\n"
        "- **FP**: ~€40–80 (consulta extra) — facilmente remediável.\n\n"
        "Por argmax simples (threshold = 0.5 para cada classe) maximizamos **accuracy**, não **valor clínico**.\n"
        "Aqui vamos:\n\n"
        "1. Calcular curvas **ROC e Precision-Recall** por classe (one-vs-rest).\n"
        "2. Ajustar threshold de cada classe para atingir **recall mínimo (95% para Disease, 80% para URTI)**.\n"
        "3. Modelar **custo total** (FN × custo_FN + FP × custo_FP) em função do threshold.\n"
        "4. Encontrar o threshold de menor custo esperado.\n"
        "5. Traduzir em **economia anual** baseada em volume típico de auscultaões em um hospital."
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
        "from sklearn.metrics import (roc_curve, auc, precision_recall_curve,\n"
        "                              confusion_matrix, classification_report)\n\n"
        "from src import config\n"
        "warnings.filterwarnings('ignore')\n"
        "sns.set_theme(style='whitegrid')"
    ),
    md("## 1. Reproduzir as probabilidades OOF do pipeline final"),
    code(
        "feats_df = pd.read_csv(config.PROCESSED_DIR / 'features.csv')\n"
        "FEATURE_COLS = [c for c in feats_df.columns\n"
        "                if c not in ('filename','patient_id','equipment','chest_location','label_multi','label_binary')]\n"
        "X = feats_df[FEATURE_COLS].values\n"
        "le = LabelEncoder(); y = le.fit_transform(feats_df['label_multi'])\n"
        "groups = feats_df['patient_id'].values\n"
        "equipment = feats_df['equipment'].values\n"
        "CLASSES = list(le.classes_)\n"
        "EQUIPMENT_CATS = sorted(set(equipment))\n\n"
        "# Pipeline com residualização + LogReg + calibração isotónica em OOF\n"
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
        "    proba_uncal[te] = clf.predict_proba(Xs_te)\n\n"
        "# Calibração isotónica\n"
        "proba_cal = np.zeros_like(proba_uncal)\n"
        "for k, cls in enumerate(CLASSES):\n"
        "    ir = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(\n"
        "        proba_uncal[:, k], (y == k).astype(float))\n"
        "    proba_cal[:, k] = ir.predict(proba_uncal[:, k])\n"
        "s = proba_cal.sum(axis=1, keepdims=True); s[s==0]=1; proba_cal/=s\n"
        "print(f'Probas calibradas OOF prontas: {proba_cal.shape}')"
    ),
    md(
        "## 2. Curvas ROC e Precision-Recall por classe (one-vs-rest)\n\n"
        "Mostram o trade-off ao variar o threshold."
    ),
    code(
        "fig, axes = plt.subplots(2, 3, figsize=(16, 9))\n"
        "for k, cls in enumerate(CLASSES):\n"
        "    yk = (y == k).astype(int)\n"
        "    pk = proba_cal[:, k]\n"
        "    # ROC\n"
        "    fpr, tpr, thr = roc_curve(yk, pk)\n"
        "    roc_auc = auc(fpr, tpr)\n"
        "    axes[0, k].plot(fpr, tpr, lw=2)\n"
        "    axes[0, k].plot([0,1],[0,1],'--',color='gray')\n"
        "    axes[0, k].set_title(f'ROC — {cls}  AUC={roc_auc:.3f}')\n"
        "    axes[0, k].set_xlabel('FPR (1 - especificidade)')\n"
        "    axes[0, k].set_ylabel('TPR (sensibilidade / recall)')\n"
        "    # PR\n"
        "    prec, rec, thr_pr = precision_recall_curve(yk, pk)\n"
        "    pr_auc = auc(rec, prec)\n"
        "    axes[1, k].plot(rec, prec, lw=2, color='darkorange')\n"
        "    axes[1, k].set_title(f'Precision-Recall — {cls}  AUC-PR={pr_auc:.3f}')\n"
        "    axes[1, k].set_xlabel('Recall'); axes[1, k].set_ylabel('Precision')\n"
        "    # baseline = prevalência\n"
        "    axes[1, k].axhline(yk.mean(), color='gray', linestyle='--',\n"
        "                       label=f'baseline (prev={yk.mean():.2f})')\n"
        "    axes[1, k].legend()\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## 3. Curvas ROC úteis para escolher operação por classe\n\n"
        "**Atenção:** thresholds independentes por classe em multiclasse é problemático. "
        "A abordagem correta é **cost-sensitive decision rule** (seção 5) — minimiza custo esperado "
        "global. Mas antes, é informativo ver, classe a classe, o trade-off recall × precision."
    ),
    code(
        "rows = []\n"
        "for k, cls in enumerate(CLASSES):\n"
        "    yk = (y == k).astype(int)\n"
        "    pk = proba_cal[:, k]\n"
        "    # default 0.5\n"
        "    pred = (pk >= 0.5).astype(int)\n"
        "    tn, fp, fn, tp = confusion_matrix(yk, pred, labels=[0,1]).ravel()\n"
        "    rec_d = tp/(tp+fn) if tp+fn else 0\n"
        "    prec_d = tp/(tp+fp) if tp+fp else 0\n"
        "    rows.append(dict(classe=cls, thr_default=0.5,\n"
        "                      recall_default=rec_d, precision_default=prec_d,\n"
        "                      FN_default=fn, FP_default=fp,\n"
        "                      total_class=yk.sum()))\n"
        "thr_df = pd.DataFrame(rows)\n"
        "print('Operação atual (argmax / threshold 0.5 por classe):')\n"
        "thr_df.round(3)"
    ),
    md(
        "## 4. Matriz de custos — valores reais do SNS português\n\n"
        "Custos fundamentados em dados do Serviço Nacional de Saúde (Portugal):\n\n"
        "| Patologia | Volume anual | Custo unitário | Impacto |\n"
        "|---|---|---|---|\n"
        "| **DPOC** | +10.000 internamentos | **€2.500–3.400** | Internamentos prolongados, oxigenoterapia domiciliária, medicação crónica |\n"
        "| **URTI** | +800.000 ocorrências | **€35–110** | Sobrecarga Linha SNS 24, urgências, absentismo laboral |\n\n"
        "Derivação da matriz de custos C[real, predito] (€):\n\n"
        "- **FN_DPOC (COPD → Healthy)**: €2.950 — média do internamento evitável; doente vai embora sem diagnóstico\n"
        "- **DPOC → URTI**: €1.475 — tratamento parcial, metade do risco do internamento\n"
        "- **FN_URTI (URTI → Healthy)**: €70 — média da consulta/urgência repetida\n"
        "- **FP (Healthy → doente)**: €70 — consulta extra + auscultação presencial\n\n"
        "Como temos **multiclasse**, definimos uma matriz de custos C[real, predito]."
    ),
    code(
        "# Custos em € — dados reais do SNS português (linhas=real, colunas=predito)\n"
        "# DPOC: €2.500-3.400/internamento (média 2.950) | URTI: €35-110/ocorrência (média 70)\n"
        "COSTS = pd.DataFrame(\n"
        "    [[0,     2950,  1475],   # real=COPD: FN grave (internamento evitável)\n"
        "     [70,    0,     70],     # real=Healthy: FP = consulta extra\n"
        "     [70,    70,    0]],     # real=URTI: FN = consulta/urgência repetida\n"
        "    index=CLASSES, columns=CLASSES)\n"
        "print('Matriz de custos clínicos — SNS Portugal (€):')\n"
        "print(COSTS)\n"
        "print()\n"
        "print('Volumes anuais de referência: DPOC ~10.000 internamentos | URTI ~800.000 ocorrências')"
    ),
    code(
        "# Função de custo total\n"
        "def total_cost(y_true, y_pred, costs):\n"
        "    total = 0\n"
        "    for i in range(len(CLASSES)):\n"
        "        for j in range(len(CLASSES)):\n"
        "            n = ((y_true == i) & (y_pred == j)).sum()\n"
        "            total += n * costs.iloc[i, j]\n"
        "    return total\n\n"
        "# A — argmax (otimiza accuracy, ignora custos)\n"
        "y_pred_argmax = np.argmax(proba_cal, axis=1)\n"
        "cost_argmax = total_cost(y, y_pred_argmax, COSTS)\n\n"
        "# B — cost-sensitive decision rule: predito = argmin_j Σ_i C[i,j] * P(real=i|x)\n"
        "# Para cada amostra x, escolhemos a classe que MINIMIZA o custo esperado.\n"
        "C_mat = COSTS.values   # (n_classes_real, n_classes_pred)\n"
        "expected_cost = proba_cal @ C_mat    # shape (n_samples, n_classes_pred)\n"
        "y_pred_cs = np.argmin(expected_cost, axis=1)\n"
        "cost_cs = total_cost(y, y_pred_cs, COSTS)\n\n"
        "print(f'A — argmax (otimiza acc):       € {cost_argmax:,.0f}')\n"
        "print(f'B — cost-sensitive (otimiza €): € {cost_cs:,.0f}')\n"
        "delta = cost_argmax - cost_cs\n"
        "print(f'\\nDelta: € {delta:+,.0f}  ({100*delta/cost_argmax:+.1f}%)')\n"
        "if delta > 0:\n"
        "    print(f'→ Cost-sensitive ECONOMIZA € {delta:,.0f} ({len(y)} pacientes do CV)')\n"
        "else:\n"
        "    print(f'→ Cost-sensitive piorou em € {-delta:,.0f}')"
    ),
    md(
        "## 5. Matriz de confusão lado a lado\n\n"
        "Vemos visualmente onde os erros migram quando ajustamos o threshold."
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n"
        "for ax, y_pred, title in [(axes[0], y_pred_argmax, f'Argmax (custo € {cost_argmax:,.0f})'),\n"
        "                            (axes[1], y_pred_cs,    f'Cost-sensitive (custo € {cost_cs:,.0f})')]:\n"
        "    cm = confusion_matrix(y, y_pred, labels=range(len(CLASSES)))\n"
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
        "                xticklabels=CLASSES, yticklabels=CLASSES, cbar=False,\n"
        "                annot_kws={'size': 14}, ax=ax)\n"
        "    ax.set_xlabel('Predito'); ax.set_ylabel('Verdadeiro')\n"
        "    ax.set_title(title)\n"
        "plt.tight_layout(); plt.show()\n\n"
        "# Comparativo FN/FP por classe doente\n"
        "comparativo = []\n"
        "for cls in CLASSES:\n"
        "    k = CLASSES.index(cls)\n"
        "    fn_a = ((y == k) & (y_pred_argmax != k)).sum()\n"
        "    fp_a = ((y != k) & (y_pred_argmax == k)).sum()\n"
        "    fn_b = ((y == k) & (y_pred_cs != k)).sum()\n"
        "    fp_b = ((y != k) & (y_pred_cs == k)).sum()\n"
        "    comparativo.append(dict(classe=cls,\n"
        "                              FN_argmax=fn_a, FP_argmax=fp_a,\n"
        "                              FN_cs=fn_b, FP_cs=fp_b,\n"
        "                              delta_FN=fn_b - fn_a, delta_FP=fp_b - fp_a))\n"
        "print('\\nFN/FP por classe (cost-sensitive vs argmax):')\n"
        "pd.DataFrame(comparativo)"
    ),
    md(
        "## 6. Projeção de impacto financeiro anual (escala nacional SNS)\n\n"
        "Considerando o volume nacional de **~810.000 ocorrências respiratórias/ano** em Portugal "
        "(10.000 internamentos DPOC + 800.000 ocorrências URTI). Usamos 50.000 como cenário de "
        "**adoção parcial** (~6% do volume nacional, ou um conjunto de ULS de médio porte)."
    ),
    code(
        "VOLUME_ANUAL = 50_000  # cenário de adoção parcial (~6% do volume nacional respiratório)\n"
        "n = len(y)\n"
        "cost_per_pat_argmax = cost_argmax / n\n"
        "cost_per_pat_cs     = cost_cs / n\n"
        "annual_argmax = cost_per_pat_argmax * VOLUME_ANUAL\n"
        "annual_cs     = cost_per_pat_cs     * VOLUME_ANUAL\n"
        "print(f'Custo médio por paciente:')\n"
        "print(f'  argmax          : € {cost_per_pat_argmax:,.2f}')\n"
        "print(f'  cost-sensitive  : € {cost_per_pat_cs:,.2f}')\n"
        "print()\n"
        "print(f'Projeção anual ({VOLUME_ANUAL:,} pacientes):')\n"
        "print(f'  argmax          : € {annual_argmax:,.0f}')\n"
        "print(f'  cost-sensitive  : € {annual_cs:,.0f}')\n"
        "print(f'  ECONOMIA        : € {(annual_argmax - annual_cs):,.0f} / ano')"
    ),
    code(
        "# Sensitivity: e se o custo de FN_COPD for diferente?\n"
        "# IMPORTANTE: re-otimizar a decisão para cada custo (a regra cost-sensitive depende da matriz)\n"
        "fn_costs = [500, 1000, 1500, 2500, 4000]\n"
        "savings = []\n"
        "for c in fn_costs:\n"
        "    C2 = COSTS.copy()\n"
        "    C2.loc['COPD', 'Healthy'] = c\n"
        "    C2.loc['COPD', 'URTI']    = c // 2\n"
        "    # recomputar decisão cost-sensitive com nova matriz\n"
        "    exp_c = proba_cal @ C2.values\n"
        "    y_pred_cs2 = np.argmin(exp_c, axis=1)\n"
        "    s_arg = total_cost(y, y_pred_argmax, C2)\n"
        "    s_cs  = total_cost(y, y_pred_cs2,    C2)\n"
        "    annual_save = (s_arg - s_cs) / n * VOLUME_ANUAL\n"
        "    fn_copd = ((y == CLASSES.index('COPD')) & (y_pred_cs2 != CLASSES.index('COPD'))).sum()\n"
        "    savings.append((c, annual_save, fn_copd))\n"
        "sens_df = pd.DataFrame(savings, columns=['custo_FN_COPD_EUR', 'economia_anual_EUR', 'FN_COPD_cs']).round(0)\n"
        "print('Sensibilidade ao custo de FN_COPD (decisão re-otimizada):')\n"
        "print(sens_df)\n\n"
        "fig, ax = plt.subplots(figsize=(9, 4))\n"
        "ax.plot(sens_df['custo_FN_COPD_EUR'], sens_df['economia_anual_EUR'], 'o-', lw=2)\n"
        "ax.set_xlabel('Custo assumido de FN_COPD (€)')\n"
        "ax.set_ylabel('Economia anual (€)')\n"
        "ax.set_title(f'Economia esperada vs custo assumido de FN_COPD (volume={VOLUME_ANUAL:,}/ano)')\n"
        "ax.grid(alpha=0.3)\n"
        "ax.axhline(0, color='black', linewidth=0.5)\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## 7. Resumo executivo\n\n"
        "**Estratégia correta: Cost-Sensitive Decision Rule.**\n"
        "Em vez de thresholds isolados por classe (quebra em multiclasse), usamos:\n"
        "$$\\hat y = \\arg\\min_j \\sum_i C_{ij}\\,P(y=i\\mid x)$$\n"
        "ou seja, predizemos a classe com **menor custo esperado** dada a matriz de custos clínicos.\n\n"
        "**Por que isso funciona melhor:**\n"
        "- Considera **as 3 probabilidades simultaneamente**.\n"
        "- Decisão final coerente: se P(COPD) é 30% mas custo de errar é alto, ainda pode "
        "compensar predizer COPD.\n"
        "- Não \"força\" recall alto a qualquer custo — o trade-off vem dos parâmetros da matriz.\n\n"
        "**Para gestão / decisão clínica:**\n"
        "1. O modelo pode ser **deployado como triagem**: pacientes preditos como COPD pelo cost-sensitive "
        "rule passam por avaliação prioritária.\n"
        "2. Pacientes com `confidence == 'alta'` e `outlier == False` (do nb09) podem ser **fast-tracked**.\n"
        "3. Pacientes com confiança baixa ou outlier: **encaminhados para auscultaão presencial** com médico.\n"
        "4. **A matriz de custos é um parâmetro de governança** — não é só técnica, é decisão "
        "gerencial/clínica que define o ponto de operação. Pode ser ajustada conforme realidade local "
        "do hospital (custos de internação reais, capacidade ambulatorial, etc.)."
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
