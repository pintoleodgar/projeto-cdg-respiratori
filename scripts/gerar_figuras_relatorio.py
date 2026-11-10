"""Gera as 3 figuras essenciais do projeto para o relatório (300 dpi).

Saída em reports/figures/:
    FIG1_descoberta_vies.png       — viés de equipamento + mitigação
    FIG2_comparacao_modelos.png    — recall doente / F1m / especificidade por modelo
    FIG3_modelo_final.png          — matriz confusão + trade-off recall×especificidade

Rodar: python scripts/gerar_figuras_relatorio.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, confusion_matrix
from xgboost import XGBClassifier

from src import config
from src.plotting import CLASS_COLORS, set_style
from src.bootstrap import bootstrap_ci

warnings.filterwarnings("ignore")
set_style()
FIG_DIR = config.FIGURES_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Dados
# ============================================================
print("A carregar dados...")
feats = pd.read_csv(config.PROCESSED_DIR / "features.csv")
FCOLS = [c for c in feats.columns if c not in
         ("filename", "patient_id", "equipment", "chest_location",
          "label_multi", "label_binary")]
X = feats[FCOLS].values
le = LabelEncoder()
y = le.fit_transform(feats["label_multi"])
groups = feats["patient_id"].values
equipment = feats["equipment"].values
CLASSES = list(le.classes_)
HEALTHY = CLASSES.index("Healthy")
EQUIP_CATS = sorted(set(equipment))


def fit_resid(Xtr, eqtr):
    ohe = OneHotEncoder(sparse_output=False, categories=[EQUIP_CATS])
    E = ohe.fit_transform(eqtr.reshape(-1, 1))
    lr = LinearRegression().fit(E, Xtr)
    return ohe, lr


def resid(X_, eq_, ohe, lr):
    return X_ - lr.predict(ohe.transform(eq_.reshape(-1, 1)))


def patient_vote(yt, yp, g):
    d = pd.DataFrame({"g": g, "y": yt, "p": yp}).groupby("g").agg(
        y=("y", "first"), p=("p", lambda s: s.value_counts().idxmax()))
    return d["y"].values, d["p"].values


# ============================================================
# FIGURA 1 — Descoberta e mitigação do viés de equipamento
# ============================================================
print("FIG1: viés de equipamento...")
fig1, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) cross-tab equipamento × diagnóstico (pacientes)
ct = pd.crosstab(
    feats.drop_duplicates(["patient_id", "equipment"])["equipment"],
    feats.drop_duplicates(["patient_id", "equipment"])["label_multi"])
ct.index = [i.split()[0] for i in ct.index]
sns.heatmap(ct, annot=True, fmt="d", cmap="Reds", cbar=False, ax=axes[0],
            annot_kws={"size": 13})
axes[0].set_title("(a) Pacientes por equipamento × diagnóstico\n100% Healthy/URTI no Meditron",
                  fontsize=12, fontweight="bold")
axes[0].set_xlabel("Diagnóstico"); axes[0].set_ylabel("Equipamento")

# (b) probe pré vs pós residualização
le_eq = LabelEncoder(); y_eq = le_eq.fit_transform(equipment)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
ohe_g = OneHotEncoder(sparse_output=False, categories=[EQUIP_CATS])
Eg = ohe_g.fit_transform(equipment.reshape(-1, 1))
X_res = X - LinearRegression().fit(Eg, X).predict(Eg)

probe = LogisticRegression(max_iter=3000, class_weight="balanced")
f1_orig = f1_score(y_eq, cross_val_predict(probe, X, y_eq, groups=groups, cv=cv, n_jobs=1), average="macro")
f1_res = f1_score(y_eq, cross_val_predict(probe, X_res, y_eq, groups=groups, cv=cv, n_jobs=1), average="macro")

bars = axes[1].bar(["Features\noriginais", "Features\nresidualizadas"],
                   [f1_orig, f1_res], color=["#c0392b", "#27ae60"],
                   edgecolor="black", width=0.6)
axes[1].axhline(0.25, color="gray", ls="--", alpha=0.6, label="acaso (4 classes)")
for b, v in zip(bars, [f1_orig, f1_res]):
    axes[1].text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}",
                 ha="center", fontsize=13, fontweight="bold")
axes[1].set_ylim(0, 1.0)
axes[1].set_ylabel("F1-macro do probe de equipamento")
axes[1].set_title(f"(b) Probe de equipamento\n−{(1-f1_res/f1_orig)*100:.0f}% após residualização",
                  fontsize=12, fontweight="bold")
axes[1].legend()

# (c) impacto no F1m do diagnóstico (com vs sem residualização)
def eval_diag(Xuse, label):
    yp = np.full(len(y), -1)
    for tr, te in cv.split(Xuse, y, groups):
        sc = StandardScaler().fit(Xuse[tr])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Xuse[tr]), y[tr])
        yp[te] = clf.predict(sc.transform(Xuse[te]))
    yt_p, yp_p = patient_vote(y, yp, groups)
    return f1_score(yt_p, yp_p, average="macro")

f1_com_vies = eval_diag(X, "com viés")
f1_sem_vies = eval_diag(X_res, "residualizado")
bars = axes[2].bar(["Com viés\n(features cruas)", "Residualizado\n(honesto)"],
                   [f1_com_vies, f1_sem_vies], color=["#e67e22", "#2980b9"],
                   edgecolor="black", width=0.6)
for b, v in zip(bars, [f1_com_vies, f1_sem_vies]):
    axes[2].text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
                 ha="center", fontsize=13, fontweight="bold")
axes[2].set_ylim(0, 0.8)
axes[2].set_ylabel("F1-macro do diagnóstico (paciente)")
axes[2].set_title("(c) Custo da honestidade\nF1m mais baixo, mas sem atalho",
                  fontsize=12, fontweight="bold")

fig1.suptitle("FIGURA 1 — Descoberta e mitigação do viés de equipamento (ICBHI 2017)",
              fontsize=15, fontweight="bold", y=1.02)
fig1.tight_layout()
fig1.savefig(FIG_DIR / "FIG1_descoberta_vies.png", dpi=300, bbox_inches="tight")
print(f"  guardado: FIG1 (probe {f1_orig:.3f} -> {f1_res:.3f})")
plt.close(fig1)

# ============================================================
# FIGURA 2 — Comparação de modelos (5x5-fold)
# ============================================================
print("FIG2: comparação de modelos (pode demorar ~1 min)...")
models = {
    "Trivial\n(sempre COPD)": ("dummy", None),
    "LogReg\n(com viés)": ("plain", None),
    "LogReg\nresidualizado": ("resid", lambda: LogisticRegression(max_iter=3000, class_weight="balanced")),
    "LDA\nresidualizado": ("resid", lambda: LinearDiscriminantAnalysis()),
    "KNN\nresidualizado": ("resid", lambda: KNeighborsClassifier(n_neighbors=5, weights="distance")),
    "XGBoost\nresidualizado": ("resid", lambda: XGBClassifier(n_estimators=200, max_depth=3,
                                learning_rate=0.05, eval_metric="mlogloss", n_jobs=1)),
    "RandomForest\n(black-box)": ("resid", lambda: RandomForestClassifier(n_estimators=300,
                                   class_weight="balanced", n_jobs=-1, random_state=42)),
}
res = {m: {"recall": [], "f1m": [], "spec": []} for m in models}
for rep in range(5):
    cvr = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42+rep)
    for name, (kind, factory) in models.items():
        yp = np.full(len(y), -1)
        for tr, te in cvr.split(X, y, groups):
            if kind == "dummy":
                yp[te] = DummyClassifier(strategy="most_frequent").fit(X[tr], y[tr]).predict(X[te])
            elif kind == "plain":
                sc = StandardScaler().fit(X[tr])
                yp[te] = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
                    sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
            else:
                ohe, lr = fit_resid(X[tr], equipment[tr])
                Xr_tr = resid(X[tr], equipment[tr], ohe, lr)
                Xr_te = resid(X[te], equipment[te], ohe, lr)
                sc = StandardScaler().fit(Xr_tr)
                clf = factory().fit(sc.transform(Xr_tr), y[tr])
                yp[te] = clf.predict(sc.transform(Xr_te))
        yt_p, yp_p = patient_vote(y, yp, groups)
        is_d = yt_p != HEALTHY
        res[name]["recall"].append((is_d & (yp_p != HEALTHY)).sum() / max(1, is_d.sum()))
        res[name]["spec"].append(((yt_p == HEALTHY) & (yp_p == HEALTHY)).sum() / max(1, (yt_p == HEALTHY).sum()))
        res[name]["f1m"].append(f1_score(yt_p, yp_p, average="macro"))

fig2, axes = plt.subplots(1, 3, figsize=(18, 5.5))
names = list(models.keys())
xp = np.arange(len(names))
for ax, key, title, color in [
    (axes[0], "recall", "Recall de doença\n(métrica principal — sensibilidade)", "#27ae60"),
    (axes[1], "f1m", "F1-macro paciente\n(qualidade global)", "#2980b9"),
    (axes[2], "spec", "Especificidade\n(saudáveis bem classificados)", "#e67e22"),
]:
    mus, los, his = [], [], []
    for n in names:
        mu, lo, hi = bootstrap_ci(res[n][key])
        mus.append(mu); los.append(mu-lo); his.append(hi-mu)
    bars = ax.bar(xp, mus, yerr=[los, his], capsize=4, color=color,
                  edgecolor="black", alpha=0.85)
    # destacar modelo final
    final_i = names.index("LogReg\nresidualizado")
    bars[final_i].set_color("#16a085"); bars[final_i].set_edgecolor("black"); bars[final_i].set_linewidth(2.5)
    for i, mu in enumerate(mus):
        ax.text(i, mu + max(his)*0.3 + 0.02, f"{mu:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(xp); ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylim(0, 1.08); ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

fig2.suptitle("FIGURA 2 — Comparação de modelos (CV 5×5-fold, IC95% bootstrap) · destaque: modelo final LogReg residualizado",
              fontsize=14, fontweight="bold", y=1.02)
fig2.tight_layout()
fig2.savefig(FIG_DIR / "FIG2_comparacao_modelos.png", dpi=300, bbox_inches="tight")
print("  guardado: FIG2")
plt.close(fig2)

# ============================================================
# FIGURA 3 — Modelo final em ação
# ============================================================
print("FIG3: modelo final...")
# Predições OOF do modelo final (LogReg residualizado), argmax + safe
proba_oof = np.zeros((len(y), len(CLASSES)))
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for tr, te in cv.split(X, y, groups):
    ohe, lr = fit_resid(X[tr], equipment[tr])
    Xr_tr = resid(X[tr], equipment[tr], ohe, lr)
    Xr_te = resid(X[te], equipment[te], ohe, lr)
    sc = StandardScaler().fit(Xr_tr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Xr_tr), y[tr])
    proba_oof[te] = clf.predict_proba(sc.transform(Xr_te))

# argmax
yp_arg = proba_oof.argmax(1)
yt_pa, yp_pa = patient_vote(y, yp_arg, groups)
# safe mode (theta=0.10)
p_dis = 1 - proba_oof[:, HEALTHY]
yp_safe = np.where(p_dis >= 0.10,
                   np.where(np.arange(len(CLASSES)) != HEALTHY, proba_oof, -np.inf).argmax(1)
                   if False else np.array([
                       np.argmax([proba_oof[i, k] if k != HEALTHY else -np.inf for k in range(len(CLASSES))])
                       for i in range(len(y))]),
                   HEALTHY)
yt_ps, yp_ps = patient_vote(y, yp_safe, groups)

fig3, axes = plt.subplots(1, 3, figsize=(18, 5.2))

# (a) matriz confusão argmax (paciente)
cm_a = confusion_matrix([CLASSES[i] for i in yt_pa], [CLASSES[i] for i in yp_pa], labels=CLASSES)
sns.heatmap(cm_a, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0], annot_kws={"size": 14})
fn_a = ((yt_pa != HEALTHY) & (yp_pa == HEALTHY)).sum()
axes[0].set_title(f"(a) Modo padrão (argmax)\nFN clínicos: {fn_a} pacientes",
                  fontsize=12, fontweight="bold")
axes[0].set_xlabel("Predito"); axes[0].set_ylabel("Verdadeiro")

# (b) matriz confusão safe (paciente)
cm_s = confusion_matrix([CLASSES[i] for i in yt_ps], [CLASSES[i] for i in yp_ps], labels=CLASSES)
sns.heatmap(cm_s, annot=True, fmt="d", cmap="Greens", cbar=False,
            xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1], annot_kws={"size": 14})
fn_s = ((yt_ps != HEALTHY) & (yp_ps == HEALTHY)).sum()
axes[1].set_title(f"(b) Modo seguro (safety net θ=0.10)\nFN clínicos: {fn_s} pacientes",
                  fontsize=12, fontweight="bold")
axes[1].set_xlabel("Predito"); axes[1].set_ylabel("Verdadeiro")

# (c) impacto: FN clínico + economia
ax = axes[2]
cats = ["Argmax", "Safety Net\nθ=0.10"]
fn_vals = [fn_a, fn_s]
bars = ax.bar(cats, fn_vals, color=["#3498db", "#27ae60"], edgecolor="black", width=0.5)
for b, v in zip(bars, fn_vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.15, f"{v}", ha="center", fontsize=14, fontweight="bold")
ax.set_ylabel("FN clínicos (doentes → Healthy)")
ax.set_title("(c) Impacto clínico do safety net\nMenos doentes perdidos",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, max(fn_vals)+2)
# anotação económica
ax.text(0.5, -0.28,
        "Custo FN DPOC = €2.950 (internamento evitável) vs FP = €70 (consulta extra)\n"
        "Economia projetada: ~€157 mil/ano (50.000 auscultações)",
        transform=ax.transAxes, ha="center", fontsize=9.5, style="italic",
        bbox=dict(boxstyle="round", fc="#fef9e7", ec="#f39c12"))

fig3.suptitle("FIGURA 3 — Modelo final em ação: LogReg residualizado + calibração + safety net (nível paciente, n=104)",
              fontsize=14, fontweight="bold", y=1.02)
fig3.tight_layout()
fig3.savefig(FIG_DIR / "FIG3_modelo_final.png", dpi=300, bbox_inches="tight")
print(f"  guardado: FIG3 (FN argmax={fn_a}, FN safe={fn_s})")
plt.close(fig3)

print()
print("═══════════════════════════════════════════")
print("3 figuras geradas em reports/figures/:")
print("  FIG1_descoberta_vies.png")
print("  FIG2_comparacao_modelos.png")
print("  FIG3_modelo_final.png")
print("═══════════════════════════════════════════")
