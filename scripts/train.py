#!/usr/bin/env python
"""Treino e avaliação reprodutíveis do modelo de produção (v3).

    python scripts/train.py            # usa data/processed/features_v3.csv se existir
    python scripts/train.py --extract  # força nova extração a partir dos .wav

Produz:
    data/processed/features_v3.csv     features com áudio cortado a MAX_DURATION_S
    models/lung_assistant_v3.npz       modelo de produção
    reports/metrics_v3.json            todas as métricas (máquina)
    reports/AVALIACAO_V3.md            as mesmas métricas (humano)

Diferenças face ao v2 (nbs 08/09):
  * áudio cortado a 20 s e `duration_s` fora das features (era atalho);
  * calibração ANINHADA: o calibrador é ajustado só com dados do treino de
    cada fold e medido no teste desse fold;
  * residualizador e scaler reajustados dentro de cada fold interno;
  * métricas por PACIENTE e por classe, com IC95% por bootstrap de pacientes;
  * avaliação separada dentro do Meditron (único equipamento com as 3 classes);
  * sonda de equipamento linear E não-linear;
  * custos reponderados por prevalências assumidas, não pela do dataset;
  * limiar de outlier = quantil empírico p99, calculado aqui.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.inference import LungAssistant  # noqa: E402

warnings.filterwarnings("ignore")

FEATURES_CSV = config.PROCESSED_DIR / "features_v3.csv"
MODEL_PATH = config.MODEL_PATH
METRICS_JSON = config.REPORTS_DIR / "metrics_v3.json"
REPORT_MD = config.REPORTS_DIR / "AVALIACAO_V3.md"

CLASSES = ["COPD", "Healthy", "URTI"]          # ordem alfabética = ordem do modelo
HEALTHY = CLASSES.index("Healthy")
MEDITRON = "WelchAllyn Meditron"
N_REPS, N_FOLDS, N_BOOT = 5, 5, 2000
SAFE_THETAS = (0.30, 0.10, 0.05)

# Custos em € (linhas = real, colunas = predito), iguais aos do nb11.
COSTS = np.array([[0, 2950, 1475],
                  [70, 0, 70],
                  [70, 70, 0]], dtype=float)
VOLUME_ANUAL = 50_000
# Prevalências ASSUMIDAS por cenário (COPD, Healthy, URTI). São hipóteses de
# trabalho para análise de sensibilidade — não foram medidas.
PREVALENCE_SCENARIOS = {
    "dataset ICBHI (não realista)": None,  # preenchido com a do dataset
    "rastreio população geral (5/85/10)": (0.05, 0.85, 0.10),
    "cuidados primários (10/70/20)": (0.10, 0.70, 0.20),
    "consulta de pneumologia (40/40/20)": (0.40, 0.40, 0.20),
}


# ------------------------------------------------------------------ dados
def extract_features() -> pd.DataFrame:
    from joblib import Parallel, delayed
    from src import features, metadata

    df = metadata.add_target_labels(
        metadata.filter_target_classes(metadata.build_recordings_dataframe())
    )
    print(f"A extrair features de {len(df)} gravações (corte a {config.MAX_DURATION_S:.0f} s)…")
    feats = Parallel(n_jobs=-1, verbose=1)(
        delayed(features.extract_from_file)(p) for p in df["wav_path"]
    )
    out = pd.DataFrame(feats)
    for c in ("filename", "patient_id", "equipment", "chest_location",
              "label_multi", "label_binary"):
        out[c] = df[c].values
    out.to_csv(FEATURES_CSV, index=False)
    return out


def load_data(force_extract: bool):
    df = extract_features() if (force_extract or not FEATURES_CSV.exists()) \
        else pd.read_csv(FEATURES_CSV)
    cols = [c for c in df.columns if c not in config.NON_FEATURE_COLS]
    X = df[cols].to_numpy(float)
    y = np.array([CLASSES.index(c) for c in df["label_multi"]])
    g = df["patient_id"].to_numpy()
    eq_cats = sorted(df["equipment"].unique())
    E = (df["equipment"].to_numpy()[:, None] == np.array(eq_cats)[None, :]).astype(float)
    return df, cols, X, y, g, eq_cats, E


# ------------------------------------------------------------- pipeline
def fit_pre(X, E):
    lr = LinearRegression().fit(E, X)
    sc = StandardScaler().fit(X - lr.predict(E))
    return lr, sc


def apply_pre(X, E, lr, sc):
    return sc.transform(X - lr.predict(E))


def fit_clf(Xs, y):
    return LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xs, y)


def oof_proba(X, E, y, g, seed):
    """Probabilidades out-of-fold com TODO o pré-processamento dentro do fold."""
    out = np.zeros((len(y), len(CLASSES)))
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for tr, va in cv.split(X, y, g):
        lr, sc = fit_pre(X[tr], E[tr])
        out[va] = fit_clf(apply_pre(X[tr], E[tr], lr, sc), y[tr]).predict_proba(
            apply_pre(X[va], E[va], lr, sc))
    return out


# ----------------------------------------------------------- calibração
_GRID = np.concatenate([[0.0], 1 / (1 + np.exp(-np.linspace(-12, 12, 2401))), [1.0]])


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_calibrator(method: str, proba_oof, y):
    """Devolve {classe: (x, y)} — tabelas para np.interp, uma por classe (OvR)."""
    tables = {}
    for k, cls in enumerate(CLASSES):
        t = (y == k).astype(float)
        if method == "nenhuma":
            tables[cls] = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        elif method == "isotonica":
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(proba_oof[:, k], t)
            tables[cls] = (ir.X_thresholds_.copy(), ir.y_thresholds_.copy())
        elif method == "sigmoide":
            pl = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(proba_oof[:, [k]]), t)
            tables[cls] = (_GRID, pl.predict_proba(_logit(_GRID[:, None]))[:, 1])
        else:
            raise ValueError(method)
    return tables


def apply_calibrator(tables, proba):
    out = np.column_stack([np.interp(proba[:, k], *tables[c]) for k, c in enumerate(CLASSES)])
    s = out.sum(axis=1, keepdims=True)
    return np.where(s > 0, out / np.where(s > 0, s, 1), proba)


CAL_METHODS = ("nenhuma", "isotonica", "sigmoide")


# -------------------------------------------------------------- métricas
def log_loss_brier(proba, y):
    p = np.clip(proba, 1e-6, 1)
    p = p / p.sum(axis=1, keepdims=True)
    ll = -np.mean(np.log(p[np.arange(len(y)), y]))
    brier = np.mean(((proba - np.eye(len(CLASSES))[y]) ** 2).sum(axis=1))
    return float(ll), float(brier)


def patient_table(proba, y, g, mask=None):
    """Agrega por paciente: média das probabilidades das suas gravações."""
    d = pd.DataFrame(proba, columns=CLASSES).assign(g=g, y=y)
    if mask is not None:
        d = d[mask]
    agg = d.groupby("g").agg({**{c: "mean" for c in CLASSES}, "y": "first"})
    return agg[CLASSES].to_numpy(), agg["y"].to_numpy()


def decide(P, theta=None):
    """argmax (theta=None) ou safety net: Healthy só se P(doente) < theta."""
    pred = P.argmax(axis=1)
    if theta is not None:
        dis = P.copy()
        dis[:, HEALTHY] = -np.inf
        pred = np.where(1 - P[:, HEALTHY] >= theta, dis.argmax(axis=1), HEALTHY)
    return pred


def cm_metrics(cm):
    """Métricas a partir de matrizes de confusão (..., 3, 3) [real, predito]."""
    cm = cm.astype(float)
    row = cm.sum(-1)
    col = cm.sum(-2)
    diag = np.einsum("...ii->...i", cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where(row > 0, diag / row, np.nan)
        prec = np.where(col > 0, diag / col, 0.0)
        f1 = np.where(rec + prec > 0, 2 * rec * prec / (rec + prec), 0.0)
        dis = [i for i in range(len(CLASSES)) if i != HEALTHY]
        fn_clin = cm[..., dis, HEALTHY].sum(-1)
        n_dis = row[..., dis].sum(-1)
        rec_doente = np.where(n_dis > 0, 1 - fn_clin / n_dis, np.nan)
    return {
        "recall_doente": rec_doente,
        "recall_COPD": rec[..., 0],
        "recall_URTI": rec[..., 2],
        "especificidade": rec[..., HEALTHY],
        "f1_macro": np.nanmean(f1, axis=-1),
        "fn_clinico": fn_clin,
    }


def onehot_cm(y_true, preds):
    """preds (reps, P) -> one-hot (reps, P, 9) da célula [real, predito]."""
    cell = y_true[None, :] * len(CLASSES) + preds
    return np.eye(len(CLASSES) ** 2)[cell]


def boot_indices(y_true, rng):
    """Reamostragem de PACIENTES, estratificada por classe."""
    idx = np.empty((N_BOOT, len(y_true)), dtype=int)
    pos = 0
    for k in range(len(CLASSES)):
        members = np.where(y_true == k)[0]
        idx[:, pos:pos + len(members)] = rng.choice(members, size=(N_BOOT, len(members)))
        pos += len(members)
    return idx


def summarize(y_true, preds, idx):
    """Estimativa (média das repetições) + IC95% bootstrap de pacientes."""
    oh = onehot_cm(y_true, preds)                              # (reps, P, 9)
    k = len(CLASSES)
    point = {m: float(np.nanmean(v)) for m, v in
             cm_metrics(oh.sum(1).reshape(-1, k, k)).items()}
    boot_cm = oh[:, idx].sum(2).reshape(len(preds), N_BOOT, k, k)
    out = {}
    for m, v in cm_metrics(boot_cm).items():
        lo, hi = np.nanpercentile(np.nanmean(v, axis=0), [2.5, 97.5])
        out[m] = {"valor": point[m], "ic95": [float(lo), float(hi)]}
    return out, oh, boot_cm


def cost_per_patient(cm, prev):
    """Custo esperado por paciente com taxas por classe do modelo e prevalência dada."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = cm / cm.sum(-1, keepdims=True)
    return np.nansum(np.asarray(prev)[:, None] * rates * COSTS, axis=(-1, -2))


# ---------------------------------------------------------------- sondas
def equipment_probes(X, E, g):
    """Uma sonda consegue adivinhar o equipamento? Linear vs não-linear."""
    ye = E.argmax(axis=1)
    res = {k: [] for k in ("linear_cru", "linear_residualizado",
                           "randomforest_cru", "randomforest_residualizado")}
    for tr, te in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=0).split(X, ye, g):
        lr = LinearRegression().fit(E[tr], X[tr])
        for name, a, b in (("cru", X[tr], X[te]),
                           ("residualizado", X[tr] - lr.predict(E[tr]), X[te] - lr.predict(E[te]))):
            sc = StandardScaler().fit(a)
            lin = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(a), ye[tr])
            rf = RandomForestClassifier(300, random_state=0, class_weight="balanced").fit(a, ye[tr])
            res[f"linear_{name}"].append(f1_score(ye[te], lin.predict(sc.transform(b)), average="macro"))
            res[f"randomforest_{name}"].append(f1_score(ye[te], rf.predict(b), average="macro"))
    return {k: float(np.mean(v)) for k, v in res.items()}


# --------------------------------------------------------------- avaliação
def evaluate(X, E, y, g, meditron_mask):
    n = len(y)
    proba = {m: np.zeros((N_REPS, n, len(CLASSES))) for m in CAL_METHODS}
    outlier_flags = []
    for rep in range(N_REPS):
        cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=42 + rep)
        for tr, te in cv.split(X, y, g):
            lr, sc = fit_pre(X[tr], E[tr])
            Xs_tr, Xs_te = apply_pre(X[tr], E[tr], lr, sc), apply_pre(X[te], E[te], lr, sc)
            raw = fit_clf(Xs_tr, y[tr]).predict_proba(Xs_te)
            inner = oof_proba(X[tr], E[tr], y[tr], g[tr], seed=1000 + rep)
            for m in CAL_METHODS:
                proba[m][rep, te] = apply_calibrator(fit_calibrator(m, inner, y[tr]), raw)
            if rep == 0:  # taxa real de falso alarme do detetor de outlier
                mu, cinv, thr = fit_outlier(Xs_tr)
                d = Xs_te - mu
                outlier_flags.append(np.einsum("ij,jk,ik->i", d, cinv, d) > thr)
        print(f"  repetição {rep + 1}/{N_REPS} concluída")

    cal = {}
    for m in CAL_METHODS:
        v = np.array([log_loss_brier(proba[m][r], y) for r in range(N_REPS)])
        cal[m] = {"log_loss": float(v[:, 0].mean()), "brier": float(v[:, 1].mean())}
    best = min(cal, key=lambda m: cal[m]["log_loss"])

    rng = np.random.default_rng(0)
    results, cost_inputs = {}, {}
    for subset, mask in (("todos", None), ("so_meditron", meditron_mask)):
        tabs = [patient_table(proba[best][r], y, g, mask) for r in range(N_REPS)]
        y_pat = tabs[0][1]
        idx = boot_indices(y_pat, rng)
        results[subset] = {"n_pacientes": {c: int((y_pat == k).sum()) for k, c in enumerate(CLASSES)},
                           "estrategias": {}}
        for name, theta in [("argmax", None)] + [(f"safe θ={t:.2f}", t) for t in SAFE_THETAS]:
            preds = np.stack([decide(P, theta) for P, _ in tabs])
            summ, oh, boot_cm = summarize(y_pat, preds, idx)
            results[subset]["estrategias"][name] = summ
            if subset == "todos":
                k = len(CLASSES)
                cost_inputs[name] = (oh.sum(1).reshape(-1, k, k), boot_cm, y_pat)
    return {
        "calibracao_aninhada": cal,
        "calibracao_escolhida": best,
        "pacientes": results,
        "outlier_falso_alarme_fora_do_treino": float(np.concatenate(outlier_flags).mean()),
    }, best, cost_inputs


def cost_analysis(cost_inputs):
    out = {}
    a_cm, a_boot, y_pat = cost_inputs["argmax"]
    s_cm, s_boot, _ = cost_inputs["safe θ=0.10"]
    prev_ds = np.bincount(y_pat, minlength=len(CLASSES)) / len(y_pat)
    for name, prev in PREVALENCE_SCENARIOS.items():
        prev = prev_ds if prev is None else np.asarray(prev)
        ca, cs = cost_per_patient(a_cm, prev).mean(), cost_per_patient(s_cm, prev).mean()
        diff = (cost_per_patient(a_boot, prev) - cost_per_patient(s_boot, prev)).mean(0) * VOLUME_ANUAL
        lo, hi = np.percentile(diff, [2.5, 97.5])
        out[name] = {
            "prevalencia": [float(p) for p in prev],
            "custo_por_paciente_argmax": float(ca),
            "custo_por_paciente_safe_0.10": float(cs),
            "poupanca_anual_safe_vs_argmax": float((ca - cs) * VOLUME_ANUAL),
            "poupanca_anual_ic95": [float(lo), float(hi)],
        }
    return out


# --------------------------------------------------------- modelo final
def fit_outlier(Xs):
    mu = Xs.mean(axis=0)
    cinv = np.linalg.pinv(np.cov(Xs.T) + 1e-4 * np.eye(Xs.shape[1]))
    d = Xs - mu
    thr = float(np.quantile(np.einsum("ij,jk,ik->i", d, cinv, d), 0.99))  # p99 empírico
    return mu, cinv, thr


def train_final(X, E, y, g, cols, eq_cats, cal_method):
    lr, sc = fit_pre(X, E)
    Xs = apply_pre(X, E, lr, sc)
    clf = fit_clf(Xs, y)
    tables = fit_calibrator(cal_method, oof_proba(X, E, y, g, seed=42), y)
    mu, cinv, thr = fit_outlier(Xs)
    return LungAssistant(
        feature_cols=cols,
        equipment_categories=eq_cats,
        residualizer_coef_=lr.coef_,
        residualizer_intercept_=lr.intercept_,
        scaler_mean_=sc.mean_,
        scaler_scale_=sc.scale_,
        logreg_classes_=np.array([CLASSES[i] for i in clf.classes_]),
        logreg_coef_=clf.coef_,
        logreg_intercept_=clf.intercept_,
        isotonic_calibrators_=tables,
        train_mean_=mu,
        train_cov_inv_=cinv,
        outlier_threshold_99_=thr,
    )


# ---------------------------------------------------------------- relatório
def _fmt(d, pct=True):
    v, (lo, hi) = d["valor"], d["ic95"]
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]" if pct else f"{v:.1f} [{lo:.1f}, {hi:.1f}]"


def _eur(x):
    return ("−€" if x < 0 else "€") + f"{abs(x):,.0f}".replace(",", ".")


def write_report(m):
    L = ["# Avaliação do modelo v3", "",
         "Gerado por `python scripts/train.py` — não editar à mão.", "",
         f"- Gravações: {m['dados']['n_gravacoes']} | pacientes: {m['dados']['n_pacientes']} "
         f"| features: {m['dados']['n_features']} (áudio cortado a {config.MAX_DURATION_S:.0f} s, sem `duration_s`)",
         f"- Validação: {N_REPS}×{N_FOLDS}-fold estratificado por paciente; "
         f"IC95% por bootstrap de pacientes ({N_BOOT} reamostragens).",
         "- Decisão por paciente: média das probabilidades calibradas das suas gravações.", ""]
    for subset, title in (("todos", "Todos os pacientes"),
                          ("so_meditron", "Só gravações Meditron (único equipamento com as 3 classes)")):
        r = m["avaliacao"]["pacientes"][subset]
        L += [f"## {title}", "",
              "Pacientes: " + ", ".join(f"{c} {n}" for c, n in r["n_pacientes"].items()), "",
              "| Estratégia | Recall doente | Recall COPD | Recall URTI | Especificidade | F1-macro | FN clínicos |",
              "|---|---|---|---|---|---|---|"]
        for name, s in r["estrategias"].items():
            L.append(f"| {name} | {_fmt(s['recall_doente'])} | {_fmt(s['recall_COPD'])} | "
                     f"{_fmt(s['recall_URTI'])} | {_fmt(s['especificidade'])} | "
                     f"{_fmt(s['f1_macro'])} | {_fmt(s['fn_clinico'], pct=False)} |")
        L.append("")
    L += ["Recall doente = doentes (COPD ou URTI) não classificados como Healthy. "
          "FN clínicos = n.º médio de pacientes doentes dados como Healthy.", "",
          "## Calibração (aninhada, por gravação)", "",
          "| Método | Log loss | Brier |", "|---|---|---|"]
    for k, v in m["avaliacao"]["calibracao_aninhada"].items():
        star = " ← usado" if k == m["avaliacao"]["calibracao_escolhida"] else ""
        L.append(f"| {k}{star} | {v['log_loss']:.3f} | {v['brier']:.3f} |")
    p = m["sondas_equipamento_f1_macro"]
    L += ["", "## Viés de equipamento — sonda que tenta adivinhar o equipamento", "",
          "| Sonda | Features cruas | Features residualizadas |", "|---|---|---|",
          f"| Linear (LogReg) | {p['linear_cru']:.3f} | {p['linear_residualizado']:.3f} |",
          f"| Não-linear (RandomForest) | {p['randomforest_cru']:.3f} | {p['randomforest_residualizado']:.3f} |",
          "", "A residualização linear só remove a componente linear do viés.", "",
          "## Detetor de outlier", "",
          f"- Limiar (p99 empírico no treino): {m['modelo']['outlier_threshold_p99']:.1f}",
          f"- Falso alarme medido fora do treino (CV): "
          f"{100 * m['avaliacao']['outlier_falso_alarme_fora_do_treino']:.1f}%", "",
          f"## Custos — safe θ=0.10 vs argmax, {_eur(VOLUME_ANUAL)[1:]} pacientes/ano", "",
          "As prevalências são **hipóteses**, não medições. Valores negativos = o modo safe sai mais caro.", "",
          "| Cenário (COPD/Healthy/URTI) | €/paciente argmax | €/paciente safe | Poupança anual [IC95%] |",
          "|---|---|---|---|"]
    for name, c in m["custos"].items():
        lo, hi = c["poupanca_anual_ic95"]
        L.append(f"| {name} | {c['custo_por_paciente_argmax']:.2f} | "
                 f"{c['custo_por_paciente_safe_0.10']:.2f} | "
                 f"{_eur(c['poupanca_anual_safe_vs_argmax'])} [{_eur(lo)}, {_eur(hi)}] |")
    L += ["", "## Limitações que estes números não resolvem", "",
          "- Healthy e URTI só existem em gravações Meditron; fora dele o modelo só viu COPD.",
          "- 14 pacientes URTI e 26 Healthy: os intervalos são largos e devem ser reportados.",
          "- Não há conjunto de teste independente: todos os dados participaram na escolha do modelo.",
          "- Desempenho pediátrico fraco (nb15): não usar em crianças.", ""]
    REPORT_MD.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", action="store_true", help="reextrair features dos .wav")
    args = ap.parse_args()

    config.ensure_dirs()
    df, cols, X, y, g, eq_cats, E = load_data(args.extract)
    print(f"X={X.shape} | pacientes={len(np.unique(g))} | equipamentos={eq_cats}")
    meditron_mask = (df["equipment"] == MEDITRON).to_numpy()

    print("Avaliação com calibração aninhada…")
    avaliacao, best, cost_inputs = evaluate(X, E, y, g, meditron_mask)
    print("Sondas de equipamento…")
    probes = equipment_probes(X, E, g)
    print(f"Modelo final (calibração: {best})…")
    model = train_final(X, E, y, g, cols, eq_cats, best)
    model.save(MODEL_PATH)

    metrics = {
        "dados": {"n_gravacoes": int(len(y)), "n_pacientes": int(len(np.unique(g))),
                  "n_features": len(cols), "max_duration_s": config.MAX_DURATION_S},
        "avaliacao": avaliacao,
        "sondas_equipamento_f1_macro": probes,
        "custos": cost_analysis(cost_inputs),
        "modelo": {"ficheiro": MODEL_PATH.name, "calibracao": best,
                   "outlier_threshold_p99": model.outlier_threshold_99_},
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(metrics)
    print(f"Guardado: {MODEL_PATH.name}, {METRICS_JSON.name}, {REPORT_MD.name}")


if __name__ == "__main__":
    main()
