"""Paleta de cores e utilitários de visualização consistentes.

Importar em todos os notebooks:
    from src.plotting import CLASS_COLORS, EQUIPMENT_COLORS, set_style, savefig
    set_style()  # aplica seaborn whitegrid + fontes
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Cores fixas por classe — mantidas em TODOS os notebooks e na app Streamlit
CLASS_COLORS = {
    "Healthy": "#2ecc71",   # verde
    "COPD":    "#e74c3c",   # vermelho
    "URTI":    "#f39c12",   # laranja
    "Disease": "#9b59b6",   # roxo (binário)
}

# Cores fixas por equipamento
EQUIPMENT_COLORS = {
    "AKG C417L Microphone":     "#3498db",  # azul
    "WelchAllyn Meditron":      "#e67e22",  # laranja escuro
    "Littmann Classic II SE":   "#16a085",  # verde-azulado
    "Littmann 3200 Electronic": "#8e44ad",  # roxo escuro
}

# Cores por nível de urgência clínica
URGENCY_COLORS = {
    "NENHUMA":         "#27ae60",
    "BAIXA":           "#f1c40f",
    "MODERADA":        "#e67e22",
    "ALTA":            "#c0392b",
    "EMERGÊNCIA":      "#c0392b",
    "REVISÃO MANUAL":  "#e67e22",
}

# Paleta para modelos (ordem fixa para boxplots comparativos)
MODEL_PALETTE = {
    "Dummy":         "#95a5a6",
    "KNN":           "#3498db",
    "LDA":           "#1abc9c",
    "LogReg":        "#9b59b6",
    "XGBoost":       "#e67e22",
    "CNN":           "#34495e",
    "Cascata":       "#e74c3c",
    "Ensemble":      "#f39c12",
}


def set_style():
    """Aplica estilo consistente (seaborn whitegrid + fontes legíveis)."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
    })


def savefig(name: str, fig=None, figures_dir: Path | None = None, tight: bool = True):
    """Salva figura em reports/figures/ com nome padronizado.

    Args:
        name: nome do ficheiro sem extensão (ex: 'nb04_boxplot_recall')
        fig: figura matplotlib (default: gcf())
        figures_dir: pasta destino (default: src.config.FIGURES_DIR)
    """
    if figures_dir is None:
        from src.config import FIGURES_DIR
        figures_dir = FIGURES_DIR
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    f = fig or plt.gcf()
    if tight:
        f.tight_layout()
    path = figures_dir / f"{name}.png"
    f.savefig(path, dpi=150, bbox_inches="tight")
    return path
