"""Utilitários de bootstrap para intervalos de confiança não-paramétricos.

Uso típico:
    from src.bootstrap import bootstrap_ci, paired_bootstrap_diff

    # IC95% da média de uma métrica
    mu, lo, hi = bootstrap_ci(recall_values_per_fold, n=2000)

    # Diferença pareada entre modelos
    delta, lo, hi, p = paired_bootstrap_diff(recall_A_per_fold, recall_B_per_fold, n=2000)
"""
from __future__ import annotations
import numpy as np


def bootstrap_ci(
    values, n: int = 2000, ci: float = 95.0, seed: int = 0, agg=np.mean
) -> tuple[float, float, float]:
    """IC bootstrap percentil de uma agregação (default: média).

    Returns:
        (estimativa_pontual, lo, hi)
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = rng.choice(arr, size=(n, len(arr)), replace=True)
    samples = np.apply_along_axis(agg, 1, boot)
    alpha = (100 - ci) / 2
    lo, hi = np.percentile(samples, [alpha, 100 - alpha])
    return float(agg(arr)), float(lo), float(hi)


def paired_bootstrap_diff(
    a, b, n: int = 2000, ci: float = 95.0, seed: int = 0
) -> tuple[float, float, float, float]:
    """Bootstrap pareado para diff(a - b). Útil quando a e b foram avaliados
    nos mesmos folds (pareamento real).

    Returns:
        (delta_observado, lo, hi, p_value_bilateral)

    p-value bilateral = 2 × proporção de boots em que (sign(diff_boot) != sign(diff_observed))
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"a e b devem ter mesmo tamanho ({len(a)} vs {len(b)})")
    diffs = a - b
    delta_obs = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    boot_diffs = rng.choice(diffs, size=(n, len(diffs)), replace=True).mean(axis=1)
    alpha = (100 - ci) / 2
    lo, hi = np.percentile(boot_diffs, [alpha, 100 - alpha])
    # p-value bilateral: proporção de boots cruzando 0 vs sinal observado
    if delta_obs >= 0:
        p = 2 * np.mean(boot_diffs <= 0)
    else:
        p = 2 * np.mean(boot_diffs >= 0)
    p = float(min(p, 1.0))
    return delta_obs, float(lo), float(hi), p


def fmt_ci(mu: float, lo: float, hi: float, digits: int = 3) -> str:
    """Formata como '0.647 [0.589, 0.631]'."""
    return f"{mu:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
