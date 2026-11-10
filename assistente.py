#!/usr/bin/env python
"""
Assistente de Diagnóstico Respiratório — modelo final
======================================================

Pipeline: librosa (áudio cortado a 20 s) → 77 features → residualização
         (equipamento) → scaler → LogReg multinomial → calibração sigmoide
         → safety net opcional

Modelo gerado por `python scripts/train.py`; métricas em reports/AVALIACAO_V3.md.

Uso:
----
    # 1 áudio
    python assistente.py paciente.wav --equipment "WelchAllyn Meditron"

    # modo conservador (reduz FN clínico à custa de especificidade)
    python assistente.py paciente.wav --equipment "WelchAllyn Meditron" --safe

    # batch (vários áudios)
    python assistente.py audio1.wav audio2.wav --equipment "AKG C417L Microphone"

    # como módulo
    >>> from assistente import diagnose
    >>> r = diagnose("paciente.wav", equipment="WelchAllyn Meditron", safe=True)

Equipamentos suportados:
    - "AKG C417L Microphone"
    - "WelchAllyn Meditron"
    - "Littmann Classic II SE"
    - "Littmann 3200 Electronic"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# --- garantir que src está no path ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import CONTRAINDICATIONS, LungAssistant  # noqa: E402
from src.metadata import EQUIPMENTS  # noqa: E402
from src import config  # noqa: E402

DEFAULT_MODEL_PATH = config.MODEL_PATH
DEFAULT_DISEASE_THRESHOLD = 0.10


# ===================================================================
# API programática
# ===================================================================
_assistant_cache: dict[Path, LungAssistant] = {}


def load_assistant(model_path: str | Path = DEFAULT_MODEL_PATH) -> LungAssistant:
    """Carrega o modelo (uma vez por ficheiro e por processo)."""
    key = Path(model_path).resolve()
    if key not in _assistant_cache:
        _assistant_cache[key] = LungAssistant.load(key)
    return _assistant_cache[key]


def diagnose(
    audio_path: str | Path,
    equipment: str,
    safe: bool = False,
    disease_threshold: float = DEFAULT_DISEASE_THRESHOLD,
    explain: bool = True,
    top_k: int = 5,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Diagnóstico completo de um áudio.

    Retorna dict com: diagnosis, confidence, top_probability, probabilities,
    outlier, mode, e (se explain=True) explanation (top features por classe).
    """
    asst = load_assistant(model_path)
    mode = "safe" if safe else "argmax"
    feats = asst.extract(audio_path)  # extração (lenta) feita uma só vez
    result = asst.predict_features(
        feats,
        equipment=equipment,
        mode=mode,
        disease_threshold=disease_threshold,
        check_outlier=True,
    )
    if explain:
        result["explanation"] = asst.explain_features(feats, equipment=equipment, top_k=top_k)
    result["audio_path"] = str(audio_path)
    return result


def diagnose_batch(
    audios: list[tuple[str | Path, str]],
    safe: bool = False,
    disease_threshold: float = DEFAULT_DISEASE_THRESHOLD,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> list[dict[str, Any]]:
    """Roda diagnose em vários áudios. `audios` = lista de (path, equipment)."""
    return [
        diagnose(p, equipment=eq, safe=safe, disease_threshold=disease_threshold,
                 explain=False, model_path=model_path)
        for p, eq in audios
    ]


# ===================================================================
# Formatação para CLI
# ===================================================================
COLOR = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "red": "\033[91m", "yellow": "\033[93m", "green": "\033[92m",
    "blue": "\033[94m", "magenta": "\033[95m", "cyan": "\033[96m",
}


def _color(text: str, c: str, use_color: bool = True) -> str:
    if not use_color:
        return text
    return f"{COLOR[c]}{text}{COLOR['reset']}"


def _bar(p: float, width: int = 30) -> str:
    return "█" * int(round(p * width)) + "░" * (width - int(round(p * width)))


def _conf_color(conf: str) -> str:
    if "alta" in conf:
        return "green"
    if "média" in conf:
        return "yellow"
    return "red"


def format_report(result: dict[str, Any], use_color: bool = True) -> str:
    """Relatório formatado de uma predição."""
    out = []
    diag = result["diagnosis"]
    conf = result["confidence"]
    top_p = result["top_probability"]
    mode = result["mode"]
    outlier = result.get("outlier") or {}
    is_out = outlier.get("is_outlier", False)

    diag_color = "green" if diag == "Healthy" else "yellow" if diag == "URTI" else "red"
    out.append("═" * 70)
    out.append(f"  DIAGNÓSTICO: {_color(diag, diag_color, use_color)}".center(80))
    out.append("═" * 70)
    out.append(f"Áudio       : {result.get('audio_path', '?')}")
    out.append(f"Equipamento : {result['equipment_used']}")
    out.append(f"Modo        : {mode}")
    if result.get("safe_blocked_healthy"):
        out.append(_color("ℹ️  Safety net: a classe mais provável era Healthy; indicada a doença "
                          "mais provável.", "yellow", use_color))
    out.append(f"Confiança   : {_color(conf, _conf_color(conf), use_color)} (P={top_p:.3f})")

    if is_out:
        out.append(_color("⚠️  OUTLIER: amostra fora da distribuição — revisar manualmente.",
                          "red", use_color))
    out.append(f"  Mahalanobis² = {outlier.get('mahalanobis_sq', float('nan')):.1f}  "
               f"(threshold p99 = {outlier.get('threshold_99', float('nan')):.1f})")

    out.append("")
    out.append(_color("Probabilidades calibradas:", "bold", use_color))
    for cls, p in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        marker = "  ← predito" if cls == diag else ""
        col = diag_color if cls == diag else "dim"
        out.append(f"  {cls:8s} {p:.3f}  {_color(_bar(p), col, use_color)}{marker}")

    if "explanation" in result:
        out.append("")
        out.append(_color(f"Top contribuições para '{diag}':", "bold", use_color))
        for item in result["explanation"][diag]:
            contrib = item["contribution"]
            sign_str = f"{contrib:+.3f}"
            col = "green" if contrib >= 0 else "red"
            out.append(
                f"  {item['feature']:25s}  "
                f"coef={item['coef']:+.3f} × x={item['value_scaled']:+.3f}  "
                f"→ {_color(sign_str, col, use_color)}"
            )
    out.append("")
    out.append(_color("Limitações (ler sempre):", "bold", use_color))
    for item in result.get("contraindications", CONTRAINDICATIONS):
        out.append(f"  ⚠ {item}")
    out.append("═" * 70)
    return "\n".join(out)


# ===================================================================
# CLI
# ===================================================================
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Assistente de Diagnóstico Respiratório (ICBHI 2017)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("audios", nargs="+", help="Caminho(s) para arquivo(s) .wav")
    p.add_argument(
        "--equipment", "-e", required=True, choices=EQUIPMENTS,
        help="Equipamento usado na gravação (obrigatório)",
    )
    p.add_argument(
        "--safe", action="store_true",
        help="Modo conservador: bloqueia predição Healthy se P(doente) ≥ threshold",
    )
    p.add_argument(
        "--threshold", "-t", type=float, default=DEFAULT_DISEASE_THRESHOLD,
        help=f"Threshold de P(doente) no modo safe (default {DEFAULT_DISEASE_THRESHOLD})",
    )
    p.add_argument(
        "--model", "-m", type=str, default=str(DEFAULT_MODEL_PATH),
        help=f"Caminho do modelo .npz (default {DEFAULT_MODEL_PATH})",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Saída em JSON (sem formatação humana)",
    )
    p.add_argument(
        "--no-color", action="store_true",
        help="Desativar cores ANSI no terminal",
    )
    p.add_argument(
        "--no-explain", action="store_true",
        help="Não calcular explicação por feature (mais rápido)",
    )
    p.add_argument(
        "--top-k", type=int, default=5,
        help="Quantas features mostrar na explicação (default 5)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    # validar arquivos
    audios = [Path(a) for a in args.audios]
    missing = [str(a) for a in audios if not a.exists()]
    if missing:
        print(f"Erro: arquivo(s) não encontrado(s): {missing}", file=sys.stderr)
        return 2

    if not Path(args.model).exists():
        print(f"Erro: modelo não encontrado em {args.model}", file=sys.stderr)
        return 2

    if not 0.0 < args.threshold < 1.0:
        print(f"Erro: --threshold deve estar entre 0 e 1 (recebido {args.threshold})", file=sys.stderr)
        return 2

    results = []
    for audio in audios:
        try:
            r = diagnose(
                audio,
                equipment=args.equipment,
                safe=args.safe,
                disease_threshold=args.threshold,
                explain=not args.no_explain,
                top_k=args.top_k,
                model_path=args.model,
            )
        except ValueError as exc:  # inclui InvalidAudioError
            print(f"Erro em {audio}: {exc}", file=sys.stderr)
            return 2
        results.append(r)

    if args.json:
        # JSON-safe: remover possíveis numpy arrays
        print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    else:
        for r in results:
            print(format_report(r, use_color=not args.no_color))
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
