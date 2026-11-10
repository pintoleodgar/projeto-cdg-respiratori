"""Pipeline final de inferência do assistente respiratório.

Modelo: LogReg sobre features residualizadas pelo equipamento, com
probabilidades calibradas. O modelo de produção (v3) é gerado por
`python scripts/train.py`; as métricas estão em reports/AVALIACAO_V3.md.

Notas:
- Probe de equipamento (nb06) atinge F1-macro ~0.9 nas features cruas → forte vazamento.
- A residualização remove a componente LINEAR desse viés; uma sonda não-linear
  ainda recupera o equipamento (F1-macro ~0.87). O viés está mitigado, não eliminado.
- O áudio é cortado a config.MAX_DURATION_S e `duration_s` não é feature.

Uso:
    >>> assistant = LungAssistant.load("models/lung_assistant_v3.npz")
    >>> result = assistant.predict("audio.wav", equipment="WelchAllyn Meditron")
    >>> result["diagnosis"], result["probabilities"], result["confidence"]
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .config import MIN_DURATION_S
from .features import extract_from_file

MODES = ("argmax", "safe")

# Avisos que acompanham TODAS as predições (app, CLI e API).
CONTRAINDICATIONS = (
    "Não usar em crianças: o desempenho pediátrico é fraco (nb15).",
    "Um resultado 'Healthy' não exclui URTI (recall URTI por paciente 0,24–0,59).",
    "Healthy e URTI só foram vistos em gravações Meditron; noutros equipamentos "
    "o modelo só conhece COPD.",
)


class InvalidAudioError(ValueError):
    """Áudio que não pode ser analisado (curto demais, silencioso, corrompido)."""


@dataclass
class LungAssistant:
    """Modelo treinado + utilitários de inferência."""

    feature_cols: list[str]
    equipment_categories: list[str]    # ordem das categorias na residualização
    residualizer_coef_: np.ndarray     # (n_features, n_equipment) — sklearn lr.coef_
    residualizer_intercept_: np.ndarray  # (n_features,)
    scaler_mean_: np.ndarray
    scaler_scale_: np.ndarray
    logreg_classes_: np.ndarray         # ['COPD','Healthy','URTI'] (labels)
    logreg_coef_: np.ndarray            # (n_classes, n_features)
    logreg_intercept_: np.ndarray       # (n_classes,)
    # class_label -> (x, y): tabela de calibração aplicada com np.interp.
    # Pode vir de isotónica (v2) ou de sigmoide/Platt amostrada numa grelha (v3);
    # o nome do campo mantém-se por compatibilidade com os modelos já guardados.
    isotonic_calibrators_: dict
    # Outlier detector (Mahalanobis no espaço residualizado+escalado)
    train_mean_: np.ndarray | None = None       # (n_features,)
    train_cov_inv_: np.ndarray | None = None    # (n_features, n_features)
    # Quantil EMPÍRICO p99 de d² no treino (scripts/train.py). Não é χ²(0.99):
    # as features não são gaussianas e o χ² dava ~19% de falsos alarmes.
    outlier_threshold_99_: float | None = None

    CONFIDENCE_THRESHOLDS = (0.6, 0.8)  # baixa < 0.6 ≤ média < 0.8 ≤ alta

    # ----------------------------------------------------------------
    def _residualize(self, x: np.ndarray, equipment: str) -> np.ndarray:
        """Remove a componente de x explicada pelo equipamento."""
        eq_onehot = np.zeros(len(self.equipment_categories))
        if equipment not in self.equipment_categories:
            raise ValueError(
                f"equipment '{equipment}' não está nas categorias treinadas: "
                f"{self.equipment_categories}"
            )
        eq_onehot[self.equipment_categories.index(equipment)] = 1.0
        # predição linear da feature dado o equipamento (sklearn: coef_ shape = (n_out, n_in))
        predicted = self.residualizer_intercept_ + self.residualizer_coef_ @ eq_onehot
        return x - predicted

    def _scale(self, x: np.ndarray) -> np.ndarray:
        return (x - self.scaler_mean_) / self.scaler_scale_

    def _logreg_proba(self, x: np.ndarray) -> np.ndarray:
        """Softmax sobre logits."""
        logits = self.logreg_coef_ @ x + self.logreg_intercept_
        m = logits.max()
        e = np.exp(logits - m)
        return e / e.sum()

    def _apply_isotonic(self, raw_proba: np.ndarray) -> np.ndarray:
        out = np.zeros_like(raw_proba)
        for i, cls in enumerate(self.logreg_classes_):
            x_thr, y_thr = self.isotonic_calibrators_[cls]
            out[i] = float(np.interp(raw_proba[i], x_thr, y_thr))
        s = out.sum()
        return out / s if s > 0 else raw_proba

    # ----------------------------------------------------------------
    def _mahalanobis_sq(self, x: np.ndarray) -> float:
        """Distância de Mahalanobis ao quadrado no espaço residualizado+escalado."""
        if self.train_mean_ is None or self.train_cov_inv_ is None:
            return float("nan")
        d = x - self.train_mean_
        return float(d @ self.train_cov_inv_ @ d)

    # ----------------------------------------------------------------
    # Extração e validação
    # ----------------------------------------------------------------
    def extract(self, audio_path: str | Path) -> dict:
        """Extrai e valida as features de um ficheiro (passo lento: fazer uma só vez).

        Levanta InvalidAudioError se o áudio não puder ser analisado.
        """
        try:
            feats = extract_from_file(audio_path)
        except Exception as exc:  # ficheiro ilegível, vazio ou não-áudio
            raise InvalidAudioError(f"Não foi possível ler o áudio '{audio_path}': {exc}") from exc
        self._validate(feats, audio_path)
        return feats

    def _validate(self, feats: dict, source="áudio") -> None:
        dur = feats.get("duration_s", float("nan"))
        if not dur >= MIN_DURATION_S:
            raise InvalidAudioError(
                f"'{source}': duração {dur:.2f} s inferior ao mínimo de {MIN_DURATION_S:.0f} s."
            )
        if not feats.get("rms_mean", 0.0) > 0.0:
            raise InvalidAudioError(f"'{source}': gravação silenciosa (energia nula).")
        missing = [c for c in self.feature_cols if c not in feats]
        if missing:
            raise InvalidAudioError(f"'{source}': features em falta: {missing[:5]}")
        bad = [c for c in self.feature_cols if not np.isfinite(feats[c])]
        if bad:
            raise InvalidAudioError(f"'{source}': features não finitas: {bad[:5]}")

    def _vector(self, feats: dict, equipment: str) -> np.ndarray:
        """features (dict) → vetor residualizado e escalado."""
        self._validate(feats)
        x = np.array([feats[c] for c in self.feature_cols], dtype=float)
        return self._scale(self._residualize(x, equipment))

    # ----------------------------------------------------------------
    # Decisão
    # ----------------------------------------------------------------
    @staticmethod
    def _check_mode(mode: str, disease_threshold: float) -> None:
        # Um modo mal escrito não pode cair silenciosamente em argmax:
        # o utilizador julgaria ter a safety net ligada.
        if mode not in MODES:
            raise ValueError(f"mode '{mode}' inválido; usar um de {MODES}")
        if mode == "safe" and not 0.0 < disease_threshold < 1.0:
            raise ValueError(f"disease_threshold deve estar em (0, 1); recebido {disease_threshold}")

    def _decide(self, proba: np.ndarray, mode: str, disease_threshold: float,
                healthy_label: str = "Healthy") -> tuple[int, bool]:
        """Devolve (índice da classe, safe_blocked_healthy).

        safe_blocked_healthy é True só quando o argmax era Healthy e a safety
        net o substituiu por uma doença.
        """
        idx = int(np.argmax(proba))
        if mode != "safe":
            return idx, False
        healthy_idx = [str(c) for c in self.logreg_classes_].index(healthy_label)
        if 1.0 - proba[healthy_idx] < disease_threshold:
            return healthy_idx, False
        masked = proba.astype(float).copy()
        masked[healthy_idx] = -np.inf
        return int(np.argmax(masked)), idx == healthy_idx

    def _confidence(self, top: float) -> str:
        lo, hi = self.CONFIDENCE_THRESHOLDS
        return "baixa" if top < lo else "média" if top < hi else "alta"

    def _outlier_info(self, x: np.ndarray) -> dict | None:
        if self.train_cov_inv_ is None:
            return None
        d2 = self._mahalanobis_sq(x)
        thr = self.outlier_threshold_99_ if self.outlier_threshold_99_ is not None else float("inf")
        return {"is_outlier": bool(d2 > thr), "mahalanobis_sq": d2, "threshold_99": thr}

    # ----------------------------------------------------------------
    # API a partir de features já extraídas
    # ----------------------------------------------------------------
    def predict_features(
        self,
        feats: dict,
        equipment: str,
        check_outlier: bool = True,
        mode: str = "argmax",
        disease_threshold: float = 0.30,
        healthy_label: str = "Healthy",
    ) -> dict:
        """Como `predict`, mas a partir do dict devolvido por `extract`."""
        self._check_mode(mode, disease_threshold)
        x = self._vector(feats, equipment)
        proba_raw = self._logreg_proba(x)
        proba_cal = self._apply_isotonic(proba_raw)
        idx, blocked = self._decide(proba_cal, mode, disease_threshold, healthy_label)

        top = float(proba_cal[idx])
        conf = self._confidence(top)
        outlier_info = self._outlier_info(x) if check_outlier else None
        if outlier_info and outlier_info["is_outlier"]:
            conf = "baixa (amostra fora da distribuição)"

        return {
            "diagnosis": str(self.logreg_classes_[idx]),
            "confidence": conf,
            "top_probability": top,
            "probabilities": {
                str(c): float(p) for c, p in zip(self.logreg_classes_, proba_cal)
            },
            "probabilities_raw": {
                str(c): float(p) for c, p in zip(self.logreg_classes_, proba_raw)
            },
            "equipment_used": equipment,
            "outlier": outlier_info,
            "mode": mode,
            "safe_blocked_healthy": blocked,
            "contraindications": list(CONTRAINDICATIONS),
        }

    def explain_features(self, feats: dict, equipment: str, top_k: int = 8) -> dict:
        """Como `explain`, mas a partir do dict devolvido por `extract`."""
        x = self._vector(feats, equipment)
        contribs = self.logreg_coef_ * x  # (n_classes, n_features)
        out = {}
        for i, cls in enumerate(self.logreg_classes_):
            ranking = np.argsort(np.abs(contribs[i]))[::-1][:top_k]
            out[str(cls)] = [
                {
                    "feature": self.feature_cols[j],
                    "value_scaled": float(x[j]),
                    "coef": float(self.logreg_coef_[i, j]),
                    "contribution": float(contribs[i, j]),
                }
                for j in ranking
            ]
        return out

    # ----------------------------------------------------------------
    # API a partir de ficheiros
    # ----------------------------------------------------------------
    def is_outlier(self, audio_path: str | Path, equipment: str) -> dict:
        """Detecta se a amostra está fora da distribuição de treino."""
        info = self._outlier_info(self._vector(self.extract(audio_path), equipment))
        return info or {"is_outlier": False, "mahalanobis_sq": float("nan"),
                        "threshold_99": float("inf")}

    def explain(
        self, audio_path: str | Path, equipment: str, top_k: int = 8
    ) -> dict:
        """Explica a predição: contribuição de cada feature ao logit de cada classe.

        Contribuição da feature i para a classe k = coef_[k, i] * x[i] (com x já escalado).
        """
        return self.explain_features(self.extract(audio_path), equipment, top_k=top_k)

    def predict(
        self,
        audio_path: str | Path,
        equipment: str,
        check_outlier: bool = True,
        mode: str = "argmax",
        disease_threshold: float = 0.30,
        healthy_label: str = "Healthy",
    ) -> dict:
        """Inferência completa: áudio → diagnóstico + probabilidades calibradas + outlier.

        Parâmetros:
            mode:
                - 'argmax': decisão padrão (mais provável)
                - 'safe':   modo conservador — só prediz Healthy se P(qualquer doença) < disease_threshold.
                            Caso contrário, escolhe a doença mais provável.
            disease_threshold: threshold da P(doente_total) para autorizar predição Healthy (modo 'safe').

        Levanta ValueError para modo/equipamento inválidos e InvalidAudioError
        (subclasse de ValueError) para áudio não analisável.
        """
        self._check_mode(mode, disease_threshold)  # falhar antes da extração (lenta)
        return self.predict_features(
            self.extract(audio_path), equipment, check_outlier=check_outlier,
            mode=mode, disease_threshold=disease_threshold, healthy_label=healthy_label,
        )

    def predict_patient(
        self,
        audio_paths: list[str | Path],
        equipment: str | list[str],
        mode: str = "argmax",
        disease_threshold: float = 0.30,
        healthy_label: str = "Healthy",
    ) -> dict:
        """Diagnóstico por PACIENTE a partir de várias gravações.

        Usa a mesma regra da avaliação (reports/AVALIACAO_V3.md): média das
        probabilidades calibradas das gravações, seguida de argmax ou safety net.
        `equipment` pode ser um nome único ou uma lista com um por gravação.
        """
        self._check_mode(mode, disease_threshold)
        paths = list(audio_paths)
        if not paths:
            raise ValueError("predict_patient precisa de pelo menos uma gravação")
        eqs = [equipment] * len(paths) if isinstance(equipment, str) else list(equipment)
        if len(eqs) != len(paths):
            raise ValueError(f"{len(paths)} gravações mas {len(eqs)} equipamentos")

        recs = [self.predict_features(self.extract(p), e) for p, e in zip(paths, eqs)]
        classes = [str(c) for c in self.logreg_classes_]
        proba = np.mean([[r["probabilities"][c] for c in classes] for r in recs], axis=0)
        idx, blocked = self._decide(proba, mode, disease_threshold, healthy_label)
        top = float(proba[idx])
        n_out = sum(bool(r["outlier"] and r["outlier"]["is_outlier"]) for r in recs)
        conf = self._confidence(top)
        if n_out:
            conf = f"baixa ({n_out}/{len(recs)} gravações fora da distribuição)"
        return {
            "diagnosis": classes[idx],
            "confidence": conf,
            "top_probability": top,
            "probabilities": dict(zip(classes, map(float, proba))),
            "n_recordings": len(recs),
            "n_outliers": n_out,
            "mode": mode,
            "safe_blocked_healthy": blocked,
            "contraindications": list(CONTRAINDICATIONS),
            "recordings": [
                {"audio_path": str(p), **r} for p, r in zip(paths, recs)
            ],
        }

    # ----------------------------------------------------------------
    # Persistência
    # ----------------------------------------------------------------
    # Formato .npz: só arrays numéricos + um cabeçalho JSON. Ao contrário do
    # pickle/joblib, não executa código ao carregar e não depende do caminho
    # da classe nem da versão do scikit-learn.
    FORMAT_VERSION = 1
    _ARRAYS = ("residualizer_coef_", "residualizer_intercept_", "scaler_mean_",
               "scaler_scale_", "logreg_coef_", "logreg_intercept_",
               "train_mean_", "train_cov_inv_")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("Guardar em .npz (o formato .joblib só é lido, por compatibilidade)")
        classes = [str(c) for c in self.logreg_classes_]
        header = {
            "format_version": self.FORMAT_VERSION,
            "feature_cols": list(self.feature_cols),
            "equipment_categories": list(self.equipment_categories),
            "classes": classes,
            "outlier_threshold_99": self.outlier_threshold_99_,
        }
        arrays = {k: np.asarray(getattr(self, k), dtype=float)
                  for k in self._ARRAYS if getattr(self, k) is not None}
        for i, c in enumerate(classes):
            x, y = self.isotonic_calibrators_[c]
            arrays[f"cal_x_{i}"], arrays[f"cal_y_{i}"] = np.asarray(x, float), np.asarray(y, float)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, header=np.array(json.dumps(header, ensure_ascii=False)), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "LungAssistant":
        path = Path(path)
        if path.suffix == ".joblib":      # modelos antigos (v1/v2) — só ficheiros de confiança
            import joblib
            obj = joblib.load(path)
            if not isinstance(obj, cls):
                raise TypeError(f"Arquivo não contém um {cls.__name__}")
            return obj
        with np.load(path, allow_pickle=False) as z:
            header = json.loads(str(z["header"]))
            if header.get("format_version") != cls.FORMAT_VERSION:
                raise ValueError(f"Versão de formato não suportada: {header.get('format_version')}")
            classes = header["classes"]
            return cls(
                feature_cols=header["feature_cols"],
                equipment_categories=header["equipment_categories"],
                logreg_classes_=np.array(classes),
                isotonic_calibrators_={c: (z[f"cal_x_{i}"], z[f"cal_y_{i}"])
                                       for i, c in enumerate(classes)},
                outlier_threshold_99_=header["outlier_threshold_99"],
                **{k: (z[k] if k in z.files else None) for k in cls._ARRAYS},
            )
