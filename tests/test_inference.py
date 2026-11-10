"""Testes unitários básicos para o pipeline de inferência.

Rodar: pytest tests/ -v
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
from src.inference import LungAssistant
from src import config


MODEL_PATH = config.MODEL_PATH
SAMPLE_AUDIO = (
    config.AUDIO_DIR / "104_1b1_Al_sc_Litt3200.wav"  # COPD conhecido
)
SAMPLE_EQUIPMENT = "Littmann 3200 Electronic"


# ---------------------------------------------------------------------
# Testes da lógica de decisão — modelo sintético, não precisam do dataset
# ---------------------------------------------------------------------
def _toy_assistant() -> LungAssistant:
    """2 features; f1 alto → COPD, f2 alto → URTI, ambos baixos → Healthy."""
    return LungAssistant(
        feature_cols=["f1", "f2"],
        equipment_categories=["A", "B"],
        residualizer_coef_=np.zeros((2, 2)),
        residualizer_intercept_=np.zeros(2),
        scaler_mean_=np.zeros(2),
        scaler_scale_=np.ones(2),
        logreg_classes_=np.array(["COPD", "Healthy", "URTI"]),
        logreg_coef_=np.array([[2.0, 0.0], [-1.0, -1.0], [0.0, 2.0]]),
        logreg_intercept_=np.zeros(3),
        isotonic_calibrators_={c: (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
                               for c in ("COPD", "Healthy", "URTI")},
    )


def _feats(f1, f2, duration=10.0, rms=0.1):
    return {"f1": f1, "f2": f2, "duration_s": duration, "rms_mean": rms}


def test_safe_mode_blocks_healthy_when_disease_probable():
    a = _toy_assistant()
    f = _feats(-0.3, -0.1)  # logits: COPD −0.6, Healthy +0.4, URTI −0.2
    r_arg = a.predict_features(f, "A", mode="argmax")
    assert r_arg["diagnosis"] == "Healthy" and r_arg["safe_blocked_healthy"] is False
    p_disease = 1 - r_arg["probabilities"]["Healthy"]

    r_safe = a.predict_features(f, "A", mode="safe", disease_threshold=p_disease - 0.01)
    assert r_safe["diagnosis"] == "URTI"          # doença mais provável
    assert r_safe["safe_blocked_healthy"] is True

    r_lax = a.predict_features(f, "A", mode="safe", disease_threshold=p_disease + 0.01)
    assert r_lax["diagnosis"] == "Healthy" and r_lax["safe_blocked_healthy"] is False


def test_safe_mode_not_flagged_when_argmax_already_disease():
    r = _toy_assistant().predict_features(_feats(3.0, 0.0), "A", mode="safe", disease_threshold=0.1)
    assert r["diagnosis"] == "COPD" and r["safe_blocked_healthy"] is False


@pytest.mark.parametrize("mode", ["Safe", "seguro", "", "argmax "])
def test_invalid_mode_raises(mode):
    with pytest.raises(ValueError, match="inválido"):
        _toy_assistant().predict_features(_feats(0, 0), "A", mode=mode)


@pytest.mark.parametrize("thr", [0.0, 1.0, -0.1, 10])
def test_invalid_threshold_raises(thr):
    with pytest.raises(ValueError, match="disease_threshold"):
        _toy_assistant().predict_features(_feats(0, 0), "A", mode="safe", disease_threshold=thr)


@pytest.mark.parametrize("feats", [
    _feats(0, 0, duration=0.3),        # curto demais
    _feats(0, 0, rms=0.0),             # silêncio
    _feats(float("nan"), 0),           # feature não finita
    {"f1": 0.0, "duration_s": 10.0, "rms_mean": 0.1},  # feature em falta
])
def test_invalid_audio_features_raise(feats):
    from src.inference import InvalidAudioError
    with pytest.raises(InvalidAudioError):
        _toy_assistant().predict_features(feats, "A")


def test_extract_rejects_short_and_silent_files(tmp_path):
    import soundfile as sf
    from src.inference import InvalidAudioError
    a, sr = _toy_assistant(), 22050
    short, silent, junk = tmp_path / "short.wav", tmp_path / "silent.wav", tmp_path / "junk.wav"
    sf.write(short, np.random.default_rng(0).normal(0, 0.1, sr // 4).astype(np.float32), sr)
    sf.write(silent, np.zeros(sr * 3, dtype=np.float32), sr)
    junk.write_bytes(b"isto nao e um wav")
    for path in (short, silent, junk):
        with pytest.raises(InvalidAudioError):
            a.extract(path)


def test_predict_patient_averages_recordings(monkeypatch):
    a = _toy_assistant()
    table = {"r1": _feats(3.0, 0.0), "r2": _feats(-1.0, -1.0), "r3": _feats(-1.0, -1.0)}
    monkeypatch.setattr(LungAssistant, "extract", lambda self, p: table[p])
    r = a.predict_patient(["r1", "r2", "r3"], equipment="A")
    assert r["n_recordings"] == 3 and len(r["recordings"]) == 3
    expected = np.mean([[x["probabilities"][c] for c in ("COPD", "Healthy", "URTI")]
                        for x in r["recordings"]], axis=0)
    assert np.allclose(list(r["probabilities"].values()), expected)
    assert r["diagnosis"] == "Healthy"            # 2 de 3 gravações claramente Healthy
    # a safety net atua sobre a média do paciente
    r_safe = a.predict_patient(["r1", "r2", "r3"], equipment="A", mode="safe", disease_threshold=0.2)
    assert r_safe["diagnosis"] == "COPD" and r_safe["safe_blocked_healthy"] is True
    with pytest.raises(ValueError):
        a.predict_patient([], equipment="A")
    with pytest.raises(ValueError):
        a.predict_patient(["r1", "r2"], equipment=["A"])


def test_contraindications_in_every_result():
    r = _toy_assistant().predict_features(_feats(0, 0), "A")
    assert any("crianças" in c for c in r["contraindications"])


def test_equipment_list_single_source():
    from src.metadata import EQUIPMENTS, EQUIPMENT_MAP, DEFAULT_EQUIPMENT
    import assistente
    assert assistente.EQUIPMENTS is EQUIPMENTS
    assert set(EQUIPMENTS) == set(EQUIPMENT_MAP.values()) and DEFAULT_EQUIPMENT in EQUIPMENTS


def test_model_cache_respects_path(tmp_path):
    import assistente
    p1, p2 = tmp_path / "a.npz", tmp_path / "b.npz"
    a1, a2 = _toy_assistant(), _toy_assistant()
    a2.feature_cols = ["f1", "f2", "f3"]
    a1.save(p1); a2.save(p2)
    assert assistente.load_assistant(p1).feature_cols == ["f1", "f2"]
    assert assistente.load_assistant(p2).feature_cols == ["f1", "f2", "f3"]


def test_npz_roundtrip_and_no_pickle(tmp_path):
    a = _toy_assistant()
    p = tmp_path / "m.npz"
    a.save(p)
    b = LungAssistant.load(p)
    f = _feats(0.4, -0.2)
    assert a.predict_features(f, "A") == b.predict_features(f, "A")
    np.load(p, allow_pickle=False).close()      # carrega sem pickle
    with pytest.raises(ValueError):
        a.save(tmp_path / "m.joblib")


def test_import_config_has_no_side_effects_and_env_override(tmp_path):
    import subprocess, sys as _sys
    code = ("import os, pathlib; os.environ['ICBHI_ROOT'] = r'%s'; "
            "from src import config; print(config.ICBHI_ROOT)" % (tmp_path / "icbhi"))
    out = subprocess.run([_sys.executable, "-c", code], cwd=config.PROJECT_ROOT,
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == str(tmp_path / "icbhi")
    assert not (tmp_path / "icbhi").exists()


def test_safe_note_only_when_blocked():
    from src.clinical_guide import get_guidance
    plain = get_guidance("COPD", top_probability=0.95, mode="safe")
    assert "Modo conservador" not in plain["encaminhamento"]
    for p in (0.95, 0.30):  # também em baixa confiança, o caso típico do bloqueio
        g = get_guidance("URTI", top_probability=p, mode="safe", safe_blocked=True)
        assert "Modo conservador" in g["encaminhamento"]


@pytest.fixture(scope="module")
def assistant():
    if not MODEL_PATH.exists():
        pytest.skip(f"Modelo não encontrado em {MODEL_PATH}")
    return LungAssistant.load(MODEL_PATH)


# --- testes de estrutura ---
def test_assistant_loads(assistant):
    assert assistant is not None
    assert len(assistant.feature_cols) == 77
    assert "duration_s" not in assistant.feature_cols  # atalho do protocolo de gravação
    assert set(assistant.logreg_classes_) == {"COPD", "Healthy", "URTI"}


def test_audio_is_cropped_to_fixed_window(tmp_path):
    """Gravações longas não podem distinguir-se das curtas pela duração."""
    import soundfile as sf
    from src.features import extract_from_file
    sr = 22050
    p = tmp_path / "long.wav"
    sf.write(p, np.random.default_rng(0).normal(0, 0.1, sr * 40).astype(np.float32), sr)
    assert extract_from_file(p)["duration_s"] == pytest.approx(config.MAX_DURATION_S, abs=0.01)


def test_assistant_has_outlier_detector(assistant):
    assert assistant.train_mean_ is not None
    assert assistant.train_cov_inv_ is not None
    assert assistant.outlier_threshold_99_ > 0


def test_equipment_categories(assistant):
    expected = {
        "AKG C417L Microphone",
        "WelchAllyn Meditron",
        "Littmann Classic II SE",
        "Littmann 3200 Electronic",
    }
    assert set(assistant.equipment_categories) == expected
    from src.metadata import EQUIPMENTS
    assert set(EQUIPMENTS) == expected  # app/CLI oferecem exatamente o que o modelo conhece


# --- testes de inferência ---
def test_predict_returns_dict_with_required_keys(assistant):
    if not SAMPLE_AUDIO.exists():
        pytest.skip(f"Áudio de teste não encontrado em {SAMPLE_AUDIO}")
    r = assistant.predict(SAMPLE_AUDIO, equipment=SAMPLE_EQUIPMENT)
    for key in [
        "diagnosis", "confidence", "top_probability",
        "probabilities", "probabilities_raw", "equipment_used",
        "outlier", "mode",
    ]:
        assert key in r, f"chave em falta: {key}"


def test_probabilities_sum_to_one(assistant):
    if not SAMPLE_AUDIO.exists():
        pytest.skip()
    r = assistant.predict(SAMPLE_AUDIO, equipment=SAMPLE_EQUIPMENT)
    s = sum(r["probabilities"].values())
    assert abs(s - 1.0) < 1e-6, f"probabilidades não somam 1: {s}"


def test_diagnosis_is_valid_class(assistant):
    if not SAMPLE_AUDIO.exists():
        pytest.skip()
    r = assistant.predict(SAMPLE_AUDIO, equipment=SAMPLE_EQUIPMENT)
    assert r["diagnosis"] in {"COPD", "Healthy", "URTI"}


def test_known_copd_predicts_copd(assistant):
    """Sanity check: este áudio é COPD conhecido."""
    if not SAMPLE_AUDIO.exists():
        pytest.skip()
    r = assistant.predict(SAMPLE_AUDIO, equipment=SAMPLE_EQUIPMENT)
    assert r["diagnosis"] == "COPD"
    assert r["top_probability"] > 0.5


def test_safe_mode_changes_decision_when_disease_likely(assistant):
    """Modo safe deve bloquear Healthy se P(doente) >= disease_threshold."""
    if not SAMPLE_AUDIO.exists():
        pytest.skip()
    # COPD claro: ambos os modos devem dar COPD
    r_argmax = assistant.predict(SAMPLE_AUDIO, equipment=SAMPLE_EQUIPMENT, mode="argmax")
    r_safe = assistant.predict(SAMPLE_AUDIO, equipment=SAMPLE_EQUIPMENT,
                                 mode="safe", disease_threshold=0.10)
    assert r_argmax["diagnosis"] == "COPD"
    assert r_safe["diagnosis"] == "COPD"  # consistência em caso claro


# --- testes de equipment unknown ---
def test_unknown_equipment_raises(assistant):
    if not SAMPLE_AUDIO.exists():
        pytest.skip()
    with pytest.raises(ValueError, match="não está nas categorias treinadas"):
        assistant.predict(SAMPLE_AUDIO, equipment="Estetoscópio Inventado")


# --- testes de outlier ---
def test_outlier_field_structure(assistant):
    if not SAMPLE_AUDIO.exists():
        pytest.skip()
    r = assistant.predict(SAMPLE_AUDIO, equipment=SAMPLE_EQUIPMENT)
    out = r["outlier"]
    assert "is_outlier" in out
    assert "mahalanobis_sq" in out
    assert "threshold_99" in out
    assert isinstance(out["is_outlier"], (bool, np.bool_))
    assert out["mahalanobis_sq"] > 0


def test_synthetic_noise_is_outlier(assistant, tmp_path):
    """Ruído gaussiano puro deve ser flagged como outlier."""
    import soundfile as sf
    rng = np.random.default_rng(0)
    sr = 22050
    noise = rng.normal(0, 1, size=sr * 5).astype(np.float32)
    p = tmp_path / "noise.wav"
    sf.write(p, noise, sr)
    r = assistant.predict(str(p), equipment=SAMPLE_EQUIPMENT)
    assert r["outlier"]["is_outlier"], \
        f"ruído puro não foi flagged (d²={r['outlier']['mahalanobis_sq']:.1f})"


# --- testes de explicação ---
def test_explain_returns_per_class(assistant):
    if not SAMPLE_AUDIO.exists():
        pytest.skip()
    exp = assistant.explain(SAMPLE_AUDIO, equipment=SAMPLE_EQUIPMENT, top_k=5)
    assert set(exp.keys()) == {"COPD", "Healthy", "URTI"}
    for cls, items in exp.items():
        assert len(items) == 5
        for it in items:
            assert "feature" in it
            assert "coef" in it
            assert "value_scaled" in it
            assert "contribution" in it


# --- testes de clinical_guide ---
def test_clinical_guide_returns_for_each_class():
    from src.clinical_guide import get_guidance
    for cls in ("COPD", "Healthy", "URTI"):
        g = get_guidance(cls, top_probability=0.95)
        for key in ("urgencia", "encaminhamento", "exames", "tratamento",
                     "sinais_alarme", "educacao", "resumo", "nivel"):
            assert key in g


def test_outlier_overrides_diagnosis_guidance():
    """is_outlier=True deve dar guidance de revisão manual independente do diagnóstico."""
    from src.clinical_guide import get_guidance
    g = get_guidance("COPD", top_probability=0.99, is_outlier=True)
    assert g["nivel"] == "OUTLIER"
    assert "REVISÃO" in g["urgencia"] or "OUTLIER" in g["urgencia"]


def test_low_confidence_overrides():
    from src.clinical_guide import get_guidance
    g = get_guidance("URTI", top_probability=0.40)
    assert g["nivel"] == "BAIXA_CONFIANCA"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
