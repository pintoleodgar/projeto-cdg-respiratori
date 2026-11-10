"""Streamlit app do Assistente de Diagnóstico Respiratório.

Rodar:
    streamlit run app_streamlit.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display

from src.inference import CONTRAINDICATIONS, InvalidAudioError, LungAssistant
from src.clinical_guide import get_guidance, urgency_color
from src.metadata import DEFAULT_EQUIPMENT, EQUIPMENTS
from src import config

MODEL_PATH = config.MODEL_PATH

CLASS_COLORS = {"Healthy": "#2ecc71", "COPD": "#e74c3c", "URTI": "#f39c12"}


# -----------------------------------------------------------------
# Setup
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Assistente Respiratório",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="A carregar modelo…")
def load_model() -> LungAssistant:
    return LungAssistant.load(MODEL_PATH)


@contextmanager
def _temp_wav(data: bytes):
    """Ficheiro temporário apagado no fim — o áudio do paciente não fica em disco."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# Cache pelo CONTEÚDO do ficheiro: mudar o modo ou o slider não repete a
# extração de features nem a leitura do áudio.
@st.cache_data(show_spinner=False, max_entries=8)
def extract_cached(data: bytes) -> dict:
    with _temp_wav(data) as path:
        try:
            return load_model().extract(path)
        except InvalidAudioError as exc:  # não mostrar o caminho temporário ao utilizador
            raise InvalidAudioError(str(exc).replace(path, "ficheiro carregado")) from None


@st.cache_data(show_spinner=False, max_entries=8)
def load_audio_cached(data: bytes) -> tuple[np.ndarray, int]:
    with _temp_wav(data) as path:
        return librosa.load(path, sr=config.SAMPLE_RATE, mono=True,
                            duration=config.MAX_DURATION_S)


# -----------------------------------------------------------------
# Sidebar — configurações
# -----------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuração")
    equipment = st.selectbox(
        "Equipamento de gravação",
        EQUIPMENTS,
        index=EQUIPMENTS.index(DEFAULT_EQUIPMENT),
        help="Equipamento usado para capturar o áudio — obrigatório para residualização correta.",
    )
    st.divider()
    st.subheader("Modo de decisão")
    mode = st.radio(
        "Estratégia",
        ["argmax (padrão)", "safe (conservador)"],
        index=0,
        help="O modo 'safe' aplica o princípio da não-maleficência: bloqueia o "
        "diagnóstico 'Healthy' se P(doente) ≥ threshold — prioriza não perder doentes "
        "(FN clínico é o erro mais grave em diagnóstico médico)."
    )
    use_safe = mode.startswith("safe")
    disease_threshold = st.slider(
        "θ — P(doente) mínimo para bloquear 'Healthy'",
        min_value=0.02, max_value=0.50, value=0.10, step=0.01,
        disabled=not use_safe,
        help="Apenas relevante no modo 'safe'.",
    )
    st.divider()
    st.markdown("### 📚 Sobre o modelo")
    st.markdown(
        "- **77 features** extraídas com `librosa` (áudio cortado a 20 s)\n"
        "- **Residualização** sobre o equipamento (mitiga a parte linear do viés)\n"
        "- **Regressão Logística** multinomial\n"
        "- **Calibração sigmoide** validada de forma aninhada\n"
        "- **Mahalanobis** para detecção de outlier\n"
    )
    st.markdown(f"`{MODEL_PATH.name}`")
    st.divider()
    st.markdown("### 🩺 Decision support")
    st.markdown(
        "O assistente inclui um **guia clínico contextual** "
        "(GOLD para DPOC, recomendações gerais para URTI) "
        "com **5 camadas**: urgência, encaminhamento, exames, tratamento, sinais de alarme, "
        "educação do paciente.\n\n"
        "Overrides automáticos: confiança baixa ou outlier → **revisão manual obrigatória**."
    )


# -----------------------------------------------------------------
# Header
# -----------------------------------------------------------------
st.title("🫁 Assistente de Diagnóstico Respiratório")
st.caption(
    "Classifica um áudio de auscultação como **Healthy** / **COPD** / **URTI** "
    "com probabilidades calibradas, detecção de outlier e explicabilidade."
)


# -----------------------------------------------------------------
# Upload
# -----------------------------------------------------------------
col_upload, col_info = st.columns([2, 1])
with col_upload:
    uploaded = st.file_uploader(
        "Carregue um arquivo .wav de auscultação",
        type=["wav"],
        accept_multiple_files=False,
        help="Use os áudios do ICBHI 2017 ou qualquer .wav de auscultação respiratória.",
    )
with col_info:
    st.info(
        f"**Equipamentos suportados** ({len(EQUIPMENTS)}):\n"
        + "\n".join(f"- {e}" for e in EQUIPMENTS)
    )

if uploaded is None:
    st.warning("⬆️ Carregue um arquivo .wav para começar.")
    st.stop()


# -----------------------------------------------------------------
# Salvar temp e processar
# -----------------------------------------------------------------
audio_bytes = uploaded.getvalue()
st.audio(audio_bytes, format="audio/wav")

assistant = load_model()
try:
    with st.spinner("A analisar áudio…"):
        feats = extract_cached(audio_bytes)
except InvalidAudioError as exc:
    st.error(f"⚠️ Não foi possível analisar este áudio: {exc}")
    st.stop()

result = assistant.predict_features(
    feats,
    equipment=equipment,
    mode="safe" if use_safe else "argmax",
    disease_threshold=disease_threshold,
)
explanation = assistant.explain_features(feats, equipment=equipment, top_k=8)

st.warning("**Limitações (ler sempre):**\n" + "\n".join(f"- {c}" for c in CONTRAINDICATIONS))


# -----------------------------------------------------------------
# Resultado principal
# -----------------------------------------------------------------
diag = result["diagnosis"]
conf = result["confidence"]
top_p = result["top_probability"]
outlier = result.get("outlier") or {}
is_outlier = outlier.get("is_outlier", False)

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    st.markdown(
        f"## Diagnóstico: <span style='color:{CLASS_COLORS[diag]}'>**{diag}**</span>",
        unsafe_allow_html=True,
    )
    st.caption(f"Modo: `{result['mode']}` | equipamento: `{equipment}`")
with c2:
    conf_color = "🟢" if "alta" in conf else "🟡" if "média" in conf else "🔴"
    st.metric("Confiança", f"{conf_color} {conf}", f"P={top_p:.3f}")
with c3:
    if is_outlier:
        st.metric("⚠️ Outlier", "SIM", f"d²={outlier['mahalanobis_sq']:.0f}",
                  delta_color="inverse")
    else:
        st.metric("Outlier", "Não", f"d²={outlier['mahalanobis_sq']:.0f}/{outlier['threshold_99']:.0f}")

if is_outlier:
    st.error(
        "⚠️ **Amostra fora da distribuição de treino.** "
        "A confiança foi rebaixada automaticamente. Recomenda-se revisão manual."
    )

st.divider()


# -----------------------------------------------------------------
# 🩺 Orientação clínica (decision support)
# -----------------------------------------------------------------
guidance = get_guidance(
    diagnosis=diag,
    top_probability=top_p,
    is_outlier=is_outlier,
    mode=result["mode"],
    safe_blocked=result["safe_blocked_healthy"],
)

urg = guidance["urgencia"]
urg_col = urgency_color(urg)

st.subheader("🩺 Conduta clínica sugerida")

# Banner de urgência colorido
st.markdown(
    f"""<div style="background:{urg_col};color:white;padding:14px 18px;border-radius:10px;
              font-size:18px;font-weight:bold;margin-bottom:12px;">
       Nível de urgência: {urg}
       </div>""",
    unsafe_allow_html=True,
)

# Resumo executivo para o médico
st.info(f"**Resumo:** {guidance['resumo']}")

# Encaminhamento
st.markdown(f"**📋 Encaminhamento:** {guidance['encaminhamento']}")

# Layout em 2 colunas para detalhes
col_left, col_right = st.columns(2)

with col_left:
    with st.expander("🧪 Exames complementares sugeridos", expanded=True):
        for item in guidance["exames"]:
            st.markdown(f"- {item}")

    with st.expander("💊 Tratamento / abordagem imediata"):
        for item in guidance["tratamento"]:
            st.markdown(f"- {item}")

with col_right:
    with st.expander("🚨 Sinais de alarme — verificar SEMPRE", expanded=True):
        for item in guidance["sinais_alarme"]:
            st.markdown(f"- {item}")

    with st.expander("📚 Educação do paciente"):
        for item in guidance["educacao"]:
            st.markdown(f"- {item}")

# Aviso ético reforçado
st.warning(
    "⚖️ **Importante**: estas recomendações são **gerais** (baseadas em diretrizes GOLD/IDSA) "
    "e não substituem o julgamento clínico. Adaptar à idade, comorbilidades, sintomas referidos, "
    "exame físico e contexto epidemiológico do paciente."
)

st.divider()


# -----------------------------------------------------------------
# Probabilidades — gráfico
# -----------------------------------------------------------------
st.subheader("Probabilidades calibradas")
probs = result["probabilities"]
probs_raw = result["probabilities_raw"]
df_p = pd.DataFrame({
    "Classe": list(probs.keys()),
    "Calibrada": list(probs.values()),
    "Bruta (sem calibração)": [probs_raw[c] for c in probs.keys()],
}).sort_values("Calibrada", ascending=True)

fig_p, ax_p = plt.subplots(figsize=(9, 3.5))
y_pos = np.arange(len(df_p))
colors = [CLASS_COLORS[c] for c in df_p["Classe"]]
ax_p.barh(y_pos - 0.2, df_p["Calibrada"], height=0.4, color=colors, label="Calibrada")
ax_p.barh(y_pos + 0.2, df_p["Bruta (sem calibração)"], height=0.4,
          color=colors, alpha=0.4, label="Bruta")
ax_p.set_yticks(y_pos)
ax_p.set_yticklabels(df_p["Classe"])
ax_p.set_xlabel("Probabilidade")
ax_p.set_xlim(0, 1)
ax_p.legend(loc="lower right")
ax_p.axvline(0.5, color="gray", linestyle="--", alpha=0.3)
for i, (cal, raw) in enumerate(zip(df_p["Calibrada"], df_p["Bruta (sem calibração)"])):
    ax_p.text(cal + 0.02, i - 0.2, f"{cal:.3f}", va="center", fontsize=9)
    ax_p.text(raw + 0.02, i + 0.2, f"{raw:.3f}", va="center", fontsize=9, alpha=0.6)
plt.tight_layout()
st.pyplot(fig_p)


# -----------------------------------------------------------------
# Tabs: visualização do áudio | explicação | detalhes
# -----------------------------------------------------------------
tab_audio, tab_exp, tab_detail = st.tabs([
    "🎵 Visualização do áudio", "🔍 Explicação", "📋 Detalhes técnicos"
])

with tab_audio:
    y, sr = load_audio_cached(audio_bytes)
    fig_a, axes = plt.subplots(2, 1, figsize=(12, 6))
    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title(f"Waveform — {uploaded.name} (primeiros {len(y)/sr:.1f}s, os que o modelo analisa)")
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=sr//2)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, hop_length=512,
                                    y_axis="mel", x_axis="time", ax=axes[1])
    axes[1].set_title("Mel-spectrogram (dB)")
    fig_a.colorbar(img, ax=axes[1], format="%+2.0f dB")
    plt.tight_layout()
    st.pyplot(fig_a)

with tab_exp:
    st.markdown(
        f"**Por que '{diag}'?** As features abaixo são as que mais empurraram "
        "a decisão (coeficiente × valor da feature, no espaço residualizado+escalado)."
    )
    exp_df = pd.DataFrame(explanation[diag])
    exp_df = exp_df[["feature", "coef", "value_scaled", "contribution"]]
    exp_df.columns = ["Feature", "Coef. LogReg", "Valor (escalado)", "Contribuição"]
    exp_df = exp_df.sort_values("Contribuição", key=lambda s: s.abs(), ascending=False)
    st.dataframe(
        exp_df.style.background_gradient(
            subset=["Contribuição"], cmap="RdYlGn", vmin=-3, vmax=3
        ).format({"Coef. LogReg": "{:+.3f}", "Valor (escalado)": "{:+.3f}", "Contribuição": "{:+.3f}"}),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown(f"**Contribuições para as outras classes:**")
    for cls in [c for c in explanation if c != diag]:
        with st.expander(f"Top contribuições para {cls}"):
            other = pd.DataFrame(explanation[cls])
            other = other[["feature", "coef", "value_scaled", "contribution"]]
            other.columns = ["Feature", "Coef.", "Valor", "Contrib."]
            st.dataframe(
                other.style.background_gradient(
                    subset=["Contrib."], cmap="RdYlGn", vmin=-3, vmax=3
                ).format({"Coef.": "{:+.3f}", "Valor": "{:+.3f}", "Contrib.": "{:+.3f}"}),
                use_container_width=True,
            )

with tab_detail:
    st.json({k: v for k, v in result.items() if k != "explanation"})


# -----------------------------------------------------------------
# Footer com aviso clínico
# -----------------------------------------------------------------
st.divider()
st.caption(
    "⚕️ **Princípio ético:** este assistente prioriza **recall de doença** "
    "(minimizar Falsos Negativos clínicos) por aplicação do princípio bioético da "
    "**não-maleficência** (*primum non nocere*). Em saúde respiratória, classificar "
    "erradamente um doente como saudável tem custo cumulativo irreversível no paciente; "
    "classificar erradamente um saudável como doente apenas gera uma consulta extra "
    "remediável. Por isso, o sistema é deliberadamente conservador a favor da deteção."
)
st.caption(
    "⚠️ **Limitações conhecidas:** modelo treinado em 851 gravações ICBHI 2017, "
    "104 pacientes (Healthy, COPD, URTI). Em validação cruzada por paciente: "
    "recall COPD ≈ 100%, mas **recall URTI ≈ 24% (argmax) a 46% (safe θ=0,10)** e "
    "especificidade ≈ 52% a 28% — um resultado 'Healthy' **não exclui URTI**. "
    "Healthy e URTI só foram vistos em gravações Meditron; noutros equipamentos o "
    "modelo só conhece COPD. **Não usar em crianças** (desempenho pediátrico fraco). "
    "Esta app classifica uma gravação de cada vez; as métricas referem-se à média "
    "das gravações do paciente. "
    "Não substitui anamnese clínica, auscultação manual nem exame físico. "
    "Use como **apoio à decisão**, sempre revisando casos com confiança baixa ou outlier. "
    "Diagnósticos fora do treino (Pneumonia, Asma, Bronquiectasias) podem ser classificados "
    "incorretamente — confiar sempre na clínica."
)
