# Manual Técnico do Código — Assistente Respiratório

> ## ⚠️ Revisão de novembro de 2026
>
> Este manual descreve o código do modelo **v2**. O que mudou desde então:
>
> - **Features:** áudio cortado a 20 s (`config.MAX_DURATION_S`); `duration_s` deixou de ser feature → **77**, não 78.
> - **Treino:** `python scripts/train.py` gera o modelo, as métricas e `reports/AVALIACAO_V3.md`.
> - **`LungAssistant`:** novos `extract`, `predict_features`, `explain_features`, `predict_patient`;
>   `mode` inválido levanta `ValueError`; áudio inválido levanta `InvalidAudioError`;
>   o resultado inclui `safe_blocked_healthy` e `contraindications`.
> - **Persistência:** `models/lung_assistant_v3.npz` (sem pickle). `load` ainda lê `.joblib` antigos.
> - **`config.py`:** dataset em `ICBHI_ROOT` (variável de ambiente); `ensure_dirs()` em vez de criar pastas no import;
>   `MODEL_PATH`, `MIN_DURATION_S`, `NON_FEATURE_COLS`.
> - **Equipamentos:** lista única em `src.metadata.EQUIPMENTS`.
>
> Detalhes em `CHANGELOG.md` (secção "Revisão v3").

---

> Walkthrough do código por dentro: **o que cada ficheiro faz, porquê, e como modificar.**
> Complementa o `RELATORIO.md` (científico) e o `README.md` (executivo).

---

## Índice

1. [Visão geral da arquitetura](#1-visão-geral-da-arquitetura)
2. [Fluxo de dados — do .wav ao diagnóstico](#2-fluxo-de-dados--do-wav-ao-diagnóstico)
3. [`src/config.py` — caminhos e parâmetros](#3-srcconfigpy--caminhos-e-parâmetros)
4. [`src/metadata.py` — parsing do ICBHI](#4-srcmetadatapy--parsing-do-icbhi)
5. [`src/features.py` — extração de áudio com librosa](#5-srcfeaturespy--extração-de-áudio-com-librosa)
6. [`src/inference.py` — a classe `LungAssistant`](#6-srcinferencepy--a-classe-lungassistant)
7. [`src/clinical_guide.py` — decision support](#7-srcclinical_guidepy--decision-support)
8. [`assistente.py` — CLI + API Python](#8-assistentepy--cli--api-python)
9. [`app_streamlit.py` — interface web](#9-app_streamlitpy--interface-web)
10. [`scripts/build_notebook_NN.py` — geração programática](#10-scriptsbuild_notebook_nnpy--geração-programática)
11. [Notebooks — mapa de implementação](#11-notebooks--mapa-de-implementação)
12. [Como executar tudo](#12-como-executar-tudo)
13. [Como debugar / estender](#13-como-debugar--estender)

---

## 1. Visão geral da arquitetura

```
┌────────────────────────────────────────────────────────────────┐
│                    INTERFACES DE USO                            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ assistente  │  │ app_streamlit │  │ from inference       │  │
│  │   .py (CLI) │  │   .py (web)   │  │ import LungAssistant │  │
│  └──────┬──────┘  └───────┬──────┘  └──────────┬───────────┘  │
│         └────────────┬────┴───────────────────┘                │
└───────────────────────┬────────────────────────────────────────┘
                        │
                        ▼
        ┌─────────────────────────────────────┐
        │      src/inference.py               │
        │      class LungAssistant            │
        │  ┌───────────────────────────────┐  │
        │  │ residualize → scale →         │  │
        │  │ logreg → isotonic → safety    │  │
        │  │ + mahalanobis + explain       │  │
        │  └───────────────────────────────┘  │
        └────────┬────────────────────┬───────┘
                 │                    │
                 ▼                    ▼
   ┌─────────────────────┐  ┌────────────────────────┐
   │  src/features.py    │  │ src/clinical_guide.py  │
   │  extract_from_file()│  │   get_guidance()       │
   │  → 78 features      │  │   → recomendação       │
   └──────────┬──────────┘  └────────────────────────┘
              │
              ▼
   ┌─────────────────────┐
   │  src/metadata.py    │
   │  parse ICBHI files  │
   │  (treino apenas)    │
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │   src/config.py     │
   │  caminhos + params  │
   └─────────────────────┘
```

**Princípios de design:**

| Princípio | Implementação |
|---|---|
| **Separação de responsabilidades** | Cada módulo faz 1 coisa: `metadata` lê, `features` extrai, `inference` prediz, `clinical_guide` orienta |
| **Modelo é dado, não código** | Pesos serializados em `models/*.joblib`. `LungAssistant` é uma `@dataclass` simples |
| **Inferência sem treino** | `LungAssistant.predict()` não importa sklearn — só numpy. Reduz dependências em produção |
| **3 interfaces, 1 backend** | CLI, web e API partilham o mesmo `assistente.diagnose()` ou `LungAssistant.predict()` |
| **Configuração centralizada** | `src/config.py` é a única fonte de paths e parâmetros |

---

## 2. Fluxo de dados — do .wav ao diagnóstico

Quando você chama `assistant.predict("audio.wav", equipment="WelchAllyn Meditron")`:

```
audio.wav (qualquer SR, mono/stereo)
    │
    │ librosa.load(sr=22050, mono=True)
    ▼
y: np.array (1D, ~22050*N samples)
    │
    │ extract_feature_vector(y, sr)  — 78 features
    ▼
feats: dict {'mfcc1_mean': -528.4, 'mfcc1_std': 28.4, ...}
    │
    │ np.array([feats[c] for c in feature_cols])
    ▼
x: np.array (78,)                                          ← features cruas
    │
    │ _residualize(x, equipment)
    │   predicted = intercept + coef_ @ one_hot(equip)
    │   return x - predicted
    ▼
x: np.array (78,)                                          ← resíduos
    │
    │ _scale(x) = (x - mean) / std
    ▼
x: np.array (78,) com mean=0, std=1                        ← escalado
    │
    ├─→ _mahalanobis_sq(x) → d² → is_outlier?
    │
    │ _logreg_proba(x)
    │   logits = coef @ x + intercept
    │   proba = softmax(logits)
    ▼
proba_raw: np.array (3,) [P_COPD, P_Healthy, P_URTI]
    │
    │ _apply_isotonic(proba_raw)
    │   para cada classe k: P_cal_k = np.interp(P_raw_k, x_thr_k, y_thr_k)
    │   normalize para somar 1
    ▼
proba_cal: np.array (3,)
    │
    │ decisão:
    │   se mode='argmax': diag = classes[argmax(proba_cal)]
    │   se mode='safe' e P(doente) >= θ: bloqueia Healthy
    ▼
diagnosis: str  + confidence ('alta'|'média'|'baixa') + probabilities + outlier
```

Cada passo é detalhado nas secções 5 (features) e 6 (inference).

---

## 3. `src/config.py` — caminhos e parâmetros

**Tamanho:** 21 linhas. **Propósito:** centralizar paths e hiperparâmetros para que mudar uma vez funcione em todos os lados.

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ICBHI_ROOT = Path(r"C:\Datasets\Respiratory_Sound_Database")
AUDIO_DIR = ICBHI_ROOT / "audio_and_txt_files"
DIAGNOSIS_CSV = ICBHI_ROOT / "patient_diagnosis.csv"
DEMOGRAPHIC_TXT = ICBHI_ROOT / "demographic_info.txt"

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (PROCESSED_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 22050
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
```

**Decisões importantes:**

- **`Path(__file__).resolve().parents[1]`** — `PROJECT_ROOT` é a pasta-mãe de `src/`. Permite que o código funcione **independentemente do diretório de onde foi chamado** (CLI, notebook, Streamlit).
- **`r"C:\..."`** — raw string para evitar problemas com `\` em Windows. Em outro SO, ajustar para `/`.
- **`for _d in (...): _d.mkdir(parents=True, exist_ok=True)`** — cria pastas no import. Evita erros tipo "directory not found" mais tarde.
- **`SAMPLE_RATE = 22050`** — escolha clássica para áudio falado/respiratório. Razão de Nyquist permite até ~11kHz, suficiente para sons pulmonares (até ~2kHz tipicamente).
- **`N_MFCC = 13`** — convenção da literatura de speech processing; captura envelope espectral sem ruído de alta-ordem.
- **`N_FFT = 2048, HOP_LENGTH = 512`** — janela de ~93ms com 75% overlap a 22050Hz; bom compromisso resolução tempo-frequência.

**Como modificar:** se mudar `SAMPLE_RATE` ou `N_MFCC`, **tem de retreinar o modelo** (features ficam diferentes). Não é um simples ajuste de runtime.

---

## 4. `src/metadata.py` — parsing do ICBHI

**Tamanho:** 160 linhas. **Propósito:** transformar os ficheiros texto do ICBHI em DataFrames pandas estruturados.

### 4.1. Constantes

```python
TARGET_CLASSES = ["Healthy", "COPD", "URTI"]
DISEASE_CLASSES = ["COPD", "URTI"]
```

Definem o escopo do projeto. Mudá-las requer retreinar.

```python
CHEST_LOCATION_MAP = {
    "Tc": "Trachea", "Al": "Anterior left", "Ar": "Anterior right",
    "Pl": "Posterior left", "Pr": "Posterior right",
    "Ll": "Lateral left", "Lr": "Lateral right",
}
```

Decodifica os códigos dos nomes de ficheiro. O ICBHI usa formato `<paciente>_<rec>_<local>_<modo>_<equipamento>.wav` (ex: `101_1b1_Al_sc_Meditron.wav`).

### 4.2. `parse_filename(stem)` — o coração do parser

```python
def parse_filename(stem: str) -> dict:
    """`101_1b1_Al_sc_Meditron` -> dict com campos parseados."""
    parts = stem.split("_")
    if len(parts) != 5:
        return {"patient_id": None, ...}
    pid, recidx, loc, mode, equip = parts
    return {
        "patient_id": int(pid),
        "recording_idx": recidx,
        "chest_location_code": loc,
        "chest_location": CHEST_LOCATION_MAP.get(loc, loc),
        ...
    }
```

**Por que `if len(parts) != 5: return dict_vazio`?**
Defensivo — se um nome de ficheiro for malformado, não crasha; retorna nulls que o pandas trata bem.

**Por que `.get(loc, loc)` em vez de `[loc]`?**
Fallback: se aparecer um código novo (futuro), usa o próprio código em vez de KeyError.

### 4.3. `build_recordings_dataframe()` — entry point principal

```python
def build_recordings_dataframe(audio_dir, diagnosis_csv, demographic_txt) -> pd.DataFrame:
    rec = list_recordings(audio_dir)      # 1 linha por wav
    diag = load_diagnosis(diagnosis_csv)  # paciente → diagnóstico
    demo = load_demographics(demographic_txt)  # paciente → idade, sexo, BMI
    df = rec.merge(diag, on="patient_id", how="left")
    df = df.merge(demo, on="patient_id", how="left")
    return df
```

**`how="left"`** — preserva todas as gravações mesmo se um paciente não tiver demografia (acontece em ~5% dos casos).

### 4.4. `build_cycles_dataframe(recordings_df)` — ciclos respiratórios

Cada `.wav` tem um `.txt` parceiro com **ciclos respiratórios anotados**: linhas com `start, end, crackles, wheezes`. Esta função expande para uma linha por ciclo (~6.900 ciclos no total).

```python
for _, row in recordings_df.iterrows():
    if not row["txt_path"]:
        continue
    cycles = load_cycles(row["txt_path"])
    cycles["filename"] = row["filename"]
    cycles["patient_id"] = row["patient_id"]
    cycles["diagnosis"] = row.get("diagnosis")
    cycles["duration"] = cycles["end"] - cycles["start"]
    frames.append(cycles)
out = pd.concat(frames, ignore_index=True)
out["cycle_idx"] = out.groupby("filename").cumcount()
```

Usado no nb07 (caminho B — classificação por ciclo).

### 4.5. `filter_target_classes()` e `add_target_labels()`

Aplicam o filtro de 3 classes e criam as colunas `label_multi` (multiclasse) e `label_binary` (Healthy vs Disease).

```python
def add_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label_multi"] = df["diagnosis"]
    df["label_binary"] = df["diagnosis"].apply(
        lambda d: "Healthy" if d == "Healthy" else "Disease"
    )
    return df
```

**Por que `.copy()`?** Evitar `SettingWithCopyWarning` do pandas — pequeno detalhe defensivo.

---

## 5. `src/features.py` — extração de áudio com librosa

**Tamanho:** 94 linhas. **Propósito:** transformar um array numérico de áudio em **78 features** que descrevem o som.

### 5.1. `load_audio()` — wrapper sobre librosa

```python
def load_audio(path, sr=SAMPLE_RATE, offset=0.0, duration=None) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=sr, offset=offset, duration=duration, mono=True)
    return y, sr
```

**`sr=22050`** força resampling — todos os áudios ficam na mesma taxa, independentemente do ficheiro original (alguns do ICBHI estão a 4kHz, outros a 44.1kHz).

**`mono=True`** colapsa stereo para mono (estetoscópios são mono, mas alguns ficheiros têm 2 canais idênticos).

### 5.2. `extract_feature_vector()` — as 78 features

Esta função é o **coração da engenharia de features**. Vou explicar por blocos:

```python
def extract_feature_vector(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC) -> dict:
    feats: dict = {}
    
    # === Bloco 1: MFCC (13 coef.) + Delta + Delta² ===
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    for i in range(n_mfcc):
        feats[f"mfcc{i+1}_mean"] = float(mfcc[i].mean())
        feats[f"mfcc{i+1}_std"] = float(mfcc[i].std())
        feats[f"mfcc{i+1}_d_mean"] = float(delta[i].mean())
        feats[f"mfcc{i+1}_d2_mean"] = float(delta2[i].mean())
```

**Por que MFCCs?**
MFCC (Mel-Frequency Cepstral Coefficients) é o **padrão de facto** para análise de áudio respiratório/voz desde os anos 80. Captura o **envelope espectral** numa escala perceptual (Mel) que aproxima a audição humana.

**Por que mean + std + Δ + Δ²?**
- `mean` captura a "cor" média do som no tempo
- `std` captura variabilidade (sibilância varia, voz estável)
- `delta` (1ª derivada) captura taxa de variação
- `delta2` (2ª derivada) captura aceleração

Total: 13 × 4 = **52 features**.

```python
    # === Bloco 2: Chroma STFT (12 bins) ===
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    for i in range(chroma.shape[0]):
        feats[f"chroma{i+1}_mean"] = float(chroma[i].mean())
```

**Por que chroma?**
Originalmente para música (12 notas), mas em sons respiratórios captura distribuição de energia harmónica. Wheezes têm padrões tonais característicos.

**12 features.**

```python
    # === Bloco 3: Spectral contrast (7 bandas) ===
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    for i in range(contrast.shape[0]):
        feats[f"contrast{i+1}_mean"] = float(contrast[i].mean())
```

**Por que spectral contrast?**
Mede a diferença entre picos e vales por banda de frequência. Importante para distinguir tons (wheezes — alto contraste) de ruído branco (crackles — baixo contraste).

**7 features.** Top contribuição no modelo final.

```python
    # === Bloco 4: Espectrais globais + duração ===
    feats["zcr_mean"] = float(librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH).mean())
    feats["rms_mean"] = float(librosa.feature.rms(y=y, hop_length=HOP_LENGTH).mean())
    feats["centroid_mean"] = float(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH).mean())
    feats["rolloff_mean"] = float(librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH).mean())
    feats["bandwidth_mean"] = float(librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH).mean())
    feats["flatness_mean"] = float(librosa.feature.spectral_flatness(y=y, hop_length=HOP_LENGTH).mean())
    feats["duration_s"] = float(len(y) / sr)
```

| Feature | O que mede | Relevância clínica |
|---|---|---|
| `zcr_mean` | Quão "rugoso" é o sinal | Crackles (eventos abruptos) ↑ |
| `rms_mean` | Energia média | Intensidade da respiração |
| `centroid_mean` | "Centro de massa" do espectro | Wheezes (alta frequência) ↑ |
| `rolloff_mean` | Frequência abaixo da qual está 85% da energia | Idem |
| `bandwidth_mean` | Dispersão espectral | Ruído branco ↑ |
| `flatness_mean` | Quão "plano" é o espectro | Whisper-like vs tonal |
| `duration_s` | Comprimento do áudio | COPD tem áudios mais longos (artefacto do dataset!) |

**6 + 1 features.**

**Total: 52 + 12 + 7 + 7 = 78 features.** ✅

### 5.3. `extract_from_file(path)` — atalho

```python
def extract_from_file(path) -> dict:
    y, sr = load_audio(path)
    return extract_feature_vector(y, sr)
```

É o que o `LungAssistant.predict()` chama.

### 5.4. `extract_segment(y_full, sr, start_s, end_s)` — para ciclos

```python
def extract_segment(y_full, sr, start_s, end_s, min_samples=1024):
    i0 = int(start_s * sr)
    i1 = int(end_s * sr)
    seg = y_full[i0:i1]
    if len(seg) < min_samples:
        return None
    return extract_feature_vector(seg, sr)
```

**Por que esta função?**
Otimização do nb07 (Parte B — classificação por ciclo): carregar 1 ficheiro `.wav` e cortar 7 ciclos é muito mais rápido que carregar o ficheiro 7 vezes. Ganho ~10×.

**`min_samples=1024`** rejeita segmentos curtos demais para o STFT funcionar (FFT precisa de pelo menos N_FFT samples — usamos 1024 como compromisso).

---

## 6. `src/inference.py` — a classe `LungAssistant`

**Tamanho:** 213 linhas. **Propósito:** encapsular o pipeline completo de inferência num único objeto serializável.

### 6.1. Por que `@dataclass` e não `class` normal?

```python
@dataclass
class LungAssistant:
    feature_cols: list[str]
    equipment_categories: list[str]
    residualizer_coef_: np.ndarray
    ...
```

**Vantagens:**
- `__init__` automático
- Atributos declarados como contrato (você vê tudo num só sítio)
- Funciona perfeitamente com `joblib` (serializa diretamente)
- `joblib.dump(asst, path)` → recupera idêntico

**Trade-off:** não é totalmente imutável (`frozen=False` por defeito), mas isso é OK porque nunca mutamos depois de treinado.

### 6.2. Estrutura dos atributos

```python
# Pipeline residualização → scale → logreg → isotonic
feature_cols: list[str]              # nomes das 78 features
equipment_categories: list[str]      # ordem fixa de equipamentos
residualizer_coef_: np.ndarray       # (78, 4) — coef LinearRegression
residualizer_intercept_: np.ndarray  # (78,)
scaler_mean_: np.ndarray             # (78,) — StandardScaler.mean_
scaler_scale_: np.ndarray            # (78,) — StandardScaler.scale_
logreg_classes_: np.ndarray          # ['COPD', 'Healthy', 'URTI']
logreg_coef_: np.ndarray             # (3, 78)
logreg_intercept_: np.ndarray        # (3,)
isotonic_calibrators_: dict          # cls → (x_thresholds, y_thresholds)
# Outlier detector
train_mean_: np.ndarray | None = None
train_cov_inv_: np.ndarray | None = None
outlier_threshold_99_: float | None = None
```

**Decisão de design:** guardamos **apenas os parâmetros aprendidos**, não os objetos sklearn inteiros. Isso permite que `predict()` rode **só com numpy**, sem importar sklearn — útil para deployment em ambientes minimais.

**`isotonic_calibrators_` como dict de tuples**, não objetos `IsotonicRegression` — porque `np.interp(x, xp, yp)` faz o mesmo e é mais leve.

### 6.3. `_residualize(x, equipment)` — remoção de viés

```python
def _residualize(self, x: np.ndarray, equipment: str) -> np.ndarray:
    eq_onehot = np.zeros(len(self.equipment_categories))
    if equipment not in self.equipment_categories:
        raise ValueError(f"equipment '{equipment}' não está nas categorias treinadas")
    eq_onehot[self.equipment_categories.index(equipment)] = 1.0
    predicted = self.residualizer_intercept_ + self.residualizer_coef_ @ eq_onehot
    return x - predicted
```

**O que faz matematicamente:**

Para cada feature `i`, treinámos no notebook 08:
```
feature_i = β_i,0 + β_i,1·1{AKG} + β_i,2·1{Litt3200} + β_i,3·1{LittClassic} + β_i,4·1{Meditron} + resíduo_i
```

Em inferência, conhecendo o equipamento, calculamos o `predicted` e subtraímos. Restam apenas as componentes **não explicadas pelo equipamento**.

**`raise ValueError`** se equipamento desconhecido — explícito para não falhar silenciosamente.

**`coef_ @ eq_onehot`** — multiplicação matriz × vetor. `coef_.shape = (78, 4)`, `eq_onehot.shape = (4,)`, resultado `(78,)`.

⚠️ **Bug corrigido:** versão inicial usava `.T @ eq_onehot` (transposta errada). Sklearn segue convenção `coef_[output, input]`, então não precisa transpor.

### 6.4. `_scale(x)` e `_logreg_proba(x)` — núcleos numéricos

```python
def _scale(self, x): return (x - self.scaler_mean_) / self.scaler_scale_

def _logreg_proba(self, x):
    logits = self.logreg_coef_ @ x + self.logreg_intercept_
    m = logits.max()
    e = np.exp(logits - m)
    return e / e.sum()
```

**Por que `logits.max()` antes do exp?**
Truque de estabilidade numérica do **softmax**: subtrair o máximo evita overflow de `exp()` quando os logits são grandes. O resultado é matematicamente idêntico mas computacionalmente estável.

### 6.5. `_apply_isotonic(raw_proba)` — calibração

```python
def _apply_isotonic(self, raw_proba):
    out = np.zeros_like(raw_proba)
    for i, cls in enumerate(self.logreg_classes_):
        x_thr, y_thr = self.isotonic_calibrators_[cls]
        out[i] = float(np.interp(raw_proba[i], x_thr, y_thr))
    s = out.sum()
    return out / s if s > 0 else raw_proba
```

**`np.interp`** faz interpolação linear entre os "joelhos" da escada isotónica aprendida. Equivalente ao `IsotonicRegression.predict()` mas sem dependência sklearn.

**Por que normalizar (`out / s`)?** Após calibração one-vs-rest, as probabilidades não somam mais 1 — renormalizamos para manter a interpretação de "distribuição".

### 6.6. `_mahalanobis_sq(x)` — detecção de outlier

```python
def _mahalanobis_sq(self, x):
    if self.train_mean_ is None or self.train_cov_inv_ is None:
        return float("nan")
    d = x - self.train_mean_
    return float(d @ self.train_cov_inv_ @ d)
```

**Fórmula:** `d² = (x − μ)ᵀ Σ⁻¹ (x − μ)`

Mede "distância" em desvios padrão, contabilizando correlações entre features. Threshold no nb09: percentil 99 empírico das `d²` observadas no treino (≈ 353), porque a distribuição teórica χ²(78) ≈ 110 subestima brutalmente devido a features não-gaussianas.

### 6.7. `predict(...)` — orquestração

```python
def predict(self, audio_path, equipment, check_outlier=True,
            mode="argmax", disease_threshold=0.30, healthy_label="Healthy"):
    feats = extract_from_file(audio_path)              # 1. extrai 78 features
    x = np.array([feats[c] for c in self.feature_cols], dtype=float)
    x = self._residualize(x, equipment)                # 2. tira viés equipamento
    x = self._scale(x)                                 # 3. padroniza
    proba_raw = self._logreg_proba(x)                  # 4. softmax
    proba_cal = self._apply_isotonic(proba_raw)        # 5. calibra
    
    # 6. decisão: argmax OU safety net
    if mode == "safe":
        healthy_idx = list(self.logreg_classes_).index(healthy_label)
        p_disease = 1.0 - proba_cal[healthy_idx]
        if p_disease >= disease_threshold:
            mask = np.ones_like(proba_cal, dtype=bool)
            mask[healthy_idx] = False
            idx = int(np.argmax(np.where(mask, proba_cal, -np.inf)))
        else:
            idx = healthy_idx
    else:
        idx = int(np.argmax(proba_cal))
    
    # 7. níveis de confiança
    top = float(proba_cal[idx])
    if top < self.CONFIDENCE_THRESHOLDS[0]: conf = "baixa"
    elif top < self.CONFIDENCE_THRESHOLDS[1]: conf = "média"
    else: conf = "alta"
    
    # 8. outlier
    outlier_info = None
    if check_outlier and self.train_cov_inv_ is not None:
        d2 = self._mahalanobis_sq(x)
        thr = self.outlier_threshold_99_ or float("inf")
        outlier_info = {"is_outlier": d2 > thr, ...}
        if outlier_info["is_outlier"]:
            conf = "baixa (amostra fora da distribuição)"
    
    return {"diagnosis": ..., "confidence": ..., ...}
```

**O truque do `np.where(mask, proba_cal, -np.inf)`:** no modo safe, "anulamos" a probabilidade de Healthy substituindo por −∞ antes do argmax. Garante que Healthy nunca é escolhido se foi bloqueado.

### 6.8. `explain(audio_path, equipment, top_k=5)`

```python
def explain(self, audio_path, equipment, top_k=8):
    feats = extract_from_file(audio_path)
    x_raw = np.array([feats[c] for c in self.feature_cols], dtype=float)
    x = self._scale(self._residualize(x_raw, equipment))
    contribs = self.logreg_coef_ * x   # broadcast: (3,78) * (78,) = (3,78)
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
```

**Linearidade do LogReg dá explicabilidade grátis:**
```
logit_k = coef_k · x + intercept_k = Σ_i (coef_k[i] * x[i]) + intercept_k
```

Cada feature `i` contribui `coef_k[i] * x[i]` para a classe `k`. Esta decomposição é **exata** (não aproximação como SHAP), e é o que mostramos no Streamlit como "Top contribuições".

**`np.argsort(np.abs(contribs[i]))[::-1][:top_k]`** — ordena por magnitude absoluta e pega os top-k.

---

## 7. `src/clinical_guide.py` — decision support

**Tamanho:** 278 linhas. **Propósito:** transformar `{"diagnosis": "COPD", "P": 0.95}` em orientação clínica acionável.

### 7.1. Estrutura do dicionário `CLINICAL_GUIDANCE`

```python
CLINICAL_GUIDANCE: dict[str, dict[str, dict]] = {
    "COPD": {
        "alta_confianca": {
            "urgencia": "MODERADA",
            "encaminhamento": "Consulta de **Pneumologia** em 2–4 semanas. ...",
            "exames": ["Espirometria com prova broncodilatadora ...", ...],
            "tratamento": ["SABA: salbutamol 100 µg ...", ...],
            "sinais_alarme": ["🚨 Dispneia em repouso ...", ...],
            "educacao": ["Cessação tabágica ...", ...],
            "resumo": "Diagnóstico provável de DPOC. ..."
        },
    },
    "URTI": {...},
    "Healthy": {...},
}
```

**Por que `dict[str, dict[str, dict]]` aninhado?**
Primeiro nível = diagnóstico, segundo nível = "alta_confianca" (apenas uma chave por enquanto). Estrutura aberta para no futuro adicionar `"baixa_confianca"`, `"exacerbacao"`, etc.

**Por que strings com Markdown (`**`, `🚨`)?**
O Streamlit renderiza diretamente. CLI mostra raw, mas legível. JSON exporta sem perder formatação.

### 7.2. `_build_low_confidence_guidance()` e `_build_outlier_guidance()`

Funções que **constroem** o guidance para casos especiais, em vez de ter no dicionário estático.

```python
def _build_low_confidence_guidance(original_diag: str) -> dict:
    return {
        "urgencia": "REVISÃO MANUAL OBRIGATÓRIA",
        "encaminhamento": f"⚠️ **Confiança baixa do modelo na predição '{original_diag}'.** ..."
        "exames": [...],
        ...
    }
```

**Por que função e não constante?**
A mensagem inclui `original_diag` — precisa ser construída dinamicamente.

### 7.3. `get_guidance()` — hierarquia de overrides

```python
def get_guidance(diagnosis, top_probability, is_outlier=False, mode="argmax",
                 confidence_threshold_high=0.8, confidence_threshold_med=0.6) -> dict:
    # PRIORIDADE 1: outlier
    if is_outlier:
        g = _build_outlier_guidance(diagnosis).copy()
        g["nivel"] = "OUTLIER"
        return g
    
    # PRIORIDADE 2: baixa confiança
    if top_probability < confidence_threshold_med:
        g = _build_low_confidence_guidance(diagnosis).copy()
        g["nivel"] = "BAIXA_CONFIANCA"
        return g
    
    # PRIORIDADE 3: guidance padrão do diagnóstico
    diag_block = CLINICAL_GUIDANCE.get(diagnosis)
    if not diag_block:
        return _build_low_confidence_guidance(diagnosis) | {"nivel": "DESCONHECIDO"}
    
    g = diag_block["alta_confianca"].copy()
    g["nivel"] = "ALTA_CONFIANCA" if top_probability >= confidence_threshold_high else "MEDIA_CONFIANCA"
    
    # Modificador: modo safe redirecionou
    if mode == "safe" and diagnosis != "Healthy":
        g = g.copy()
        g["encaminhamento"] = "ℹ️ **Modo conservador ativo**: ..." + g["encaminhamento"]
    
    return g
```

**A ordem das prioridades importa:**
1. Outlier sempre vence (não importa se confiança alta — modelo está fora do seu domínio)
2. Baixa confiança vence sobre guidance padrão (incerteza > certeza ruim)
3. Modificador safe é aplicado *depois* do guidance padrão (acrescenta nota, não substitui)

**`g | {"nivel": "..."}`** — merge de dicts (Python 3.9+). Equivalente a `{**g, "nivel": "..."}`.

### 7.4. `urgency_color()` — para o Streamlit

```python
def urgency_color(urgency: str) -> str:
    if "EMERGÊNCIA" in urgency or "ALTA" in urgency: return "#c0392b"  # vermelho
    if "REVISÃO" in urgency or "OUTLIER" in urgency or "MODERADA" in urgency: return "#e67e22"  # laranja
    if "BAIXA" in urgency: return "#f1c40f"  # amarelo
    return "#27ae60"  # verde
```

**Match por substring** (`in`) — robusto a variações tipo "REVISÃO MANUAL OBRIGATÓRIA — AMOSTRA ATÍPICA".

---

## 8. `assistente.py` — CLI + API Python

**Tamanho:** 273 linhas. **Propósito:** interface de linha de comando + função importável (`diagnose()`).

### 8.1. Cache global do modelo

```python
_assistant_cache: LungAssistant | None = None

def load_assistant(model_path=DEFAULT_MODEL_PATH) -> LungAssistant:
    global _assistant_cache
    if _assistant_cache is None:
        _assistant_cache = LungAssistant.load(model_path)
    return _assistant_cache
```

**Por que cache?** Carregar o modelo joblib leva ~50ms. Em batch (vários áudios), reusar é 50× mais rápido.

**`global` é OK aqui** porque `LungAssistant` é imutável após `load()`.

### 8.2. `diagnose()` — API principal

```python
def diagnose(audio_path, equipment, safe=False, disease_threshold=0.10,
             explain=True, top_k=5, model_path=DEFAULT_MODEL_PATH) -> dict:
    asst = load_assistant(model_path)
    mode = "safe" if safe else "argmax"
    result = asst.predict(audio_path, equipment=equipment, mode=mode,
                          disease_threshold=disease_threshold, check_outlier=True)
    if explain:
        result["explanation"] = asst.explain(audio_path, equipment=equipment, top_k=top_k)
    result["audio_path"] = str(audio_path)
    return result
```

**API limpa:** booleano `safe` em vez de string `mode="safe"`. Mais Pythônico.

### 8.3. Formatação CLI — barras de progresso com Unicode

```python
COLOR = {"reset": "\033[0m", "bold": "\033[1m", "red": "\033[91m", ...}

def _bar(p: float, width: int = 30) -> str:
    return "█" * int(round(p * width)) + "░" * (width - int(round(p * width)))
```

**Códigos ANSI escapados** funcionam em terminais Unix, PowerShell moderno e Windows Terminal. Em CMD antigo aparecem códigos brutos — daí a flag `--no-color`.

**Barras `█░`** funcionam em qualquer terminal Unicode.

### 8.4. `format_report()` — o relatório bonito

```python
def format_report(result, use_color=True) -> str:
    out = []
    diag = result["diagnosis"]
    diag_color = "green" if diag == "Healthy" else "yellow" if diag == "URTI" else "red"
    out.append("═" * 70)
    out.append(f"  DIAGNÓSTICO: {_color(diag, diag_color, use_color)}".center(80))
    ...
```

Construção por linhas + `"\n".join()`. Simples e testável.

### 8.5. `argparse` — CLI declarativa

```python
def _build_argparser():
    p = argparse.ArgumentParser(...)
    p.add_argument("audios", nargs="+", help="Caminho(s) para arquivo(s) .wav")
    p.add_argument("--equipment", "-e", required=True, choices=EQUIPMENTS, ...)
    p.add_argument("--safe", action="store_true", ...)
    p.add_argument("--threshold", "-t", type=float, default=0.10, ...)
    p.add_argument("--json", action="store_true", ...)
    p.add_argument("--no-color", action="store_true", ...)
    ...
    return p
```

**`choices=EQUIPMENTS`** valida automaticamente — erro útil se usuário escrever errado.

**`nargs="+"`** aceita 1 ou mais áudios (batch).

### 8.6. `main()` — orquestração

```python
def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    
    # Validação prévia
    audios = [Path(a) for a in args.audios]
    missing = [str(a) for a in audios if not a.exists()]
    if missing:
        print(f"Erro: arquivo(s) não encontrado(s): {missing}", file=sys.stderr)
        return 2
    
    # Inferência
    results = []
    for audio in audios:
        r = diagnose(audio, equipment=args.equipment, ...)
        results.append(r)
    
    # Output
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    else:
        for r in results:
            print(format_report(r, use_color=not args.no_color))
            print()
    return 0
```

**Exit codes:**
- `0`: sucesso
- `2`: erro de input (convenção POSIX)

**`if __name__ == "__main__": raise SystemExit(main())`** — padrão para CLI Python: testável (`main(["audio.wav", "-e", "..."])`) e executável.

---

## 9. `app_streamlit.py` — interface web

**Tamanho:** 355 linhas. **Propósito:** app web interativa para clínicos.

### 9.1. Setup e cache

```python
st.set_page_config(page_title="Assistente Respiratório", page_icon="🫁", layout="wide", ...)

@st.cache_resource(show_spinner="A carregar modelo…")
def load_model() -> LungAssistant:
    return LungAssistant.load(MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_audio_cached(path_str: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path_str, sr=sr, mono=True)
    return y, sr
```

**`@st.cache_resource`** vs **`@st.cache_data`**:
- `cache_resource`: para objetos pesados que nunca mudam (modelo). Partilhado entre sessões.
- `cache_data`: para outputs computacionais. Re-executa se inputs mudarem.

### 9.2. Sidebar — configurações

```python
with st.sidebar:
    equipment = st.selectbox("Equipamento", EQUIPMENTS, index=1, help="...")
    mode = st.radio("Estratégia", ["argmax (padrão)", "safe (conservador)"], index=0, ...)
    use_safe = mode.startswith("safe")
    disease_threshold = st.slider("θ", 0.02, 0.50, 0.10, 0.01, disabled=not use_safe)
```

**`disabled=not use_safe`** — o slider só fica clicável no modo safe. UX limpa.

### 9.3. Upload + processamento

```python
uploaded = st.file_uploader("Carregue um .wav", type=["wav"])
if uploaded is None:
    st.warning("⬆️ Carregue um arquivo .wav para começar.")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
    tmp.write(uploaded.getbuffer())
    tmp_path = tmp.name

st.audio(uploaded, format="audio/wav")
```

**`st.stop()`** interrompe a execução se não há ficheiro — evita executar o resto da página.

**`tempfile.NamedTemporaryFile(delete=False)`** — escrever bytes uploaded num ficheiro real porque `librosa` precisa de path, não de bytes.

### 9.4. Inferência + métricas em destaque

```python
assistant = load_model()
with st.spinner("A analisar áudio…"):
    result = assistant.predict(tmp_path, equipment=equipment, mode="safe" if use_safe else "argmax", ...)
    explanation = assistant.explain(tmp_path, equipment=equipment, top_k=8)

# 3 métricas em colunas
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    st.markdown(f"## Diagnóstico: <span style='color:{CLASS_COLORS[diag]}'>**{diag}**</span>",
                unsafe_allow_html=True)
with c2:
    conf_color = "🟢" if "alta" in conf else "🟡" if "média" in conf else "🔴"
    st.metric("Confiança", f"{conf_color} {conf}", f"P={top_p:.3f}")
with c3:
    if is_outlier:
        st.metric("⚠️ Outlier", "SIM", f"d²={outlier['mahalanobis_sq']:.0f}", delta_color="inverse")
    else:
        st.metric("Outlier", "Não", ...)
```

**`unsafe_allow_html=True`** permite o `<span style="...">` colorido. Em produção real teríamos cuidado com XSS, mas o conteúdo é controlado pelo modelo.

### 9.5. Decision support integrado

```python
guidance = get_guidance(diagnosis=diag, top_probability=top_p, is_outlier=is_outlier, mode=result["mode"])
urg = guidance["urgencia"]
urg_col = urgency_color(urg)

st.markdown(
    f"""<div style="background:{urg_col};color:white;padding:14px 18px;border-radius:10px;
              font-size:18px;font-weight:bold;margin-bottom:12px;">
       Nível de urgência: {urg}
       </div>""",
    unsafe_allow_html=True,
)

col_left, col_right = st.columns(2)
with col_left:
    with st.expander("🧪 Exames complementares sugeridos", expanded=True):
        for item in guidance["exames"]:
            st.markdown(f"- {item}")
    with st.expander("💊 Tratamento / abordagem imediata"):
        for item in guidance["tratamento"]:
            st.markdown(f"- {item}")
with col_right:
    with st.expander("🚨 Sinais de alarme", expanded=True):
        ...
    with st.expander("📚 Educação do paciente"):
        ...
```

**Banner colorido custom em HTML** porque `st.warning/info/error` têm cores fixas.

**`st.expander(expanded=True)`** — alguns abertos por padrão (exames, alarme), outros colapsados (tratamento, educação) para reduzir scroll inicial.

### 9.6. Tabs para detalhes

```python
tab_audio, tab_exp, tab_detail = st.tabs(["🎵 Visualização", "🔍 Explicação", "📋 Detalhes"])

with tab_audio:
    y, sr = load_audio_cached(tmp_path)
    fig_a, axes = plt.subplots(2, 1, figsize=(12, 6))
    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=sr//2)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, hop_length=512,
                                    y_axis="mel", x_axis="time", ax=axes[1])
    ...
    st.pyplot(fig_a)

with tab_exp:
    exp_df = pd.DataFrame(explanation[diag])[["feature", "coef", "value_scaled", "contribution"]]
    st.dataframe(
        exp_df.style.background_gradient(subset=["Contribuição"], cmap="RdYlGn", vmin=-3, vmax=3)
                    .format({"Coef.": "{:+.3f}", ...}),
        use_container_width=True,
    )
```

**`st.dataframe(style.background_gradient(...))`** dá heatmap coloridos na tabela — vermelho para contribuições negativas, verde para positivas. Insight visual imediato.

---

## 10. `scripts/build_notebook_NN.py` — geração programática

**Por que gerar notebooks via script Python?**

Tentei inicialmente usar a ferramenta `NotebookEdit` direta, mas houve **bug de aspas escapadas** (cada `"` virava `\"` literal, causando `SyntaxError` na execução). Resolvi com:

```python
def code(src):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }

cells = [
    md("# 04 — Repeated Cross-Validation\n\n..."),
    code("import numpy as np\n..."),
    md("## Resultado\n..."),
    ...
]

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {...}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
```

**Vantagens:**
- Código Python normal, sem hacks de escape
- Versionável (diff legível no git)
- Reproduzível (regenera notebook idêntico)
- Permite **lógica programática** (loops para gerar células similares)

**Trade-off:** edição interativa do notebook (no Jupyter) **não é persistida** — alterações são perdidas se regerar. Workflow: editar o `build_*.py`, executar para regenerar.

---

## 11. Notebooks — mapa de implementação

Cada notebook resolve um problema específico. Tabela de **o que importar de onde**:

| Notebook | Imports principais | Função-chave usada |
|---|---|---|
| 01 EDA | `metadata`, `features`, `librosa.display` | `build_recordings_dataframe()`, `load_audio()` |
| 02 Extração | `metadata`, `features`, `tqdm` | `extract_from_file()` em loop |
| 03 Modelos baseline | `sklearn` (LDA, LogReg, KNN), `xgboost` | `cross_val_predict(cv=StratifiedGroupKFold)` |
| 04 Repeated CV | Idem + `scipy.stats` | Loop manual + `wilcoxon()` |
| 05 Nested CV | + `GridSearchCV` | Loop com inner CV manual |
| 06 Calibração + probe | + `IsotonicRegression` | `calibration_curve()`, probe = LogReg(`equipment`) |
| 07 Caminhos A/B/C | + `LinearRegression`, `OneHotEncoder` | Residualização |
| 08 Pipeline final | Todos os anteriores | Construção do `LungAssistant` |
| 09 Holdout + outlier | + `scipy.stats.chi2` | Mahalanobis |
| 10 CNN | `torch`, `torch.nn` | Custom `SmallCNN` |
| 11 Custo clínico | `sklearn.metrics` | `proba @ cost_matrix` |
| 12 Safety net | Idem | Função `safety_net_pred()` |

---

## 12. Como executar tudo

### Setup único
```bash
pip install -r requirements.txt
```

### Pipeline completo (do zero ao modelo final)
```bash
# 1. extrai features (se features.csv não existir) — ~5min
python -m nbconvert --to notebook --execute --inplace notebooks/02_extracao_completa.ipynb

# 2. treina pipeline e salva modelo — ~30s
python -m nbconvert --to notebook --execute --inplace notebooks/08_pipeline_final.ipynb

# 3. holdout + outlier (atualiza v2)
python -m nbconvert --to notebook --execute --inplace notebooks/09_holdout_explain_outliers.ipynb
```

### Usar o modelo treinado

**CLI:**
```bash
python assistente.py audio.wav --equipment "WelchAllyn Meditron"
```

**Python:**
```python
from assistente import diagnose
r = diagnose("audio.wav", equipment="WelchAllyn Meditron", safe=True)
print(r["diagnosis"], r["confidence"])
```

**Web:**
```bash
python -m streamlit run app_streamlit.py
```

---

## 13. Como debugar / estender

### Adicionar um equipamento novo

1. **Não é trivial** — a residualização precisa ser **retreinada** porque o `OneHotEncoder` tem categorias fixas.
2. Conseguir gravações suficientes do novo equipamento (≥ 20 pacientes idealmente).
3. Atualizar `EQUIPMENT_MAP` em `metadata.py`.
4. Re-rodar nb02 (extração) e nb08 (treino) com o dataset expandido.
5. Novo modelo substitui `lung_assistant_v2.joblib`.

### Adicionar uma classe nova (ex: Pneumonia)

1. Atualizar `TARGET_CLASSES` em `metadata.py`.
2. Garantir ≥ 10 pacientes (preferencialmente ≥ 20) da nova classe.
3. Re-rodar nb08 — `LogisticRegression` multinomial aceita N classes nativamente.
4. **Atualizar `CLINICAL_GUIDANCE`** em `clinical_guide.py` com protocolos para a nova classe.
5. Treshold de safety net pode precisar de retuning (nb12).

### Trocar LogReg por XGBoost no modelo final

1. No nb08, substituir `LogisticRegression(...)` por `XGBClassifier(...)`.
2. **Problema:** `LungAssistant` espera `coef_` e `intercept_` (lineares). XGBoost não tem.
3. Solução: criar uma `LungAssistantXGB` que serializa o modelo XGB inteiro via `joblib`.
4. Trade-off: perde explicabilidade linear (precisa de SHAP).

### Adicionar features novas

1. Modificar `extract_feature_vector()` em `features.py`.
2. Retreinar tudo (nb02 + nb08).
3. **O `LungAssistant` antigo deixa de funcionar** (espera 78 features, encontra N).

### Debugar predição inesperada

```python
from assistente import diagnose
r = diagnose("audio.wav", equipment="WelchAllyn Meditron", explain=True)
# Imprime as 5 features que mais contribuíram para cada classe
for cls, items in r["explanation"].items():
    print(f"\n{cls}:")
    for it in items[:5]:
        print(f"  {it['feature']:25s} contrib={it['contribution']:+.3f}")
```

Se `is_outlier=True`, é sinal de que o áudio está fora da distribuição de treino — provável causa: ruído, equipamento diferente, ou doença não-coberta.

### Ajustar agressividade do safety net

```python
# Conservador (zera FN mas muitos FP)
diagnose("audio.wav", equipment="...", safe=True, disease_threshold=0.02)

# Moderado (recomendado)
diagnose("audio.wav", equipment="...", safe=True, disease_threshold=0.10)

# Próximo de argmax
diagnose("audio.wav", equipment="...", safe=True, disease_threshold=0.40)
```

### Testar com áudio sintético / fora de domínio

```python
import numpy as np, soundfile as sf
# Ruído branco — deve ser flagged como outlier
sf.write("test_noise.wav", np.random.randn(22050 * 5), 22050)
r = diagnose("test_noise.wav", equipment="WelchAllyn Meditron")
assert r["outlier"]["is_outlier"], "Outlier detector falhou"
```

### Profiling de tempo

```python
import time
t0 = time.time()
for _ in range(100):
    diagnose("audio.wav", equipment="WelchAllyn Meditron", explain=False)
print(f"Avg: {(time.time()-t0)/100*1000:.1f}ms por predição")
# Esperado: ~200-400ms (dominado por librosa.load e extração)
```

---

## Apêndice — Convenções de código

| Convenção | Aplicação |
|---|---|
| Type hints | Em todas as funções públicas (`def f(x: int) -> str:`) |
| Docstrings | Triple-quoted, primeiro line = resumo, blank line, detalhes |
| Imports | Stdlib → 3rd-party → local (separados por blank lines) |
| Constantes | UPPER_SNAKE_CASE no topo do módulo |
| Variáveis privadas | `_underscore_prefix` |
| Type hints modernos | `list[str]` (Python 3.9+), `X | None` (3.10+) |
| F-strings | Sempre que possível |
| `Path` em vez de `str` para caminhos | Funciona cross-platform |

---

*Fim do manual. Para a ciência por trás das decisões, ver `RELATORIO.md`. Para overview executivo, ver `README.md`.*
