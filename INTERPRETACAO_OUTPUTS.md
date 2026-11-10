# Guia de Interpretação de Outputs — Assistente Respiratório

> ## ⚠️ Revisão de novembro de 2026 — ler antes do resto
>
> Este documento descreve o modelo **v2** e contém números que foram **revistos**.
> Os valores de referência são os de `reports/AVALIACAO_V3.md` (gerado por `python scripts/train.py`)
> e estão resumidos na secção 6 do `README.md`. Em caso de conflito, vale o v3.
>
> | Afirmação neste documento | O que se verificou |
> |---|---|
> | Recall de doença 99,5% (modo safe) | Era por **gravação** (793 das 816 gravações de doentes são COPD). Por **paciente**: 0,94 com θ=0,10; recall URTI 0,46; especificidade 0,28 |
> | Residualização reduz o viés em 89,7% (probe 0,914 → 0,089) | Só para sonda **linear**. Uma sonda RandomForest ainda identifica o equipamento com F1-macro ≈ 0,85–0,87. Viés mitigado, não eliminado |
> | Holdout "virgem" F1m 0,778 confirma ausência de overfitting | O holdout foi separado depois da escolha do modelo e tem ~3 pacientes URTI. É ilustrativo, não confirmatório |
> | Calibração isotónica: log loss −49% / −44% | Era medida nos mesmos dados em que foi ajustada. Aninhada: 0,228 → 0,196 (isotónica) ou 0,150 (sigmoide, usada no v3) |
> | Poupança de ~€157 mil a >€10 milhões/ano | Não se confirma. Por paciente, o modo safe **custa mais** que o argmax em todos os cenários de prevalência testados |
> | 78 features, incluindo `duration_s` | `duration_s` era um atalho (no Meditron separa COPD de Healthy/URTI na perfeição). O v3 corta o áudio a 20 s e usa 77 features |
> | Limiar de outlier χ²(0,99) / FPR 1% | O v3 usa o p99 empírico (346,0); falso alarme medido fora do treino: 5,5% |
> | Modelo de produção `lung_assistant_v2.joblib` | Substituído por `lung_assistant_v3.npz` |

---

> **Para quem usa o sistema** (clínico, avaliador, gestor): "vejo este output — o que significa cada número e como devo agir?"
>
> Complementa: `README.md` (executivo), `RELATORIO.md` (científico), `MANUAL_CODIGO.md` (engenharia).

---

## ⚕️ Princípio ético subjacente — leitura obrigatória

Todo o sistema prioriza **recall de doença** (minimizar Falsos Negativos clínicos). Esta escolha **não é técnica nem económica — é ética**:

| Erro | Natureza | Remediabilidade |
|---|---|---|
| **FN clínico** (doente → "Healthy") | Dano por omissão; paciente vai para casa doente; doença progride; pode ser irreversível | **Não remediável** após dano consolidado |
| **FP** (saudável → "doente") | Inconveniência; consulta extra; ansiedade temporária | **Sempre remediável** pelo médico na revisão |

**Justificação bioética:**
1. **Não-maleficência** (*primum non nocere*): sistema clínico não pode causar dano por omissão.
2. **Subdiagnóstico em Portugal**: 86% dos doentes com DPOC estão por diagnosticar (ONDR). Um modelo que gera FN **alimenta** este problema em vez de resolvê-lo.
3. **Tempo é tecido pulmonar**: DPOC é progressiva e irreversível; cada mês sem tratamento = perda funcional permanente.
4. **Autonomia do paciente**: direito a saber o seu diagnóstico — FN nega este direito.

A análise económica (€/ano) é **consequência mensurável** desta priorização ética, **não a sua causa**. Quando lê valores em euros nos notebooks 11 e 12, leia-os como métrica de gestão sobre o dano clínico, não como justificação primária do modelo.

---

## Índice

1. [Output do modelo (`predict`)](#1-output-do-modelo-predict)
2. [Output da explicação (`explain`)](#2-output-da-explicação-explain)
3. [Output do detetor de outlier (`is_outlier`)](#3-output-do-detetor-de-outlier-is_outlier)
4. [Output do CLI (`assistente.py`)](#4-output-do-cli-assistentepy)
5. [Output do Streamlit](#5-output-do-streamlit)
6. [Métricas dos notebooks](#6-métricas-dos-notebooks)
7. [Gráficos dos notebooks](#7-gráficos-dos-notebooks)
8. [Árvore de decisão clínica](#8-árvore-de-decisão-clínica)
9. [Casos práticos anotados](#9-casos-práticos-anotados)

---

## 1. Output do modelo (`predict`)

Quando chamas `asst.predict("audio.wav", equipment="WelchAllyn Meditron")`, recebes um `dict`:

```python
{
    'diagnosis': 'COPD',
    'confidence': 'alta',
    'top_probability': 0.994,
    'probabilities': {'COPD': 0.994, 'Healthy': 0.000, 'URTI': 0.006},
    'probabilities_raw': {'COPD': 0.991, 'Healthy': 0.001, 'URTI': 0.008},
    'equipment_used': 'WelchAllyn Meditron',
    'outlier': {'is_outlier': False, 'mahalanobis_sq': 96.1, 'threshold_99': 353.1},
    'mode': 'argmax'
}
```

### 1.1. `diagnosis` — a predição final

Sempre uma de: **`'COPD'`, `'Healthy'`, `'URTI'`**.

| Diagnosis | Significado clínico |
|---|---|
| `'COPD'` | Padrão acústico compatível com **Doença Pulmonar Obstrutiva Crónica** |
| `'URTI'` | Padrão acústico compatível com **infeção respiratória alta** (rinite, sinusite, faringite) |
| `'Healthy'` | Padrão acústico **dentro dos parâmetros normais** — sem evidência audível de doença |

⚠️ **Atenção:** `Healthy` significa apenas "não detecto doença a partir do som" — **não exclui** outras patologias que o modelo não foi treinado para detectar (Pneumonia, Asma, Bronquiectasias, etc.).

### 1.2. `confidence` — nível de confiança discreto

Calculado a partir de `top_probability`:

| Valor | Critério | Interpretação |
|---|---|---|
| `'alta'` | P ≥ 0.80 | Modelo está confiante. **Diagnóstico provável.** Pode prosseguir conforme protocolo (mas não dispensa exame físico). |
| `'média'` | 0.60 ≤ P < 0.80 | Modelo está **moderadamente** confiante. Verificar achados clínicos antes de tratar. |
| `'baixa'` | P < 0.60 | Modelo está **incerto**. **Revisão manual obrigatória.** Não tomar decisão clínica apenas com base no output. |
| `'baixa (amostra fora da distribuição)'` | (qualquer P) + outlier | Áudio **estatisticamente atípico**. Pode ser ruído, técnica de captura, ou doença não-coberta. Revisar. |

### 1.3. `top_probability` — número entre 0 e 1

A probabilidade **calibrada** da classe predita.

**Interpretação prática:**

| P | Frequência esperada de acerto* |
|---|---|
| 0.50 | ~50% das predições com esta P estão corretas |
| 0.70 | ~70% |
| 0.85 | ~85% |
| 0.95 | ~95% |
| 0.99 | ~99% |

*\*Em média, no nosso CV. Graças à calibração isotónica (Brier −9%, log loss −49%), estes números têm correspondência razoável com a frequência real — ao contrário das probabilidades brutas do modelo, que sub/sobre-estimam.*

⚠️ **Não confundir com "probabilidade de doença".** É a probabilidade da **classe predita** ser correta — pode ser qualquer das 3.

### 1.4. `probabilities` (calibradas) vs `probabilities_raw` (brutas)

```python
'probabilities':     {'COPD': 0.994, 'Healthy': 0.000, 'URTI': 0.006}   # ← USAR
'probabilities_raw': {'COPD': 0.991, 'Healthy': 0.001, 'URTI': 0.008}   # ← debug
```

**Sempre olha para `probabilities`** (calibradas). As `raw` estão lá só para auditoria — comparar antes/depois da calibração.

**Como ler a distribuição:**

| Padrão | Interpretação |
|---|---|
| Uma classe ≈ 1.00, outras ≈ 0.00 | Modelo muito confiante. Provável caso "clássico" |
| Uma classe ~0.70, outra ~0.30 | Modelo prefere uma mas reconhece ambiguidade |
| Três classes ~0.33 cada | Modelo está perdido. **Não confiar.** |
| Healthy ~0.60, COPD ~0.40 | Próximo da fronteira. Considerar safety net |

### 1.5. `equipment_used` — eco do input

Apenas confirma qual equipamento foi passado. Útil para auditoria e logs.

### 1.6. `mode` — modo de decisão usado

- `'argmax'`: decisão padrão (classe mais provável)
- `'safe'`: aplicado safety net. Se `diagnosis != 'Healthy'` mas P(Healthy) é alta, foi **redirecionado** para a doença mais provável.

### 1.7. `outlier` — secção dedicada (ver §3)

---

## 2. Output da explicação (`explain`)

Quando chamas `asst.explain("audio.wav", equipment="...", top_k=5)`:

```python
{
    'COPD': [
        {'feature': 'duration_s',     'coef': +1.566, 'value_scaled': -0.535, 'contribution': -0.838},
        {'feature': 'mfcc5_std',      'coef': -0.304, 'value_scaled': +2.021, 'contribution': -0.614},
        {'feature': 'mfcc6_mean',     'coef': -0.425, 'value_scaled': -1.334, 'contribution': +0.566},
        {'feature': 'mfcc3_d_mean',   'coef': -0.155, 'value_scaled': -3.153, 'contribution': +0.488},
        {'feature': 'contrast4_mean', 'coef': +0.150, 'value_scaled': +2.583, 'contribution': +0.388},
    ],
    'Healthy': [...],
    'URTI': [...],
}
```

### 2.1. Lendo uma linha

Exemplo: `{'feature': 'duration_s', 'coef': +1.566, 'value_scaled': -0.535, 'contribution': -0.838}`

| Campo | Significado |
|---|---|
| `feature` | Nome da feature (ex: `duration_s` = duração da gravação) |
| `coef` | Coeficiente **fixo** do LogReg para esta classe. **+** = empurra para a classe; **−** = afasta |
| `value_scaled` | Valor desta feature neste áudio, **já normalizado** (mean=0, std=1 no treino). +2 = 2 desvios padrão acima da média do treino |
| `contribution` | `coef × value_scaled`. **+** = empurra **para** a classe; **−** = empurra **contra** |

**Soma das contribuições de todas as features + intercept = logit da classe.** É a decomposição linear exata do LogReg.

### 2.2. Como interpretar — exemplo

```
duration_s   coef=+1.566   value_scaled=-0.535   contribution=-0.838
```

- `coef=+1.566`: gravações longas tendem a ser COPD (coef positivo). Razão: artefacto do dataset — COPD tem áudios mais longos
- `value_scaled=-0.535`: este áudio está **0.5 desvios padrão abaixo** da duração média do treino — é mais **curto** que o típico
- `contribution=-0.838`: porque é curto, **empurra contra COPD** (apesar do coef positivo)

### 2.3. Tipos de contribuições

| Padrão | Interpretação |
|---|---|
| Top features quase todas + para a classe predita | Modelo "decidiu cedo" — todas as evidências alinhadas |
| Mistura de + e − para a classe predita | Sinal misto; modelo decidiu pela soma. **Mais sujeito a erro** |
| Top features apontam para classe **diferente** da predita | Suspeito. Pode ser que o intercept (bias) esteja a empurrar |
| Features dominantes são `duration_s` ou `contrast*` | Pode ser sinal de viés residual (estes features capturaram equipamento) |
| Features dominantes são MFCCs | Bom — captura textura do som, mais clinicamente relevante |

### 2.4. Quando suspeitar da explicação

🚩 Se as top contribuições para a classe predita são **todas pequenas** (`|contribution| < 0.3`), o modelo está a decidir por **pequenas margens cumulativas** — predição menos robusta.

🚩 Se há features com `|value_scaled| > 5`, este áudio é estatisticamente **extremo** — provável outlier (verificar §3).

---

## 3. Output do detetor de outlier (`is_outlier`)

```python
'outlier': {
    'is_outlier': False,
    'mahalanobis_sq': 96.1,
    'threshold_99': 353.1
}
```

### 3.1. Os 3 campos

| Campo | O que é |
|---|---|
| `is_outlier` | Booleano: `True` se `mahalanobis_sq > threshold_99` |
| `mahalanobis_sq` | Distância de Mahalanobis ao quadrado deste áudio ao "centro" do treino |
| `threshold_99` | Limiar empírico: 99% dos áudios de treino têm d² abaixo deste valor (≈ 353) |

### 3.2. Como interpretar `mahalanobis_sq`

| Valor d² | Interpretação |
|---|---|
| 0–50 | **Muito típico**: parece muito com áudios de treino |
| 50–150 | **Típico**: dentro da distribuição |
| 150–353 | **No limite**: estatisticamente raro mas ainda aceitável |
| > 353 (threshold p99) | **Outlier**: 99% dos áudios de treino estão abaixo. Suspeitar |
| > 1000 | **Outlier extremo**: provavelmente ruído, dispositivo desconhecido, ou doença fora do treino |

### 3.3. O que fazer com cada caso

| `is_outlier` | Confidence | Ação recomendada |
|---|---|---|
| False | alta | Pode usar a predição como apoio normal |
| False | média | Verificar achados clínicos antes de tratar |
| False | baixa | **Revisão manual obrigatória** |
| **True** | (qualquer) | **Suspender uso automatizado.** Repetir auscultação. Considerar diagnósticos fora do treino (Pneumonia, Asma, Bronquiectasias, Bronquiolite) |

### 3.4. Causas comuns de outlier

- Ruído ambiente (TV ligada, conversa, equipamento médico)
- Técnica de captura inadequada (estetoscópio mal posicionado)
- Paciente atípico (pediatria extrema, adultos muito idosos com comorbilidades)
- Doença não coberta pelo modelo (Pneumonia em particular tem padrão acústico distintivo)
- Equipamento desconhecido ao modelo (residualização falha)
- Áudio demasiado curto (< 5 segundos)

---

## 4. Output do CLI (`assistente.py`)

```
══════════════════════════════════════════════════════════════════════
                          DIAGNÓSTICO: COPD
══════════════════════════════════════════════════════════════════════
Áudio       : C:\Datasets\...\104_1b1_Al_sc_Litt3200.wav
Equipamento : Littmann 3200 Electronic
Modo        : argmax
Confiança   : alta (P=0.994)
  Mahalanobis² = 96.1  (threshold p99 = 353.1)

Probabilidades calibradas:
  COPD     0.994  ██████████████████████████████  ← predito
  URTI     0.006  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  Healthy  0.000  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

Top contribuições para 'COPD':
  duration_s                 coef=+1.566 × x=-0.535  → -0.838
  mfcc5_std                  coef=-0.304 × x=+2.021  → -0.614
  mfcc6_mean                 coef=-0.425 × x=-1.334  → +0.566
══════════════════════════════════════════════════════════════════════
```

### 4.1. Anatomia visual

| Elemento | O que mostra | Significado |
|---|---|---|
| **Cor do diagnóstico** | Verde = Healthy / Amarelo = URTI / Vermelho = COPD | Codificação visual rápida |
| **`══════` superior** | Banner principal | Captura atenção |
| **`Áudio` / `Equipamento`** | Eco dos inputs | Auditoria + verificação |
| **`Modo`** | `argmax` ou `safe` | Saber qual decisão foi aplicada |
| **`Confiança (P=...)`** | Discreto + numérico | Resumido + preciso |
| **`Mahalanobis²`** | Numérico vs threshold | Distância da distribuição |
| **`██` (barras)** | Visualização das probabilidades | Comparação rápida das 3 classes |
| **`←  predito`** | Marcador | Reforça qual venceu |
| **`Top contribuições`** | 5 features mais relevantes | Justificação do diagnóstico |
| **Cor das contribuições** | Verde = + / Vermelho = − | Para a classe predita |

### 4.2. Quando ⚠️ aparece

Se o áudio é outlier, aparece uma linha extra:

```
⚠️  OUTLIER: amostra fora da distribuição — revisar manualmente.
  Mahalanobis² = 458.2  (threshold p99 = 353.1)
```

A confiança automaticamente muda para `baixa (amostra fora da distribuição)`.

### 4.3. Output em JSON (`--json`)

```bash
python assistente.py audio.wav -e "Meditron" --json
```

Devolve a estrutura raw, sem cores nem formatação. Use para integração com outros sistemas (logging, dashboards, EHR).

---

## 5. Output do Streamlit

### 5.1. Métricas no topo (3 colunas)

```
┌────────────────────────┬──────────────┬──────────────┐
│ Diagnóstico: COPD      │ 🟢 alta      │ Outlier: Não │
│ Modo: argmax           │ P=0.994      │ d²=96/353    │
└────────────────────────┴──────────────┴──────────────┘
```

| Coluna | Significado |
|---|---|
| **Diagnóstico** | Cor verde/amarelo/vermelho conforme classe |
| **Confiança** | Emoji 🟢 alta / 🟡 média / 🔴 baixa + P numérica |
| **Outlier** | "Não" (verde) ou "SIM" (vermelho com seta para baixo) |

### 5.2. Banner de urgência clínica

Logo abaixo, banner colorido a ocupar a largura toda:

| Cor | Urgência | O que significa |
|---|---|---|
| 🟢 Verde | NENHUMA | Sem ação imediata necessária (geralmente Healthy alta confiança) |
| 🟡 Amarelo | BAIXA | Acompanhamento ambulatorial normal (URTI típica) |
| 🟠 Laranja | MODERADA | Encaminhamento para especialista em 2–4 semanas (COPD típica) |
| 🟠 Laranja | REVISÃO MANUAL OBRIGATÓRIA | Modelo incerto ou outlier — não automatizar |
| 🔴 Vermelho | ALTA / EMERGÊNCIA | Encaminhar urgência hospitalar |

### 5.3. Gráfico de probabilidades duplo (calibrada vs bruta)

```
COPD     |█████████████████████████████  0.994
         |████████████████████████████   0.991 (bruto)
Healthy  |  0.000
         |  0.001
URTI     |▌ 0.006
         |▌ 0.008
```

**Para que serve:**
- Barra **cheia** = calibrada (use esta)
- Barra **transparente** (alpha 0.4) = bruta (referência)
- Se a diferença é grande, a calibração está a fazer trabalho importante

### 5.4. Aba "Visualização do áudio"

**Waveform** (em cima): mostra a forma de onda no tempo. Picos = sons mais fortes.

**Mel-spectrogram** (em baixo): tempo no eixo X, frequência (mel) no eixo Y, intensidade em dB nas cores.

**Como ler o mel-spec:**

| Padrão visual | Sugestão clínica |
|---|---|
| Bandas horizontais persistentes a ~200–800 Hz | Wheezes (tons sustentados) |
| Manchas verticais curtas e brilhantes | Crackles (eventos transitórios) |
| Energia distribuída uniformemente | Respiração normal ou ruído |
| Energia muito concentrada em baixa frequência | Tipicamente COPD (respiração mais lenta, congesta) |

### 5.5. Aba "Explicação"

Tabela das features mais influentes com **gradiente RdYlGn** (vermelho-amarelo-verde) na coluna Contribuição:

- **Verde escuro**: contribuição +3 (forte a favor)
- **Amarelo**: contribuição ≈ 0 (irrelevante)
- **Vermelho escuro**: contribuição -3 (forte contra)

Permite ler "qual feature decidiu" em segundos.

### 5.6. Aba "Detalhes técnicos"

JSON cru do output do `predict()`. Para debug ou para clínicos avançados que querem ver tudo.

### 5.7. Painéis de orientação clínica

4 painéis colapsáveis (expander):

| Painel | Conteúdo | Default |
|---|---|---|
| 🧪 Exames complementares | Lista priorizada (espirometria, Rx, etc.) | **Aberto** |
| 💊 Tratamento imediato | Doses sugeridas (broncodilatador, etc.) | Fechado |
| 🚨 Sinais de alarme | O que verificar para escalar urgência | **Aberto** |
| 📚 Educação do paciente | O que dizer ao paciente | Fechado |

Os 2 abertos por defeito são os mais críticos para decisão imediata.

---

## 6. Métricas dos notebooks

### 6.1. Métricas de classificação básica

| Métrica | Fórmula | O que mede | Quando usar |
|---|---|---|---|
| **Accuracy** | (TP+TN)/total | % de acertos | Apenas com classes balanceadas |
| **Precision** | TP/(TP+FP) | Dos que eu disse "positivo", quantos eram | Quando custo de FP é alto |
| **Recall (sensibilidade)** | TP/(TP+FN) | Dos positivos reais, quantos eu peguei | **Crítico em medicina** (não perder doentes) |
| **F1** | 2·P·R/(P+R) | Média harmónica precision/recall | Compromisso |
| **F1-macro** | Média das F1 por classe | Trata classes igualmente | **Métrica principal aqui** |
| **F1-weighted** | F1 ponderada por suporte | Reflete prevalência | Otimista com desbalanceamento |

### 6.2. Por que F1-macro é a métrica principal

Com **COPD = 76% dos pacientes**, um modelo trivial "sempre COPD" tem accuracy 76%. **F1-macro castiga** este modelo porque calcula F1 separado para cada classe e tira média simples — Healthy e URTI baixos puxam para baixo.

| F1-macro | Interpretação |
|---|---|
| < 0.33 | Pior que dummy (sempre uma classe) |
| 0.33–0.50 | Fraco; modelo praticamente não distingue classes |
| 0.50–0.65 | Razoável para problema difícil |
| **0.65–0.78** | **Bom** (estamos aqui) |
| 0.78–0.90 | Muito bom |
| > 0.90 | Excelente (raro em medicina real) |

### 6.3. Métricas de calibração

| Métrica | Fórmula | O que mede | Bom |
|---|---|---|---|
| **Brier multiclasse** | mean(Σ_k (P_k − y_k)²) | Erro quadrático das probabilidades | Próximo de 0 |
| **Log loss** | −mean(Σ_k y_k · log(P_k)) | Penaliza muito predições confiantes erradas | Próximo de 0 |

**Comparação típica antes/depois calibração:**

| Estado | Brier | Log loss |
|---|---|---|
| Sem calibração | 0.357 | 0.862 |
| Com isotónica | 0.326 (−9%) | 0.487 (−44%) |

**Interpretação:** quanto **menor**, melhor. Log loss é mais sensível a probabilidades sobre-confiantes erradas.

### 6.4. Métricas estatísticas

#### Wilcoxon p-value (nb04)

Tabela tipo:

```
                KNN    LDA    LogReg   XGBoost
KNN              —     0.56   0.32    *0.002
LDA             0.56    —     0.85    *0.002
LogReg          0.32   0.85    —      *0.002
XGBoost        *0.002 *0.002 *0.002    —
```

| p-value | Interpretação |
|---|---|
| < 0.001 | Diferença muito significativa |
| 0.001–0.01 | **Significativa** (typical threshold) |
| 0.01–0.05 | Marginalmente significativa |
| 0.05–0.10 | Inconclusivo |
| > 0.10 | **Sem evidência de diferença** |

**No nosso caso:** KNN, LDA, LogReg são **estatisticamente equivalentes** (p > 0.30 entre eles). XGBoost é **significativamente pior** que os 3 (p = 0.002 contra cada).

#### IC95% bootstrap

Exemplo: `KNN: 0.610 [0.589, 0.631]` significa:
- Média da métrica: 0.610
- 95% de confiança que o verdadeiro valor está em [0.589, 0.631]
- Se os ICs de 2 modelos **se sobrepõem**, não dá para afirmar que um é melhor

### 6.5. Métricas de custo (nb11, nb12)

#### Custo total

```
Argmax:         €6.480
Cost-sensitive: €3.300 (−49%)
```

**Como calcular:** soma de `count_ij × custo_ij` para cada par (real, predito) na matriz de custos.

**Interpretação:** menor = melhor. **Diferença percentual** é mais útil que absoluto (varia com volume).

#### Recall doente / Especificidade

| Métrica | Fórmula | Significado |
|---|---|---|
| **Recall doente** | TP_doente / (TP_doente + FN_doente) | % de doentes corretamente identificados |
| **Especificidade** | TN_doente / (TN_doente + FP_doente) | % de saudáveis corretamente identificados |

**Trade-off fundamental em clínica:**

| Recall | Especificidade | Cenário |
|---|---|---|
| 100% | 0% | "Sempre doente" — zero FN mas todos vão pra médico |
| 99.5% | 31% | **Safety net θ=0.10 (recomendado)** |
| 97.8% | 65.7% | Argmax (mais equilibrado mas perde 2.2% de doentes) |
| 50% | 95% | Modelo tímido — perde metade dos doentes |

---

## 7. Gráficos dos notebooks

### 7.1. Matriz de confusão

```
                  Predito
              COPD  Healthy  URTI
Real  COPD    [62]     1      1
      Healthy   3    [17]     6
      URTI      2      8     [4]
```

**Como ler:**

- **Diagonal `[]`** = acertos
- **Fora da diagonal** = erros
- **Linha** = casos reais dessa classe; **soma da linha** = n total dessa classe
- **Coluna** = predições dessa classe

**No nosso caso (paciente, n=104, modelo argmax):**

| Erro | Quantidade | Custo |
|---|---|---|
| COPD → Healthy (FN clínico crítico) | 0 | — |
| COPD → URTI | 1 | Tratamento parcial |
| URTI → Healthy (FN clínico) | 18 | Pior tipo de erro |
| Healthy → COPD | 3 | Consulta extra |

### 7.2. Reliability diagram

Gráfico tipo:

```
1.0 |                            *
    |                       *
    |                   *      ← linha pontilhada = perfeito (y=x)
0.5 |           *  ←  curva calibrada (azul)
    |       *   ←  curva bruta (vermelha)
    |   *
0.0 |____________________________
    0.0       0.5            1.0
       Probabilidade predita
```

| Padrão | Interpretação |
|---|---|
| Curva **sobre a diagonal** | Modelo **sub-confiante** (P=0.7 mas acerta 80%) |
| Curva **sob a diagonal** | Modelo **sobre-confiante** (P=0.9 mas acerta 70%) |
| Curva **próxima da diagonal** | Calibrado ✓ |

A calibração isotónica empurra a curva vermelha (bruta) para a diagonal (azul calibrada).

### 7.3. Boxplot de F1m (nb04)

```
KNN     ┃───[ █████ ]───┃    (mediana, IQR, whiskers)
LDA     ┃───[ █████ ]───┃
LogReg  ┃───[ █████ ]───┃
XGBoost ┃─[█████]─┃           ← claramente mais baixo
```

**Pontos pretos** sobrepostos = valores individuais das 10 repetições.

| Padrão | Interpretação |
|---|---|
| Boxes que **se sobrepõem** | Não dá para distinguir modelos |
| Box estreita | Modelo estável, baixa variância |
| Box larga | Modelo instável, alta variância |
| Outliers (pontos longe do box) | Repetições atípicas — verificar seed |

### 7.4. Heatmap de Wilcoxon

```
       KNN   LDA  LogReg  XGB
KNN     —   0.56  0.32   0.002  ← VERDE escuro
LDA    0.56   —   0.85   0.002
LogReg 0.32 0.85    —    0.002
XGB   0.002 0.002 0.002    —
```

**Cores:**
- 🟢 Verde (p baixo) = diferença significativa
- 🟡 Amarelo (p ~0.05) = limítrofe
- 🔴 Vermelho (p alto) = sem diferença

### 7.5. Sensibilidade ao custo de FN_COPD (nb11)

```
Economia anual (€)
   400k |              ●
   300k |          ●
   200k |       ●          ← cresce com custo de FN_COPD
   150k |    ●
   100k |  ●
        |____________________
        500  1k  1.5k  2.5k  4k
              Custo FN_COPD (€)
```

**Interpretação:** quanto **maior** o custo assumido de FN_COPD, **mais** a estratégia cost-sensitive economiza vs argmax. Robusto em toda a gama testada.

### 7.6. Mel-spectrogram

```
Frequência (mel)
8000  |   ░░  ▓▓░░     ← wheezes (bandas horizontais)
4000  |░░ ▓▓░░░░░░
2000  |░░░░░░░░░░░░    ← respiração base
 500  |░▓░░░▓░░░▓      ← crackles (manchas verticais)
      |____________
        0   2   4  Tempo (s)
```

Cores mais claras/quentes = mais energia em dB. Permite identificação visual de wheezes e crackles antes mesmo do modelo decidir.

---

## 8. Árvore de decisão clínica

**Como agir baseado no output completo:**

```
                       ┌─────────────────────┐
                       │ predict(audio, equip)│
                       └──────────┬──────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  │ outlier.is_outlier == True?   │
                  └───────┬──────────────┬────────┘
                          │SIM           │NÃO
                          ▼              │
       ┌───────────────────────────┐    │
       │ REVISÃO MANUAL OBRIGATÓRIA│    │
       │ - Repetir auscultação     │    │
       │ - Considerar Pneumonia,   │    │
       │   Asma, outras patologias │    │
       │ - Avaliação clínica       │    │
       │   completa                │    │
       └───────────────────────────┘    │
                                        ▼
                       ┌─────────────────────────┐
                       │ confidence == 'baixa'?  │
                       └────┬──────────────┬─────┘
                            │SIM           │NÃO
                            ▼              │
            ┌──────────────────────────┐  │
            │ REVISÃO MANUAL            │  │
            │ - Não iniciar tratamento │  │
            │ - Auscultação manual     │  │
            │ - Exames complementares  │  │
            └──────────────────────────┘  │
                                          ▼
                         ┌──────────────────────────┐
                         │ diagnosis == ?           │
                         └────┬──────┬──────┬───────┘
                              │      │      │
                           COPD   Healthy  URTI
                              │      │      │
                              ▼      ▼      ▼
         ┌────────────────────┐ ┌─────────┐ ┌────────────────────┐
         │ Encaminhar         │ │ Alta com│ │ Tratamento         │
         │ Pneumologia        │ │ orient. │ │ sintomático        │
         │ 2-4 semanas        │ │ Manter  │ │ Acompanhamento     │
         │ + espirometria     │ │ rastr.  │ │ MGF/CSP            │
         │ + Rx tórax         │ │ ⚠ Não   │ │ ⚠ Antibiótico só   │
         │ + cessação tabag.  │ │ ignorar │ │   com critério     │
         │                    │ │ sintom. │ │                    │
         │ 🚨 Se dispneia em │ │ refer.  │ │ 🚨 Se dispneia →   │
         │ repouso: URGÊNCIA │ │         │ │   descartar PNM    │
         └────────────────────┘ └─────────┘ └────────────────────┘
```

### Casos especiais

| Situação | Acrescentar |
|---|---|
| `mode='safe'` e diagnosis ≠ Healthy | Verificar que modo conservador foi aplicado. Confirmar achados antes de tratar |
| `confidence='média'` | Verificar sintomas referidos antes de seguir guidance |
| Sintomas clínicos divergem do output | **Sempre prevalece a clínica.** Output é apoio, não decisão |

---

## 9. Casos práticos anotados

### Caso 1 — COPD clássico (confiança alta)

**Input:** `104_1b1_Al_sc_Litt3200.wav`, equipamento `Littmann 3200 Electronic`

**Output:**
```python
{
    'diagnosis': 'COPD',
    'confidence': 'alta',
    'top_probability': 0.994,
    'probabilities': {'COPD': 0.994, 'Healthy': 0.000, 'URTI': 0.006},
    'outlier': {'is_outlier': False, 'mahalanobis_sq': 96.1, 'threshold_99': 353.1},
    'mode': 'argmax'
}
```

**Leitura:**
- ✅ **Predição: COPD** com 99.4% — modelo muito confiante
- ✅ Healthy = 0%, URTI = 0.6% → nenhuma ambiguidade
- ✅ Mahalanobis 96 << 353 → áudio típico, dentro da distribuição
- ✅ Confiança "alta" → pode prosseguir conforme guidance COPD

**Ação clínica:**
1. Aplicar guidance COPD: encaminhar Pneumologia em 2–4 semanas
2. Solicitar espirometria com prova broncodilatadora (confirma)
3. Avaliar cessação tabágica
4. Verificar sinais de alarme (dispneia em repouso → urgência)

**Confiança no output: ALTA.**

---

### Caso 2 — Healthy ambíguo (confiança média)

**Input:** `102_1b1_Ar_sc_Meditron.wav`, equipamento `WelchAllyn Meditron`

**Output:**
```python
{
    'diagnosis': 'Healthy',
    'confidence': 'média',
    'top_probability': 0.692,
    'probabilities': {'COPD': 0.000, 'Healthy': 0.692, 'URTI': 0.308},
    'outlier': {'is_outlier': False, 'mahalanobis_sq': 140.9, 'threshold_99': 353.1},
    'mode': 'argmax'
}
```

**Leitura:**
- 🟡 **Predição: Healthy** mas só com 69%
- ⚠️ URTI = 30.8% → modelo reconhece **ambiguidade significativa**
- ⚠️ Mahalanobis 141 ≈ threshold/2.5 → áudio um pouco atípico
- Confiança "média" → verificar antes de confirmar Healthy

**Ação clínica:**
1. **Não dar alta automática.** Investigar sintomas referidos
2. Se o paciente refere coriza, faringite, febre baixa → tratar como URTI provável
3. Se assintomático → manter Healthy mas agendar reavaliação em 2 semanas
4. Considerar repetir auscultação para confirmar

**Confiança no output: MÉDIA. Decisão final cabe ao clínico.**

**Se usado modo `safe` com θ=0.10:**
```python
{'diagnosis': 'URTI', 'confidence': 'baixa', ...}
```
→ Sistema redireciona para URTI porque P(doente)=30.8% > 10%. **Comportamento ético do safety net.**

---

### Caso 3 — Outlier (revisão obrigatória)

**Input:** `audio_pediatrico_chorando.wav`, equipamento `WelchAllyn Meditron`

**Output hipotético:**
```python
{
    'diagnosis': 'URTI',
    'confidence': 'baixa (amostra fora da distribuição)',
    'top_probability': 0.456,
    'probabilities': {'COPD': 0.234, 'Healthy': 0.310, 'URTI': 0.456},
    'outlier': {'is_outlier': True, 'mahalanobis_sq': 678.4, 'threshold_99': 353.1},
    'mode': 'argmax'
}
```

**Leitura:**
- 🚨 **Outlier confirmado** (d²=678 > 353)
- 🚨 Probabilidades quase distribuídas igualmente (23/31/46%) → modelo **confuso**
- 🚨 Confiança automaticamente rebaixada para "baixa (fora da distribuição)"

**Ação clínica:**
1. **Suspender uso automatizado**
2. Investigar causa do outlier:
   - Áudio com ruído (criança a chorar, equipamento ambiente)?
   - Equipamento mal posicionado?
   - Paciente atípico (pediatria, adulto muito idoso)?
   - Doença não-coberta? (suspeitar Pneumonia, Bronquiolite especialmente em crianças)
3. **Repetir gravação** em condições controladas
4. Se outlier persiste → encaminhar para auscultação manual + Rx tórax

**Confiança no output: NULA.** Sistema reconheceu que está fora do seu domínio.

---

### Caso 4 — URTI claro (recall do safety net)

**Input:** `101_1b1_Al_sc_Meditron.wav`, equipamento `WelchAllyn Meditron`, `mode='safe'`, `θ=0.05`

**Output:**
```python
{
    'diagnosis': 'URTI',
    'confidence': 'média',
    'top_probability': 0.753,
    'probabilities': {'COPD': 0.000, 'Healthy': 0.247, 'URTI': 0.753},
    'outlier': {'is_outlier': False, 'mahalanobis_sq': 78.4, 'threshold_99': 353.1},
    'mode': 'safe'
}
```

**Leitura:**
- ✅ **Predição: URTI** com 75.3%
- 🟡 Healthy = 24.7% → ambiguidade moderada, mas P(doente) = 75% > 5% → safety net **não bloqueou** (já era URTI)
- ✅ Mahalanobis 78 << 353 → típico
- Confiança "média" → tratar com cautela

**Ação clínica:**
1. Aplicar guidance URTI: sintomáticos, hidratação, repouso
2. ⚠️ **Não prescrever antibiótico** sem critério clínico (Centor, sinusite documentada)
3. Verificar sinais de alarme: dispneia → reavaliar (descartar pneumonia)
4. Reavaliar em 7–10 dias se sintomas persistem

**Confiança no output: MÉDIA. Acompanhar evolução clínica.**

---

## Apêndice — Glossário rápido

| Termo | Significado |
|---|---|
| **FN clínico** | Falso Negativo clínico = doente classificado como Healthy. **Erro eticamente inaceitável** sob o princípio da não-maleficência (*primum non nocere*) — sistema clínico não pode dar alta a um doente. Custo cumulativo e potencialmente irreversível no paciente. |
| **FP** | Falso Positivo = saudável classificado como doente. **Erro remediável**: médico corrige na consulta seguinte, sem dano permanente. |
| **Calibração** | Processo de fazer P=0.8 corresponder a 80% de acerto real. |
| **Mahalanobis²** | Distância estatística que considera correlações entre features. |
| **Argmax** | Decisão padrão: classe com maior probabilidade. |
| **Safety net** | Modo conservador: bloqueia Healthy se P(doente) > θ. |
| **Outlier** | Áudio estatisticamente fora da distribuição de treino. |
| **F1-macro** | Média não-ponderada das F1 por classe (favorece desempenho equilibrado). |
| **Brier** | Erro quadrático das probabilidades (menor = melhor). |
| **Log loss** | Penaliza muito predições confiantes erradas. |
| **Wilcoxon** | Teste estatístico para comparar 2 modelos pareados. |
| **IC95%** | Intervalo de confiança a 95% (bootstrap). |
| **Cost-sensitive** | Decisão que minimiza custo esperado, não erro. |
| **Residualização** | Remoção da componente linear explicada por variável confusora (equipamento). |

---

*Fim do guia de interpretação. Para detalhes científicos das métricas, ver `RELATORIO.md` §4. Para implementação técnica, ver `MANUAL_CODIGO.md`.*
