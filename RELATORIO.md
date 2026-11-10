# Relatório Técnico — Assistente de Diagnóstico Respiratório

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

**Projeto:** Sistema de apoio à decisão clínica para classificação de sons de auscultação pulmonar
**Dataset:** ICBHI 2017 Respiratory Sound Database
**Pipeline:** Ciência de Dados + Machine Learning + Decision Support Clínico
**Data:** Maio de 2026

---

## Resumo Executivo

Foi desenvolvido um sistema de classificação automática de sons respiratórios capaz de distinguir três classes clínicas — **Healthy**, **COPD** (Doença Pulmonar Obstrutiva Crónica) e **URTI** (Upper Respiratory Tract Infection) — a partir de gravações de auscultação realizadas com estetoscópio digital.

O pipeline foi construído ao longo de **12 notebooks Jupyter**, com avaliação progressivamente mais rigorosa: comparação de 4 algoritmos clássicos, validação cruzada repetida, *nested cross-validation* com tuning de hiperparâmetros, deteção e mitigação de viés de equipamento, calibração de probabilidades, validação em *holdout* virgem, comparação com CNN, análise de custo clínico em euros e implementação de mecanismo de *safety net* para minimização de falsos negativos clínicos.

**Resultados principais:**

- **Recall de doença (métrica clínica primária): 99,5%** no modo conservador — minimiza Falsos Negativos clínicos, que são **inaceitáveis sob o princípio bioético da não-maleficência** (*primum non nocere*).
- **F1-macro a nível de paciente**: 0,647 ± 0,033 (10×5-fold CV) / 0,778 (*holdout* virgem)
- **Probabilidades calibradas**: log loss reduzido em 49%
- **Viés de equipamento identificado e mitigado** (probe atingia F1=0,91 prevendo equipamento a partir das mesmas features)
- **Modelo final empacotado** com API Python, interface CLI e aplicação web (Streamlit) com módulo integrado de orientação clínica
- **Análise económica como dimensão secundária** (custos reais do SNS): economia projetada de **~€157 mil/ano** (adoção parcial, 50 mil auscultações) a **>€10 milhões/ano** se contabilizado o custo de FN de DPOC a €10k. O custo financeiro é **consequência** do dano clínico, não a justificação primária do modelo.

---

## 1. Contexto e Objetivos

### 1.1. Problema

As doenças respiratórias crónicas (DPOC) e agudas (infeções do trato respiratório) representam carga substancial nos sistemas de saúde europeus. O **diagnóstico precoce** depende crucialmente da auscultação pulmonar, mas:

- Existe variabilidade interobservador entre médicos
- Em contextos de triagem (cuidados primários, telemedicina), a auscultação pode ser tecnicamente limitada
- A escassez de pneumologistas obriga a triagem por clínicos gerais

**Hipótese:** algoritmos de aprendizagem automática treinados em gravações anotadas podem servir como **apoio à decisão** (não substituto), permitindo:
- Triagem mais sistemática
- Sinalização de casos prováveis de DPOC para encaminhamento prioritário
- Redução do número de auscultações não diagnosticadas que evoluem para internamentos evitáveis

### 1.2. Objetivos do projeto

1. **Construir um classificador** robusto sobre sons respiratórios
2. **Validar honestamente** o desempenho com técnicas que evitem vazamento de informação
3. **Identificar e mitigar viés** intrínseco ao dataset
4. **Calibrar probabilidades** para uso clínico (P=0,8 deve significar 80% de risco)
5. **Quantificar impacto económico** em valores realistas europeus
6. **Implementar safety net** para minimizar falsos negativos clínicos
7. **Empacotar** num sistema utilizável (API, CLI, app web) com orientação clínica integrada

---

## 2. Dataset — ICBHI 2017 Respiratory Sound Database

### 2.1. Composição original

- **920 gravações** (.wav) com 920 ficheiros de anotação (.txt)
- **126 pacientes**
- Duração média: 21 segundos (mínimo 10s, máximo 90s)
- 4 equipamentos de captura: AKG C417L Microphone, WelchAllyn Meditron, 3M Littmann Classic II SE, 3M Littmann 3200 Electronic

### 2.2. Diagnósticos disponíveis

| Diagnóstico | Gravações | Pacientes |
|---|---:|---:|
| COPD | 793 | 64 |
| Pneumonia | 37 | 6 |
| Healthy | 35 | 26 |
| URTI | 23 | 14 |
| Bronchiectasis | 16 | — |
| Bronchiolitis | 13 | — |
| LRTI | 2 | — |
| Asthma | 1 | — |

### 2.3. Filtros aplicados

**Decisão metodológica:** focar nas 3 classes com **n ≥ 14 pacientes** para garantir viabilidade estatística em k-fold CV. Pneumonia (6 pacientes) foi inicialmente considerada mas removida — com 6 pacientes, 5-fold CV implicava 1–2 pacientes por fold de teste, gerando variância intolerável.

**Dataset final:** 851 gravações, 104 pacientes, 3 classes:
- COPD: 793 grav. / 64 pac.
- Healthy: 35 grav. / 26 pac.
- URTI: 23 grav. / 14 pac.

### 2.4. Achado crítico — viés de equipamento

| Equipamento | COPD | Healthy | URTI |
|---|---:|---:|---:|
| AKG C417L | 32 pac. (646 grav.) | **0** | **0** |
| Littmann 3200 | 11 pac. | **0** | **0** |
| Littmann Classic | 16 pac. | **0** | **0** |
| **WelchAllyn Meditron** | 8 pac. | **26 pac.** | **14 pac.** |

**100% dos pacientes Healthy e URTI foram gravados com Meditron.** Os outros três equipamentos só contêm COPD. Este desequilíbrio gera **shortcut learning**: o modelo pode aprender a identificar o equipamento (que correlaciona com o diagnóstico) em vez do som respiratório em si.

---

## 3. Metodologia

### 3.1. Pipeline geral

```
áudio .wav + equipamento
        ↓
   extração 78 features (librosa)
        ↓
   residualização sobre equipamento (LinearRegression one-hot)
        ↓
   StandardScaler
        ↓
   detector outlier (Mahalanobis)
        ↓
   Regressão Logística multinomial (class_weight='balanced')
        ↓
   calibração isotónica (one-vs-rest, OOF)
        ↓
   decisão: argmax OU safety net (mode='safe')
        ↓
   diagnóstico + probabilidades + outlier flag + explicação
        ↓
   módulo de orientação clínica (decision support)
```

### 3.2. Engenharia de features

Extração com `librosa`, totalizando **78 features** por gravação:

| Família | Detalhe | Nº features |
|---|---|---:|
| MFCC (13 coef.) | mean, std, Δ, Δ² | 52 |
| Chroma STFT (12 bins) | mean | 12 |
| Spectral contrast (7 bandas) | mean | 7 |
| Espectrais agregadas | centroid, rolloff, bandwidth, flatness, RMS, ZCR | 6 |
| Duração | duration_s | 1 |

Persistido em `data/processed/features.csv` (851 × 84 colunas, incluindo metadata).

### 3.3. Validação — princípios

**Princípio fundamental:** todo split de dados é realizado **por paciente** (`groups=patient_id`), nunca por gravação. Como pacientes COPD têm em média 12 gravações cada, split por gravação vazaria identidade.

Técnicas utilizadas em ordem crescente de rigor:

| Técnica | Onde | Propósito |
|---|---|---|
| `StratifiedGroupKFold(5)` | nb03 | Baseline |
| Repeated 10×5-fold | nb04 | Reduzir variância da estimativa |
| Wilcoxon pareado | nb04 | Significância estatística entre modelos |
| Nested CV (outer 3×5, inner 3) | nb05 | Tuning sem vazamento |
| Holdout 80/20 virgem | nb09 | Validação final independente |

### 3.4. Tratamento de viés — residualização

Para cada feature *i*, ajustamos:

```
feature_i = β_i,0 + β_i,1·1{AKG} + β_i,2·1{Litt3200} + β_i,3·1{LittClassic} + β_i,4·1{Meditron} + resíduo_i
```

E usamos o **resíduo** como nova feature. Intuitivamente: removemos a parte da feature explicada linearmente pelo equipamento, preservando a parte que pode discriminar diagnóstico independentemente do dispositivo.

**Crítico:** a residualização é ajustada **em cada fold de treino**, nunca no dataset inteiro, para evitar vazamento.

### 3.5. Calibração de probabilidades

A Regressão Logística produz probabilidades, mas não necessariamente *calibradas* — particularmente sob desbalanceamento extremo. Aplicámos **Regressão Isotónica** (one-vs-rest) sobre as probabilidades *out-of-fold*:

```
P_cal_k(x) = IsotonicRegression_k( P_raw_k(x) )
```

normalizando depois para somar 1. Métricas avaliadas: **Brier multiclasse** e **log loss**.

### 3.6. Deteção de outlier — Mahalanobis empírico

Distância de Mahalanobis no espaço residualizado+escalado:

```
d²(x) = (x − μ)ᵀ Σ⁻¹ (x − μ)
```

com Σ regularizada (`Σ + 1e-4·I`). Sob hipótese gaussiana, `d² ~ χ²(p=78)` com threshold a 99% ≈ 110.

**Achado:** as features **não são gaussianas** — `d² > 110` flaggea 19% do próprio treino (esperado: 1%). Corrigido usando **quantil empírico p99** das `d²` observadas no treino, que dá threshold ≈ 353 e taxa de outliers no treino ≈ 1%.

### 3.7. Safety Net — minimização de FN clínico

**Definição:** *Falso Negativo Clínico* = paciente doente (COPD ou URTI) classificado como Healthy.

Duas estratégias avaliadas:

**A) Cost-sensitive decision rule:**
```
ŷ = argmin_j Σ_i C_{ij} · P(y=i | x)
```

**B) Safety Net hierárquica:**
```
se 1 − P(Healthy | x) ≥ θ:
    ŷ = argmax entre classes doentes
senão:
    ŷ = Healthy
```

---

## 4. Resultados Detalhados (Notebook a Notebook)

### Notebook 01 — Análise exploratória e extração inicial

- Carregamento de metadata (diagnóstico, demografia, parsing dos nomes de ficheiro)
- Visualizações: distribuição de classes, equipamentos, locais de auscultação, ciclos respiratórios
- Demonstração da extração com librosa: waveform, mel-spectrogram, MFCC para 1 paciente Healthy e 1 COPD

**Decisão de modelagem:** filtrar para 3 classes (Healthy, COPD, URTI). Asthma (n=1) e LRTI (n=2) inviáveis; Pneumonia/Bronchiectasis/Bronchiolitis (n=6, 16, 13 grav. — mas só 6 pac. em Pneumonia) excluídas.

### Notebook 02 — Extração completa de features

Extração das 78 features para as 851 gravações filtradas. Tempo: ~5 minutos. Verificações:

- Shape: (851, 84) — 78 features + 6 colunas de metadata
- **Zero NaN, zero Inf** ✓
- Distribuição das features por classe visualizada em histograma 2×5
- Correlação entre MFCCs (mean) — matriz 13×13

### Notebook 03 — Baseline de 4 modelos

`StratifiedGroupKFold(5, random_state=42)` × 4 modelos. Métricas em nível de gravação (`y_pred` out-of-fold) e nível paciente (voto majoritário).

| Modelo | Acc grav. | F1m grav. | Acc pac. | F1m pac. |
|---|---:|---:|---:|---:|
| KNN | 0,949 | 0,608 | 0,760 | 0,612 |
| LogReg | 0,931 | 0,545 | 0,731 | 0,583 |
| LDA | 0,928 | 0,544 | 0,721 | 0,547 |
| XGBoost | 0,942 | 0,524 | 0,721 | 0,513 |

**KNN aparenta ser o melhor**, mas com um único *random_state* não temos noção da variância.

### Notebook 04 — Repeated Cross-Validation (10×5)

Repetimos o split 10 vezes com seeds diferentes para obter **médias ± desvios** e testar significância via **Wilcoxon pareado**.

| Modelo | F1m paciente |
|---|---|
| KNN | 0,610 ± 0,035 |
| LDA | 0,598 ± 0,043 |
| LogReg | 0,586 ± 0,049 |
| XGBoost | 0,499 ± 0,038 |
| Dummy (sempre COPD) | 0,254 ± 0,000 |

**Matriz Wilcoxon (p-values, F1m paciente, n=10):**

| | KNN | LDA | LogReg | XGBoost |
|---|---:|---:|---:|---:|
| **KNN** | — | 0,557 | 0,322 | **0,002** |
| **LDA** | 0,557 | — | 0,846 | **0,002** |
| **LogReg** | 0,322 | 0,846 | — | **0,002** |
| **XGBoost** | **0,002** | **0,002** | **0,002** | — |

**Conclusão crítica:**

- KNN, LDA, LogReg são **estatisticamente equivalentes** (p ≥ 0,32)
- XGBoost é **significativamente pior** que os três (p = 0,002 contra cada)
- A "vitória" do KNN no nb03 foi sorte do split

### Notebook 05 — Nested CV + tuning de XGBoost

Hipótese: XGBoost teria perdido por *overfit* na classe majoritária (COPD = 76% dos pacientes). Tentámos corrigir com `sample_weight` balanceado + GridSearchCV (36 combinações: n_estimators, max_depth, learning_rate, min_child_weight) em CV interno.

| Modelo | F1m paciente |
|---|---|
| KNN | 0,579 ± 0,121 |
| LDA | 0,575 ± 0,120 |
| XGB-tuned | 0,556 ± 0,140 |
| XGB-balanced | 0,529 ± 0,118 |
| XGB-default | 0,527 ± 0,116 |

**Wilcoxon XGB-tuned vs baselines:**
- vs KNN: p = 0,551 (não significativo)
- vs XGB-default: p = 0,151 (não significativo)

**Conclusão:** mesmo tunado e balanceado, XGBoost não supera os modelos lineares neste dataset. **Hiperparâmetros estáveis escolhidos pelo CV interno**: `learning_rate=0,05` (10/15 folds), `max_depth=3` (8/15), `n_estimators=200` (14/15) → preferência clara por regularização. Sinal de que o limite não é capacidade do modelo, mas sim sinal nas features.

### Notebook 06 — 🚨 Calibração isotónica + deteção de viés

#### Calibração

Aplicada Regressão Isotónica sobre `predict_proba` OOF do KNN restrito ao Meditron:

| Métrica | Bruto | Calibrado | Δ |
|---|---:|---:|---:|
| Brier multiclasse | 0,357 | 0,326 | **−8,8%** |
| Log loss | 0,862 | 0,487 | **−43,6%** |

Reliability diagram (one-vs-rest, 8 bins quantílicos) mostra aproximação à diagonal após calibração.

#### Probe de equipamento — **achado principal do projeto**

Treinámos LogReg e RandomForest a prever o **equipamento** a partir das mesmas 78 features de áudio, com `StratifiedGroupKFold(5)`:

| Sonda | Accuracy | **F1-macro** |
|---|---:|---:|
| LogReg | 0,967 | **0,914** ⚠️ |
| RandomForest | 0,959 | 0,881 |
| Baseline trivial (sempre AKG) | 0,759 | — |

**Conclusão:** as 78 features carregam informação **massiva** sobre o equipamento. Como Healthy e URTI **só existem no Meditron**, separar essas classes do COPD é parcialmente equivalente a separar Meditron dos outros equipamentos. A "alta performance" dos modelos no dataset completo é **parcialmente espúria**.

### Notebook 07 — Três caminhos para isolar o sinal real

Comparámos três abordagens para mitigar o viés:

| Modelo | BASELINE (com viés) | A — Meditron-only | C — Residualizado |
|---|---:|---:|---:|
| **KNN** | 0,610 | **0,707** ↑ | 0,461 ↓↓ |
| **LDA** | 0,598 | 0,635 ↑ | 0,523 ↓ |
| **LogReg** | 0,586 | 0,597 ≈ | **0,633** ↑ |
| **XGBoost** | 0,485 | 0,615 ↑↑ | 0,611 ↑↑ |

**Caminho B — Classificação por ciclo respiratório** (4 classes: normal/crackles/wheezes/both): F1m = 0,30–0,35. Tarefa muito mais difícil, exigiria deep learning sobre representação tempo-frequência.

**Insight:** modelos baseados em distância (KNN, LDA) **sofrem** com residualização (perdem estrutura local), mas modelos paramétricos (LogReg, XGBoost) **melhoram**. **LogReg residualizado é o melhor modelo sem viés** (F1m = 0,633), e é também interpretável (coeficientes lineares).

### Notebook 08 — Pipeline final + empacotamento

Validação final do pipeline `residualizar → scaler → LogReg → isotonia` com repeated 10×5-fold:

| Métrica | Valor |
|---|---|
| F1m paciente (não calibrado) | 0,638 ± 0,034 |
| **F1m paciente (calibrado)** | **0,647 ± 0,033** |
| Acc paciente | 0,825 ± 0,017 |
| Log loss bruto | 0,222 ± 0,029 |
| **Log loss calibrado** | **0,113 ± 0,009** (**−49%**) |

Classe a classe (paciente):
- COPD: precision 0,970 / recall 1,000 / F1 0,985 (n=64)
- Healthy: precision 0,630 / recall 0,654 / F1 0,642 (n=26)
- URTI: precision 0,273 / recall 0,214 / F1 0,240 (n=14)

**Modelo final empacotado em `models/lung_assistant.joblib`** com a classe `LungAssistant` (`src/inference.py`).

### Notebook 09 — Holdout virgem + explicabilidade + outliers

**Holdout 80/20 por paciente, estratificado por classe:** 83 pacientes treino, 21 pacientes teste.

| Métrica (paciente, n=21) | Valor |
|---|---|
| Acc | 0,905 |
| F1-macro | 0,778 |
| COPD recall | 1,000 (13/13) |
| Healthy recall | 1,000 (5/5) |
| URTI recall | 0,333 (1/3) |

**F1m do holdout (0,778) ≥ F1m do CV (0,647)** — não há indício de overfitting. URTI continua sendo o calcanhar de Aquiles devido ao n pequeno.

**Explicabilidade** integrada via decomposição linear:
```
contribuição_i,k = coef_k[i] × x_i  (no espaço escalado)
```

Top contribuições para um paciente COPD predito com P=1,000:
- `duration_s` +5,38 (COPD tem áudios mais longos)
- `contrast5_mean` +4,66
- `contrast7_mean` +4,43

**Outlier detector** com threshold empírico p99 = 353,1. Sintéticos gaussianos com std=5σ flagged a 100%; treino flagged a 1,06% (esperado).

### Notebook 10 — CNN sobre mel-spectrogramas

CNN PyTorch (3 conv blocks + GAP + dense, **72.435 parâmetros**) treinada 20 épocas com `class_weight` balanceado e SpecAugment, no mesmo holdout do nb09.

| Modelo | F1m gravação | F1m paciente | Parâmetros | Treino |
|---|---:|---:|---:|---:|
| **LogReg residualizado** | **0,747** | **0,778** | ~240 | <1s |
| CNN | 0,603 | 0,641 | 72.435 | 49s |

**Conclusão:** a CNN com 300× mais parâmetros **não supera** o LogReg. Curva de F1m no teste oscilou entre 0,09 e 0,61 durante o treino (típico de *overfitting* em dataset pequeno). **O teto não está no algoritmo, está nos dados** — 851 gravações, 14 pacientes URTI, viés de equipamento.

### Notebook 11 — Custo clínico em euros (dados reais do SNS Português)

**Fontes de custo (SNS Portugal):**

| Patologia | Volume anual | Custo unitário |
|---|---|---|
| DPOC | +10.000 internamentos | €2.500–3.400 |
| URTI | +800.000 ocorrências | €35–110 |

**Matriz de custos clínicos** (linhas = real, colunas = predito, valores em €):

| | COPD | Healthy | URTI |
|---|---:|---:|---:|
| **COPD** | 0 | **2.950** | 1.475 |
| **Healthy** | 70 | 0 | 70 |
| **URTI** | 70 | 70 | 0 |

O FN de DPOC (€2.950) é **42× mais caro** que um FP (€70) — fundamentado no custo real de um internamento evitável vs uma consulta extra.

**Cost-sensitive decision rule:**
```
ŷ = argmin_j  Σ_i  C_{ij} · P(real=i | x)
```

Resultados (851 pacientes do CV, **custos SNS**):

| Estratégia | Custo total | Custo/paciente |
|---|---:|---:|
| Argmax (otimiza acc.) | €5.120 | €6,02 |
| **Cost-sensitive (otimiza €)** | **€2.450** | **€2,88** |
| | | **−52,1%** |

A regra cost-sensitive **zera FN_DPOC** trocando por FPs em pacientes saudáveis (consultas extra a €70). A matemática fecha porque um FN_DPOC custa €2.950 (internamento evitável) vs €70 de uma consulta extra — **razão 42:1**.

**Projeção financeira anual (50.000 auscultações/ano — adoção parcial):**

| Estratégia | Anual | Economia |
|---|---:|---:|
| Argmax | €300.823 | — |
| **Cost-sensitive** | **€143.948** | **−€156.874/ano** |

**Análise de sensibilidade ao custo de FN_DPOC (decisão re-otimizada, FN_DPOC=0 em todos):**

| Custo FN_DPOC | Economia anual |
|---:|---:|
| €500 | €12.926 |
| €1.000 | €42.303 |
| €1.500 | €71.680 |
| €2.500 | €130.435 |
| €4.000 | €218.566 |

→ Com o custo real do SNS (€2.950), a economia situa-se em **~€150 mil/ano** num cenário de adoção parcial.

### Notebook 12 — Safety Net: minimizar FN clínico

**Definição:** *FN clínico* = doente classificado como Healthy.

**Justificação ética (não financeira):** este é o erro mais grave em diagnóstico médico porque:

1. **Princípio da não-maleficência** (*primum non nocere*) — um sistema clínico não pode dar alta a um doente; isso constitui **dano por omissão**.
2. **Subdiagnóstico já é a doença** — em Portugal, 86% dos doentes com DPOC estão por diagnosticar (ONDR); um assistente que gera FN **perpetua o problema** que se propõe combater.
3. **Tempo é tecido pulmonar** — DPOC é progressiva e irreversível; cada mês sem tratamento implica perda funcional permanente.
4. **Princípio da autonomia do paciente** — o doente tem direito a saber o seu diagnóstico; o FN nega este direito.
5. **Assimetria de remediabilidade** — Falso Positivo (FP) é remediável (consulta extra, médico corrige); FN tem custo cumulativo no organismo do paciente, eventualmente irreversível.

A análise económica em euros (§4 nb11) é uma **consequência mensurável** desta priorização ética — não a sua justificação primária.

**Estado argmax (baseline LogReg residualizado):**
- FN clínico: **18/816 (2,2%)**
- COPD → Healthy: 0 ✓
- URTI → Healthy: 18 ← problema concentrado em URTI

**Estratégia A — Cost-sensitive com penalidade massiva:**

| Penalidade FN | FN clínico | Especificidade |
|---:|---:|---:|
| €1.500 | 4 | 25,7% |
| €5.000 | 4 | 25,7% |
| **€10.000** | **0** | 2,9% |
| €20.000 | 0 | 0% |
| €100.000 | 0 | 0% |

**Estratégia B — Safety Net hierárquica:**

| θ | FN clínico | Especificidade |
|---:|---:|---:|
| 0,50 | 15 | 57,1% |
| 0,30 | 6 | 37,1% |
| 0,10 | 4 | 31,4% |
| 0,05 | 4 | 25,7% |
| 0,02 | 4 | 25,7% |

**Achado científico importante:** existem **4 pacientes URTI cuja P(doente) é < 2%**. Nenhuma estratégia razoável os captura. É um **limite das features**, não da estratégia — provavelmente URTIs leves/precoces sem assinatura acústica distinguível.

**Custo anual com FN_clínico = €10.000 e 50.000 pacientes/ano:**

| Estratégia | FN clínico | Custo anual |
|---|---:|---:|
| Argmax | 18 | €10.692.127 |
| **Cost-sensitive €10k** | **0** | **€144.536** |
| Safety Net θ=0,05 | 4 | €2.519.389 |

A diferença de R$ 10M parece extrema mas reflete o custo brutal de FN clínicos: 18 doentes × €10.000 = €180.000 só pelos FNs, multiplicado pela frequência amostral.

**Trade-off operacional:** Cost-sensitive €10k zera FN, mas especificidade vai a 2,9% (quase todos os saudáveis passam por médico). Safety Net θ=0,05 mantém 25,7% de especificidade ao custo de 4 FN clínicos residuais. Decisão depende da política operacional do hospital.

---

## 5. Sistema de Decision Support Clínico

Para transformar o classificador num **assistente clínico real**, foi adicionado um módulo de orientação ([src/clinical_guide.py](src/clinical_guide.py)) que devolve recomendações estruturadas baseadas em diretrizes:

### 5.1. Para cada diagnóstico

| Campo | Conteúdo |
|---|---|
| `urgencia` | NENHUMA / BAIXA / MODERADA / ALTA / EMERGÊNCIA |
| `encaminhamento` | Especialidade + prazo |
| `exames` | Exames complementares priorizados |
| `tratamento` | Conduta terapêutica inicial com doses |
| `sinais_alarme` | Critérios para escalada |
| `educacao` | Orientações para o paciente |
| `resumo` | Síntese clínica em 1 frase |

### 5.2. Protocolos implementados

**COPD** (baseado em GOLD):
- Encaminhamento: Pneumologia em 2–4 semanas; urgência se exacerbação
- Exames essenciais: espirometria com prova broncodilatadora (FEV1/FVC < 0,70 confirma diagnóstico), Rx tórax, gasometria se dispneia
- Tratamento: SABA, considerar LABA/LAMA + corticoide inalado se eosinofílico
- Educação: cessação tabágica (urgente), vacinação anual, reabilitação pulmonar

**URTI** (geralmente autolimitada):
- Encaminhamento: Clínica Geral
- Sintomáticos: paracetamol, AINEs, soro nasal
- ⚠️ Antibiótico apenas com critério clínico (Centor ≥ 3, sinusite bacteriana documentada)

**Healthy:**
- Alta com orientações
- ⚠️ Manter rastreios; não desvalorizar sintomas referidos

### 5.3. Overrides automáticos (hierarquia)

| Situação | Override |
|---|---|
| `outlier = True` | "Revisão Manual Obrigatória — Amostra Atípica" + sugere considerar diagnósticos fora do treino (Pneumonia, Asma, Bronquiectasias) |
| `P < 0,60` | "Revisão Manual Obrigatória" + "não iniciar tratamento empírico" |
| Modo `safe` + diagnóstico ≠ Healthy | Nota: "modo conservador ativo — confirmar achados antes de tratar" |

### 5.4. Integração no Streamlit

Banner colorido com nível de urgência, seguido de 4 painéis expansíveis: exames | tratamento | sinais de alarme | educação do paciente. Aviso ético reforçado de que as recomendações são gerais.

---

## 6. Sistema implementado

### 6.1. Estrutura do repositório

```
projeto_cdg1.1/
├── RELATORIO.md                   ← este documento
├── README.md                      ← guia executivo
├── assistente.py                  ← CLI executável + API Python
├── app_streamlit.py               ← aplicação web interativa
├── requirements.txt
│
├── data/processed/
│   ├── features.csv               (851 × 84)
│   ├── features_cycles.csv        (6.898 × 83)
│   └── mel_specs.npz              (851 mel-spectrogramas)
│
├── models/
│   ├── lung_assistant.joblib      (v1, 9 KB)
│   └── lung_assistant_v2.joblib   (v2 produção, 58 KB) — com outlier + safe mode
│
├── src/
│   ├── config.py
│   ├── metadata.py                ← parsing ICBHI
│   ├── features.py                ← extração librosa
│   ├── inference.py               ← classe LungAssistant
│   └── clinical_guide.py          ← decision support clínico
│
├── scripts/                       ← 11 geradores programáticos de notebooks
└── notebooks/                     ← 12 notebooks numerados
```

### 6.2. Três interfaces de uso

**Python API:**
```python
from assistente import diagnose

r = diagnose("paciente.wav",
             equipment="WelchAllyn Meditron",
             safe=True,
             disease_threshold=0.10)
# → diagnosis, confidence, probabilities, outlier, explanation
```

**CLI (linha de comando):**
```bash
python assistente.py paciente.wav --equipment "WelchAllyn Meditron" --safe
python assistente.py *.wav -e "AKG C417L Microphone" --json
```

**Aplicação Web (Streamlit):**
```bash
python -m streamlit run app_streamlit.py
# → http://localhost:8501
```

Funcionalidades do app web:
- Upload drag-and-drop de `.wav`
- Player de áudio embutido
- Diagnóstico colorido + métricas de confiança e outlier
- Gráfico comparativo de probabilidades calibradas vs brutas
- Waveform e mel-spectrogram do áudio
- Tabela de explicação por feature com gradiente
- **Painel completo de orientação clínica** (urgência, encaminhamento, exames, tratamento, sinais de alarme, educação)
- Controlos sidebar: equipamento, modo argmax/safe, threshold θ

---

## 7. Discussão

### 7.1. Achados principais

1. **O dataset ICBHI 2017, sem mitigação, leva a modelos com viés grave de equipamento.** Um classificador-sonda prevê o estetoscópio com F1=0,91, e os pacientes Healthy/URTI só foram gravados num equipamento. Esta limitação **raramente é discutida nos papers** que usam este dataset.

2. **A residualização de features sobre equipamento mitiga parcialmente o viés** para modelos paramétricos (LogReg, XGBoost), mas prejudica modelos baseados em distância (KNN, LDA). LogReg residualizado é o melhor compromisso (F1m=0,633 ± 0,030 paciente).

3. **Deep learning não quebrou o teto.** Uma CNN com 72k parâmetros sobre mel-spectrogramas teve desempenho inferior ao LogReg residualizado. **O limite é os dados**, não a capacidade do modelo. Aumentar o dataset (combinar com Coswara, FluSense) ou usar *transfer learning* (YAMNet, PaSST) são vias mais promissoras que treinar redes maiores do zero.

4. **Calibração isotónica é praticamente gratuita** e reduz log loss em 49%, transformando probabilidades em quantidades clinicamente acionáveis.

5. **O recall de doença em modo *safe* atinge 99,5%**, com 4 FN clínicos residuais (todos URTI). Estes 4 pacientes têm `P(doente) < 2%` — limite intrínseco das features, não mitigável sem perder discriminação.

6. **O impacto económico é robusto** com custos reais do SNS: economia de **~€157 mil/ano** (cenário de adoção parcial); até **>€10 milhões/ano** se o custo de FN de DPOC for valorizado a €10k. FN_DPOC zerado em todos os cenários de sensibilidade.

### 7.2. Limitações conhecidas

| Limitação | Severidade | Mitigação |
|---|:-:|---|
| Cobertura limitada a 3 classes (sem Pneumonia, Asma, etc.) | 🟠 | Outlier detector sinaliza casos atípicos; protocolo de revisão clínica |
| **Performance crítica em crianças <6 anos (acc 0,45) — nb15** | 🔴 | **Documentar como contraindicação; modelo NÃO deve ser usado em pediatria sem revalidação** |
| Performance fraca em URTI (recall 0,357) | 🟠 | Safety net + revisão clínica obrigatória se sintomas referidos |
| Equipamentos suportados restritos a 4 | 🟡 | Validar antes de aceitar gravação de outro dispositivo |
| **Holdout pequeno (n=21) — não é estimador pontual** | 🟠 | **Métrica principal é o CV repetido 10×5-fold (F1m = 0,647 ± 0,033)** |
| Calibração ajustada nos OOFs depois reportados | 🟡 | Pequeno *data leakage* possível, documentado no FAQ Q6 |
| **Avaliação retrospetiva sem validação externa** | 🔴 | **Validação externa em dataset independente é obrigatória antes de uso clínico** |
| Sem validação por pneumologista | 🔴 | Trabalho futuro (Fase 1 do roadmap) |
| 9 pacientes URTI sempre classificados Healthy (nb15) | 🔴 | Todos pediátricos com Meditron — ponto cego documentado |

**Legenda:** 🔴 grave / 🟠 média / 🟡 menor

### 7.2.1. Comparação com a literatura ICBHI

| Estudo | F1-macro | Trata viés equip.? | Split paciente? | Valid. externa? |
|---|---:|:-:|:-:|:-:|
| Aykanat et al. (2017) | 0,86 | ❌ | ❌ (split gravação) | ❌ |
| Bardou et al. (2018) | 0,87 | ❌ | ❌ | ❌ |
| Pham et al. (2020) | 0,84 | ❌ | parcial | ❌ |
| Este projeto, *baseline com viés* | ~0,75 | ❌ | ✅ | ❌ |
| **Este projeto, residualizado (modelo final)** | **0,647 ± 0,033** | ✅ | ✅ | ❌ |

**Leitura honesta:** o nosso F1-macro é mais baixo que os papers de referência. Não porque o modelo seja pior — **porque os papers raramente tratam viés de equipamento**. No nb06 demonstrámos que um probe-LogReg prediz o equipamento das features com F1=0,914; no nb15 demonstrámos que **após residualização linear o probe cai para F1=0,089** (redução de 89,7%). Os 0,86 dos papers incluem provavelmente *shortcut learning*; os nossos 0,65 são honestos sem ele.

### 7.2.2. Achados de fairness e análise de erro (nb15)

Análise por subgrupo demográfico revelou viés grave por idade:

| Subgrupo | Accuracy | n |
|---|---:|---:|
| Adultos (18–80) | 0,994 | 703 |
| Idosos (>80) | 0,976 | 84 |
| Adolescentes (6–18) | 0,556 | 18 |
| **Crianças (<6 anos)** | **0,450** ⚠️ | 40 |

→ O modelo **não está calibrado para pediatria**. Os 9 pacientes URTI que são predominantemente classificados como Healthy são **todos crianças (idades 0,67–16 anos), todos com Meditron**. Padrão claro: **URTI pediátrica é o ponto cego do modelo**.

**Em uso pediátrico real, este sistema produziria taxas inaceitáveis de FN clínico.** Documentado como contraindicação explícita em todos os outputs.

### 7.3. Uso responsável

O sistema é **apoio à decisão**, não substituto. Recomendações para deployment:

1. **Modo padrão clínico:** safe com θ=0,10 (recall 99,5%, especificidade 31%)
2. **Triagem em massa:** cost-sensitive €10k (recall 100%, todos vão para médico)
3. **Monitorização contínua:** medir FN observados em produção; rebaixar θ se subir
4. **Validação dos 4 FN residuais documentada** como limitação conhecida — protocolo de revisão obrigatória para pacientes com sintomas referidos
5. **Não substituir auscultação manual** em casos com confiança baixa ou outlier

---

## 8. Conclusão

Foi entregue um sistema completo de apoio ao diagnóstico de doenças respiratórias, partindo do processamento de sons brutos até uma aplicação web utilizável por clínicos. As **três narrativas** do projeto sustentam-se mutuamente:

- **Ciência de Dados:** pipeline metodologicamente rigoroso, com identificação e mitigação de viés, validação repetida e validação independente em holdout.
- **Clínica:** safety net com recall de doença de 99,5%, decision support com orientações estruturadas (urgência, encaminhamento, exames, tratamento, sinais de alarme).
- **Gestão:** impacto financeiro projetado de ~€157 mil/ano (adoção parcial) a >€10 milhões/ano, com custos fundamentados em dados reais do SNS português (DPOC €2.950/internamento, URTI €70/ocorrência) e análise de sensibilidade.

**O sistema está pronto para piloto clínico**, condicionado à:
1. Auditoria local dos custos clínicos para calibrar a matriz
2. Validação prospetiva com pelo menos 50 pacientes adicionais
3. Implementação de monitorização contínua de FN em produção
4. Aprovação ética e regulamentar conforme jurisdição (CE Marking se UE)

### 8.1. Trabalho futuro priorizado

| Direção | Justificação |
|---|---|
| Transfer learning (YAMNet, PaSST) | Pode quebrar o teto que CNN treinada do zero não quebrou |
| Combinar com Coswara/FluSense | Aumentar amostra de URTI e Healthy (atenuar o "n pequeno") |
| Classificação por ciclo respiratório com CNN | Tarefa objetiva (crackles/wheezes), 6,9k amostras |
| API REST (FastAPI) + Docker | Deploy production-ready |
| Auditoria contínua em produção | Detetar drift de distribuição; recalibrar |
| Estudo prospetivo com painel de pneumologistas | Validação clínica formal |

---

## Apêndice A — Especificação técnica do modelo final

| Componente | Implementação |
|---|---|
| Extração de features | `librosa` 0.10+ |
| Sample rate | 22 050 Hz |
| n_mfcc | 13 |
| n_fft / hop_length | 2048 / 512 |
| Residualização | `sklearn.linear_model.LinearRegression` (multi-output) |
| Scaler | `sklearn.preprocessing.StandardScaler` |
| Classificador | `sklearn.linear_model.LogisticRegression` (multinomial, `class_weight='balanced'`, max_iter=3000) |
| Calibração | `sklearn.isotonic.IsotonicRegression` (one-vs-rest, OOF) |
| Outlier detector | Mahalanobis (`np.linalg.pinv(cov + 1e-4·I)`), threshold = quantil p99 empírico |
| Safety net | regra hierárquica: bloqueia Healthy se `1 − P(Healthy) ≥ θ` |
| Serialização | `joblib` |

## Apêndice B — Reprodutibilidade

```bash
# Setup
pip install -r requirements.txt

# Pipeline completo
jupyter lab notebooks/02_extracao_completa.ipynb     # extrai features.csv (~5 min)
jupyter lab notebooks/08_pipeline_final.ipynb        # treina e empacota
jupyter lab notebooks/09_holdout_explain_outliers.ipynb  # valida + outlier

# Análise clínica/financeira
jupyter lab notebooks/11_custo_clinico_threshold.ipynb
jupyter lab notebooks/12_safety_net_fn_clinico.ipynb

# Regenerar qualquer notebook
python scripts/build_notebook_NN.py
```

## Apêndice C — Notebooks (mapa de leitura)

| Nº | Notebook | Tema | Output |
|---:|---|---|---|
| 01 | `01_eda_e_extracao.ipynb` | EDA + librosa demo | Visualizações |
| 02 | `02_extracao_completa.ipynb` | Extração 78 features | `features.csv` |
| 03 | `03_comparacao_modelos.ipynb` | Baseline 4 modelos | F1m por modelo |
| 04 | `04_repeated_cv.ipynb` | Repeated CV 10×5 | Wilcoxon + IC95 |
| 05 | `05_nested_cv_xgb_tuning.ipynb` | Nested CV + tuning XGB | Hiperparâmetros estáveis |
| 06 | `06_calibracao_e_vies.ipynb` | 🚨 Probe equipamento + calibração | F1m probe=0,914 + Brier −9% |
| 07 | `07_caminhos_ABC.ipynb` | 3 caminhos de mitigação | LogReg residualizado vence |
| 08 | `08_pipeline_final.ipynb` | Pipeline final empacotado | `lung_assistant.joblib` |
| 09 | `09_holdout_explain_outliers.ipynb` | Holdout (n=21) + explain + outlier | Sanity check; CV é métrica real |
| 10 | `10_cnn_mel_spectrogram.ipynb` | CNN PyTorch (CV 3×3) | CNN não bate LogReg |
| 11 | `11_custo_clinico_threshold.ipynb` | Custo clínico € (SNS real) | ~€157 k/ano economia |
| 12 | `12_safety_net_fn_clinico.ipynb` | Safety net FN clínico | 4 FN residuais documentados |
| 13 | `13_hierarquico_smote_recall.ipynb` | ⚠️ Cascata + SMOTE (invalidado pelo 14) | Contra-exemplo metodológico |
| 14 | `14_undersampling_clinico.ipynb` | Undersampling + reenquadramento ético | Substitui parcialmente nb13 |
| **15** | `15_validacao_critica.ipynb` | **Probe pós-residualização + fairness + pacientes errados + ensemble** | **Achados críticos: residualização −89.7%, pediatria acc 0.45** |
| **16** | `16_dashboard_final.ipynb` | **Dashboard único — todos modelos × métricas** | **Figura principal para defesa** |

## Apêndice D — Documentação complementar

| Ficheiro | Propósito |
|---|---|
| `README.md` | Guia executivo rápido |
| `MANUAL_CODIGO.md` | Walkthrough técnico de cada ficheiro `.py` |
| `INTERPRETACAO_OUTPUTS.md` | Como ler cada output do modelo |
| `FAQ_DEFESA.md` | **20 perguntas/respostas antecipadas para o júri** |
| `HISTORICO_DECISOES.md` | Bug fixes + mudanças de direção + lições aprendidas |
| `MELHORIAS_APLICADAS.md` | **Todas as 24 melhorias aplicadas após análise crítica** |

---

*Documento gerado a partir de pipeline reproduzível. Para detalhes técnicos, consultar os notebooks numerados em `notebooks/` e os módulos em `src/`.*
