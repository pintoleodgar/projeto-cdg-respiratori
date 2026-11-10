# Assistente de Diagnóstico Respiratório — ICBHI 2017

Pipeline de Ciência de Dados para apoio ao diagnóstico de doenças pulmonares
a partir de sons de auscultação respiratória.

**Status:** modelo v3 — métricas revistas em novembro de 2026 (ver `reports/AVALIACAO_V3.md`).
**Dataset:** ICBHI 2017 Respiratory Sound Database (920 gravações, 126 pacientes).
**Modelo final:** Regressão Logística sobre features residualizadas + calibração sigmoide + safety net clínica.
**Performance (por paciente, CV 5×5, IC95% bootstrap de pacientes):** F1-macro 0,59 [0,51–0,66]; recall COPD 1,00; **recall URTI 0,24 [0,09–0,41]**; especificidade 0,52 [0,38–0,68]. Com safety net θ=0,10: recall de doença 0,94, recall URTI 0,46, especificidade 0,28.
**Impacto financeiro:** as poupanças anteriormente anunciadas (€157 mil a >€10 M/ano) **não se confirmam** ao nível do paciente — ver secção 6.

---

## 1. Visão geral em uma frase

> Dadas 77 features de áudio extraídas com `librosa` (gravação cortada a 20 s), **classifica uma gravação como Healthy, COPD ou URTI**, com probabilidades calibradas, detecção de outlier e modo "safety net" que **prioriza não perder doentes** (princípio bioético da **não-maleficência** — *primum non nocere*). Deteta COPD de forma fiável; **distingue mal URTI de Healthy**, e essa é a principal limitação.

## 1.1. Princípio orientador

**Recall de doença é a métrica governante** de todas as decisões de modelagem. Falso Negativo clínico (doente classificado como saudável) é considerado **eticamente inaceitável** porque:

- Constitui **dano por omissão** — sistema clínico não pode dar alta a um doente.
- **Subdiagnóstico já é a doença**: 86% dos doentes com DPOC em Portugal estão por diagnosticar (ONDR).
- DPOC é **progressiva e irreversível** — atraso diagnóstico = perda funcional permanente.
- Falso Positivo (FP) é **remediável** (médico corrige na consulta seguinte); FN tem custo cumulativo no paciente.

O custo financeiro é **consequência** do dano clínico, não a justificação primária do sistema.

---

## 2. As três narrativas que se sustentam mutuamente

| Eixo | Pergunta que responde | Componente principal |
|---|---|---|
| **Ciência de Dados** | Qual modelo é o melhor e como sabemos? | nbs 03–07: comparação, repeated CV, nested CV, detecção e mitigação de viés |
| **Clínica** | Posso confiar que um doente não vai sair como saudável? | train.py, nbs 09, 12: explicabilidade, safety net (COPD: sim; URTI: só em parte — ver secção 6) |
| **Gestão** | Quanto custa cada ponto de operação? | train.py, nbs 11, 12: matriz de custos €, cenários de prevalência |

---

## 3. Lógica decisória do pipeline (passo a passo)

```
              ┌─────────────────────┐
              │ áudio .wav + equip. │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  librosa: extrair   │
              │ corte 20 s, 77 feat.│   (MFCC + Δ + Δ², chroma,
              └──────────┬──────────┘    contrast, espectrais, RMS, ZCR)
                         ▼
              ┌─────────────────────┐
              │  residualizar p/    │   (remove componente linear explicada
              │     equipamento     │    pelo dispositivo de captura)
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │   StandardScaler    │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │   Mahalanobis →     │── outlier? → confiança = "baixa"
              │  detector outlier   │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ Regressão Logística │
              │   multinomial       │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ Calibração          │   (log loss 0,228 → 0,150,
              │ sigmoide (OOF)      │    medido de forma aninhada)
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  Decisão final      │
              │ mode='argmax' OU    │
              │ mode='safe' (θ)     │   ← bloqueia Healthy se P(doente) ≥ θ
              └──────────┬──────────┘
                         ▼
        ┌──────────────────────────────────┐
        │ diagnosis + probabilidades       │
        │ + confiança + outlier flag       │
        │ + explicação por feature         │
        └──────────────────────────────────┘
```

---

## 4. Estrutura do repositório

```
projeto_cdg1.1/
├── README.md                              ← este arquivo (executivo)
├── RELATORIO.md                           Relatório técnico-científico
├── MANUAL_CODIGO.md                       Walkthrough do código
├── INTERPRETACAO_OUTPUTS.md               Como ler outputs do modelo
├── FAQ_DEFESA.md                          20 perguntas/respostas para a defesa
├── CHANGELOG.md                           Decisões, correções e melhorias
├── requirements.txt                       versões fixas
├── pytest.ini
├── .gitignore
│
├── data/
│   ├── raw/ → dataset ICBHI, fora do repositório (variável de ambiente ICBHI_ROOT)
│   └── processed/
│       ├── features.csv                   (851×84, áudio inteiro — usado pelos nbs 02–16)
│       ├── features_v3.csv                (851×84, áudio cortado a 20 s — usado pelo modelo v3)
│       ├── features_cycles.csv            (6898×83, features por ciclo)
│       └── mel_specs.npz                  (851 mel-spectrogramas)
│
├── models/
│   ├── lung_assistant.joblib              (v1: pipeline básico)
│   ├── lung_assistant_v2.joblib           (v2: obsoleto — usa duration_s, calibração não aninhada)
│   └── lung_assistant_v3.npz           (v3 PRODUÇÃO: gerado por scripts/train.py)
│
├── reports/
│   ├── AVALIACAO_V3.md                    métricas do modelo v3 (gerado)
│   ├── metrics_v3.json                    idem, para máquina
│   └── figures/                           figuras exportadas dos notebooks
│
├── src/
│   ├── config.py                          caminhos e parâmetros
│   ├── metadata.py                        parsing ICBHI
│   ├── features.py                        extração librosa
│   ├── inference.py                       classe LungAssistant
│   ├── clinical_guide.py                  decision support clínico
│   ├── plotting.py                        paleta consistente + savefig
│   └── bootstrap.py                       IC95% bootstrap utilities
│
├── tests/
│   └── test_inference.py                  38 testes (pytest); a lógica de decisão testa-se sem o dataset
│
├── scripts/
│   ├── train.py                           ★ treina e avalia o modelo de produção (v3)
│   ├── build_notebook_NN.py               geradores programáticos dos notebooks
│   └── _arquivado/                        geradores dos nbs 03, 04, 05, 07, 13
│
└── notebooks/
    ├── 01_eda_e_extracao.ipynb            EDA + librosa demo
    ├── 02_extracao_completa.ipynb         Extração das features × 851 gravações
    ├── 06_calibracao_e_vies.ipynb         🚨 Descoberta do viés de equipamento
    ├── 08_pipeline_final.ipynb            Pipeline v1 (histórico)
    ├── 09_holdout_explain_outliers.ipynb  Holdout 80/20 + outlier + explain (v2, histórico)
    ├── 10_cnn_mel_spectrogram.ipynb       CNN com CV repetido — não bate LogReg
    ├── 11_custo_clinico_threshold.ipynb   Análise económica por gravação (histórico)
    ├── 12_safety_net_fn_clinico.ipynb     Safety net por gravação (histórico)
    ├── 14_undersampling_clinico.ipynb     Undersampling
    ├── 15_validacao_critica.ipynb         🎯 Sonda linear e não-linear + fairness + erros
    ├── 16_dashboard_final.ipynb           📊 Figura mestre
    └── _arquivado/                        nbs 03, 04, 05, 07, 13 (comparação de modelos, SMOTE)
```

> Os notebooks 08–16 usam `features.csv` (áudio inteiro, com `duration_s`) e o modelo v2.
> Os números de referência do projeto são os de `reports/AVALIACAO_V3.md`.

---

## 5. Decisões metodológicas críticas (com justificativa)

| Decisão | Por quê | Onde |
|---|---|---|
| Manter só 3 classes (Healthy/COPD/URTI) | Pneumonia tinha 6 pacientes — inviabiliza k-fold; Asthma só 1 amostra | nb01 |
| Split **paciente-estratificado** (`StratifiedGroupKFold`) | Pacientes COPD têm muitas gravações cada → split por gravação vaza identidade | nb03+ |
| **Repeated CV** (10×5-fold) | Com 14 pacientes URTI, um único seed dá ±5pp; reportar média±desvio | nb04 |
| Calibração **sigmoide**, validada de forma **aninhada** | Com 35 gravações Healthy e 23 URTI a isotónica sobreajusta; medida fora do treino, a sigmoide dá log loss 0,150 vs 0,196 (isotónica) vs 0,228 (sem calibração) | train.py |
| Áudio **cortado a 20 s**, sem `duration_s` | No Meditron, todas as gravações COPD têm >32 s e todas as Healthy/URTI ~20 s: a duração era um atalho (e a feature de maior peso para COPD no v2) | train.py |
| **Residualização** de features sobre equipamento | Sonda prediz equipamento com F1m≈0,9 a partir das features → shortcut learning. A residualização remove a parte **linear** desse viés (ver secção 6) | nb06, 07, 15 |
| LogReg como modelo final (não XGB nem CNN) | LogReg residualizado venceu no nested CV (nb05) e ficou igual à CNN com 300× menos parâmetros (nb10). LogReg também é interpretável | nb05, 07, 10 |
| Holdout 80/20 por paciente — **apenas ilustrativo** | Foi separado depois de a escolha do modelo já ter usado todos os dados e tem ~3 pacientes URTI; F1m 0,78 não é estimativa fiável nem prova de ausência de overfitting. A métrica de referência é o CV repetido | nb09 |
| Outlier detector com **quantil empírico p99**, não χ²(0.99) | Features não-gaussianas → χ² dá ~19% de falsos alarmes. Com p99 empírico, o falso alarme medido fora do treino é 5,5% (não 1%) | train.py |
| Modo `safe` com `disease_threshold` ajustável | FN clínico (doente → Healthy) é o erro mais grave; gestão clínica define θ | nb12 |
| Custos em **euros** (SNS Portugal), reponderados por prevalência assumida | Internamento DPOC ≈ €2.950; consulta extra ≈ €70. A prevalência do dataset (62% COPD) não representa nenhuma população real | train.py |
| IC95% por **bootstrap de pacientes** | Reamostrar valores por fold (não independentes) dava intervalos demasiado estreitos | train.py |

---

## 6. Resultados-chave

Todos os números desta secção vêm de `python scripts/train.py` (`reports/AVALIACAO_V3.md`):
CV 5×5-fold estratificado por paciente, calibração aninhada, decisão por paciente
(média das probabilidades das suas gravações), IC95% por bootstrap de pacientes.

### Desempenho por paciente (64 COPD, 26 Healthy, 14 URTI)

| Estratégia | Recall doente | Recall COPD | Recall URTI | Especificidade | F1-macro | FN clínicos |
|---|---|---|---|---|---|---|
| argmax | 0,887 [0,854–0,921] | 1,000 | 0,243 [0,086–0,414] | 0,523 [0,377–0,677] | 0,591 [0,511–0,664] | 8,8 |
| safe θ=0,30 | 0,910 [0,872–0,949] | 1,000 | 0,329 [0,143–0,529] | 0,431 [0,284–0,585] | 0,589 [0,502–0,670] | 7,0 |
| **safe θ=0,10** | 0,938 [0,903–0,969] | 1,000 | 0,457 [0,257–0,657] | 0,277 [0,138–0,423] | 0,563 [0,478–0,641] | 4,8 |
| safe θ=0,05 | 0,962 [0,928–0,990] | 1,000 | 0,586 [0,386–0,772] | 0,231 [0,100–0,369] | 0,572 [0,487–0,650] | 3,0 |

**Como ler:** o modelo nunca falha um paciente COPD, mas isso domina o "recall de doença".
Todos os falsos negativos clínicos são **URTI dados como Healthy**: em argmax perde-se cerca
de 3 em cada 4 pacientes URTI. A safety net recupera parte deles à custa de mandar a maioria
dos saudáveis para revisão.

> O valor "recall de doença 99,5%" de versões anteriores era calculado **por gravação**
> (816 gravações de doentes, das quais 793 COPD), não por paciente.

### Só gravações Meditron (8 COPD, 26 Healthy, 14 URTI)

Healthy e URTI **só existem em gravações Meditron**; nos outros três equipamentos há apenas
COPD. O Meditron é por isso o único cenário em que as três classes são comparáveis:

| Estratégia | Recall doente | Recall URTI | Especificidade | F1-macro |
|---|---|---|---|---|
| argmax | 0,600 [0,482–0,718] | 0,243 | 0,523 | 0,538 [0,455–0,632] |
| safe θ=0,10 | 0,782 [0,664–0,891] | 0,457 | 0,277 | 0,489 [0,393–0,588] |

Fora do Meditron o modelo nunca viu um saudável nem uma URTI — o desempenho nessas
condições é **desconhecido**.

### Viés de equipamento — mitigado, não eliminado

Uma sonda tenta adivinhar o equipamento a partir das features (F1-macro, CV por paciente):

| Sonda | Features cruas | Features residualizadas |
|---|---:|---:|
| Linear (LogReg) | 0,895 | 0,216 |
| Não-linear (RandomForest) | 0,897 | **0,869** |

A residualização linear zera a média de cada feature por equipamento, por isso uma sonda
linear falha quase por construção. Uma sonda não-linear continua a identificar o equipamento:
a assinatura do dispositivo permanece nas features. O LogReg final, sendo linear, não a
explora diretamente; modelos não-lineares (XGBoost, RF, CNN) podem explorá-la. (O nb15
mostra o mesmo nas features originais: 0,089 linear vs 0,854 não-linear.)

### ⚠️ Viés por subgrupo (nb15, modelo v2, por gravação)

| Subgrupo | Accuracy | n |
|---|---:|---:|
| Adultos (18–80) | 0.994 | 703 |
| Idosos (>80) | 0.976 | 84 |
| Adolescentes (6–18) | 0.556 | 18 |
| **Crianças (<6 anos)** | **0.450** | 40 |

Esta tabela mistura idade com classe: os adultos são quase todos COPD (a classe fácil) e as
crianças são quase todas Healthy/URTI (a distinção difícil). O que ela mostra com segurança é
que **o sistema não deve ser usado em crianças** — a app avisa-o no rodapé.

### Calibração e outliers

- Log loss por gravação, medido fora do treino: sem calibração 0,228; isotónica 0,196; **sigmoide 0,150**.
- Limiar de outlier (p99 empírico): 346,0; falso alarme fora do treino: 5,5%.

### Impacto financeiro — as poupanças anunciadas não se confirmam

Custos do nb11 (FN de COPD €2.950; restantes erros €70), 50.000 pacientes/ano, safe θ=0,10
comparado com argmax. As prevalências são **hipóteses**, não medições:

| Cenário (COPD/Healthy/URTI) | €/paciente argmax | €/paciente safe | Diferença anual [IC95%] |
|---|---:|---:|---|
| Rastreio população geral (5/85/10) | 33,68 | 46,82 | −€657 mil [−€942 mil, −€391 mil] |
| Cuidados primários (10/70/20) | 33,97 | 43,03 | −€453 mil [−€696 mil, −€218 mil] |
| Consulta de pneumologia (40/40/20) | 23,95 | 27,85 | −€195 mil [−€344 mil, −€38 mil] |

Ao nível do paciente o modelo já não falha COPD em argmax, por isso a safety net não evita
nenhum internamento: só troca URTI não detetadas (€70) por mais saudáveis enviados para
consulta (€70 cada, e são muitos mais). **Com esta matriz de custos o modo safe sai mais
caro.** As poupanças de €157 mil a >€10 M/ano de versões anteriores resultavam de contar
erros por gravação e de usar a prevalência do dataset. O argumento para a safety net é
clínico (não perder URTI), não financeiro — e esta análise não compara o sistema com a
prática atual sem modelo.

---

## 7. Como usar o modelo em produção

```python
from src.inference import LungAssistant

# Carregar modelo
asst = LungAssistant.load("models/lung_assistant_v3.npz")

# Modo padrão (classe mais provável)
r = asst.predict("paciente.wav", equipment="WelchAllyn Meditron")

# Modo conservador (prioriza não perder doentes)
r = asst.predict("paciente.wav",
                 equipment="WelchAllyn Meditron",
                 mode="safe",
                 disease_threshold=0.10)

# r retorna:
# {
#   'diagnosis': 'COPD',
#   'confidence': 'alta',
#   'top_probability': 0.994,
#   'probabilities': {'COPD': 0.994, 'Healthy': 0.000, 'URTI': 0.006},
#   'outlier': {'is_outlier': False, 'mahalanobis_sq': 96.1, 'threshold_99': 346.0},
#   'mode': 'safe'
# }

# Explicar a decisão
exp = asst.explain("paciente.wav", equipment="WelchAllyn Meditron", top_k=5)
# exp['COPD'] = [{'feature': ..., 'coef': ..., 'value_scaled': ..., 'contribution': ...}, ...]

# Diagnóstico por PACIENTE (várias gravações) — é a regra a que as métricas se referem
rp = asst.predict_patient(["g1.wav", "g2.wav", "g3.wav"],
                          equipment="WelchAllyn Meditron", mode="safe", disease_threshold=0.10)

# Extrair uma vez e reutilizar (a extração é o passo lento)
feats = asst.extract("paciente.wav")          # InvalidAudioError se curto/silencioso/ilegível
r = asst.predict_features(feats, equipment="WelchAllyn Meditron")
exp = asst.explain_features(feats, equipment="WelchAllyn Meditron")

# r inclui também 'safe_blocked_healthy' (a safety net trocou Healthy por doença?)
# e 'contraindications' (avisos que acompanham todas as predições).
# Um `mode` desconhecido levanta ValueError — não cai em argmax.

# Só checar outlier
out = asst.is_outlier("paciente.wav", equipment="WelchAllyn Meditron")
```

**Equipamentos suportados** (devem ser passados exatamente):
- `"AKG C417L Microphone"`
- `"WelchAllyn Meditron"`
- `"Littmann Classic II SE"`
- `"Littmann 3200 Electronic"`

---

## 8. Limitações conhecidas e uso responsável

**O que o modelo NÃO faz:**
- Não detecta Pneumonia, Asthma, Bronchiolitis, Bronchiectasis, LRTI (excluídas por falta de dados).
- Não funciona com outros estetoscópios fora dos 4 listados (residualização falha).
- **Não deve ser usado em crianças.**
- Não substitui anamnese clínica e exame físico.

**Limitações estruturais dos dados:**
- Healthy e URTI só existem em gravações Meditron; nos outros equipamentos o modelo só conhece COPD.
- 14 pacientes URTI e 26 Healthy: todos os intervalos de confiança são largos.
- Não há conjunto de teste independente — todos os dados participaram na escolha do modelo.
- O viés de equipamento está mitigado apenas na componente linear.

**Onde erra mais:**
- URTI: recall por paciente de 0,24 (argmax) a 0,59 (safe θ=0,05). **Um resultado "Healthy" não exclui URTI.**
- Casos com `outlier=True`: confiança rebaixada automaticamente; revisar manualmente.

**Recomendações operacionais:**
1. Usar como apoio à deteção de COPD; não usar um resultado "Healthy" para dar alta.
2. Escolher θ pelo compromisso recall URTI × especificidade da tabela da secção 6.
3. A app e a CLI classificam **uma gravação**; as métricas referem-se à média das gravações do paciente (`predict_patient` na API).
4. Monitorizar FN observados em produção.

---

## 9. Reproduzir resultados

```bash
# 1. Setup (Python 3.11)
pip install -r requirements.txt
#    Se o dataset não estiver em C:\Datasets\Respiratory_Sound_Database:
#    PowerShell:  $env:ICBHI_ROOT = "D:\caminho\Respiratory_Sound_Database"
#    bash:        export ICBHI_ROOT=/caminho/Respiratory_Sound_Database

# 2. Modelo de produção + todas as métricas da secção 6 (~2 min; --extract reextrai os .wav)
python scripts/train.py
#    → models/lung_assistant_v3.npz, reports/AVALIACAO_V3.md, reports/metrics_v3.json

# 3. Testes
pytest tests/ -q

# 4. Notebooks de análise (históricos: usam features.csv e o modelo v2)
jupyter lab notebooks/

# Regenerar qualquer notebook
python scripts/build_notebook_NN.py
```

---

## 10. Trabalho futuro

| Direção | Por quê |
|---|---|
| Transfer learning (YAMNet/PaSST) | Pode quebrar o teto que CNN treinada do zero não quebrou |
| Combinar com Coswara/FluSense | Aumentar dataset de URTI e Healthy não-pediátricos |
| Classificação por ciclo (crackles/wheezes) com CNN | Tarefa objetiva, ~6.9k amostras, alvo clínico direto |
| API REST (FastAPI) + Streamlit dashboard | Deploy real |
| Auditoria contínua de FN em produção | Detectar drift de distribuição |

---

## Autor

Projeto final de Ciência de Dados — com viés de equipamento mitigado e limitações documentadas.
