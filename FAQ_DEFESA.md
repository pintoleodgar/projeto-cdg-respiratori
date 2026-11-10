# FAQ — Antecipação de perguntas do júri

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

> Respostas preparadas para perguntas previsíveis na defesa do TFC.
> Cada resposta cita evidência específica (notebook, métrica, secção do relatório).

---

## 🔬 Perguntas técnicas / metodológicas

### Q1. Porque escolheu Regressão Logística e não XGBoost ou CNN?

**R:** Três argumentos convergentes (nb05, nb07, nb10):

1. **Performance:** após tuning com Nested CV (nb05), XGBoost não superou LogReg (F1m paciente: XGB tuned 0,556 vs LogReg residualizado 0,633). CNN com 72 435 parâmetros sobre mel-spectrogramas teve F1m=0,641 — abaixo do LogReg com 240 parâmetros (nb10).

2. **Interpretabilidade clínica:** LogReg decompõe-se exatamente em `contribuição_feature = coef × valor_escalado`. Permite mostrar ao médico as features que justificaram a decisão (visível na aba "Explicação" do Streamlit). XGBoost e CNN exigem SHAP/LIME — aproximações.

3. **Operacional:** LogReg serializa em 58 KB. CNN serializa em ~300 KB. Em sistema clínico, modelos pequenos são auditáveis linha a linha.

**Conclusão:** o teto não é o algoritmo, são os dados. Aumentar capacidade não traz ganho neste tamanho de dataset (851 gravações, 14 pacientes URTI).

---

### Q2. Porque 78 features específicamente?

**R:** São features padrão da literatura de análise de sons respiratórios:

| Família | Features | Razão |
|---|---:|---|
| MFCC (13) + Δ + Δ² + mean/std | 52 | Captura envelope espectral em escala perceptual (Mel) — padrão de facto |
| Chroma STFT | 12 | Distribuição de energia harmónica — wheezes têm padrões tonais |
| Spectral contrast (7 bandas) | 7 | Distingue tons (wheezes) de ruído (crackles) |
| Centroide, rolloff, bandwidth, flatness, RMS, ZCR | 6 | Caracterização espectral global |
| Duração | 1 | Indicador da gravação |

Não foram escolhidas arbitrariamente — replicam o conjunto recomendado em Bardou et al. (2018) e Aykanat et al. (2017). Adicionar mais (ex: tonnetz, spectral_centroid_std) testámos no nb02 e não melhorou.

---

### Q3. Como justifica F1-macro de 0,65 quando papers reportam 0,80+?

**R:** Diferença metodológica explícita — não defeito do modelo. **Os papers raramente tratam viés de equipamento.**

| Estudo | F1-macro | Trata viés equip.? | Split paciente? |
|---|---:|:-:|:-:|
| Aykanat et al. (2017) | 0,86 | ❌ | ❌ (split gravação) |
| Bardou et al. (2018) | 0,87 | ❌ | ❌ |
| Pham et al. (2020) | 0,84 | ❌ | parcial |
| **Este projeto, baseline com viés** | **0,75** (CV) | ❌ | ✅ |
| **Este projeto, residualizado** | **0,65** (CV) | ✅ | ✅ |

No nb06 demonstro que um classificador-sonda prediz o equipamento a partir das features com F1=0,91. Como Healthy/URTI estão **100% no Meditron** e COPD principalmente noutros, **classificar diagnóstico é parcialmente equivalente a classificar equipamento**. Os 0,86 dos papers incluem este *shortcut learning* — o nosso 0,65 é honesto sem ele.

**Tradução:** os papers publicaram um modelo que "aprende equipamento". O nosso aprende sinal real (no que é possível com 851 gravações).

---

### Q4. O holdout virgem só tem 21 pacientes. F1m=0,778 é confiável?

**R:** **Não como estimativa pontual.** É *sanity check* qualitativo: confirma que o modelo não overfitou (F1m holdout ≥ F1m CV), mas o IC95% do F1m do holdout é amplo (~\[0,60, 0,90\]).

**A métrica que governa o relatório é o CV repetido 10×5-fold (nb04, nb08):**

| Métrica | CV (n=10 × 5 folds) | Holdout (n=21 pac.) |
|---|---|---|
| F1m paciente | **0,647 ± 0,033** ← principal | 0,778 (variável) |
| Recall doente | **0,995** (modo safe) | 0,950 (variável) |

Reconheço como limitação no `RELATORIO.md` § 7.2 e proponho **Leave-One-Patient-Out CV** como melhoria futura (104 folds, intervalo de confiança válido).

---

### Q5. A residualização sobre equipamento realmente funciona?

**R:** **Parcialmente.** Análise em três passos (nb06, nb07, nb15):

1. **Probe pré-residualização** (nb06): LogReg prediz equipamento com **F1=0,914** — viés massivo confirmado.
2. **Probe pós-residualização** (nb15): F1 cai significativamente para ~0,60–0,70 — viés *reduzido*, mas **não eliminado**.
3. **Evidência indireta de leakage residual** (nb13): no Estágio 2 (COPD vs URTI dentro dos doentes), modelos atingiram 100% perfeito — sintoma de viés não-linear que residualização linear não captura.

**Caminho de melhoria:** Domain-Adversarial Neural Network (DANN) ou *adversarial debiasing* — remove componentes não-lineares correlatados com equipamento. Documentado como trabalho futuro.

---

### Q6. Porque safety net e não cost-sensitive?

**R:** Ambos resolvem o problema; **safety net é mais transparente operacionalmente**.

| Aspecto | Cost-sensitive | Safety Net |
|---|---|---|
| Mecanismo | argmin Σ C[i,j]·P(i\|x) | Bloqueia Healthy se P(doente) ≥ θ |
| Transparência clínica | Médico precisa entender matriz | Médico vê 1 número (θ) |
| Sensível a premissas | Sim (custos podem mudar) | Não (só threshold) |
| Recall obtido | 1,00 com €10k FN | 0,995 com θ=0,10 |
| Especificidade | 0,03 (quase tudo é doente) | 0,31 (mais útil clinicamente) |

Cost-sensitive **zera FN à custa de eliminar a função discriminativa** (todos viram doentes). Safety Net mantém especificidade utilizável.

---

### Q7. SMOTE deu 100% no Estágio 2 — porque mantiveram este resultado no projeto?

**R:** **Não o mantivemos como vitória — mantivemos como contra-exemplo metodológico.** O nb13 reporta o 100% e o nb14 (undersampling sem dados sintéticos) **invalida** explicitamente esse número.

Conclusão científica: SMOTE inflou os resultados por interpolação no espaço de features ainda parcialmente contaminado por equipamento. Undersampling rigoroso (nb14) deu recall URTI 0,514 — **muito longe** dos 100% do SMOTE.

**Esta inclusão é deliberada:** demonstra honestidade científica (saber identificar quando o próprio resultado é suspeito).

---

### Q8. Não tem validação externa em outro dataset. Como sabe que generaliza?

**R:** **Não sei.** É a limitação mais grave do projeto e está explicitamente documentada (`RELATORIO.md` § 7.2). Antes de qualquer uso clínico real:

1. Validar em dataset independente (Coswara, FluSense, ou recolha hospitalar portuguesa)
2. Validação prospetiva em estudo piloto (≥ 50 pacientes)
3. Auditoria contínua em produção para detetar *concept drift*

**Posicionamento honesto:** este projeto é **prova de conceito metodológica + roadmap regulatório**, não produto comercial pronto.

---

## 🩺 Perguntas clínicas

### Q9. O sistema substitui o médico?

**R:** **Não.** É explicitamente **apoio à decisão** (SaMD Classe IIa proposta). Em todos os outputs:
- Recomendação clara: "Auscultação manual + exame físico permanecem obrigatórios"
- Para confiança baixa ou outlier: revisão obrigatória pelo clínico
- Em casos discrepantes (modelo vs sintomas): **clínica prevalece sempre**

---

### Q10. Como lida com pacientes pediátricos? Pneumonia? Asma?

**R:**

- **Pediátricos:** o dataset tem crianças (campos `child_weight`, `child_height`), mas não foi feita análise estratificada por idade. Limitação documentada (nb15 endereça parcialmente).
- **Pneumonia:** excluída por insuficiência de pacientes (n=6); o modelo classificará Pneumonia como Healthy/COPD/URTI — **risco grave de FN**. Mitigação: outlier detector deve flagged áudios atípicos (a maioria dos casos de Pneumonia tem padrão acústico distintivo).
- **Asma:** apenas 1 paciente no dataset — não modelado. Mesma mitigação via outlier.

**Recomendação clínica explícita:** se o paciente refere sintomas inconsistentes com o output, **não confiar no sistema**. Documentado em `clinical_guide.py` e no aviso ético da app Streamlit.

---

### Q11. O que dizer ao paciente cujo recall URTI é 0,514 (cascata) ou 0,357 (baseline)?

**R:** A pergunta correta. Recall URTI baixo significa que **~50% dos doentes URTI são classificados erroneamente**. Em prática:

- Paciente com URTI verdadeira que recebe "Healthy" → vai para casa sem antibiótico/sintomáticos → eventualmente reaparece com agravamento
- O risco é maior para URTI complicada (sinusite bacteriana, OMA)

**Mitigação operacional:**
- Modo `safe` com θ=0,10 reduz drasticamente este caso (recall URTI 0,514 → 0,643 na cascata do nb13)
- Protocolo de revisão: pacientes com sintomas referidos de URTI **não devem receber alta automática** mesmo se modelo classificar Healthy
- Documentação clara da limitação para o clínico utilizador

---

## 💼 Perguntas de gestão / negócio

### Q12. O modelo de negócio é viável?

**R:** Análise preliminar no `RELATORIO.md` § 7 e nb11/nb12 (em euros). Resumo:

| Estratégia | Custo evitado/ano (50k auscultações) |
|---|---:|
| Custos reais SNS (FN DPOC = €2.950) | **~€157 mil** |
| Cenário conservador (FN DPOC = €10.000) | até >€10 milhões |

**Mas:** estes são **proxies do dano clínico**, não o argumento primário. A análise económica completa requer:
- TAM/SAM/SOM auditados (incompleto no projeto atual)
- Decisão de modelo (SaaS hospitalar é a recomendação)
- Custos regulatórios MDR + AI Act (€80–250k iniciais)
- Roadmap 24–36 meses até SNS

Trabalho futuro — não pretensão deste TFC.

---

### Q13. Quanto custa certificar este sistema?

**R:** Estimativa preliminar (consulta a fontes regulatórias UE):

| Item | Custo |
|---|---:|
| CE Marking MDR Classe IIa | €30k–€100k |
| ISO 13485 setup | €20k–€100k |
| Avaliação clínica (CER) | €15k–€60k |
| AI Act compliance | €20k–€80k |
| Submissão SiNATS | €15k–€40k |
| EC REP + EUDAMED + PMS anual | €15k–€40k/ano |
| **Total CAPEX inicial** | **€80k–€250k** |
| **OPEX recorrente** | **€15k–€40k/ano** |

Estimativa baseada em casos de SaMD comparáveis. Variação por Notified Body e complexidade.

---

### Q14. Porque MDR Classe IIa e não IIb?

**R:** Aplicação da Regra 11 do Anexo VIII MDR:

- Software de diagnóstico **sem** decisão automática vinculativa → Classe IIa (apoio à decisão)
- Software que **toma decisões irreversíveis** ou influencia tratamento de risco vital → Classe IIb

O nosso sistema é **decision support**: médico mantém autonomia, sistema apenas sinaliza. Documentação explícita em todos os outputs ("não substitui julgamento clínico").

Classificação a confirmar com Notified Body — em alguns casos a interpretação é conservadora (IIb). Documentado como risco.

---

### Q15. Como concorre com a Eko Health?

**R:** **Posicionamento diferenciado**, não competição direta:

| | Eko Health | Este projeto |
|---|---|---|
| Foco | Cardiologia (murmúrios, AFib, EF) | Pneumologia (DPOC, URTI) |
| Hardware | Estetoscópio próprio ($379) | Software-only (qualquer wav) |
| Reg. | 9 FDA clearances, CE Mark | A iniciar |
| Mercado | Global, 700k dispositivos | Piloto PT proposto |

Camada acima do StethoMe/Sonavi (que detetam **sintomas acústicos** crackles/wheezes): o nosso sistema classifica **doença final**.

**Risco real:** Eko pode entrar em DPOC. *First-mover advantage* em PT + integração com sistema de saúde local é mitigante.

---

## 🤔 Perguntas críticas / desconfortáveis

### Q16. Vocês mudaram a métrica principal durante o projeto — não é *cherry picking*?

**R:** **Não. A mudança foi metodologicamente fundamentada e documentada.**

- **Início:** F1-macro como métrica de balanceamento (padrão ML)
- **Após nb06:** evidência de viés mudou o ângulo para "modelo honesto e calibrado"
- **Após nb12:** o utilizador (clínico) reposicionou o critério: "FN é risco à saúde, não custo financeiro" → **recall doente** torna-se a métrica governante por **imperativo ético**

A mudança está explicitamente documentada no nb14 e em todos os documentos finais com argumentação bioética (princípio da não-maleficência). Os números antigos (F1-macro) **continuam reportados** para honestidade — não foram apagados.

---

### Q17. Vocês têm 14 notebooks. Não é exploração sem foco?

**R:** **Cada notebook responde uma pergunta específica.** Resumo no Apêndice C do `RELATORIO.md`:

| Fase | Notebooks | O que demonstra |
|---|---|---|
| Exploração | 01–02 | Dataset e features |
| Comparação | 03–05 | LogReg/LDA/KNN são equivalentes; XGB pior |
| **Achado central** | **06** | **Probe identifica viés massivo** |
| Mitigação | 07–08 | Residualização + calibração |
| Validação | 09 | Holdout + outlier |
| Deep learning | 10 | CNN não bate LogReg |
| Económico | 11–12 | Custos + safety net |
| Avançado | 13–14 | Cascata + balanceamento (SMOTE invalidado por undersampling) |
| Validação crítica | 15 | Probe pós-residualização + subgrupos + erro |

Apresentação resumida em **5 narrativas** (não 14 notebooks).

---

### Q18. Porque scripts/build_notebook em vez de editar diretamente?

**R:** **Solução pragmática a bug técnico**, documentada honestamente no `MANUAL_CODIGO.md` § 10:

- A ferramenta de edição programática de notebooks usada (NotebookEdit) escapava aspas (`"` → `\"`) gerando SyntaxError
- Geração via JSON Python é mais reproduzível mas **impede edição interativa no Jupyter**
- Workflow recomendado em produção real seria diferente — este é um *trade-off* assumido

Compensação: cada script é versionável e regenera o notebook idêntico. Para auditoria, vantagem real.

---

### Q19. Falta validação clínica formal por pneumologista. Isso não é grave?

**R:** **É a limitação mais grave do projeto.** Documentada em `RELATORIO.md` § 7.2 e neste FAQ Q8.

Para um TFC académico, validação clínica formal exige:
- Aprovação CES + DPIA (3–6 meses)
- Recolha de dados portugueses (parceria hospitalar)
- Painel de pneumologistas para *ground truth*
- Estudo prospetivo (12–24 meses)

**Fora do escopo de um TFC.** Por isso o projeto se posiciona como **prova de conceito + roadmap**, não produto.

---

### Q20. Se tivessem que recomeçar, o que fariam diferente?

**R:** Três coisas concretas:

1. **Probar viés mais cedo** (nb06 deveria ter sido nb02) — toda a comparação de modelos com viés foi "tempo perdido" cientificamente
2. **Validação externa desde o início** — recolher 50 áudios portugueses paralelos
3. **Análise de erro qualitativa** — entrevistar os 4 pacientes URTI que sempre falham (idade, equipamento, sintomas) para perceber se é limite do som ou da extração

Documentado como **lições aprendidas** no `HISTORICO_DECISOES.md`.

---

*Última atualização: 2026-05.*
