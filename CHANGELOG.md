# Changelog — Decisões, Correções e Melhorias

> Log único consolidado do projeto: decisões metodológicas + bug fixes + melhorias aplicadas.
> Substitui `HISTORICO_DECISOES.md` e `MELHORIAS_APLICADAS.md`.

---

## 🔁 Revisão v3 — novembro de 2026 (validade das métricas)

Tudo reproduzível com `python scripts/train.py`; resultados em `reports/AVALIACAO_V3.md`.

| # | Problema encontrado | Correção |
|---|---|---|
| V1 | "Recall de doença 99,5%" era por gravação e dominado por COPD | Métricas por paciente e por classe; app e README corrigidos |
| V2 | "Viés reduzido 89,7%" só valia para sonda linear | Sonda RandomForest acrescentada (train.py e nb15 §1.1): F1m ≈ 0,85–0,87 após residualização |
| V3 | `duration_s` era atalho (Meditron: COPD >32 s, Healthy/URTI ~20 s) | Áudio cortado a 20 s (`config.MAX_DURATION_S`); `duration_s` fora das features (77) |
| V4 | Healthy/URTI só existem no Meditron | Avaliação separada "só Meditron"; limitação declarada |
| V5 | Modelo v2 em disco não era gerado por nenhum script (limiar 353 vs χ² no nb09) | `scripts/train.py` gera `lung_assistant_v3.npz` com p99 empírico |
| V6 | Calibração avaliada nos dados de ajuste; pré-processamento fora do CV interno | Calibração aninhada; sigmoide escolhida (log loss 0,150) |
| V7 | Holdout "virgem" não era independente nem conclusivo | Reclassificado como ilustrativo |
| V8 | Projeção financeira com prevalência do dataset e erros por gravação | Cenários de prevalência por paciente com IC95%; a poupança não se confirma |
| V9 | IC por bootstrap de folds (não independentes) | Bootstrap estratificado de pacientes |

**Código e app (mesma revisão):**

| # | Problema | Correção |
|---|---|---|
| C1 | Aviso pediátrico só existia no README | `contraindications` em todas as predições; mostrado na app e na CLI |
| C2 | Métricas por paciente, API só por gravação | `LungAssistant.predict_patient` (média das probabilidades) |
| C3 | `mode` mal escrito caía em argmax; limiar sem validação | `ValueError`; `--threshold` validado na CLI |
| C4 | Nota "bloqueou Healthy" aparecia sem bloqueio e desaparecia em baixa confiança | `safe_blocked_healthy` no resultado; nota só quando houve bloqueio |
| C5 | App deixava o áudio do paciente em ficheiros temporários e reextraía features a cada interação | Temporário apagado; cache por conteúdo; extração única (`extract` + `predict_features`/`explain_features`) |
| C6 | Áudio curto, silencioso ou ilegível não era validado | `InvalidAudioError` |
| C7 | Cache da CLI ignorava `model_path`; lista de equipamentos em 3 sítios | Cache por caminho; `src.metadata.EQUIPMENTS` |
| C8 | Lógica de decisão sem testes | 20 testes novos com modelo sintético (36 no total) |

**Organização (mesma revisão):**

| # | Problema | Correção |
|---|---|---|
| O1 | Projeto dentro de um repositório git na pasta do utilizador, sem `.gitignore` | Repositório próprio em `projeto_cdg1.1/` + `.gitignore` |
| O2 | `requirements.txt` sem versões e sem `pytest` | Versões fixas (as usadas para gerar o v3); `pytest.ini` |
| O3 | Caminho do dataset fixo no código; `config` criava pastas ao ser importado | Variável de ambiente `ICBHI_ROOT`; `config.ensure_dirs()` |
| O4 | Modelo em pickle (`.joblib`): frágil a versões e executa código ao carregar | `lung_assistant_v3.npz` (arrays + cabeçalho JSON); `.joblib` só é lido, para o v1/v2 |

Não alterado nesta revisão: notebooks 08–14 e 16 (continuam a usar `features.csv` e o v2),
`src/bootstrap.py`, figuras em `reports/figures/` e os ficheiros `.docx`.

---

## 🎯 Decisões metodológicas chave

| # | Decisão | Justificação | Doc |
|---|---|---|---|
| D1 | 3 classes (Healthy, COPD, URTI) | Pneumonia tinha 6 pac. → inviabiliza k-fold | nb01 |
| D2 | **Recall doente** como métrica principal (não F1) | Imperativo ético — não-maleficência | nb14, README |
| D3 | Residualização linear sobre equipamento | Probe nb06 mostrou F1=0.914 prevendo equipamento | nb06–08 |
| D4 | Calibração isotónica em OOF | Probabilidades brutas miscalibradas | nb06 |
| D5 | Threshold outlier = p99 empírico | χ²(0.99) flaggea 19% do treino | nb09 |
| D6 | LogReg residualizado como modelo final | Interpretável + iguala XGBoost/CNN | nb08, nb16 |
| D7 | Undersampling > SMOTE | SMOTE inflou Stage 2 a 100% (leakage) | nb14 |
| D8 | Holdout n=21 → sanity check, não estimador | CV repetido é métrica primária | nb09, FAQ Q4 |

---

## 🐛 Bug fixes registados honestamente

| # | Bug | Sintoma | Correção |
|---|---|---|---|
| B1 | `_residualize` com `.T` errada | Predições absurdas | `coef_ @ x` (sem transposta) — convenção sklearn |
| B2 | NotebookEdit escapava aspas | `\"texto\"` em vez de `"texto"` → SyntaxError | Geração programática via JSON Python |
| B3 | 5 células com aspas no nb01 | SyntaxError ao executar | Re-escritas via Edit |
| B4 | Encoding UTF-8 no CLI Windows | `ç`, `ã` corrompidos | `PYTHONIOENCODING=utf-8` |
| B5 | CNN instável (F1m oscila 0.09–0.61) | Overfit em 655 amostras de treino | Confirma "limite é dados, não algoritmo" |

---

## 🔄 Mudanças de direção significativas

### M1. De "otimização financeira" para "imperativo ético"
- **Antes:** análise económica como argumento central (€)
- **Agora:** FN clínico = risco à saúde primariamente; € é proxy do dano
- **Atualizou:** README, RELATORIO, INTERPRETACAO_OUTPUTS, clinical_guide, app_streamlit

### M2. De "modelo único" para "3 interfaces + decision support"
- CLI (`assistente.py`) + Web (`app_streamlit.py`) + API (`LungAssistant`)
- + módulo `clinical_guide.py` com guidance contextual

### M3. De "publicação científica" para "prova de conceito + roadmap"
- Após nb06: viés massivo reconhecido
- Após nb10: CNN não bate LogReg (limite é dados, não algoritmo)
- Reposicionamento honesto: prova de conceito + roadmap regulatório

---

## ✅ Melhorias após análise crítica auto-aplicada (24 pontos)

### 🔴 Críticos resolvidos
1. ✅ **Holdout reposicionado** como sanity check (não estimador)
2. ✅ **Comparação com literatura ICBHI** adicionada (RELATORIO § 7.2.1)
3. ✅ **Probe pós-residualização** (nb15): F1 caiu de 0.914 → 0.089 (−89.7%)
4. ✅ **Threshold 0.017 contextualizado** vs baseline trivial "sempre COPD"
5. ✅ **CNN re-avaliada em CV repetido** (nb10): confirma F1m=0.505, pior que LogReg

### 🟠 Médios resolvidos
6. ✅ Calibração documentada como limitação (FAQ Q6)
7. ✅ Bootstrap IC95% para todas as métricas (`src/bootstrap.py`)
8. ✅ Cores consistentes via `src/plotting.py`
9. ✅ Figuras exportadas para `reports/figures/` (savefig)
10. ✅ Dashboard único (nb16) — figura principal para defesa
11. ✅ Notas síntese nos nb13/14 (relação entre eles)
12. ✅ Validação externa documentada como 🔴 limitação

### 🟡 Estéticos resolvidos
13. ✅ Este CHANGELOG (bug fixes + lições)
14. ✅ **15 testes unitários** (`tests/test_inference.py`) — passam em ~5s
15. ✅ `build_notebook` documentado como decisão pragmática

### 💡 Estratégicos adicionados
16. ✅ Fairness por subgrupo (nb15) — **descoberta: pediatria acc 0.45** 🔴
17. ✅ Análise dos 9 pacientes URTI errados (nb15) — todos pediátricos com Meditron
18. ✅ Ensemble bagging testado (nb15) — marginal vs LogReg
19. ✅ Baseline trivial explícito (nb16) — referência absoluta
20. ✅ **FAQ_DEFESA.md** com 20 perguntas antecipadas
21. ✅ Limpeza: 5 notebooks redundantes arquivados (de 16 para 11)
22. ✅ Documentação consolidada (este CHANGELOG)
23. ✅ **Random Forest como black-box** adicionado ao dashboard
24. ✅ CNN com CV justo confirma: F1m 0.505 < LogReg 0.638

---

## 🧠 Lições aprendidas (se recomeçasse hoje)

1. **Probar viés mais cedo** — nb06 deveria ter sido nb02; toda comparação anterior estava contaminada
2. **Validação externa desde o início** — recolher 30–50 áudios portugueses paralelos era essencial
3. **Análise de erro qualitativa cedo** — 9 pacientes URTI errados eram todos pediátricos: descoberta crucial só feita no nb15
4. **Princípio ético declarado no dia 1** — reenquadramento "FN como risco clínico" foi *post-hoc*
5. **Versionamento explícito de modelos** — `lung_assistant.joblib` v1 vs v2 sem CHANGELOG dos pesos

---

## 📐 Convenções estabelecidas (consistentes daqui em diante)

| Convenção | Aplicação |
|---|---|
| Métrica principal | **Recall doente** |
| Métrica secundária | F1-macro, especificidade, Brier, log loss |
| Split | StratifiedGroupKFold(`groups=patient_id`) |
| Repeated CV | 10×5-fold final / 5×3 experiências rápidas |
| Holdout 80/20 | *sanity check apenas* |
| Cores | `src/plotting.py:CLASS_COLORS` |
| Idioma | Português (EU) |
| Custos | Euros (€) |
| Princípio ético | Não-maleficência (*primum non nocere*) |
| Salvamento figuras | `src.plotting.savefig('nbXX_titulo')` |
| Testes | `pytest tests/` antes de qualquer release |

---

*Última atualização: 2026-05.*
