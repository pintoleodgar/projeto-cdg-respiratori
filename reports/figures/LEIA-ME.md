# Figuras essenciais para o relatório

3 figuras (300 dpi, prontas para inserir no relatório). Geradas por
`python scripts/gerar_figuras_relatorio.py`.

---

## FIG1 — Descoberta e mitigação do viés de equipamento
**`FIG1_descoberta_vies.png`**

3 painéis:
- **(a)** Cross-tab equipamento × diagnóstico → 100% de Healthy/URTI gravados no Meditron
- **(b)** Probe de equipamento: F1-macro cai de **0,864 → 0,089** após residualização (−90%)
- **(c)** Custo da honestidade: F1m do diagnóstico (0,583 com viés → 0,644 residualizado)

**Mensagem:** o achado metodológico central — o dataset tem viés massivo de equipamento, e a residualização linear mitiga-o eficazmente.

**Legenda sugerida:** *"Figura 1 — O dataset ICBHI 2017 apresenta viés estrutural de equipamento (a): todas as gravações de pacientes saudáveis e com URTI foram feitas com o estetoscópio Meditron. Um classificador-sonda prevê o equipamento a partir das features de áudio com F1-macro de 0,864 (b), confirmando que a classificação de diagnóstico pode usar o equipamento como atalho. Após residualização linear das features sobre o equipamento, o probe cai para 0,089, demonstrando mitigação eficaz do viés."*

---

## FIG2 — Comparação de modelos
**`FIG2_comparacao_modelos.png`**

3 painéis (recall de doença / F1-macro / especificidade), 7 modelos, IC95% bootstrap.
Modelo final (LogReg residualizado) destacado a verde-escuro.

**Mensagem:** o baseline trivial "sempre COPD" tem recall 1,0 mas especificidade 0 (inútil). LogReg residualizado é o melhor compromisso. RandomForest (black-box) é marginalmente melhor mas não-interpretável.

**Legenda sugerida:** *"Figura 2 — Comparação de 7 estratégias em validação cruzada 5×5-fold paciente-estratificada, com intervalos de confiança a 95% (bootstrap). O classificador trivial 'sempre COPD' atinge recall de doença de 1,0 mas especificidade nula, ilustrando que recall elevado isolado não tem valor clínico. O modelo final (LogReg residualizado, destacado) oferece o melhor compromisso entre sensibilidade, qualidade global (F1-macro) e especificidade, mantendo interpretabilidade total."*

---

## FIG3 — Modelo final em ação
**`FIG3_modelo_final.png`**

3 painéis:
- **(a)** Matriz de confusão paciente (modo argmax): 9 FN clínicos
- **(b)** Matriz de confusão paciente (modo safety net θ=0,10): 7 FN clínicos
- **(c)** Comparação de FN clínicos + nota económica (€2.950 vs €70, ~€157 mil/ano)

**Mensagem:** o modo safety net reduz falsos negativos clínicos (doentes classificados como saudáveis) ao custo de mais falsos positivos remediáveis.

**Legenda sugerida:** *"Figura 3 — Desempenho do modelo final ao nível do paciente (n=104). O modo conservador 'safety net' (b) reduz os falsos negativos clínicos de 9 para 7 face ao modo padrão (a), aplicando o princípio bioético da não-maleficência. Como o custo de um falso negativo de DPOC (internamento evitável, ~€2.950) é 42× superior ao de um falso positivo (consulta extra, ~€70), esta troca traduz-se numa economia projetada de ~€157 mil/ano para um volume de 50.000 auscultações."*

---

## Como inserir no relatório (Markdown/LaTeX)

**Markdown:**
```markdown
![Figura 1](reports/figures/FIG1_descoberta_vies.png)
```

**LaTeX:**
```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=\textwidth]{reports/figures/FIG1_descoberta_vies.png}
  \caption{Descoberta e mitigação do viés de equipamento.}
  \label{fig:vies}
\end{figure}
```

**Word:** Inserir → Imagem → selecionar o PNG (300 dpi imprime nítido).
