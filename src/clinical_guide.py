"""Orientação clínica para o assistente respiratório.

NÃO substitui julgamento clínico. Estas recomendações são baseadas em diretrizes
gerais (GOLD para COPD, IDSA/SBP para URTI) e devem ser **adaptadas** à
realidade clínica do paciente: idade, comorbilidades, sintomas referidos,
exame físico, contexto epidemiológico, recursos locais.

Princípio ético subjacente
--------------------------
Todo o pipeline prioriza recall(doente) — minimização de Falsos Negativos clínicos —
porque o erro de classificar um doente como saudável é eticamente inaceitável:

  * Princípio da não-maleficência (primum non nocere): sistema clínico não pode
    causar dano por omissão.
  * Princípio da autonomia: paciente tem direito ao seu diagnóstico.
  * Em DPOC, atraso diagnóstico implica progressão irreversível da doença.
  * FP é remediável (consulta extra); FN tem custo cumulativo no paciente.

A linguagem das recomendações reflete esta priorização ética: alertas explícitos,
encaminhamentos conservadores, e revisão manual obrigatória em casos de incerteza
ou outlier.

Estrutura:
    CLINICAL_GUIDANCE[diagnóstico][nível] -> dict com 5 secções:
        - urgencia            (str: NENHUMA / BAIXA / MODERADA / ALTA / EMERGÊNCIA)
        - encaminhamento      (str)
        - exames              (list[str])
        - tratamento          (list[str])
        - sinais_alarme       (list[str])  — sempre verificar e documentar
        - educacao            (list[str])  — orientação para o paciente
        - resumo              (str)        — 1 frase para o médico

`get_guidance(diagnosis, top_probability, is_outlier, mode)` decide o nível
e retorna o dict apropriado, com **overrides** para baixa confiança e outlier.
"""
from __future__ import annotations
from typing import Literal


UrgencyLevel = Literal["NENHUMA", "BAIXA", "MODERADA", "ALTA", "EMERGÊNCIA"]


CLINICAL_GUIDANCE: dict[str, dict[str, dict]] = {
    # =================================================================
    "COPD": {
        "alta_confianca": {
            "urgencia": "MODERADA",
            "encaminhamento": (
                "Consulta de **Pneumologia** em 2–4 semanas. Se exacerbação aguda "
                "(dispneia em repouso, cianose), **encaminhar imediatamente para serviço de urgência**."
            ),
            "exames": [
                "**Espirometria com prova broncodilatadora** — confirma diagnóstico (FEV1/FVC < 0.70 pós-broncodilatador)",
                "Radiografia de tórax PA + perfil — exclusão de comorbilidades (neoplasia, IC)",
                "Hemograma completo + bioquímica + função renal",
                "Gasometria arterial se SpO₂ < 92% ou dispneia ao repouso",
                "Alfa-1-antitripsina sérica se < 45 anos ou história familiar",
                "ECG basal (avaliar cor pulmonale)",
            ],
            "tratamento": [
                "Broncodilatador de ação curta (SABA: salbutamol 100 µg inalado, 2 puffs até 4×/dia se dispneia)",
                "Considerar broncodilatador de longa duração (LABA/LAMA) — confirmar com espirometria antes",
                "Corticoide inalado APENAS em fenótipo eosinofílico ou exacerbações ≥ 2/ano",
                "Oxigenoterapia domiciliar se PaO₂ < 55 mmHg ou SpO₂ < 88% em repouso",
            ],
            "sinais_alarme": [
                "🚨 Dispneia em repouso ou ao falar → ENCAMINHAR URGÊNCIA",
                "🚨 Cianose central, alteração da consciência, uso de musculatura acessória → EMERGÊNCIA",
                "🚨 SpO₂ < 90% em ar ambiente → considerar internamento",
                "⚠️ Febre alta sustentada + expectoração purulenta → pneumonia sobreposta",
                "⚠️ Hemoptise — investigar neoplasia",
            ],
            "educacao": [
                "**Cessação tabágica** — fator etiológico principal; oferecer apoio (terapia substituição nicotina, varenicline)",
                "Vacinação anual da gripe + pneumocócica + COVID-19 + tosse convulsa (Tdap)",
                "Programa de reabilitação pulmonar (melhora qualidade de vida e dispneia)",
                "Técnica inalatória — rever em cada consulta",
                "Plano de ação escrito para exacerbações (quando aumentar broncodilatador, iniciar prednisolona/antibiótico)",
                "Atividade física regular adaptada à tolerância",
            ],
            "resumo": "Diagnóstico provável de DPOC. Confirmar com espirometria, otimizar broncodilatador, cessação tabágica e vacinação."
        },
    },
    # =================================================================
    "URTI": {
        "alta_confianca": {
            "urgencia": "BAIXA",
            "encaminhamento": (
                "Acompanhamento por **Clínica Geral / Medicina Familiar**. "
                "Reavaliação se persistência > 10 dias ou agravamento."
            ),
            "exames": [
                "Geralmente **não são necessários** exames complementares no início",
                "Se febre > 7 dias: hemograma + PCR/VS",
                "Se suspeita de sinusite bacteriana (>10d ou bifásica): considerar TC seios perinasais",
                "Se faringite com Centor ≥ 3: teste rápido para estreptococo / cultura",
                "Otoscopia se otalgia (descartar OMA)",
            ],
            "tratamento": [
                "**Sintomáticos**: paracetamol 1 g 8/8h se febre/dor; AINEs se inflamação",
                "Hidratação abundante, repouso",
                "Soro fisiológico nasal (lavagens 4–6×/dia)",
                "Anti-histamínicos se rinorreia importante (1ª linha: 2ª geração)",
                "Mel pode ajudar tosse (> 1 ano)",
                "⚠️ **Antibiótico apenas com critério**: faringite estreptocócica confirmada, OMA, sinusite bacteriana documentada, ou imunossupressão",
                "Antitússicos apenas tosse seca, irritativa, NUNCA tosse produtiva",
            ],
            "sinais_alarme": [
                "🚨 Dispneia, taquipneia (FR > 24) → REAVALIAR (descartar pneumonia)",
                "🚨 Dor torácica → URGÊNCIA",
                "🚨 Sonolência, confusão, rigidez nuca → URGÊNCIA (meningite?)",
                "⚠️ Sintomas > 10–14 dias → reavaliar diagnóstico",
                "⚠️ Febre > 39°C persistente > 5 dias → reavaliar",
                "⚠️ Dor de ouvido intensa, otorreia → otite média complicada",
            ],
            "educacao": [
                "Quadro geralmente **autolimitado** (5–10 dias)",
                "Lavagem frequente das mãos; etiqueta respiratória",
                "Hidratação adequada (2–3 L/dia se não houver contraindicação)",
                "Repouso conforme tolerância",
                "**Não auto-medicar com antibióticos** — só agravam o problema da resistência",
                "Vacinação anual da gripe se grupo de risco",
                "Retornar imediatamente se aparecerem sinais de alarme",
            ],
            "resumo": "Infeção respiratória alta provável — quadro habitualmente autolimitado. Sintomáticos, hidratação e vigilância de sinais de alarme."
        },
    },
    # =================================================================
    "Healthy": {
        "alta_confianca": {
            "urgencia": "NENHUMA",
            "encaminhamento": "Alta clínica. Manter rastreios periódicos conforme idade/risco.",
            "exames": [
                "Apenas exames de rastreio habituais conforme idade e fatores de risco",
                "Se paciente referiu sintomas, **não desvalorizar** apenas pelo resultado deste assistente — fazer exame clínico completo",
            ],
            "tratamento": [
                "Nenhum tratamento específico necessário neste momento",
            ],
            "sinais_alarme": [
                "ℹ️ Reavaliar se aparecerem sintomas respiratórios (tosse > 3 semanas, dispneia, hemoptise, perda de peso inexplicada)",
            ],
            "educacao": [
                "Manter hábitos saudáveis: actividade física regular, não fumar, dieta equilibrada",
                "Vacinação atualizada conforme calendário",
                "Auscultação periódica em consultas de rotina",
                "Procurar médico se aparecerem sintomas respiratórios persistentes",
            ],
            "resumo": "Auscultação dentro dos parâmetros esperados. Manter rastreios e hábitos saudáveis."
        },
    },
}


# Overrides especiais quando há incerteza
def _build_low_confidence_guidance(original_diag: str) -> dict:
    return {
        "urgencia": "REVISÃO MANUAL OBRIGATÓRIA",
        "encaminhamento": (
            f"⚠️ **Confiança baixa do modelo na predição '{original_diag}'.** "
            "O resultado é **insuficiente** como apoio à decisão. "
            "Realizar **auscultação presencial completa** + anamnese detalhada + exame físico. "
            "Considerar segunda opinião se sintomas/sinais ambíguos."
        ),
        "exames": [
            "Repetir auscultação manualmente em todos os campos pulmonares",
            "Avaliar necessidade de radiografia de tórax baseada na clínica",
            "Hemograma + PCR se febre ou sintomas sistémicos",
            "Espirometria se suspeita de doença obstrutiva",
        ],
        "tratamento": [
            "**Não iniciar tratamento empírico** com base apenas neste resultado",
            "Decisão terapêutica deve seguir auscultação manual + clínica",
        ],
        "sinais_alarme": [
            "🚨 Sinais de insuficiência respiratória (taquipneia, uso musculatura acessória, cianose) → URGÊNCIA",
            "🚨 Hipoxemia (SpO₂ < 92%) → URGÊNCIA",
            "⚠️ Qualquer dúvida persistente após reavaliação → encaminhar especialista",
        ],
        "educacao": [
            "Explicar ao paciente que será necessária avaliação clínica adicional",
            "Orientar para retornar se piorar antes da próxima avaliação",
        ],
        "resumo": "⚠️ Modelo com baixa confiança. Não usar como decisão clínica isolada; realizar avaliação presencial completa."
    }


def _build_outlier_guidance(original_diag: str) -> dict:
    return {
        "urgencia": "REVISÃO MANUAL OBRIGATÓRIA — AMOSTRA ATÍPICA",
        "encaminhamento": (
            "⚠️ **A gravação está estatisticamente fora da distribuição de treino** "
            "(detetor de outlier ativado). Possíveis causas: ruído excessivo, "
            "qualidade técnica da captura, paciente com perfil clínico raro, "
            "ou doença não coberta pelo modelo (Pneumonia, Asma, Bronquiectasias, "
            "Bronquiolite — NÃO incluídas no treino). "
            "**Repetir auscultação** com qualidade técnica adequada e **avaliação clínica completa**."
        ),
        "exames": [
            "Repetir gravação em ambiente silencioso, com técnica adequada",
            "Auscultação manual em todos os campos",
            "Considerar radiografia de tórax PA + perfil",
            "Avaliação por pneumologia se padrão atípico se confirmar",
        ],
        "tratamento": [
            "**Suspender qualquer decisão automatizada** baseada neste resultado",
            "Decisão clínica conforme avaliação manual e contexto",
        ],
        "sinais_alarme": [
            "🚨 Sinais respiratórios agudos (dispneia, dor torácica, cianose) → URGÊNCIA imediata",
            "⚠️ Padrão acústico atípico pode indicar doença não-COPD/URTI/Healthy",
            "⚠️ Considerar diagnósticos diferenciais não cobertos pelo modelo",
        ],
        "educacao": [
            "Explicar ao paciente que será feita avaliação clínica complementar",
        ],
        "resumo": "⚠️ Amostra fora da distribuição — exige avaliação manual completa. Diagnósticos fora do alcance do modelo são possíveis."
    }


def get_guidance(
    diagnosis: str,
    top_probability: float,
    is_outlier: bool = False,
    mode: str = "argmax",
    confidence_threshold_high: float = 0.8,
    confidence_threshold_med: float = 0.6,
    safe_blocked: bool = False,
) -> dict:
    """Decide o guia clínico apropriado baseado em diagnóstico + confiança + outlier.

    Hierarquia de overrides:
        1. Outlier  → orientação especial (revisão manual obrigatória)
        2. P < med_threshold (≈0.6) → orientação de baixa confiança
        3. Caso contrário → orientação padrão do diagnóstico
    `mode` é mantido por compatibilidade; a nota do modo safe depende só de
    `safe_blocked` (a safety net substituiu 'Healthy' por uma doença).

    Retorna dict com chaves:
        urgencia, encaminhamento, exames, tratamento, sinais_alarme, educacao, resumo, nivel
    """
    if is_outlier:
        g = _build_outlier_guidance(diagnosis).copy()
        g["nivel"] = "OUTLIER"
        return g

    diag_block = CLINICAL_GUIDANCE.get(diagnosis)
    if top_probability < confidence_threshold_med:
        g = _build_low_confidence_guidance(diagnosis).copy()
        g["nivel"] = "BAIXA_CONFIANCA"
    elif not diag_block:  # diagnóstico inesperado
        g = _build_low_confidence_guidance(diagnosis) | {"nivel": "DESCONHECIDO"}
    else:  # confiança média ou alta → guidance padrão do diagnóstico
        g = diag_block["alta_confianca"].copy()
        g["nivel"] = (
            "ALTA_CONFIANCA"
            if top_probability >= confidence_threshold_high
            else "MEDIA_CONFIANCA"
        )

    # Nota só quando a safety net alterou de facto a decisão (o argmax era
    # 'Healthy'). `safe_blocked` vem de result["safe_blocked_healthy"].
    # Nesses casos a probabilidade da doença escolhida costuma ser baixa, por
    # isso a nota tem de aparecer também na orientação de baixa confiança.
    if safe_blocked:
        g["encaminhamento"] = (
            "ℹ️ **Modo conservador (safe) ativo**: a classe mais provável era 'Healthy', "
            "mas P(doença) ultrapassou o limiar configurado e o sistema indicou a doença "
            f"mais provável ('{diagnosis}'). É um sinal para avaliar, não um diagnóstico.\n\n"
            + g["encaminhamento"]
        )
    return g


# Escala numérica de urgência (para ordenação / dashboard / alertas)
URGENCY_SCALE = {
    "NENHUMA": 0,
    "BAIXA": 1,
    "MODERADA": 2,
    "ALTA": 3,
    "EMERGÊNCIA": 4,
    "REVISÃO MANUAL OBRIGATÓRIA": 3,
    "REVISÃO MANUAL OBRIGATÓRIA — AMOSTRA ATÍPICA": 3,
}


def urgency_color(urgency: str) -> str:
    """Cor (hex) para representar visualmente o nível de urgência."""
    if "EMERGÊNCIA" in urgency or "ALTA" in urgency:
        return "#c0392b"  # vermelho forte
    if "REVISÃO" in urgency or "OUTLIER" in urgency or "MODERADA" in urgency:
        return "#e67e22"  # laranja
    if "BAIXA" in urgency:
        return "#f1c40f"  # amarelo
    return "#27ae60"      # verde
