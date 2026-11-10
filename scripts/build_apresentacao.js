// Apresentação anonimizada do projeto (modelo v3). Fonte dos números: reports/metrics_v3.json
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/root/.claude/skills/synced/7e0490a4-34c8-4c05-98f7-96f28abe1975_88fa8368-a176-4576-bdca-7a9fbc278293/pptx/scripts/apply_theme.js");

const THEME = {
  name: "Auscultacao",
  headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1B2A33", lt1: "FFFFFF", dk2: "0F6E6E", lt2: "EAF3F2",
            accent1: "0F6E6E", accent2: "C8553D", accent3: "5B8FA8", accent4: "8A9BA3", accent5: "E0B04B", accent6: "3E4C54",
            hlink: "0F6E6E", folHlink: "5B8FA8" } };
const H = THEME.colors;
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";               // 10 x 5.625
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Assistente de diagnóstico respiratório (ICBHI 2017) — modelo v3";
pres.author = "Projeto Aplicado";
const C = pres.SchemeColor;

const footer = (dark) => ([
  { text: "Projeto Aplicado · Ciência de Dados e Gestão · modelo v3 · reports/AVALIACAO_V3.md", options: { x: 0.5, y: 5.25, w: 7.5, h: 0.25, fontSize: 9, color: dark ? C.accent4 : C.accent4, margin: 0, isTextBox: true } } ]);

pres.defineSlideMaster({ title: "DARK", background: { color: H.dk1 },
  objects: [ { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.6, w: 8.8, h: 1.4, fontSize: 40, bold: true, color: C.background1, margin: 0 } } },
             { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.1, w: 8.8, h: 1.4, fontSize: 18, color: C.background2, margin: 0 } } },
             { text: { text: footer(true)[0].text, options: footer(true)[0].options } } ] });
pres.defineSlideMaster({ title: "LIGHT", background: { color: H.lt1 },
  objects: [ { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.35, w: 9, h: 0.7, fontSize: 28, bold: true, color: C.text2, margin: 0, align: "left" } } },
             { text: { text: footer(false)[0].text, options: footer(false)[0].options } },
             { placeholder: { options: { name: "body", type: "body", x: 0.5, y: 1.2, w: 9, h: 3.9, fontSize: 15, color: C.text1, margin: 0 } } } ],
  slideNumber: { x: 9.2, y: 5.25, w: 0.4, h: 0.25, fontSize: 9, color: H.accent4 } });

const txt = (s, t, o) => s.addText(t, Object.assign({ isTextBox: true, margin: 0, fontSize: 14, color: C.text1 }, o));
const card = (s, x, y, w, h, title, body, num) => {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "card " + title });
  if (num) s.addText(num, { x: x + 0.15, y: y + 0.12, w: 0.8, h: 0.55, fontSize: 26, bold: true, color: C.accent2, isTextBox: true, margin: 0, objectName: "num " + title });
  txt(s, title, { x: x + (num ? 0.95 : 0.2), y: y + 0.15, w: w - (num ? 1.1 : 0.4), h: 0.5, fontSize: 15, bold: true, color: C.text2 });
  txt(s, body, { x: x + 0.2, y: y + 0.7, w: w - 0.4, h: h - 0.85, fontSize: 12.5, color: C.text1, valign: "top" });
};
const stat = (s, x, y, w, big, label, color) => {
  s.addText(big, { x, y, w, h: 0.9, fontSize: 44, bold: true, color: color || C.accent1, isTextBox: true, margin: 0, align: "center", objectName: "stat " + big });
  txt(s, label, { x, y: y + 0.9, w, h: 0.6, fontSize: 12, color: C.accent6, align: "center" });
};
const tblOpts = (extra) => Object.assign({ fontSize: 11.5, color: H.dk1, border: { type: "solid", color: "D5DEE0", pt: 0.75 }, fill: { color: "FFFFFF" }, margin: 0.05, valign: "middle" }, extra || {});
const hdr = (cells) => cells.map(c => ({ text: c, options: { bold: true, color: "FFFFFF", fill: { color: H.accent1 }, fontSize: 11.5, align: "center" } }));
const row = (cells, boldFirst) => cells.map((c, i) => ({ text: c, options: { bold: boldFirst && i === 0, align: i === 0 ? "left" : "center", fontSize: 11.5 } }));
const chartBase = { chartColors: [H.accent1, H.accent2, H.accent3], catAxisLabelColor: H.accent6, valAxisLabelColor: H.accent6, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mj-lt", valGridLine: { color: "E3E8EA", size: 0.5 }, catGridLine: { style: "none" }, dataLabelColor: H.dk1, dataLabelFontSize: 10, catAxisLabelFontSize: 10, valAxisLabelFontSize: 10, legendFontSize: 10, legendColor: H.accent6, titleColor: H.dk2, titleFontSize: 12 };

// ---------- 1 Título
pres.addSection({ title: "Abertura" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Abertura" });
s.addText("Apoio ao diagnóstico respiratório a partir de sons de auscultação", { placeholder: "title" });
s.addText("Projeto Aplicado · Licenciatura em Ciência de Dados e Gestão\nICBHI 2017 · modelo v3 (revisão de novembro de 2026)", { placeholder: "body" });
s.addShape(pres.ShapeType.ellipse, { x: 8.3, y: 0.5, w: 1.1, h: 1.1, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "marca" });
s.addNotes("Deck anonimizado para o repositório público. Todos os números vêm de reports/metrics_v3.json.");

// ---------- 2 Problema e dados
pres.addSection({ title: "Problema e dados" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Problema e dados" });
s.addText("Classificar auscultações: COPD, Healthy ou URTI", { placeholder: "title" });
stat(s, 0.5, 1.3, 2.0, "851", "gravações .wav");
stat(s, 2.5, 1.3, 2.0, "104", "pacientes");
stat(s, 4.5, 1.3, 2.0, "4", "estetoscópios");
stat(s, 6.5, 1.3, 2.0, "77", "features librosa");
s.addTable([hdr(["Classe", "Gravações", "Pacientes", "Equipamentos onde existe"]),
  row(["COPD", "793", "64", "os 4"], true), row(["Healthy", "35", "26", "só Meditron"], true), row(["URTI", "23", "14", "só Meditron"], true)],
  Object.assign(tblOpts(), { x: 0.5, y: 3.05, w: 5.6, colW: [1.2, 1.2, 1.2, 2.0] }));
card(s, 6.4, 3.05, 3.1, 1.95, "Por que só 3 classes", "Pneumonia tinha 6 pacientes e asma 1: inviável em validação cruzada por paciente. Ficaram as classes com n ≥ 14 pacientes.");
s.addNotes("ICBHI 2017 Respiratory Sound Database. 920 gravações originais; 851 após filtrar para as 3 classes.");

// ---------- 3 Viés de equipamento
pres.addSection({ title: "Achados" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Achados" });
s.addText("Achado 1: o equipamento prevê a classe", { placeholder: "title" });
txt(s, "Healthy e URTI só foram gravados com o Meditron. Um modelo pode aprender a reconhecer o microfone em vez do som respiratório. Uma sonda que tenta adivinhar o equipamento a partir das features mede esse atalho (F1-macro, CV por paciente):", { x: 0.5, y: 1.15, w: 9, h: 0.8, fontSize: 13 });
s.addChart(pres.ChartType.bar, [
  { name: "Features cruas", labels: ["Sonda linear (LogReg)", "Sonda não-linear (RandomForest)"], values: [0.895, 0.897] },
  { name: "Após residualização", labels: ["Sonda linear (LogReg)", "Sonda não-linear (RandomForest)"], values: [0.216, 0.869] } ],
  Object.assign({}, chartBase, { x: 0.5, y: 2.0, w: 5.6, h: 3.0, barDir: "col", barGrouping: "clustered", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 1, showLegend: true, legendPos: "b", showTitle: true, title: "F1-macro da sonda de equipamento (1 = identifica sempre)" }));
card(s, 6.4, 2.0, 3.1, 3.0, "Mitigado, não eliminado", "A residualização linear remove a média de cada feature por equipamento. Uma sonda linear deixa de o identificar (0,22 ≈ acaso); uma sonda não-linear continua a consegui-lo (0,87).\n\nO modelo final é linear e não explora directamente o resíduo não-linear, mas o viés está nos dados.");
s.addNotes("Na v2 reportámos 0,089 para a sonda pós-residualização. Era um artefacto: a residualização era ajustada com todos os dados antes do CV, e a sonda ficava abaixo do baseline (acc 0,128 vs 0,759).");

// ---------- 4 duration_s
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Achados" });
s.addText("Achado 2: a duração da gravação era um atalho", { placeholder: "title" });
s.addChart(pres.ChartType.bar, [{ name: "Duração média (s)", labels: ["COPD", "Healthy", "URTI"], values: [58.6, 20.0, 20.0] }],
  Object.assign({}, chartBase, { x: 0.5, y: 1.2, w: 4.6, h: 3.8, barDir: "col", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 70, showLegend: false, showTitle: true, title: "Gravações Meditron: duração média por classe (s)", chartColors: [H.accent2] }));
card(s, 5.4, 1.2, 4.1, 1.75, "O que acontecia", "No Meditron, as gravações COPD duram ~59 s e as Healthy/URTI exatamente 20 s. A feature duration_s separava COPD do resto por protocolo de gravação, não por fisiologia. Sozinha igualava as 78 features (F1-macro 0,63 vs 0,61).");
card(s, 5.4, 3.15, 4.1, 1.85, "Correção na v3", "Todo o áudio é cortado aos primeiros 20 s antes de extrair features, e duration_s deixa de ser feature (77 em vez de 78). O corte é o mesmo no treino e na inferência.");
s.addNotes("Réplica independente: só duration_s dava F1m paciente 0,634; as 78 features davam 0,613.");

// ---------- 5 Pipeline
pres.addSection({ title: "Método" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Método" });
s.addText("Pipeline do modelo v3", { placeholder: "title" });
const steps = [["Áudio", "corte a 20 s, 22,05 kHz"], ["77 features", "MFCC+Δ+Δ², chroma, contraste, espectrais"], ["Residualizar", "regressão linear em equipamento"], ["Escalar + LogReg", "multinomial, class_weight balanced"], ["Calibrar", "sigmoide, ajustada fora do treino"], ["Decidir", "argmax ou safety net (θ)"]];
steps.forEach(([t, d], i) => {
  const x = 0.5 + i * 1.52;
  s.addShape(pres.ShapeType.roundRect, { x, y: 1.5, w: 1.35, h: 1.5, rectRadius: 0.08, fill: { color: i === 2 || i === 4 ? C.accent1 : C.background2 }, line: { color: C.background2 }, objectName: "step " + t });
  txt(s, t, { x: x + 0.1, y: 1.6, w: 1.15, h: 0.45, fontSize: 13, bold: true, color: i === 2 || i === 4 ? C.background1 : C.text2, align: "center" });
  txt(s, d, { x: x + 0.1, y: 2.05, w: 1.15, h: 0.9, fontSize: 10, color: i === 2 || i === 4 ? C.background2 : C.text1, align: "center", valign: "top" });
  if (i < 5) s.addShape(pres.ShapeType.rightArrow, { x: x + 1.37, y: 2.1, w: 0.15, h: 0.3, fill: { color: C.accent4 }, line: { color: C.accent4 }, objectName: "arrow " + i });
});
card(s, 0.5, 3.3, 4.4, 1.7, "Em paralelo: detetor de outliers", "Distância de Mahalanobis no espaço residualizado e escalado. Limiar = p99 empírico do treino (346). Falso alarme medido fora do treino: 5,5%. Acima do limiar a confiança baixa e o caso vai para revisão.");
card(s, 5.1, 3.3, 4.4, 1.7, "Por que Regressão Logística", "No nested CV o XGBoost não a superou, e uma CNN sobre mel-espectrogramas (72 mil parâmetros) ficou abaixo. O limite está nos dados, não no algoritmo. A LogReg é interpretável: contribuição por feature para cada classe.");

// ---------- 6 Validação
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Método" });
s.addText("Como os números foram medidos", { placeholder: "title" });
card(s, 0.5, 1.2, 2.9, 2.1, "Por paciente", "StratifiedGroupKFold 5×5: um paciente nunca está em treino e teste ao mesmo tempo. Decisão = média das probabilidades das suas gravações.", "1");
card(s, 3.55, 1.2, 2.9, 2.1, "Tudo dentro do fold", "Residualizador, scaler, modelo e calibrador são ajustados só com o treino de cada fold (calibração aninhada).", "2");
card(s, 6.6, 1.2, 2.9, 2.1, "Incerteza", "IC95% por bootstrap de pacientes (2000 reamostragens, estratificado por classe). Com 14 URTI os intervalos são largos.", "3");
s.addTable([hdr(["Calibração (aninhada)", "Log loss", "Brier"]), row(["nenhuma", "0,228", "0,073"], true), row(["isotónica", "0,196", "0,067"], true), row(["sigmoide (usada)", "0,150", "0,065"], true)],
  Object.assign(tblOpts(), { x: 0.5, y: 3.55, w: 4.4, colW: [2.2, 1.1, 1.1] }));
txt(s, "Na v2 a calibração isotónica era medida nos mesmos dados em que fora ajustada (−49% de log loss). Medida fora do treino, a isotónica sobreajusta com 35 gravações Healthy e 23 URTI; a sigmoide (2 parâmetros por classe) generaliza melhor.", { x: 5.1, y: 3.55, w: 4.4, h: 1.5, fontSize: 12, color: C.accent6 });

// ---------- 7 Resultados
pres.addSection({ title: "Resultados" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Resultados" });
s.addText("Resultados por paciente", { placeholder: "title" });
s.addTable([hdr(["Estratégia", "Recall doente", "Recall COPD", "Recall URTI", "Especificidade", "F1-macro", "FN clínicos"]),
  row(["argmax", "0,887 [0,85–0,92]", "1,000", "0,243 [0,09–0,41]", "0,523 [0,38–0,68]", "0,591 [0,51–0,66]", "8,8"], true),
  row(["safe θ=0,30", "0,910", "1,000", "0,329", "0,431", "0,589", "7,0"], true),
  row(["safe θ=0,10", "0,938 [0,90–0,97]", "1,000", "0,457 [0,26–0,66]", "0,277 [0,14–0,42]", "0,563 [0,48–0,64]", "4,8"], true),
  row(["safe θ=0,05", "0,962", "1,000", "0,586", "0,231", "0,572", "3,0"], true),
  row(["trivial: sempre COPD", "1,000", "1,000", "0,000", "0,000", "0,254", "0"], true)],
  Object.assign(tblOpts(), { x: 0.5, y: 1.2, w: 9, colW: [1.55, 1.45, 0.95, 1.45, 1.45, 1.35, 0.8] }));
card(s, 0.5, 3.55, 4.4, 1.45, "O que o modelo faz bem", "Nunca falha um paciente COPD, em nenhuma estratégia e em nenhuma reamostragem do bootstrap. Todos os falsos negativos clínicos são URTI dados como Healthy.");
card(s, 5.1, 3.55, 4.4, 1.45, "O que a safety net compra", "Baixar θ recupera URTI (0,24 → 0,59) mas manda a maioria dos saudáveis para revisão (especificidade 0,52 → 0,23). É uma decisão clínica, não técnica.");
s.addNotes("FN clínicos = número médio de pacientes doentes classificados como Healthy, em 104 pacientes. Valores entre parênteses: IC95% bootstrap.");

// ---------- 8 Healthy vs URTI
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Resultados" });
s.addText("Achado 3: Healthy e URTI são indistinguíveis", { placeholder: "title" });
stat(s, 0.5, 1.4, 3.2, "0,45", "AUC por paciente, Healthy vs URTI\n(só Meditron, 26 vs 14 pacientes; acaso = 0,50)", H.accent2);
card(s, 4.0, 1.3, 5.5, 1.8, "O que isto significa", "A separação Healthy/URTI não existe no sinal que extraímos. O recall de URTI que a safety net recupera vem de rebaixar o limiar, não de o modelo distinguir as duas classes.\n\nNa prática, o sistema é um classificador COPD vs não-COPD.");
card(s, 4.0, 3.3, 5.5, 1.7, "Dois confundidores que não conseguimos separar", "Equipamento: Healthy e URTI só existem no Meditron.\nIdade: todos os URTI têm entre 0,7 e 16 anos; todos os COPD são adultos. Falta um baseline só com idade para saber o que o modelo aprende.");
txt(s, "Dentro do Meditron (8 COPD, 26 Healthy, 14 URTI): recall de doença 0,60 em argmax e 0,78 com θ=0,10; F1-macro 0,54.", { x: 0.5, y: 3.4, w: 3.2, h: 1.5, fontSize: 12, color: C.accent6 });

// ---------- 9 Trade-off chart
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Resultados" });
s.addText("O limiar θ é uma escolha de gestão clínica", { placeholder: "title" });
s.addChart(pres.ChartType.line, [
  { name: "Recall de doença", labels: ["argmax", "θ=0,30", "θ=0,10", "θ=0,05"], values: [0.887, 0.910, 0.938, 0.962] },
  { name: "Recall URTI", labels: ["argmax", "θ=0,30", "θ=0,10", "θ=0,05"], values: [0.243, 0.329, 0.457, 0.586] },
  { name: "Especificidade", labels: ["argmax", "θ=0,30", "θ=0,10", "θ=0,05"], values: [0.523, 0.431, 0.277, 0.231] } ],
  Object.assign({}, chartBase, { x: 0.5, y: 1.2, w: 5.8, h: 3.8, lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 7, showValue: true, dataLabelPosition: "t", dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 1, showLegend: true, legendPos: "b", showTitle: true, title: "Por paciente, CV 5×5 (safe: Healthy só se P(doente) < θ)" }));
card(s, 6.6, 1.2, 2.9, 1.8, "Triagem em massa", "θ baixo: quase nenhum doente passa, mas 3 em 4 saudáveis vão a consulta. Só faz sentido se a consulta for barata e rápida.");
card(s, 6.6, 3.2, 2.9, 1.8, "Apoio em consulta", "argmax ou θ=0,30: menos ruído, mas um resultado Healthy não exclui URTI. O clínico tem de o saber.");

// ---------- 10 Económico
pres.addSection({ title: "Gestão" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Gestão" });
s.addText("Impacto económico: a safety net não paga por si", { placeholder: "title" });
txt(s, "Matriz de custos: FN de COPD €2.950 (internamento evitável), restantes erros €70 (consulta extra). 50.000 pacientes/ano. Comparação safe θ=0,10 vs argmax. As prevalências são hipóteses, não medições.", { x: 0.5, y: 1.15, w: 9, h: 0.65, fontSize: 12.5, color: C.accent6 });
s.addTable([hdr(["Cenário (COPD / Healthy / URTI)", "€/paciente argmax", "€/paciente safe", "Diferença anual [IC95%]"]),
  row(["Rastreio população geral (5/85/10)", "33,68", "46,82", "−€657 mil [−€942 mil, −€391 mil]"], true),
  row(["Cuidados primários (10/70/20)", "33,97", "43,03", "−€453 mil [−€696 mil, −€218 mil]"], true),
  row(["Consulta de pneumologia (40/40/20)", "23,95", "27,85", "−€195 mil [−€344 mil, −€38 mil]"], true),
  row(["Dataset ICBHI (62/25/13, não realista)", "15,48", "17,77", "−€114 mil [−€209 mil, −€13 mil]"], true)],
  Object.assign(tblOpts(), { x: 0.5, y: 1.9, w: 9, colW: [3.1, 1.6, 1.5, 2.8] }));
card(s, 0.5, 3.7, 4.4, 1.3, "Porquê", "Em argmax o modelo já não falha nenhum COPD por paciente, logo a safety net não evita internamentos: só troca URTI não detetadas (€70) por muitos mais saudáveis enviados a consulta (€70 cada).");
card(s, 5.1, 3.7, 4.4, 1.3, "O que falta para um VAL", "Comparar com a prática atual sem sistema, e incluir custos de certificação (MDR Classe IIa, ISO 13485), integração e formação. O argumento para a safety net é clínico, não financeiro.");
s.addNotes("As poupanças anunciadas na v2 (€157 mil a >€10 M/ano) vinham de contar erros por gravação e de usar a prevalência do dataset.");

// ---------- 11 Limitações
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Gestão" });
s.addText("Limitações a declarar", { placeholder: "title" });
const lims = [["Sem teste independente", "Todos os dados participaram na escolha do modelo. Validação externa (outro dataset, estudo prospetivo) é obrigatória antes de uso clínico."],
  ["Confundimento estrutural", "Classe, equipamento e idade estão entrelaçados no ICBHI. Fora do Meditron o modelo nunca viu um saudável nem uma URTI."],
  ["Amostra pequena", "14 pacientes URTI, 26 Healthy. Os intervalos de confiança são largos e devem acompanhar qualquer número."],
  ["Pediatria", "Accuracy 0,45 em crianças com menos de 6 anos (v2). Contraindicação explícita; a aplicação mostra o aviso, mas não pede idade."],
  ["Cobertura", "Pneumonia, asma, bronquiectasia e bronquiolite não são modeladas; o detetor de outliers sinaliza parte dos casos atípicos."],
  ["Não é dispositivo médico", "Prova de conceito. Apoio à decisão proposto como SaMD Classe IIa; o juízo clínico prevalece sempre."]];
lims.forEach(([t, d], i) => { const col = i % 3, r = Math.floor(i / 3); card(s, 0.5 + col * 3.05, 1.2 + r * 1.95, 2.9, 1.8, t, d, String(i + 1)); });

// ---------- 12 Conclusão
pres.addSection({ title: "Fecho" });
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Fecho" });
s.addText("O que fica", { placeholder: "title", options: {} });
s.addText([
  { text: "Deteta COPD com fiabilidade; não distingue URTI de Healthy. ", options: { bold: true, breakLine: true } },
  { text: "A revisão v3 trocou números melhores por números defensáveis: sem atalhos de duração, sem calibração otimista, sem poupanças que não existem.", options: { breakLine: true } },
  { text: "Próximos passos: baseline só com idade; features por ciclo respiratório (crackles/wheezes); validação num dataset externo; VAL com custos de certificação.", options: {} } ],
  { placeholder: "body" });

(async () => {
  await pres.writeFile({ fileName: "/home/claude/deck/Apresentacao_TFC.pptx" });
  await applyTheme("/home/claude/deck/Apresentacao_TFC.pptx", THEME);
  console.log("ok");
})();
