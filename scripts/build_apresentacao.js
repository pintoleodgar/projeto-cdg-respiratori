// Apresentação com o sistema visual da referência "proposta verde":
// verde escuro + verde médio + lima, bandas diagonais, pílulas, ícones em círculos,
// tabelas com cabeçalho verde escuro e linhas alternadas. Sem travessões.
const pptxgen = require("pptxgenjs");
const path = require("path");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const Fi = require("react-icons/fi");
const { applyTheme } = require(path.join(__dirname, "apply_theme.js"));

// ----------------------------------------------------------- design system
const THEME = {
  name: "PropostaVerde",
  headFontFace: "Arial", bodyFontFace: "Calibri",
  colors: { dk1: "1B3A2A", lt1: "FFFFFF", dk2: "1E5A3C", lt2: "EAF4EE",
            accent1: "2E7D4F", accent2: "8DC63F", accent3: "A9D18E", accent4: "6B7F74", accent5: "CFE8D6", accent6: "0F3D27",
            hlink: "2E7D4F", folHlink: "8DC63F" } };
const H = THEME.colors;
const CHART_GREENS = [H.dk2, H.accent1, H.accent2, H.accent5];

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";                      // 10 x 5.625
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Apoio ao diagnóstico respiratório (protótipo, modelo v3)";
pres.author = "Projeto";
const C = pres.SchemeColor;
const OUT = path.join(__dirname, "..", "relatorio", "Apresentacao.pptx");

// ----------------------------------------------------------- icons
async function icon(name, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Fi[name], { color: "#" + color, size, strokeWidth: 1.6 }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
const ICONS = {};
async function loadIcons() {
  for (const n of ["FiMic", "FiUsers", "FiRadio", "FiLayers", "FiActivity", "FiScissors", "FiGitBranch", "FiSliders", "FiCheckCircle", "FiTarget", "FiAlertTriangle", "FiDatabase", "FiShield", "FiHeart", "FiBarChart2", "FiFlag", "FiCalendar", "FiGithub", "FiEye", "FiXCircle"]) {
    ICONS[n] = { white: await icon(n, "FFFFFF"), green: await icon(n, H.dk2) };
  }
}

// ----------------------------------------------------------- layouts
const FOOTER = { x: 0.5, y: 5.2, w: 6.5, h: 0.3, fontSize: 12, color: C.accent4, margin: 0, isTextBox: true };
const footerText = "Protótipo de apoio ao diagnóstico respiratório · modelo v3 · reports/AVALIACAO_V3.md";

pres.defineSlideMaster({ title: "COVER", background: { color: H.dk2 },
  objects: [
    { rect: { x: 6.3, y: 0, w: 3.7, h: 5.625, fill: { color: H.accent6 } } },
    { parallelogram: { x: 5.6, y: 0, w: 1.4, h: 5.625, fill: { color: H.accent1 }, line: { color: H.accent1 } } },
    { parallelogram: { x: -0.4, y: 4.7, w: 5.2, h: 0.55, fill: { color: H.accent2 }, line: { color: H.accent2 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.3, w: 4.9, h: 2.0, fontSize: 40, bold: true, color: C.background1, margin: 0, valign: "middle" } } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.4, w: 4.9, h: 1.1, fontSize: 18, color: C.background2, margin: 0, valign: "top" } } } ] });

pres.defineSlideMaster({ title: "CONTENT", background: { color: H.lt1 },
  objects: [
    { parallelogram: { x: 8.9, y: 0, w: 1.5, h: 1.05, fill: { color: H.dk2 }, line: { color: H.dk2 } } },
    { parallelogram: { x: 8.1, y: 0.0, w: 0.55, h: 1.05, fill: { color: H.accent2 }, line: { color: H.accent2 } } },
    { parallelogram: { x: 7.3, y: 5.3, w: 3.1, h: 0.33, fill: { color: H.accent1 }, line: { color: H.accent1 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 7.4, h: 0.8, fontSize: 32, bold: true, color: C.text2, margin: 0, align: "left", valign: "middle" } } },
    { text: { text: footerText, options: FOOTER } } ],
  slideNumber: { x: 9.3, y: 5.2, w: 0.4, h: 0.3, fontSize: 12, color: H.lt1, align: "right" } });

pres.defineSlideMaster({ title: "BAND", background: { color: H.lt1 },
  objects: [
    { rect: { x: 0, y: 0, w: 10, h: 1.15, fill: { color: H.dk2 } } },
    { parallelogram: { x: 8.3, y: 0, w: 1.7, h: 1.15, fill: { color: H.accent1 }, line: { color: H.accent1 } } },
    { parallelogram: { x: 7.8, y: 0, w: 0.45, h: 1.15, fill: { color: H.accent2 }, line: { color: H.accent2 } } },
    { parallelogram: { x: 7.3, y: 5.3, w: 3.1, h: 0.33, fill: { color: H.accent1 }, line: { color: H.accent1 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.18, w: 7.0, h: 0.8, fontSize: 32, bold: true, color: C.background1, margin: 0, align: "left", valign: "middle" } } },
    { text: { text: footerText, options: FOOTER } } ],
  slideNumber: { x: 9.3, y: 5.2, w: 0.4, h: 0.3, fontSize: 12, color: H.lt1, align: "right" } });

// ----------------------------------------------------------- helpers
const T = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, margin: 0, fontSize: 18, color: C.text1, valign: "top" }, o));
const pill = (s, x, y, w, h, text, fill, color, fs) => {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.5, fill: { color: fill }, line: { color: fill }, objectName: "pill " + text });
  s.addText(text, { x, y, w, h, fontSize: fs || 16, bold: true, color, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: "pill text " + text });
};
const circleIcon = (s, x, y, d, name, variant) => {
  const fill = variant === "dark" ? C.text2 : C.background1;
  const line = variant === "dark" ? H.dk2 : H.accent1;
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill }, line: { color: line, width: 1.5 }, objectName: "circle " + name });
  s.addImage({ data: ICONS[name][variant === "dark" ? "white" : "green"], x: x + d * 0.25, y: y + d * 0.25, w: d * 0.5, h: d * 0.5, objectName: "icon " + name });
};
const iconItem = (s, x, y, w, name, title, body, variant) => {
  circleIcon(s, x, y, 0.75, name, variant);
  T(s, title, { x: x + 0.95, y: y - 0.02, w: w - 0.95, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  T(s, body, { x: x + 0.95, y: y + 0.4, w: w - 0.95, h: 0.8, fontSize: 15, color: C.text1 });
};
const stat = (s, x, y, w, big, label) => {
  s.addText(big, { x, y, w, h: 0.85, fontSize: 44, bold: true, color: C.text2, isTextBox: true, margin: 0, align: "center", valign: "middle", objectName: "stat " + big });
  T(s, label, { x, y: y + 0.85, w, h: 0.45, fontSize: 16, color: C.accent4, align: "center" });
};
const hdr = (cells) => cells.map(c => ({ text: c, options: { bold: true, color: "FFFFFF", fill: { color: H.dk2 }, fontSize: 15, align: "center", valign: "middle" } }));
const row = (cells, i, opts) => cells.map((c, j) => ({ text: c, options: Object.assign({ fontSize: 15, fill: { color: i % 2 ? H.lt2 : "FFFFFF" }, color: H.dk1, align: j === 0 ? "left" : "center", bold: j === 0, valign: "middle" }, opts || {}) }));
const hiRow = (cells) => cells.map((c, j) => ({ text: c, options: { fontSize: 15, fill: { color: H.accent2 }, color: H.dk1, bold: true, align: j === 0 ? "left" : "center", valign: "middle" } }));
const tbl = (extra) => Object.assign({ border: { type: "solid", color: "FFFFFF", pt: 1 }, margin: 0.06, rowH: 0.42 }, extra);
const chartBase = { chartColors: CHART_GREENS, catAxisLabelColor: H.accent4, valAxisLabelColor: H.accent4,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
  valGridLine: { color: "DDE7E0", size: 0.5 }, catGridLine: { style: "none" }, dataLabelColor: H.dk1,
  dataLabelFontSize: 14, catAxisLabelFontSize: 14, valAxisLabelFontSize: 13, legendFontSize: 14, legendColor: H.dk1, showTitle: false };
const noDash = (s) => s.replace(/[—–]/g, " a ");

// ----------------------------------------------------------- slides
async function build() {
  await loadIcons();
  let s;

  // 1 capa
  pres.addSection({ title: "Abertura" });
  s = pres.addSlide({ masterName: "COVER", sectionTitle: "Abertura" });
  s.addText("Apoio ao diagnóstico respiratório", { placeholder: "title" });
  s.addText("Protótipo treinado em sons de auscultação\nICBHI 2017 · modelo v3", { placeholder: "body" });
  pill(s, 0.6, 0.55, 2.2, 0.45, "PROTÓTIPO", H.accent2, H.dk1, 14);
  circleIcon(s, 7.4, 2.1, 1.5, "FiActivity", "light");
  pill(s, 6.8, 4.75, 2.7, 0.45, "Novembro de 2026", H.accent1, "FFFFFF", 14);
  s.addNotes("Deck para o repositório público. Todos os números vêm de reports/metrics_v3.json. Com 104 pacientes e duas classes presas a um único equipamento, não há dados para mais do que um protótipo.");

  // 2 dados
  pres.addSection({ title: "Dados" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Dados" });
  s.addText("Poucos dados para três classes", { placeholder: "title" });
  [["851", "gravações"], ["104", "pacientes"], ["4", "estetoscópios"], ["77", "features"]].forEach(([b, l], i) => stat(s, 0.5 + i * 2.3, 1.3, 2.2, b, l));
  s.addTable([hdr(["Classe", "Pacientes", "Equipamento"]),
    row(["COPD", "64", "os 4"], 0), row(["Healthy", "26", "só Meditron"], 1), row(["URTI", "14", "só Meditron"], 2)],
    tbl({ x: 0.5, y: 2.95, w: 5.4, colW: [1.8, 1.6, 2.0] }));
  circleIcon(s, 6.5, 3.0, 0.75, "FiAlertTriangle", "dark");
  T(s, "Saudáveis e URTI só existem num equipamento", { x: 7.4, y: 3.0, w: 2.3, h: 1.2, fontSize: 17, bold: true, color: C.text2 });
  s.addNotes("920 gravações no ICBHI; 851 após filtrar para as 3 classes. Pneumonia (6 pacientes) e asma (1) ficaram de fora.");

  // 3 achado 1
  pres.addSection({ title: "Achados" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Achados" });
  s.addText("O equipamento prevê a classe", { placeholder: "title" });
  s.addChart(pres.ChartType.bar, [
    { name: "Features cruas", labels: ["Sonda linear", "Sonda não linear"], values: [0.895, 0.897] },
    { name: "Após residualização", labels: ["Sonda linear", "Sonda não linear"], values: [0.216, 0.869] } ],
    Object.assign({}, chartBase, { x: 0.5, y: 1.25, w: 5.0, h: 3.8, barDir: "col", barGrouping: "clustered", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 1, showLegend: true, legendPos: "b" }));
  iconItem(s, 5.9, 1.5, 3.8, "FiEye", "Sonda acerta o aparelho", "F1 de 0,90 a partir do som", "dark");
  iconItem(s, 5.9, 3.2, 3.8, "FiShield", "Só engana a sonda linear", "A não linear fica em 0,87", "dark");
  s.addNotes("Na v2 reportámos 0,089 para a sonda pós-residualização. Era um artefacto: a residualização era ajustada com todos os dados antes do CV.");

  // 4 achado 2
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Achados" });
  s.addText("A duração era um atalho", { placeholder: "title" });
  s.addChart(pres.ChartType.bar, [{ name: "Duração média (s)", labels: ["COPD", "Healthy", "URTI"], values: [58.6, 20.0, 20.0] }],
    Object.assign({}, chartBase, { x: 0.5, y: 1.25, w: 5.0, h: 3.8, barDir: "col", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 70, showLegend: false, chartColors: [H.accent1] }));
  T(s, "Gravações Meditron, segundos por classe", { x: 0.5, y: 5.0, w: 5, h: 0.3, fontSize: 13, color: C.accent4 });
  iconItem(s, 6.0, 1.5, 3.7, "FiAlertTriangle", "Atalho de protocolo", "Sozinha valia tanto como 78 features", "dark");
  iconItem(s, 6.0, 3.2, 3.7, "FiScissors", "Correção na v3", "Áudio cortado a 20 s. duration_s fora.", "dark");

  // 5 pipeline (chevrons como a timeline da referência)
  pres.addSection({ title: "Método" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Método" });
  s.addText("Pipeline do modelo v3", { placeholder: "title" });
  const steps = [["FiMic", "Áudio", "corte a 20 s"], ["FiLayers", "77 features", "librosa"], ["FiGitBranch", "Residualizar", "por equipamento"],
                 ["FiSliders", "LogReg", "multinomial"], ["FiTarget", "Calibrar", "sigmoide"], ["FiCheckCircle", "Decidir", "argmax ou safe"]];
  steps.forEach(([ic, t, d], i) => {
    const x = 0.5 + i * 1.52, dark = i % 2 === 0;
    s.addShape(pres.ShapeType.chevron, { x, y: 1.45, w: 1.55, h: 0.7, fill: { color: dark ? H.dk2 : H.accent1 }, line: { color: "FFFFFF", width: 1 }, objectName: "chevron " + t });
    circleIcon(s, x + 0.35, 2.4, 0.8, ic, "light");
    T(s, t, { x: x - 0.1, y: 3.35, w: 1.7, h: 0.4, fontSize: 17, bold: true, color: C.text2, align: "center" });
    T(s, d, { x: x - 0.1, y: 3.75, w: 1.7, h: 0.4, fontSize: 14, color: C.accent4, align: "center" });
  });
  pill(s, 0.5, 4.45, 9.0, 0.5, "Em paralelo: detetor de outliers (Mahalanobis, p99) manda casos atípicos para revisão", H.lt2, H.dk2, 15);

  // 6 validação
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Método" });
  s.addText("Como os números foram medidos", { placeholder: "title" });
  iconItem(s, 0.5, 1.4, 4.3, "FiUsers", "Por paciente", "CV 5×5, um paciente nunca está em treino e teste", "dark");
  iconItem(s, 0.5, 2.75, 4.3, "FiShield", "Tudo dentro do fold", "Scaler, modelo e calibrador só veem o treino", "dark");
  iconItem(s, 0.5, 4.1, 4.3, "FiBarChart2", "Intervalos a 95%", "Bootstrap de pacientes, 2000 reamostragens", "dark");
  s.addTable([hdr(["Calibração", "Log loss"]), row(["nenhuma", "0,228"], 0), row(["isotónica", "0,196"], 1), hiRow(["sigmoide, usada", "0,150"])],
    tbl({ x: 5.4, y: 1.4, w: 4.3, colW: [2.5, 1.8] }));
  T(s, "Medido fora do treino. A isotónica sobreajusta com tão poucos saudáveis e URTI.", { x: 5.4, y: 3.3, w: 4.3, h: 1.2, fontSize: 15, color: C.accent4 });

  // 7 resultados
  pres.addSection({ title: "Resultados" });
  s = pres.addSlide({ masterName: "BAND", sectionTitle: "Resultados" });
  s.addText("Resultados por paciente", { placeholder: "title" });
  s.addTable([hdr(["Estratégia", "Recall doente", "Recall COPD", "Recall URTI", "Especificidade", "F1 macro"]),
    row(["argmax", "0,89", "1,00", "0,24", "0,52", "0,59"], 0),
    row(["safe θ=0,30", "0,91", "1,00", "0,33", "0,43", "0,59"], 1),
    hiRow(["safe θ=0,10", "0,94", "1,00", "0,46", "0,28", "0,56"]),
    row(["safe θ=0,05", "0,96", "1,00", "0,59", "0,23", "0,57"], 1),
    row(["sempre COPD", "1,00", "1,00", "0,00", "0,00", "0,25"], 0)],
    tbl({ x: 0.5, y: 1.45, w: 9.0, colW: [2.0, 1.4, 1.4, 1.4, 1.5, 1.3], rowH: 0.4 }));
  iconItem(s, 0.5, 4.15, 4.3, "FiCheckCircle", "Nunca falha um COPD", "Em nenhuma estratégia", "dark");
  iconItem(s, 5.2, 4.15, 4.5, "FiXCircle", "Perde 3 em 4 URTI", "Todos dados como saudáveis", "dark");
  s.addNotes("64 COPD, 26 Healthy, 14 URTI. IC95% em reports/AVALIACAO_V3.md: recall URTI argmax 0,09 a 0,41; F1 macro 0,51 a 0,66.");

  // 8 achado 3
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Resultados" });
  s.addText("Saudável e URTI soam igual", { placeholder: "title" });
  s.addShape(pres.ShapeType.ellipse, { x: 0.7, y: 1.45, w: 3.3, h: 3.3, fill: { color: C.text2 }, line: { color: C.text2 }, objectName: "auc circle" });
  s.addText("0,45", { x: 0.7, y: 2.0, w: 3.3, h: 1.3, fontSize: 60, bold: true, color: C.background1, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: "auc" });
  T(s, "AUC por paciente\nacaso = 0,50", { x: 0.7, y: 3.3, w: 3.3, h: 0.9, fontSize: 16, color: C.background2, align: "center" });
  iconItem(s, 4.7, 1.5, 5.0, "FiRadio", "Só Meditron, 26 vs 14 pacientes", "A separação não existe no sinal", "dark");
  iconItem(s, 4.7, 2.85, 5.0, "FiTarget", "Na prática: COPD contra o resto", "A safety net só baixa o limiar", "dark");
  iconItem(s, 4.7, 4.2, 5.0, "FiUsers", "Idade é outro confundidor", "URTI são crianças, COPD são adultos", "dark");

  // 9 limiar
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Resultados" });
  s.addText("O limiar θ é uma escolha clínica", { placeholder: "title" });
  s.addChart(pres.ChartType.line, [
    { name: "Recall de doença", labels: ["argmax", "θ=0,30", "θ=0,10", "θ=0,05"], values: [0.887, 0.910, 0.938, 0.962] },
    { name: "Recall URTI", labels: ["argmax", "θ=0,30", "θ=0,10", "θ=0,05"], values: [0.243, 0.329, 0.457, 0.586] },
    { name: "Especificidade", labels: ["argmax", "θ=0,30", "θ=0,10", "θ=0,05"], values: [0.523, 0.431, 0.277, 0.231] } ],
    Object.assign({}, chartBase, { x: 0.5, y: 1.25, w: 5.8, h: 3.85, lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 9, showValue: true, dataLabelPosition: "t", dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 1, showLegend: true, legendPos: "b", chartColors: [H.dk2, H.accent2, H.accent4] }));
  iconItem(s, 6.6, 1.5, 3.2, "FiFlag", "θ baixo: triagem", "Quase nenhum doente escapa, 3 em 4 saudáveis vão a consulta", "dark");
  iconItem(s, 6.6, 3.3, 3.2, "FiHeart", "θ alto: consulta", "Menos ruído. Saudável não exclui URTI.", "dark");

  // 10 económico
  pres.addSection({ title: "Gestão" });
  s = pres.addSlide({ masterName: "BAND", sectionTitle: "Gestão" });
  s.addText("A safety net custa mais", { placeholder: "title" });
  s.addTable([hdr(["Cenário (COPD / Healthy / URTI)", "€ argmax", "€ safe", "Diferença por ano"]),
    row(["Rastreio geral (5/85/10)", "33,68", "46,82", "menos €657 mil"], 0),
    row(["Cuidados primários (10/70/20)", "33,97", "43,03", "menos €453 mil"], 1),
    row(["Pneumologia (40/40/20)", "23,95", "27,85", "menos €195 mil"], 0),
    hiRow(["Dataset ICBHI (62/25/13)", "15,48", "17,77", "menos €114 mil"])],
    tbl({ x: 0.5, y: 1.45, w: 9.0, colW: [3.6, 1.6, 1.6, 2.2], rowH: 0.42 }));
  iconItem(s, 0.5, 4.0, 4.4, "FiDatabase", "COPD falhado: €2.950", "Qualquer outro erro: €70", "dark");
  iconItem(s, 5.2, 4.0, 4.5, "FiXCircle", "Não evita internamentos", "Só troca URTI por saudáveis em consulta", "dark");
  s.addNotes("50.000 pacientes por ano; safe θ=0,10 contra argmax; IC95% em reports/AVALIACAO_V3.md. As prevalências são hipóteses. Falta comparar com a prática sem sistema e incluir certificação (MDR Classe IIa, ISO 13485) para um VAL.");

  // 11 protótipo
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Gestão" });
  s.addText("Por que lhe chamamos protótipo", { placeholder: "title" });
  const lims = [["FiDatabase", "Poucos áudios", "14 URTI, 26 saudáveis"], ["FiRadio", "Um só equipamento", "para saudáveis e URTI"], ["FiXCircle", "Sem teste independente", "todos os dados escolheram o modelo"],
                ["FiUsers", "Pediatria", "accuracy 0,45 abaixo dos 6 anos"], ["FiLayers", "Cobertura", "sem pneumonia nem asma"], ["FiShield", "Não é dispositivo médico", "o clínico decide sempre"]];
  lims.forEach(([ic, t, d], i) => iconItem(s, 0.5 + (i % 2) * 4.7, 1.4 + Math.floor(i / 2) * 1.3, 4.4, ic, t, d, i % 2 ? "light" : "dark"));

  // 12 fecho
  pres.addSection({ title: "Fecho" });
  s = pres.addSlide({ masterName: "COVER", sectionTitle: "Fecho" });
  s.addText("Um protótipo que deteta COPD", { placeholder: "title" });
  s.addText("Ainda não distingue URTI de saudável.\nPara ir mais longe faltam mais áudios, de mais pacientes e equipamentos.", { placeholder: "body" });
  pill(s, 0.6, 0.55, 2.6, 0.45, "PRÓXIMOS PASSOS", H.accent2, H.dk1, 14);
  [["FiUsers", "Baseline só com idade"], ["FiActivity", "Features por ciclo respiratório"], ["FiGithub", "Validar num dataset externo"]].forEach(([ic, t], i) => {
    circleIcon(s, 7.0, 1.2 + i * 1.25, 0.7, ic, "light");
    T(s, t, { x: 7.85, y: 1.3 + i * 1.25, w: 2.0, h: 0.6, fontSize: 15, bold: true, color: C.background1, valign: "middle" });
  });

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("ok", OUT);
}
build();
