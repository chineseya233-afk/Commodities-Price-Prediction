const assert = require("assert");
const fs = require("fs");
const path = require("path");

const charts = require("../frontend/src/chartOptions.js");

const theme = {
  backgroundColor: "transparent",
  textStyle: { color: "#949494" },
  grid: { top: 42, right: 20, bottom: 38, left: 56, containLabel: true },
};
const colors = {
  mint: "#2bc6b2",
  mintMid: "rgba(43,198,178,0.62)",
  mintSoft: "rgba(43,198,178,0.34)",
  purple: "#6da5c0",
  amber: "#ffa502",
  white: "#ffffff",
};

function optionsBase(extra = {}) {
  return { theme, colors, emptyText: "empty", ...extra };
}

function testPriceChartSeriesStayAligned() {
  const history = [
    { date: "2026-05-20", price: 5800 },
    { date: "2026-05-21", price: 5810 },
    { date: "2026-05-22", price: 5820 },
  ];
  const predictionRows = [
    { target_date: "2026-05-25", p10: 5790, p50: 5830, p90: 5870 },
    { target_date: "2026-05-26", p10: 5800, p50: 5840, p90: 5880 },
    { target_date: "2026-05-27", p10: 5810, p50: 5850, p90: 5890 },
  ];

  const option = charts.buildPriceChartOption(optionsBase({ history, predictionRows, range: 30, tier: "all" }));
  const dates = option.xAxis.data;

  assert.deepStrictEqual(dates, [
    "2026-05-20",
    "2026-05-21",
    "2026-05-22",
    "2026-05-25",
    "2026-05-26",
    "2026-05-27",
  ]);
  for (const series of option.series) {
    if (Array.isArray(series.data) && series.type !== "custom") {
      assert.strictEqual(series.data.length, dates.length, `${series.name} length should match xAxis`);
    }
  }
  assert.strictEqual(option.series.find((s) => s.name === "精确预测").data[history.length - 1], 5820);
}

function testPriceChartIntervalUsesNativeEchartsSeries() {
  const predictionRows = [
    { target_date: "2026-05-23", p10: 5790, p50: 5830, p90: 5870 },
    { target_date: "2026-05-24", p10: 5800, p50: 5840, p90: 5880 },
  ];
  const option = charts.buildPriceChartOption(optionsBase({
    history: [{ date: "2026-05-22", price: 5820 }],
    predictionRows,
    range: 30,
    tier: "all",
  }));
  const bandSeries = option.series.find((series) => series.name === "预测区间");
  const lowerSeries = option.series.find((series) => series.name === "_预测区间下界");

  assert.ok(bandSeries, "interval band should be an ECharts series, not a DOM overlay");
  assert.ok(lowerSeries, "interval band needs an invisible lower-bound line");
  assert.strictEqual(bandSeries.type, "line");
  assert.strictEqual(lowerSeries.type, "line");
  assert.strictEqual(bandSeries.stack, "预测区间");
  assert.strictEqual(lowerSeries.stack, "预测区间");
  assert.ok(bandSeries.areaStyle, "upper stacked line should fill the interval area");
  assert.ok(option.legend.data.includes("预测区间"));
  assert.ok(!("intervalBandData" in option), "price chart should not expose DOM overlay interval metadata");
  assert.ok(!("dataZoom" in option), "price chart should not use ECharts dataZoom with a detached overlay");
  assert.strictEqual(lowerSeries.data[1], 5790);
  assert.strictEqual(bandSeries.data[1].value, 80);
  assert.strictEqual(bandSeries.data[1].intervalLower, 5790);
  assert.strictEqual(bandSeries.data[1].intervalUpper, 5870);

  const widerHistoryOption = charts.buildPriceChartOption(optionsBase({
    history: [
      { date: "2026-05-20", price: 5800 },
      { date: "2026-05-21", price: 5810 },
      { date: "2026-05-22", price: 5820 },
    ],
    predictionRows,
    range: 30,
    tier: "all",
  }));
  const widerBand = widerHistoryOption.series.find((series) => series.name === "预测区间");
  const widerLower = widerHistoryOption.series.find((series) => series.name === "_预测区间下界");
  assert.strictEqual(widerLower.data.length, widerHistoryOption.xAxis.data.length);
  assert.strictEqual(widerBand.data.length, widerHistoryOption.xAxis.data.length);
  assert.strictEqual(widerLower.data[2], 5820, "interval band should start from the last historical point");
  assert.strictEqual(widerLower.data[3], 5790, "first forecast interval should keep its x-axis position");
}

function testPriceChartTooltipShowsDateAndValues() {
  const option = charts.buildPriceChartOption(optionsBase({
    history: [{ date: "2026-05-22", price: 5820 }],
    predictionRows: [
      { target_date: "2026-05-23", p10: 5790, p50: 5830, p90: 5870 },
      { target_date: "2026-05-24", p10: 5800, p50: 5840, p90: 5880 },
    ],
    range: 30,
    tier: "all",
    today: "2026-05-24",
  }));

  const html = option.tooltip.formatter([
    { axisValue: "2026-05-25", seriesName: "精确预测", value: 5830, color: "#2bc6b2" },
    { axisValue: "2026-05-25", seriesName: "_lower", value: 5790, color: "transparent" },
    {
      axisValue: "2026-05-25",
      seriesName: "预测区间",
      value: { value: 5830, intervalLower: 5790, intervalUpper: 5870 },
      dataIndex: 1,
      color: "#2bc6b2",
    },
  ]);

  assert.ok(html.includes("2026-05-25"));
  assert.ok(html.includes("精确预测"));
  assert.ok(html.includes("5830"));
  assert.ok(html.includes("预测区间"));
  assert.ok(html.includes("5790-5870"));
  assert.ok(html.includes("5870"));
  assert.ok(!html.includes("_lower"));
  assert.strictEqual(option.tooltip.axisPointer.type, "none");
}

function testPriceChartExposesForecastBoundaryMarkLine() {
  const option = charts.buildPriceChartOption(optionsBase({
    history: [
      { date: "2026-05-21", price: 5810 },
      { date: "2026-05-22", price: 5820 },
    ],
    predictionRows: [
      { target_date: "2026-05-23", p10: 5790, p50: 5830, p90: 5870 },
      { target_date: "2026-05-24", p10: 5800, p50: 5840, p90: 5880 },
    ],
    range: 30,
    tier: "all",
    today: "2026-05-24",
  }));

  const historySeries = option.series.find((series) => series.name === "历史价格");
  const markerSeries = option.series.find((series) => series.name === "_日期标记");
  assert.ok(!("forecastBoundaryRatio" in option));
  assert.ok(!("forecastBoundaryIndex" in option));
  assert.ok(!("markLine" in historySeries));
  assert.ok(markerSeries);
  assert.strictEqual(markerSeries.data[0].value[0], "2026-05-23");
  assert.strictEqual(markerSeries.data[0].value[6], "预测起点");
  assert.strictEqual(markerSeries.data[1].value[0], "2026-05-24");
  assert.strictEqual(markerSeries.data[1].value[6], "今日");
  assert.strictEqual(markerSeries.data[0].value[2], 0);
  assert.strictEqual(markerSeries.data[1].value[2], 0);
}

function testPriceChartOffsetsTodayAndForecastStartWhenSameDay() {
  const option = charts.buildPriceChartOption(optionsBase({
    history: [
      { date: "2026-05-21", price: 5810 },
      { date: "2026-05-22", price: 5820 },
    ],
    predictionRows: [
      { target_date: "2026-05-23", p10: 5790, p50: 5830, p90: 5870 },
      { target_date: "2026-05-24", p10: 5800, p50: 5840, p90: 5880 },
    ],
    range: 30,
    tier: "all",
    today: "2026-05-23",
  }));

  const markerSeries = option.series.find((series) => series.name === "_日期标记");
  assert.ok(markerSeries);
  assert.strictEqual(markerSeries.data[0].value[0], "2026-05-23");
  assert.strictEqual(markerSeries.data[0].value[6], "预测起点");
  assert.strictEqual(markerSeries.data[1].value[0], "2026-05-23");
  assert.strictEqual(markerSeries.data[1].value[6], "今日");
  assert.notStrictEqual(markerSeries.data[0].value[2], markerSeries.data[1].value[2]);
  assert.ok(!JSON.stringify(option).includes("今日 / 预测起点"));
}

function testCompactChartsHaveEnoughGridSpace() {
  const backtest = [
    { dates: ["2026-05-21", "2026-05-22"], actual: [5810, 5820], predicted: [5805, 5835] },
  ];
  const bt = charts.buildBacktestChartOption(optionsBase({ backtestResults: backtest }));
  assert.ok(bt.grid.left >= 64);
  assert.ok(bt.grid.right >= 32);
  assert.strictEqual(bt.series[0].data.length, 2);
  assert.strictEqual(bt.series[1].data.length, 2);

  const model = charts.buildModelChartOption(optionsBase({
    modelMetrics: [
      { model_name: "prophet", mape: 1.1, directional_accuracy: 57, coverage_rate: 71 },
      { model_name: "xgboost", mape: 2.2, directional_accuracy: 43, coverage_rate: 14 },
    ],
  }));
  assert.ok(model.grid.left >= 64);
  assert.ok(model.grid.right >= 54);
  assert.strictEqual(model.series.length, 3);
}

function testModelChartTooltipUsesPercentUnits() {
  const model = charts.buildModelChartOption(optionsBase({
    modelMetrics: [
      { model_name: "prophet", mape: 1.1, directional_accuracy: 57, coverage_rate: 71 },
    ],
  }));

  assert.strictEqual(model.tooltipUnit, "%");
  assert.strictEqual(typeof model.tooltip.formatter, "function");
  const html = model.tooltip.formatter([
    { axisValue: "prophet", seriesName: "价格偏差率(%)", value: 1.1, color: "#2bc6b2" },
    { axisValue: "prophet", seriesName: "方向准确率(%)", value: 57, color: "#6da5c0" },
  ]);
  assert.ok(html.includes("prophet"));
  assert.ok(html.includes("1.10%"));
  assert.ok(html.includes("57.00%"));
  assert.ok(!html.includes("RMB/吨"));
}

function testModelChartUsesPriceAccuracyInsteadOfDeviation() {
  const model = charts.buildModelChartOption(optionsBase({
    modelMetrics: [
      { model_name: "prophet", mape: 1.1, price_accuracy: 98.9, directional_accuracy: 67, coverage_rate: 71 },
    ],
  }));

  assert.ok(model.legend.data.includes("价格准确率(%)"));
  assert.ok(!model.legend.data.includes("价格偏差率(%)"));
  assert.strictEqual(model.series[0].name, "价格准确率(%)");
  assert.strictEqual(model.series[0].data[0], 98.9);
  assert.ok(model.yAxis.every((axis) => axis.min === 0 && axis.max === 100));
}

function testModelChartTreatsFlatNaiveDirectionAsNotApplicable() {
  const model = charts.buildModelChartOption(optionsBase({
    modelMetrics: [
      { model_name: "naive", mape: 8.1, price_accuracy: 91.9, directional_accuracy: 0, directional_accuracy_applicable: false, coverage_rate: 50 },
      { model_name: "prophet", mape: 1.1, price_accuracy: 98.9, directional_accuracy: 67, coverage_rate: 71 },
    ],
  }));

  const directionSeries = model.series.find((series) => series.name === "方向准确率(%)");
  assert.strictEqual(directionSeries.data[0], null);
  assert.strictEqual(directionSeries.data[1], 67);
  const html = model.tooltip.formatter([
    { axisValue: "naive", seriesName: "方向准确率(%)", value: null, color: "#6da5c0" },
  ]);
  assert.ok(!html.includes("0.00%"));
}

function testMainDoesNotInitializeHiddenViewCharts() {
  const main = fs.readFileSync(path.join(__dirname, "../frontend/src/main.js"), "utf8");
  assert.ok(main.includes('v-if="state.view === \'dashboard\'"'));
  assert.ok(main.includes('v-else-if="state.view === \'procurement\'"'));
  assert.ok(!main.includes('v-show="state.view === \'procurement\'"'));
  assert.ok(main.includes("采购动作看板"));
  assert.ok(main.includes("固定验证"));
  assert.ok(main.includes("tooltipUnit, tooltipDigits, ...echartOption"));
  assert.ok(main.includes("cloneChartOption(echartOption)"));
  assert.ok(main.includes("const raw = toRaw(value);"));
  assert.ok(main.includes("function hasRenderableOption"));
  assert.ok(main.includes("!hasRenderableOption(props.option)"));
  assert.ok(main.includes("function scheduleApplyOption"));
  assert.ok(main.includes("cancelAnimationFrame(applyFrame)"));
  assert.ok(main.includes("el.value?.clientWidth > 0"));
  assert.ok(main.includes("el.value?.clientHeight > 0"));
  assert.ok(!main.includes("chart.value.clear();"));
  assert.ok(main.includes("notMerge: true"));
  assert.ok(main.includes("function initChart()"));
  assert.ok(main.includes("requestAnimationFrame(initChart)"));
  assert.ok(main.includes("requestAnimationFrame(() => {"));
  assert.ok(!main.includes("{ deep: true }"));
  assert.ok(main.includes("chart-tooltip"));
  assert.ok(main.includes("plotWidth"));
  assert.ok(main.includes("tooltipUnit"));
  assert.ok(!main.includes("boundaryState"));
  assert.ok(!main.includes("bandState"));
  assert.ok(!main.includes("intervalBandData"));
  assert.ok(!main.includes("interval-band-overlay"));
  assert.ok(!main.includes("forecast-boundary"));
  assert.ok(!main.includes("updateIntervalBand"));
  assert.ok(!main.includes("axisPixel"));
  assert.ok(main.includes("tooltipUnit, tooltipDigits, ...echartOption"));
  assert.ok(!main.includes('chart.value.on("finished"'));
  assert.ok(!main.includes('chart.value.on("dataZoom"'));
  assert.ok(!main.includes("window.dispatchEvent(new Event(\"resize\"))"));
}

function testFrontendAuthDoesNotExposeSecretsOrPersistPasswords() {
  const main = fs.readFileSync(path.join(__dirname, "../frontend/src/main.js"), "utf8");
  const index = fs.readFileSync(path.join(__dirname, "../frontend/index.html"), "utf8");
  const appJsPath = path.join(__dirname, "../frontend/app.js");
  const legacyApp = fs.existsSync(appJsPath) ? fs.readFileSync(appJsPath, "utf8") : "";

  assert.ok(main.includes("function restoreSession"));
  assert.ok(main.includes("/auth/session"));
  assert.ok(main.includes('credentials: "same-origin"'));
  assert.ok(main.includes("remember14Days"));
  assert.ok(main.includes("rememberPassword"));
  assert.ok(main.includes("/auth/logout"));
  assert.ok(!main.includes("access_token"));
  assert.ok(!main.includes("Authorization"));
  assert.ok(!main.includes("localStorage.setItem(\"token\""));
  assert.ok(!main.includes("localStorage.setItem(\"password\""));
  assert.ok(!main.includes("localStorage.setItem('password'"));
  assert.ok(!main.includes("sessionStorage"));
  assert.ok(!main.includes("password: \"exec123\""));
  assert.ok(!main.includes("admin:admin123"));
  assert.ok(!main.includes("executive:exec123"));
  assert.ok(!main.includes("procurement:proc123"));
  assert.ok(!legacyApp.includes("admin123"));
  assert.ok(!legacyApp.includes("exec123"));
  assert.ok(!legacyApp.includes("proc123"));
  assert.ok(/\/src\/styles\.css\?v=\d{8}[a-z]?/.test(index));
  assert.ok(/\/src\/chartOptions\.js\?v=\d{8}[a-z]?/.test(index));
  assert.ok(/\/src\/main\.js\?v=\d{8}[a-z]?/.test(index));
}

function testSessionTimeoutRequiresLeavingPageInsteadOfIdleActivePage() {
  const main = fs.readFileSync(path.join(__dirname, "../frontend/src/main.js"), "utf8");

  assert.ok(main.includes("PAGE_AWAY_TIMEOUT_MS"));
  assert.ok(main.includes("PAGE_AWAY_STORAGE_KEY"));
  assert.ok(main.includes("markPageLeft"));
  assert.ok(main.includes("pageAwayTimedOut"));
  assert.ok(main.includes("visibilitychange"));
  assert.ok(main.includes("pagehide"));
  assert.ok(main.includes("beforeunload"));
  assert.ok(main.includes("离开页面超过30分钟，请重新登录"));
  assert.ok(!main.includes("scheduleIdleLogout"));
  assert.ok(!main.includes("idleTimer"));
  assert.ok(main.includes('const ACTIVITY_EVENTS = ["scroll"];'));
  assert.ok(!main.includes("长时间未操作，请重新登录"));
}

function testNavigationLoadingSidebarAndBackToTopUiExist() {
  const main = fs.readFileSync(path.join(__dirname, "../frontend/src/main.js"), "utf8");
  const css = fs.readFileSync(path.join(__dirname, "../frontend/src/styles.css"), "utf8");

  assert.ok(main.includes("authStage"));
  assert.ok(main.includes("loading-screen"));
  assert.ok(main.includes("loading-spinner"));
  assert.ok(main.includes("loading-dashboard-shell"));
  assert.ok(main.includes("state.loading && !state.data"));
  assert.ok(!main.includes("正在恢复安全会话"));
  const sessionStart = main.indexOf("async function startAuthenticatedSession");
  const appStage = main.indexOf('state.authStage = "app"', sessionStart);
  const firstDashboardLoad = main.indexOf("await Promise.all", sessionStart);
  assert.ok(appStage > sessionStart && appStage < firstDashboardLoad);
  assert.ok(main.includes("sidebarOpen"));
  assert.ok(main.includes("commodity-sidebar"));
  assert.ok(main.includes("hamburger-btn"));
  assert.ok(main.includes("commodities"));
  assert.ok(main.includes("0# 柴油"));
  assert.ok(main.includes("showBackTop"));
  assert.ok(main.includes("scrollToTop"));
  assert.ok(main.includes("back-top-btn"));
  assert.ok(main.includes("window.scrollTo({ top: 0, behavior: \"smooth\" })"));
  assert.ok(css.includes(".back-top-btn"));
  assert.ok(css.includes("right: calc(24px + 56px + 16px);"));
  assert.ok(css.includes("bottom: 62px;"));
  assert.ok(css.includes("width: 56px;"));
  assert.ok(css.includes("height: 56px;"));
  assert.ok(css.includes(".loading-dashboard-shell"));
  assert.ok(css.includes(".commodity-sidebar"));
  assert.ok(css.includes(".hamburger-btn"));
  assert.ok(css.includes(".loading-spinner"));
}

function testProcurementTriggersUseExistingForecastData() {
  const triggers = charts.buildProcurementTriggers({
    currentPrice: 5800,
    predictionRows: [
      { p10: 5750, p50: 5790, p90: 5860 },
      { p10: 5740, p50: 5780, p90: 5870 },
    ],
    advice: { confidence: "中" },
    newsSentiment: { price_adjustment_pct: 0.003 },
  });

  assert.strictEqual(triggers.length, 4);
  assert.ok(triggers[0].value.includes("5740"));
  assert.ok(triggers[2].value.includes("5870"));
  assert.ok(triggers[3].value.includes("+0.30%"));
}

function testPriceSummaryCardsUseFourCardsAndSeparateNewsAdjustment() {
  const rows = Array.from({ length: 30 }, (_, index) => ({
    p50: index < 7 ? 6291 : 6222,
  }));

  const cards = charts.buildPriceSummaryCards({
    currentPrice: 6318,
    currentChange: 8,
    predictionRows: rows,
    modelMetrics: { price_accuracy: 99.65, coverage_rate: 88 },
    newsSentiment: { price_adjustment_pct: 0.0023 },
  });

  assert.strictEqual(cards.length, 4);
  assert.deepStrictEqual(cards.map((card) => card.label), ["实时价格", "7天预测均价", "30天预测均价", "新闻调优"]);
  assert.strictEqual(cards[1].accuracy, "准确率 99.7%");
  assert.strictEqual(cards[2].accuracy, "准确率 99.7%");
  assert.ok(cards[3].detail.includes("+0.23%"));
  assert.ok(!JSON.stringify(cards).includes("覆盖率"));
}

function testDashboardRefreshAndSingleScreenControlsExist() {
  const main = fs.readFileSync(path.join(__dirname, "../frontend/src/main.js"), "utf8");
  const css = fs.readFileSync(path.join(__dirname, "../frontend/src/styles.css"), "utf8");

  assert.ok(main.includes("dashboardSlide"));
  assert.ok(main.includes("loadDashboard(true)"));
  assert.ok(main.includes('/dashboard/summary?refresh=1'));
  assert.ok(main.includes("refreshTimer = setInterval(() => loadDashboard(true), 60 * 1000)"));
  assert.ok(main.includes("price-live-tile"));
  assert.ok(main.includes("forecast-accuracy-badge"));
  assert.ok(main.includes("dashboard-carousel"));
  assert.ok(main.includes("dashboard-slide-track"));
  assert.ok(main.includes("dashboard-slide-page"));
  assert.ok(main.includes("--dashboard-slide"));
  assert.ok(main.includes("slide-dot"));
  assert.ok(main.includes("dashboard-slide-next"));
  assert.ok(main.includes("procurement-live-card"));
  assert.ok(main.includes("purchasePlan"));
  assert.ok(!main.includes("class=\"kpi-row\""));
  assert.ok(css.includes(".dashboard-single-screen"));
  assert.ok(css.includes(".price-live-tile"));
  assert.ok(css.includes("grid-template-columns: repeat(4, minmax(0, 1fr));"));
  assert.ok(css.includes(".dashboard-carousel"));
  assert.ok(css.includes(".dashboard-slide-track"));
  assert.ok(css.includes("transition: transform"));
  assert.ok(css.includes(".analysis-slide-panel"));
  assert.ok(css.includes("overflow: auto"));
  assert.ok(css.includes("--dashboard-control-gutter: 72px;"));
  assert.ok(css.includes(".dashboard-single-screen .market-brief"));
  assert.ok(css.includes("margin-right: var(--dashboard-control-gutter);"));
  assert.ok(css.includes("right: -62px;"));
  assert.ok(!css.includes("right: -24px;"));
  assert.ok(!css.includes("right: -10px;"));
  assert.ok(css.includes(".dashboard-slide-next"));
  assert.ok(css.includes(".procurement-live-card"));
  assert.ok(!css.includes(".price-live-tile::after"));
  assert.ok(!css.includes(".procurement-live-price::after"));
  assert.ok(!css.includes("liveScan"));
}

function testChatFabIsDraggableAndViewportPositioned() {
  const main = fs.readFileSync(path.join(__dirname, "../frontend/src/main.js"), "utf8");
  const css = fs.readFileSync(path.join(__dirname, "../frontend/src/styles.css"), "utf8");

  assert.ok(main.includes("CHAT_FAB_STORAGE_KEY"));
  assert.ok(main.includes("commodity_chat_fab_position"));
  assert.ok(main.includes("chatFabPosition"));
  assert.ok(main.includes("chatFabStyle"));
  assert.ok(main.includes("chatPanelStyle"));
  assert.ok(main.includes("placeLeft"));
  assert.ok(main.includes("onChatFabPointerDown"));
  assert.ok(main.includes("@pointermove=\"onChatFabPointerMove\""));
  assert.ok(main.includes("@pointerup=\"onChatFabPointerUp\""));
  assert.ok(main.includes("@click=\"onChatFabClick\""));
  assert.ok(main.includes("v-if=\"state.view === 'procurement' && state.chatFabReady\""));
  assert.ok(main.includes("v-if=\"state.view === 'procurement' && state.chatOpen && state.chatFabReady\""));
  assert.ok(main.includes('if (view !== "procurement") state.chatOpen = false;'));
  assert.ok(main.includes("state.chatSuppressClick"));
  assert.ok(css.includes(".chat-fab.dragging"));
  assert.ok(css.includes("touch-action: none;"));
  assert.ok(css.includes("z-index: 45;"));
  assert.ok(css.includes(".chat-panel"));
  assert.ok(css.includes("z-index: 46;"));
  assert.ok(!main.includes("@click=\"state.chatOpen = !state.chatOpen\""));
}

function testDisplayTextNormalizesVisibleEnglishWithoutCorruptingTokens() {
  const main = fs.readFileSync(path.join(__dirname, "../frontend/src/main.js"), "utf8");

  assert.ok(main.includes("function displayText"));
  assert.ok(main.includes("function riskTitleText"));
  assert.ok(main.includes("isEnglishDominantText"));
  assert.ok(main.includes("RMB/吨"));
  assert.ok(main.includes("medium: \"中\""));
  assert.ok(main.includes("watch: \"关注\""));
  assert.ok(main.includes("Forecast uncertainty remains material"));
  assert.ok(main.includes("预测不确定性仍然较高"));
  assert.ok(main.includes("Policy adjustment window"));
  assert.ok(main.includes("政策调整窗口"));
  assert.ok(main.includes("一级校验 FAILED"));
  assert.ok(!main.includes('replaceAll("AI"'));
  assert.ok(main.includes("displayText(advice.confidence || '中')"));
  assert.ok(main.includes("displayText(advice.suggested_price_range || 'N/A')"));
  assert.ok(main.includes("displayText(advice.timing || 'N/A')"));
  assert.ok(main.includes("市场消息风险"));
  assert.ok(main.includes("riskTitleText(item, dim.title)"));
}

function testFrontendVisualLanguageIsLessNeonAndAiBranded() {
  const main = fs.readFileSync(path.join(__dirname, "../frontend/src/main.js"), "utf8");
  const css = fs.readFileSync(path.join(__dirname, "../frontend/src/styles.css"), "utf8");
  const chartsSource = fs.readFileSync(path.join(__dirname, "../frontend/src/chartOptions.js"), "utf8");

  assert.ok(main.includes("策略研判"));
  assert.ok(main.includes("采购问答"));
  assert.ok(main.includes("模型服务"));
  assert.ok(main.includes("function actionLabel"));
  assert.ok(main.includes("hold_or_stage: \"观望分批\""));
  assert.ok(main.includes("function switchView"));
  assert.ok(main.includes("nextTick(scrollToTop)"));
  assert.ok(main.includes("LLM(?=$|[^A-Za-z])"));
  assert.ok(!main.includes('replaceAll("AI"'));
  assert.ok(css.includes("@media (prefers-reduced-motion: reduce)"));
  assert.ok(!main.includes("AI 策略建议"));
  assert.ok(!main.includes("AI 采购助手"));
  assert.ok(!main.includes("LLM:"));
  assert.ok(!chartsSource.includes("AI情绪修正"));
  assert.ok(!css.includes("pulseGlow"));
  assert.ok(!css.includes("#3cffd0"));
  assert.ok(!css.includes("#5200ff"));
}

testPriceChartSeriesStayAligned();
testPriceChartIntervalUsesNativeEchartsSeries();
testPriceChartTooltipShowsDateAndValues();
testPriceChartExposesForecastBoundaryMarkLine();
testPriceChartOffsetsTodayAndForecastStartWhenSameDay();
testCompactChartsHaveEnoughGridSpace();
testModelChartTooltipUsesPercentUnits();
testModelChartUsesPriceAccuracyInsteadOfDeviation();
testModelChartTreatsFlatNaiveDirectionAsNotApplicable();
testMainDoesNotInitializeHiddenViewCharts();
testFrontendAuthDoesNotExposeSecretsOrPersistPasswords();
testSessionTimeoutRequiresLeavingPageInsteadOfIdleActivePage();
testNavigationLoadingSidebarAndBackToTopUiExist();
testProcurementTriggersUseExistingForecastData();
testPriceSummaryCardsUseFourCardsAndSeparateNewsAdjustment();
testDashboardRefreshAndSingleScreenControlsExist();
testChatFabIsDraggableAndViewportPositioned();
testDisplayTextNormalizesVisibleEnglishWithoutCorruptingTokens();
testFrontendVisualLanguageIsLessNeonAndAiBranded();
