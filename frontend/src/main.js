const { createApp, ref, reactive, computed, watch, onMounted, onBeforeUnmount, nextTick, toRaw } = Vue;

const API = "/api";
const CHART_THEME = {
  backgroundColor: "transparent",
  textStyle: { fontFamily: "'Space Grotesk', sans-serif", color: "#949494" },
  grid: { top: 42, right: 20, bottom: 38, left: 56, containLabel: true },
};
const COLORS = {
  mint: "#2bc6b2",
  mintMid: "rgba(43,198,178,0.62)",
  mintSoft: "rgba(43,198,178,0.34)",
  purple: "#6da5c0",
  amber: "#ffa502",
  white: "#ffffff",
  red: "#ff4757",
  canvas: "#0f1111",
};
const DashboardCharts = window.DashboardCharts;
const PAGE_AWAY_TIMEOUT_MS = 30 * 60 * 1000;
const ACTIVITY_EVENTS = ["scroll"];
const USERNAME_STORAGE_KEY = "commodity_forecast_username";
const PAGE_AWAY_STORAGE_KEY = "commodity_forecast_last_left_at";
const CHAT_FAB_STORAGE_KEY = "commodity_chat_fab_position";
const CHAT_FAB_SIZE = 56;
const CHAT_FAB_MARGIN = 16;
const CHAT_PANEL_GAP = 14;

function asNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function formatNumber(value, digits = 0) {
  return asNumber(value).toFixed(digits);
}

function pickPrediction(predictions) {
  if (!predictions || typeof predictions !== "object") return null;
  return predictions.ensemble || Object.values(predictions).find((item) => item?.predictions?.length) || null;
}

function uniquePredictionRows(rows) {
  const seen = new Set();
  return (Array.isArray(rows) ? rows : []).filter((row, index) => {
    const date = row?.target_date || row?.date || `row-${index}`;
    if (seen.has(date)) return false;
    seen.add(date);
    return Number.isFinite(asNumber(row?.p50, NaN));
  });
}

function tierLabel(index) {
  if (index < 7) return "精确";
  if (index < 14) return "标准";
  return "趋势";
}

function tierClass(index) {
  if (index < 7) return "tier-precise";
  if (index < 14) return "tier-standard";
  return "tier-fuzzy";
}

function emptyChartOption(text) {
  return {
    ...CHART_THEME,
    title: { text, left: "center", top: "center", textStyle: { color: "#949494", fontSize: 13 } },
  };
}

function cloneChartOption(value) {
  const raw = toRaw(value);
  if (Array.isArray(raw)) return raw.map((item) => cloneChartOption(item));
  if (raw && typeof raw === "object") {
    return Object.fromEntries(Object.entries(raw).map(([key, item]) => [key, cloneChartOption(item)]));
  }
  return raw;
}

function hasRenderableOption(option) {
  const raw = toRaw(option);
  if (!raw || typeof raw !== "object") return false;
  if (raw.title || raw.xAxis || raw.yAxis) return true;
  return Array.isArray(raw.series) && raw.series.length > 0;
}

const ChartBox = {
  props: { option: Object, height: { type: String, default: "320px" } },
  template: `
    <div class="chart-shell" :style="{height}">
      <div ref="el" class="chart-box"></div>
      <div
        v-if="tooltipState.show"
        class="chart-tooltip"
        :style="{ left: tooltipState.left + 'px', top: tooltipState.top + 'px' }"
      >
        <div class="chart-tooltip-date" v-text="tooltipState.title"></div>
        <div v-for="row in tooltipState.rows" :key="row.name" class="chart-tooltip-row">
          <span :style="{ background: row.color }"></span>
          <em v-text="row.name"></em>
          <strong v-text="row.value"></strong>
        </div>
      </div>
    </div>
  `,
  setup(props) {
    const el = ref(null);
    const chart = ref(null);
    const tooltipState = reactive({ show: false, left: 0, top: 0, title: "", rows: [] });
    let resizeHandler = null;
    let moveHandler = null;
    let outHandler = null;
    let initFrame = null;
    let applyFrame = null;

    function applyOption() {
      if (!chart.value || !hasRenderableOption(props.option)) return;
      const { tooltipUnit, tooltipDigits, ...echartOption } = props.option;
      chart.value.setOption(cloneChartOption(echartOption), { notMerge: true, lazyUpdate: false });
    }

    function scheduleApplyOption() {
      if (applyFrame != null) cancelAnimationFrame(applyFrame);
      applyFrame = requestAnimationFrame(() => {
        applyFrame = null;
        applyOption();
      });
    }

    function formatTooltipValue(raw) {
      if (raw == null) return null;
      const unit = props.option?.tooltipUnit || "RMB/吨";
      const digits = asNumber(props.option?.tooltipDigits, 0);
      const suffix = unit === "%" ? "%" : ` ${unit}`;
      if (Array.isArray(raw)) {
        const lower = Number(raw[1]);
        const upper = Number(raw[2]);
        if (Number.isFinite(lower) && Number.isFinite(upper)) {
          return `${lower.toFixed(digits)}-${upper.toFixed(digits)}${suffix}`;
        }
      }
      if (typeof raw === "object") {
        const lower = Number(raw.intervalLower);
        const upper = Number(raw.intervalUpper);
        if (Number.isFinite(lower) && Number.isFinite(upper)) {
          return `${lower.toFixed(digits)}-${upper.toFixed(digits)}${suffix}`;
        }
        if (Number.isFinite(Number(raw.intervalUpper))) return `${Number(raw.intervalUpper).toFixed(digits)}${suffix}`;
        if (Number.isFinite(Number(raw.value))) return `${Number(raw.value).toFixed(digits)}${suffix}`;
      }
      const value = Number(raw);
      return Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : null;
    }

    function updateTooltip(event) {
      if (!chart.value || !props.option?.xAxis?.data?.length) return;
      const grid = props.option.grid || {};
      const left = asNumber(grid.left, 56);
      const right = asNumber(grid.right, 20);
      const plotWidth = Math.max(chart.value.getWidth() - left - right, 1);
      const ratio = (event.offsetX - left) / plotWidth;
      const index = Math.max(0, Math.min(
        props.option.xAxis.data.length - 1,
        Math.round(ratio * (props.option.xAxis.data.length - 1)),
      ));
      if (!Number.isFinite(index)) return;
      const rows = (props.option.series || [])
        .filter((series) => !String(series.name || "").startsWith("_"))
        .map((series) => {
          const rawValue = series.type === "custom"
            ? series.data?.find((point) => Array.isArray(point) && Number(point[0]) === index)
            : series.data?.[index];
          const displayValue = formatTooltipValue(rawValue);
          return displayValue == null ? null : {
            name: series.name,
            value: displayValue,
            color: series.lineStyle?.color || series.itemStyle?.color || COLORS.mint,
          };
        })
        .filter(Boolean);
      if (!rows.length) {
        tooltipState.show = false;
        return;
      }
      tooltipState.show = true;
      tooltipState.title = props.option.xAxis.data[index];
      tooltipState.rows = rows;
      tooltipState.left = Math.min(event.offsetX + 12, Math.max(12, el.value.clientWidth - 180));
      tooltipState.top = Math.max(12, event.offsetY - 18);
    }

    function initChart() {
      if (!el.value || chart.value) return;
      chart.value = echarts.init(el.value);
      resizeHandler = () => {
        if (
          chart.value &&
          !chart.value.isDisposed() &&
          hasRenderableOption(props.option) &&
          el.value?.clientWidth > 0 &&
          el.value?.clientHeight > 0
        ) {
          try {
            chart.value.resize();
          } catch {
            scheduleApplyOption();
          }
        }
      };
      window.addEventListener("resize", resizeHandler);
      moveHandler = (event) => updateTooltip(event);
      outHandler = () => { tooltipState.show = false; };
      chart.value.getZr().on("mousemove", moveHandler);
      chart.value.getZr().on("globalout", outHandler);
      scheduleApplyOption();
    }

    onMounted(() => {
      nextTick(() => {
        initFrame = requestAnimationFrame(initChart);
      });
    });

    watch(() => props.option, () => nextTick(scheduleApplyOption));

    onBeforeUnmount(() => {
      if (initFrame != null) cancelAnimationFrame(initFrame);
      if (applyFrame != null) cancelAnimationFrame(applyFrame);
      if (resizeHandler) window.removeEventListener("resize", resizeHandler);
      if (chart.value && !chart.value.isDisposed()) {
        chart.value.getZr().off("mousemove", moveHandler);
        chart.value.getZr().off("globalout", outHandler);
      }
      if (chart.value && !chart.value.isDisposed()) chart.value.dispose();
      chart.value = null;
    });
    return { el, tooltipState };
  },
};

createApp({
  components: { ChartBox },
  setup() {
    const state = reactive({
      username: "",
      role: "",
      rememberMe: false,
      authStage: "checking",
      sidebarOpen: false,
      showBackTop: false,
      selectedCommodity: "diesel_0",
      view: "dashboard",
      dashboardSlide: 0,
      range: 90,
      tier: "all",
      data: null,
      backtest: null,
      loading: false,
      loginError: "",
      statusError: "",
      now: new Date(),
      chatOpen: false,
      chatInput: "",
      chatSending: false,
      messages: [{ who: "bot", text: "采购问答已就绪。可以咨询价格趋势、采购时机、成本影响和模型风险。" }],
      chatFabReady: false,
      chatFabPosition: { x: 0, y: 0 },
      chatDrag: { active: false, pointerId: null, startX: 0, startY: 0, originX: 0, originY: 0, moved: false },
      chatSuppressClick: false,
      viewportWidth: window.innerWidth,
      viewportHeight: window.innerHeight,
    });
    const loginForm = reactive({ username: "", password: "", rememberPassword: false, remember14Days: false });
    const commodities = [
      { id: "diesel_0", name: "0# 柴油", status: "已接入" },
    ];
    let refreshTimer = null;
    let clockTimer = null;

    const isAuthed = computed(() => Boolean(state.username && state.role));
    const selectedCommodity = computed(() => commodities.find((item) => item.id === state.selectedCommodity) || commodities[0]);
    const passwordAutocomplete = computed(() => loginForm.rememberPassword ? "current-password" : "off");
    const data = computed(() => state.data || {});
    const prediction = computed(() => pickPrediction(data.value.predictions));
    const predictionRows = computed(() => uniquePredictionRows(prediction.value?.predictions).slice(0, 30));
    const currentPrice = computed(() => asNumber(data.value.current_price?.price));
    const riskAlerts = computed(() => Array.isArray(data.value.risk_alerts) ? data.value.risk_alerts : []);
    const kpis = computed(() => Array.isArray(data.value.kpis) ? data.value.kpis : []);
    const ensembleMetrics = computed(() => {
      const rows = Array.isArray(data.value.model_metrics) ? data.value.model_metrics : [];
      return rows.find((m) => m.model_name === "ensemble") || {};
    });
    const currentChange = computed(() => asNumber(data.value.current_price?.change));

    const marketBrief = computed(() => {
      const dir = data.value.ensemble_direction || "震荡";
      const pct = asNumber(data.value.ensemble_change_pct);
      const highAlerts = riskAlerts.value.filter((item) => item.severity === "high").length;
      const level = highAlerts > 1 || Math.abs(pct) > 3 ? "警戒" : highAlerts > 0 || Math.abs(pct) > 1.5 ? "关注" : "安全";
      const text = dir === "上涨"
        ? `综合模型预测未来 7 天均价上行 ${Math.abs(pct).toFixed(1)}%，建议评估提前锁价。`
        : dir === "下跌"
          ? `综合模型预测未来 7 天均价下行 ${Math.abs(pct).toFixed(1)}%，建议等待低位分批采购。`
          : "综合模型预测未来 7 天价格震荡运行，建议维持常规采购节奏。";
      return { level, text, arrow: pct > 0.3 ? "上行" : pct < -0.3 ? "下行" : "震荡" };
    });

    const riskDimensions = computed(() => {
      const rr = data.value.risk_report || {};
      const dims = [
        ["dimension_1_market", "市场舆情风险"],
        ["dimension_2_model", "模型预测风险"],
        ["dimension_3_policy", "数据与政策风险"],
      ];
      return dims.map(([key, title]) => ({
        title,
        items: (Array.isArray(rr[key]) ? rr[key] : []).filter((item) => {
          const level = String(item.level || "").toLowerCase();
          return !["通过", "pass", "低", "low"].includes(level);
        }),
      })).filter((dim) => dim.items.length);
    });

    const analysis = computed(() => data.value.analysis_report || {});
    const advice = computed(() => analysis.value.procurement_advice || {});
    const modelMetrics = computed(() => (Array.isArray(data.value.model_metrics) ? data.value.model_metrics : []).filter((m) => m.model_name !== "ensemble"));
    const fixedSplit = computed(() => data.value.fixed_split_evaluation || {});
    const llmModelLabel = computed(() => {
      const models = data.value.llm_models || {};
      const provider = models.provider ? `${models.provider}: ` : "";
      const model = models.model || models.primary || "未配置";
      return `${provider}${model}${models.available === false ? "（不可用）" : ""}`;
    });
    const priceSummaryCards = computed(() => DashboardCharts.buildPriceSummaryCards({
      currentPrice: currentPrice.value,
      currentChange: currentChange.value,
      predictionRows: predictionRows.value,
      modelMetrics: ensembleMetrics.value,
      newsSentiment: data.value.news_sentiment || {},
    }));
    const procurementTriggers = computed(() => DashboardCharts.buildProcurementTriggers({
      currentPrice: currentPrice.value,
      predictionRows: predictionRows.value,
      advice: advice.value,
      newsSentiment: data.value.news_sentiment || {},
    }));
    const purchasePlan = computed(() => {
      const rows = predictionRows.value;
      const firstWeek = rows.slice(0, 7);
      const avg = (items) => items.reduce((sum, row) => sum + asNumber(row.p50), 0) / Math.max(items.length, 1);
      const avg7 = avg(firstWeek);
      const low = Math.min(...firstWeek.map((row) => asNumber(row.p10, Infinity)));
      const high = Math.max(...firstWeek.map((row) => asNumber(row.p90, -Infinity)));
      const triggerLow = Number.isFinite(low) ? low : currentPrice.value;
      const triggerHigh = Number.isFinite(high) ? high : currentPrice.value;
      const confidence = displayText(advice.value.confidence || "中");
      const batchText = confidence === "高" ? "50% / 30% / 20%" : confidence === "低" ? "20% / 40% / 40%" : "30% / 40% / 30%";
      const trendText = avg7 < currentPrice.value ? "预测均价低于现价，优先等待回落成交。" : "预测均价不低于现价，先锁定基础用量。";
      return [
        { step: "首批", action: `价格≤${formatNumber(triggerLow)} RMB/吨时执行${batchText.split("/")[0].trim()}采购`, note: trendText },
        { step: "二批", action: `若价格接近7天均价${formatNumber(avg7)} RMB/吨，补足${batchText.split("/")[1]?.trim() || "40%"}`, note: "按库存下限和订单刚需执行，不追高补单。" },
        { step: "尾批", action: `价格≥${formatNumber(triggerHigh)} RMB/吨暂停追单`, note: `剩余${batchText.split("/")[2]?.trim() || "30%"}等待新闻与库存信号确认。` },
      ];
    });
    const qaRows = computed(() => {
      const checks = [
        ["positive_values", "预测值正数校验"],
        ["price_bounds", "价格合理范围校验"],
        ["daily_change", "单日涨跌幅校验"],
        ["cumulative_change", "7 日累计偏差校验"],
        ["sigma_bounds", "历史波动范围校验"],
        ["interval_ordering", "预测区间排序校验"],
        ["interval_width", "预测区间宽度校验"],
      ];
      const result = Object.fromEntries(checks.map(([key]) => [key, { passed: true, failModels: [] }]));
      Object.entries(data.value.predictions || {}).filter(([name]) => name !== "ensemble").forEach(([name, pred]) => {
        (pred.qa_checks || []).forEach((check) => {
          if (result[check.check] && !check.passed) {
            result[check.check].passed = false;
            result[check.check].failModels.push({
              name,
              detail: check.detail || "",
              excluded: Boolean(pred.excluded_from_ensemble),
              repaired: Boolean(pred.qa_auto_repaired),
            });
          }
        });
      });
      return checks.map(([key, name]) => ({ key, name, ...result[key] }));
    });
    function qaStatusText(row) {
      if (row.passed) return "通过";
      const names = (row.failModels || []).map((item) => {
        const suffix = item.repaired ? "已修复" : item.excluded ? "已剔除" : "待复核";
        return `${item.name}(${suffix})`;
      }).join(",");
      return `异常 ${names}`;
    }
    function formatMetricPercent(value, applicable = true, digits = 1) {
      const numeric = Number(value);
      if (!applicable || !Number.isFinite(numeric)) return "N/A";
      return `${numeric.toFixed(digits)}%`;
    }

    const costRows = computed(() => {
      const rows = predictionRows.value;
      const volume = 100;
      const avg = (items) => items.reduce((sum, row) => sum + asNumber(row.p50), 0) / Math.max(items.length, 1);
      const avg7 = avg(rows.slice(0, 7));
      const avg30 = avg(rows);
      return [
        { label: "当前价格", value: currentPrice.value * volume, detail: `${formatNumber(currentPrice.value)} RMB/吨 x ${volume} 吨`, delta: 0 },
        { label: "7 天均价", value: avg7 * volume, detail: `${formatNumber(avg7)} RMB/吨`, delta: (avg7 - currentPrice.value) * volume },
        { label: "30 天均价", value: avg30 * volume, detail: `${formatNumber(avg30)} RMB/吨`, delta: (avg30 - currentPrice.value) * volume },
      ];
    });

    const deviationRows = computed(() => {
      const rows = [];
      (state.backtest?.backtest_results || []).slice().reverse().forEach((period) => {
        (period.dates || []).forEach((date, i) => {
          const actual = asNumber(period.actual?.[i], NaN);
          const predicted = asNumber(period.predicted?.[i], NaN);
          if (Number.isFinite(actual) && Number.isFinite(predicted)) {
            rows.push({ date, actual, predicted, dev: actual - predicted, devPct: actual ? ((actual - predicted) / actual) * 100 : 0 });
          }
        });
      });
      return rows.slice(-30);
    });

    const priceChartOption = computed(() => {
      const history = Array.isArray(data.value.price_history) ? data.value.price_history.slice(-state.range) : [];
      const preds = predictionRows.value.slice(0, state.tier === "precise" ? 7 : state.tier === "standard" ? 14 : 30);
      return DashboardCharts.buildPriceChartOption({
        history,
        predictionRows: preds,
        range: state.range,
        tier: state.tier,
        today: data.value.today,
        theme: CHART_THEME,
        colors: COLORS,
      });
    });

    const backtestChartOption = computed(() => {
      return DashboardCharts.buildBacktestChartOption({
        backtestResults: state.backtest?.backtest_results || [],
        theme: CHART_THEME,
        colors: COLORS,
      });
    });

    const modelChartOption = computed(() => {
      return DashboardCharts.buildModelChartOption({
        modelMetrics: modelMetrics.value,
        theme: CHART_THEME,
        colors: COLORS,
      });
    });

    function severityLabel(value) {
      const key = String(value || "").toLowerCase();
      if (key === "high") return "高";
      if (key === "medium") return "中";
      if (key === "low") return "低";
      return value || "关注";
    }

    function displayText(value, fallback = "") {
      const raw = String(value ?? fallback).trim();
      if (!raw) return String(fallback || "");
      const exactLabels = {
        high: "高",
        medium: "中",
        low: "低",
        watch: "关注",
        warning: "预警",
        warn: "预警",
        pass: "通过",
        passed: "通过",
        failed: "未通过",
        up: "上行",
        down: "下行",
        flat: "震荡",
        reduce: "减少采购",
        chase_less: "控制追高",
      };
      const exact = exactLabels[raw.toLowerCase()];
      if (exact) return exact;
      return raw
        .replace(/\bRMB\/ton\b/g, "RMB/吨")
        .replace(/\byuan\/ton\b/gi, "元/吨")
        .replace(/\bhigh\b/g, "高")
        .replace(/\bmedium\b/g, "中")
        .replace(/\blow\b/g, "低")
        .replace(/\bwatch\b/g, "关注")
        .replace(/\bwarning\b/g, "预警")
        .replace(/\bpass\b/g, "通过")
        .replace(/\bLayer\s*1 FAILED\b/g, "一级校验 FAILED")
        .replace(/\bLayer\s*1\b/g, "一级")
        .replace(/\bPolicy adjustment window\b/g, "政策调整窗口")
        .replace(/\bDomestic refined oil policy windows may affect procurement prices\./g, "国内成品油调价窗口可能影响采购价格。")
        .replace(/\bForecast uncertainty remains material, so procurement should be staged around the lower part of the forecast interval while preserving risk limits\./g, "预测不确定性仍然较高，建议围绕预测区间低位分批执行，并保留风险边界。")
        .replace(/\bForecast uncertainty remains material\b/g, "预测不确定性仍然较高")
        .replace(/\bReview during the next 1-7 days; consider staged execution near ([0-9.]+) RMB\/吨\./g, "未来 1-7 天复核，接近 $1 RMB/吨时分批执行。")
        .replace(/\bReview during the next 1-7 days and avoid one-time large purchases without confirmation\./g, "未来 1-7 天滚动复核，未得到确认前避免一次性大额采购。")
        .replace(/\bUse staged procurement because forecast intervals and external policy risks remain uncertain\./g, "由于预测区间和外部政策风险仍不确定，建议分批采购。")
        .replace(/\bMarket news risk\b/g, "市场消息风险")
        .replace(/\bNews impact requires monitoring\b/g, "新闻影响需要持续跟踪")
        .replace(/\bModel QA completed\b/g, "模型质量校验完成")
        .replace(/\bForecast output passed deterministic validation or used a guarded fallback\./g, "预测输出已通过确定性校验，或已使用受控兜底结果。")
        .replace(/\bOPEC\+ production policy uncertainty\b/g, "OPEC+ 产量政策不确定性")
        .replace(/\bPolicy shifts may affect upstream cost and procurement timing\./g, "产量政策变化可能影响上游成本和采购时点。")
        .replace(/\bSeasonal diesel demand support\b/g, "柴油季节性需求支撑")
        .replace(/\bRegional demand may limit near-term downside\./g, "区域需求可能限制短期下行空间。")
        .replace(/\bExternal oil prices, exchange rates, inventory and policy windows require monitoring\./g, "外盘原油、汇率、库存和政策窗口仍需持续跟踪。")
        .replace(/\bExchange-rate changes can affect imported cost and domestic pricing expectations\./g, "汇率变化可能影响进口成本和国内定价预期。")
        .replace(/\bPolicy adjustment windows can create discrete price jumps\./g, "政策调价窗口可能带来跳跃式价格变化。")
        .replace(/\bDeterministic fallback report used because LLM output was unavailable or invalid\./g, "模型服务暂不可用或输出未通过校验，当前使用确定性兜底报告。")
        .replace(/\bGenerated from model outputs and available market evidence\./g, "基于模型输出和可用市场证据生成。")
        .replace(/\bDiesel short-term forecast direction is up\./g, "柴油短期预测方向偏上行。")
        .replace(/\bDiesel short-term forecast direction is down\./g, "柴油短期预测方向偏下行。")
        .replace(/\bDiesel short-term forecast direction is flat\./g, "柴油短期预测方向震荡。")
        .replace(/(^|[^A-Za-z])LLM(?=$|[^A-Za-z])/g, "$1模型服务")
        .replace(/(^|[^A-Za-z])AI(?=$|[^A-Za-z])/g, "$1模型")
        .replaceAll("后台模型服务报告", "后台模型报告");
    }

    function isEnglishDominantText(value) {
      const text = String(value || "").trim();
      if (!text || /[\u4e00-\u9fff]/.test(text)) return false;
      const letters = (text.match(/[A-Za-z]/g) || []).length;
      return letters >= 12 && letters / Math.max(text.length, 1) > 0.45;
    }

    function riskTitleText(item, dimensionTitle = "风险项") {
      const fallback = dimensionTitle.includes("市场")
        ? "市场消息风险"
        : dimensionTitle.includes("模型")
          ? "模型校验提示"
          : "政策与数据风险";
      const raw = item?.title || item?.interpretation || "";
      return displayText(isEnglishDominantText(raw) ? fallback : raw, fallback);
    }

    function riskImpactText(item) {
      const raw = item?.impact || "";
      return displayText(isEnglishDominantText(raw) ? "该风险可能影响采购价格，建议继续跟踪。" : raw);
    }

    function actionLabel(value, fallback = "策略建议") {
      const raw = String(value || "").trim();
      const labels = {
        hold_or_stage: "观望分批",
        staged_buy: "逢低分批",
        staged_purchase: "分批采购",
        wait: "等待低位",
        hold: "持仓观望",
        buy: "执行采购",
        lock_price: "锁价采购",
      };
      return labels[raw.toLowerCase()] || displayText(raw || fallback);
    }

    function chatSafeBottom() {
      return state.viewportWidth <= 760 ? CHAT_FAB_MARGIN : 70;
    }

    function clampChatFabPosition(position) {
      const maxX = Math.max(CHAT_FAB_MARGIN, state.viewportWidth - CHAT_FAB_SIZE - CHAT_FAB_MARGIN);
      const maxY = Math.max(CHAT_FAB_MARGIN, state.viewportHeight - CHAT_FAB_SIZE - chatSafeBottom());
      return {
        x: Math.min(Math.max(Number(position?.x) || 0, CHAT_FAB_MARGIN), maxX),
        y: Math.min(Math.max(Number(position?.y) || 0, CHAT_FAB_MARGIN), maxY),
      };
    }

    function defaultChatFabPosition() {
      return clampChatFabPosition({
        x: state.viewportWidth - CHAT_FAB_SIZE - 28,
        y: Math.max(CHAT_FAB_MARGIN, Math.round(state.viewportHeight * 0.58)),
      });
    }

    function restoreChatFabPosition() {
      const stored = localStorage.getItem(CHAT_FAB_STORAGE_KEY);
      if (stored) {
        try {
          state.chatFabPosition = clampChatFabPosition(JSON.parse(stored));
          state.chatFabReady = true;
          return;
        } catch {
          localStorage.removeItem(CHAT_FAB_STORAGE_KEY);
        }
      }
      state.chatFabPosition = defaultChatFabPosition();
      state.chatFabReady = true;
    }

    function persistChatFabPosition() {
      localStorage.setItem(CHAT_FAB_STORAGE_KEY, JSON.stringify(state.chatFabPosition));
    }

    function updateViewport() {
      state.viewportWidth = window.innerWidth;
      state.viewportHeight = window.innerHeight;
      state.chatFabPosition = state.chatFabReady
        ? clampChatFabPosition(state.chatFabPosition)
        : defaultChatFabPosition();
    }

    const chatFabStyle = computed(() => ({
      left: `${state.chatFabPosition.x}px`,
      top: `${state.chatFabPosition.y}px`,
    }));

    const chatPanelStyle = computed(() => {
      const width = Math.min(400, Math.max(280, state.viewportWidth - 32));
      const height = Math.min(480, Math.max(280, state.viewportHeight - 130));
      const placeLeft = state.chatFabPosition.x >= width + CHAT_PANEL_GAP + CHAT_FAB_MARGIN;
      const leftCandidate = placeLeft
        ? state.chatFabPosition.x - width - CHAT_PANEL_GAP
        : state.chatFabPosition.x + CHAT_FAB_SIZE + CHAT_PANEL_GAP;
      const left = Math.min(
        Math.max(CHAT_FAB_MARGIN, leftCandidate),
        Math.max(CHAT_FAB_MARGIN, state.viewportWidth - width - CHAT_FAB_MARGIN),
      );
      const topCandidate = state.chatFabPosition.y + (CHAT_FAB_SIZE / 2) - (height / 2);
      const top = Math.min(
        Math.max(CHAT_FAB_MARGIN, topCandidate),
        Math.max(CHAT_FAB_MARGIN, state.viewportHeight - height - chatSafeBottom()),
      );
      return {
        left: `${left}px`,
        top: `${top}px`,
        width: `${width}px`,
        height: `${height}px`,
      };
    });

    function onChatFabPointerDown(event) {
      if (event.button != null && event.button !== 0) return;
      state.chatDrag = {
        active: true,
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        originX: state.chatFabPosition.x,
        originY: state.chatFabPosition.y,
        moved: false,
      };
      event.currentTarget?.setPointerCapture?.(event.pointerId);
    }

    function onChatFabPointerMove(event) {
      if (!state.chatDrag.active || event.pointerId !== state.chatDrag.pointerId) return;
      const dx = event.clientX - state.chatDrag.startX;
      const dy = event.clientY - state.chatDrag.startY;
      if (Math.abs(dx) + Math.abs(dy) > 6) state.chatDrag.moved = true;
      state.chatFabPosition = clampChatFabPosition({
        x: state.chatDrag.originX + dx,
        y: state.chatDrag.originY + dy,
      });
    }

    function onChatFabPointerUp(event) {
      if (!state.chatDrag.active || event.pointerId !== state.chatDrag.pointerId) return;
      state.chatSuppressClick = state.chatDrag.moved;
      state.chatDrag.active = false;
      event.currentTarget?.releasePointerCapture?.(event.pointerId);
      persistChatFabPosition();
    }

    function onChatFabPointerCancel(event) {
      if (event.pointerId !== state.chatDrag.pointerId) return;
      state.chatDrag.active = false;
      state.chatSuppressClick = false;
    }

    function onChatFabClick() {
      if (state.chatSuppressClick) {
        state.chatSuppressClick = false;
        return;
      }
      state.chatOpen = !state.chatOpen;
    }

    function clearRefreshTimer() {
      if (refreshTimer) clearInterval(refreshTimer);
      refreshTimer = null;
    }

    function startRefreshTimer() {
      clearRefreshTimer();
      refreshTimer = setInterval(() => loadDashboard(true), 60 * 1000);
    }

    function clearPageAwayMarker() {
      localStorage.removeItem(PAGE_AWAY_STORAGE_KEY);
    }

    function markPageLeft() {
      if (!isAuthed.value || state.rememberMe) return;
      localStorage.setItem(PAGE_AWAY_STORAGE_KEY, String(Date.now()));
    }

    function pageAwayTimedOut(now = Date.now()) {
      const leftAt = Number(localStorage.getItem(PAGE_AWAY_STORAGE_KEY));
      return Number.isFinite(leftAt) && leftAt > 0 && now - leftAt > PAGE_AWAY_TIMEOUT_MS;
    }

    function pageAwayTimeoutMessage() {
      return "离开页面超过30分钟，请重新登录";
    }

    function showPageAwayLoginIfNeeded() {
      if (!pageAwayTimedOut()) return false;
      clearPageAwayMarker();
      state.authStage = "login";
      state.loginError = pageAwayTimeoutMessage();
      return true;
    }

    async function enforcePageAwayTimeout() {
      if (!isAuthed.value || state.rememberMe) {
        clearPageAwayMarker();
        return false;
      }
      if (!pageAwayTimedOut()) {
        clearPageAwayMarker();
        return false;
      }
      await logout(pageAwayTimeoutMessage());
      return true;
    }

    async function handlePageVisibilityChange() {
      if (document.visibilityState === "hidden") {
        markPageLeft();
        clearRefreshTimer();
        return;
      }
      if (document.visibilityState === "visible") {
        if (await enforcePageAwayTimeout()) return;
        if (isAuthed.value) {
          startRefreshTimer();
          loadDashboard(true);
        }
        updateScrollState();
      }
    }

    function handleUserActivity() {
      updateScrollState();
    }

    function bindActivityListeners() {
      ACTIVITY_EVENTS.forEach((eventName) => {
        window.addEventListener(eventName, handleUserActivity, { passive: true });
      });
      document.addEventListener("visibilitychange", handlePageVisibilityChange);
      window.addEventListener("pagehide", markPageLeft);
      window.addEventListener("beforeunload", markPageLeft);
    }

    function unbindActivityListeners() {
      ACTIVITY_EVENTS.forEach((eventName) => {
        window.removeEventListener(eventName, handleUserActivity);
      });
      document.removeEventListener("visibilitychange", handlePageVisibilityChange);
      window.removeEventListener("pagehide", markPageLeft);
      window.removeEventListener("beforeunload", markPageLeft);
    }

    function clearTimers() {
      clearRefreshTimer();
      if (clockTimer) clearInterval(clockTimer);
      clockTimer = null;
      window.removeEventListener("resize", updateViewport);
      unbindActivityListeners();
    }

    function updateScrollState() {
      state.showBackTop = window.scrollY > 360;
    }

    function scrollToTop() {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }

    function switchView(view) {
      state.view = view;
      if (view !== "procurement") state.chatOpen = false;
      nextTick(scrollToTop);
    }

    function selectCommodity(id) {
      state.selectedCommodity = id;
      state.sidebarOpen = false;
      scrollToTop();
    }

    async function apiFetch(path, options = {}) {
      const res = await fetch(`${API}${path}`, {
        ...options,
        credentials: "same-origin",
        headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      });
      if (res.status === 401) {
        await logout("登录已过期，请重新登录");
        throw new Error("登录已过期");
      }
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `请求失败: ${res.status}`);
      }
      return res.json();
    }

    async function loadDashboard(silent = false) {
      if (!silent) state.loading = true;
      try {
        state.data = await apiFetch(silent ? "/dashboard/summary" : "/dashboard/summary?refresh=1");
        state.statusError = "";
      } catch (err) {
        state.statusError = err.message || "数据加载失败";
      } finally {
        state.loading = false;
      }
    }

    async function loadBacktest() {
      try {
        state.backtest = await apiFetch("/backtest/results");
      } catch {
        state.backtest = { backtest_results: [] };
      }
    }

    async function startAuthenticatedSession(payload) {
      state.role = payload.role || "";
      state.username = payload.username || "";
      state.rememberMe = Boolean(payload.remember_me);
      state.view = state.role === "procurement" ? "procurement" : "dashboard";
      state.authStage = "app";
      clearPageAwayMarker();
      await Promise.all([loadDashboard(), loadBacktest()]);
      clearRefreshTimer();
      startRefreshTimer();
      updateScrollState();
    }

    async function login() {
      state.loginError = "";
      state.authStage = "loading";
      try {
        const res = await fetch(`${API}/auth/login`, {
          method: "POST",
          credentials: "same-origin",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            username: loginForm.username,
            password: loginForm.password,
            remember_me: loginForm.remember14Days,
          }),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || "登录失败");
        }
        const payload = await res.json();
        if (loginForm.rememberPassword) {
          localStorage.setItem(USERNAME_STORAGE_KEY, loginForm.username);
        } else {
          localStorage.removeItem(USERNAME_STORAGE_KEY);
        }
        loginForm.password = "";
        await startAuthenticatedSession(payload);
      } catch (err) {
        state.authStage = "login";
        state.loginError = err.message || "登录失败";
      }
    }

    async function restoreSession() {
      try {
        const res = await fetch(`${API}/auth/session`, {
          method: "GET",
          credentials: "same-origin",
          headers: { "Content-Type": "application/json" },
        });
        if (!res.ok) {
          showPageAwayLoginIfNeeded();
          return;
        }
        const payload = await res.json();
        if (!payload.authenticated) {
          if (!showPageAwayLoginIfNeeded()) state.authStage = "login";
          return;
        }
        if (!payload.remember_me && pageAwayTimedOut()) {
          await logout(pageAwayTimeoutMessage());
          return;
        }
        await startAuthenticatedSession(payload);
      } catch {
        // 没有有效会话 cookie 时停留在登录页。
        if (!showPageAwayLoginIfNeeded()) state.authStage = "login";
      } finally {
        if (!isAuthed.value && state.authStage !== "app") state.authStage = "login";
      }
    }

    function clearSessionState(message = "") {
      clearRefreshTimer();
      clearPageAwayMarker();
      state.role = "";
      state.username = "";
      state.rememberMe = false;
      state.authStage = "login";
      state.data = null;
      state.backtest = null;
      state.loginError = message;
    }

    async function logout(message = "") {
      clearSessionState(message);
      try {
        await fetch(`${API}/auth/logout`, {
          method: "POST",
          credentials: "same-origin",
          headers: { "Content-Type": "application/json" },
        });
      } catch {
        // 本地状态已经清空；网络失败不应让界面保持登录。
      }
    }

    async function sendChat() {
      const question = state.chatInput.trim();
      if (!question || state.chatSending) return;
      state.chatInput = "";
      state.messages.push({ who: "user", text: question });
      state.chatSending = true;
      try {
        const answer = await apiFetch("/chat", { method: "POST", body: JSON.stringify({ question }) });
        state.messages.push({ who: "bot", text: answer.answer || "暂时无法回答" });
      } catch (err) {
        state.messages.push({ who: "bot", text: err.message || "网络错误，请稍后重试" });
      } finally {
        state.chatSending = false;
        nextTick(() => {
          const box = document.querySelector(".chat-messages");
          if (box) box.scrollTop = box.scrollHeight;
        });
      }
    }

    onMounted(() => {
      clockTimer = setInterval(() => { state.now = new Date(); }, 1000);
      updateViewport();
      restoreChatFabPosition();
      window.addEventListener("resize", updateViewport);
      const rememberedUsername = localStorage.getItem(USERNAME_STORAGE_KEY) || "";
      if (rememberedUsername) {
        loginForm.username = rememberedUsername;
        loginForm.rememberPassword = true;
      }
      bindActivityListeners();
      updateScrollState();
      restoreSession();
    });

    onBeforeUnmount(clearTimers);

    return {
      state, loginForm, commodities, selectedCommodity, passwordAutocomplete, isAuthed, data, currentPrice, currentChange, ensembleMetrics, kpis, marketBrief, riskDimensions, riskAlerts,
      analysis, advice, predictionRows, modelMetrics, qaRows, costRows, deviationRows, llmModelLabel,
      priceSummaryCards, procurementTriggers, purchasePlan, fixedSplit, priceChartOption, backtestChartOption, modelChartOption, qaStatusText,
      login, logout, sendChat, scrollToTop, switchView, selectCommodity, formatNumber, formatMetricPercent, tierLabel, tierClass, severityLabel, displayText, riskTitleText, riskImpactText, actionLabel,
      chatFabStyle, chatPanelStyle, onChatFabPointerDown, onChatFabPointerMove, onChatFabPointerUp, onChatFabPointerCancel, onChatFabClick,
    };
  },
  template: `
    <section v-if="state.authStage === 'checking' || state.authStage === 'loading'" class="loading-screen">
      <nav class="top-nav loading-nav">
        <div class="nav-left">
          <span class="hamburger-btn loading-hamburger" aria-hidden="true"><span></span><span></span><span></span></span>
          <span class="nav-wordmark">COMMODITY FORECAST</span>
        </div>
      </nav>
      <main class="view-container loading-dashboard-shell" aria-label="加载中">
        <div class="loading-dashboard-header">
          <span class="loading-spinner"></span>
          <div class="loading-lines">
            <span></span>
            <span></span>
          </div>
        </div>
        <div class="loading-grid">
          <span></span><span></span><span></span><span></span>
        </div>
      </main>
      <footer class="status-bar loading-status">
        <div class="status-left"><span class="status-dot"></span><span class="status-text">系统准备中</span></div>
        <div class="status-right"><span>数据源: 加载中</span><span>模型服务: 通用接口</span></div>
      </footer>
    </section>

    <section v-else-if="state.authStage === 'login'" class="login-screen">
      <div class="login-container">
        <div class="login-brand">
          <h1 class="brand-wordmark">COMMODITY<br>FORECAST</h1>
          <p class="brand-tagline">大宗商品采购价格预测 SaaS 系统</p>
        </div>
        <form class="login-form" @submit.prevent="login">
          <label class="form-group">
            <span class="form-label">用户名</span>
            <input v-model="loginForm.username" class="form-input" autocomplete="username" required>
          </label>
          <label class="form-group">
            <span class="form-label">密码</span>
            <input v-model="loginForm.password" class="form-input" type="password" :autocomplete="passwordAutocomplete" required>
          </label>
          <label class="login-check">
            <input v-model="loginForm.rememberPassword" type="checkbox">
            <span>记住密码（使用浏览器密码管理器）</span>
          </label>
          <label class="login-check">
            <input v-model="loginForm.remember14Days" type="checkbox">
            <span>14 天内自动登录（不保存密码）</span>
          </label>
          <button class="btn-primary" type="submit">登录</button>
          <p v-if="state.loginError" class="login-error" v-text="state.loginError"></p>
          <div class="login-hint">
            <span class="hint-label">POC 测试账号</span>
            <span class="hint-detail">默认账号：admin / executive / procurement。密码请使用后端 .env 配置或当前 POC 强密码。</span>
          </div>
        </form>
      </div>
    </section>

    <section v-else-if="state.authStage === 'app' && isAuthed" class="app-shell">
      <nav class="top-nav">
        <div class="nav-left">
          <button class="hamburger-btn" type="button" aria-label="打开商品选择" @click="state.sidebarOpen = true">
            <span></span><span></span><span></span>
          </button>
          <span class="nav-wordmark">COMMODITY FORECAST</span>
          <div class="nav-links">
            <button class="nav-link" :class="{active: state.view === 'dashboard'}" @click="switchView('dashboard')">决策大屏</button>
            <button class="nav-link" :class="{active: state.view === 'procurement'}" @click="switchView('procurement')">采购看板</button>
          </div>
        </div>
        <div class="nav-right">
          <span class="nav-role" v-text="(state.role || '').toUpperCase()"></span>
          <span class="nav-user" v-text="state.username"></span>
          <button class="btn-outline-mint" @click="logout()">退出</button>
        </div>
      </nav>

      <div v-if="state.sidebarOpen" class="sidebar-backdrop" @click="state.sidebarOpen = false"></div>
      <aside v-if="state.sidebarOpen" class="commodity-sidebar open" aria-label="商品选择">
        <div class="sidebar-header">
          <span>商品选择</span>
          <button class="sidebar-close" type="button" @click="state.sidebarOpen = false">×</button>
        </div>
        <button
          v-for="item in commodities"
          :key="item.id"
          class="commodity-item"
          :class="{active: state.selectedCommodity === item.id}"
          type="button"
          @click="selectCommodity(item.id)"
        >
          <strong v-text="item.name"></strong>
          <span v-text="item.status"></span>
        </button>
      </aside>

      <main v-if="state.loading && !state.data" class="view-container loading-dashboard-shell" aria-label="加载中">
        <div class="loading-dashboard-header">
          <span class="loading-spinner"></span>
          <div class="loading-lines">
            <span></span>
            <span></span>
          </div>
        </div>
        <div class="loading-grid">
          <span></span><span></span><span></span><span></span>
        </div>
      </main>

      <template v-else>
      <main v-if="state.view === 'dashboard'" class="view-container dashboard-single-screen">
        <section class="market-brief">
          <div class="brief-signal" :class="marketBrief.level" v-text="marketBrief.level"></div>
          <div class="brief-text" v-text="marketBrief.text"></div>
          <div class="brief-arrow" v-text="marketBrief.arrow"></div>
        </section>

        <section class="card card-feature">
          <div class="card-header">
            <div>
              <span class="card-kicker">价格趋势总览</span>
              <h2 class="card-title">{{ selectedCommodity.name }} - 历史走势与未来预测</h2>
            </div>
            <div class="segmented">
              <button v-for="r in [30,60,90]" :key="r" class="btn-pill" :class="{active: state.range === r}" @click="state.range = r">{{ r }}天</button>
            </div>
          </div>
          <div class="dashboard-carousel" :style="{'--dashboard-slide': state.dashboardSlide}">
            <div class="dashboard-slide-track">
              <div class="dashboard-slide-page">
                <div class="dashboard-slide dashboard-chart-slide">
                  <div class="tier-switcher">
                    <button class="tier-btn" :class="{active: state.tier === 'all'}" @click="state.tier = 'all'">30天全景</button>
                    <button class="tier-btn" :class="{active: state.tier === 'precise'}" @click="state.tier = 'precise'">精确预测</button>
                    <button class="tier-btn" :class="{active: state.tier === 'standard'}" @click="state.tier = 'standard'">标准预测</button>
                  </div>
                  <div class="price-stat-strip">
                    <div v-for="item in priceSummaryCards" :key="item.label" class="price-stat-item" :class="{'price-live-tile': item.live}">
                      <span v-text="item.label"></span>
                      <strong :key="item.value" class="animated-number" v-text="item.value"></strong>
                      <em v-text="item.unit"></em>
                      <small v-if="item.accuracy" class="forecast-accuracy-badge" v-text="item.accuracy"></small>
                      <small v-if="item.detail" class="price-stat-detail" :class="item.direction" v-text="item.detail"></small>
                    </div>
                  </div>
                  <ChartBox :option="priceChartOption" height="420px" />
                  <div class="tier-legend">
                    <span class="tier-precise">精确预测: 1-7天</span>
                    <span class="tier-standard">标准预测: 8-14天</span>
                    <span class="tier-fuzzy">趋势参考: 15-30天</span>
                  </div>
                </div>
              </div>
              <div class="dashboard-slide-page">
                <div class="dashboard-slide dashboard-risk-slide">
                  <section class="risk-slide-panel">
                    <div class="card-header compact"><span class="card-kicker">风险研判</span><h2 class="card-title">三维度风险报告</h2></div>
                    <div v-if="!riskDimensions.length && !riskAlerts.length" class="empty-state">当前无活跃风险预警。</div>
                    <div v-for="dim in riskDimensions" :key="dim.title" class="risk-dimension">
                      <h3 class="risk-dimension-title" v-text="dim.title"></h3>
                      <div v-for="(item, index) in dim.items" :key="index" class="risk-item">
                        <span class="risk-level" v-text="displayText(item.level || '关注')"></span>
                        <div><p class="risk-item-text" v-text="riskTitleText(item, dim.title)"></p><p v-if="item.impact" class="risk-item-impact" v-text="riskImpactText(item)"></p></div>
                      </div>
                    </div>
                    <div v-for="alert in riskAlerts" :key="alert.message" class="risk-item"><span class="risk-level high" v-text="severityLabel(alert.severity)"></span><p class="risk-item-text" v-text="displayText(alert.message)"></p></div>
                  </section>
                  <section class="analysis-slide-panel">
                    <div class="card-header compact"><span class="card-kicker dark">策略研判</span><h2 class="card-title dark">专家研判报告</h2></div>
                    <div class="analysis-summary" v-text="displayText(analysis.summary || '分析报告生成中...')"></div>
                    <p class="analysis-body" v-text="displayText(analysis.trend_analysis || '')"></p>
                    <div class="analysis-risks"><span v-for="risk in (analysis.risk_factors || [])" :key="risk" class="analysis-risk-tag" v-text="displayText(risk)"></span></div>
                    <div v-if="advice.action || advice.reasoning || advice.confidence || advice.suggested_price_range || advice.timing" class="advice-box">
                      <strong v-text="actionLabel(advice.action, '策略建议')"></strong>
                      <span v-text="'置信度: ' + displayText(advice.confidence || '中')"></span>
                      <p v-text="displayText(advice.reasoning || '')"></p>
                    </div>
                  </section>
                </div>
              </div>
            </div>
          </div>
          <button class="dashboard-slide-next" type="button" :class="{back: state.dashboardSlide === 1}" @click="state.dashboardSlide = state.dashboardSlide === 0 ? 1 : 0">{{ state.dashboardSlide === 0 ? '▶' : '◀' }}</button>
          <div class="dashboard-slide-dots" aria-label="看板页切换">
            <button class="slide-dot" type="button" :class="{active: state.dashboardSlide === 0}" @click="state.dashboardSlide = 0"></button>
            <button class="slide-dot" type="button" :class="{active: state.dashboardSlide === 1}" @click="state.dashboardSlide = 1"></button>
          </div>
        </section>

        <div v-if="false" class="dashboard-grid">
          <section class="card">
            <div class="card-header compact"><span class="card-kicker">风险研判</span><h2 class="card-title">三维度风险报告</h2></div>
            <div v-if="!riskDimensions.length && !riskAlerts.length" class="empty-state">当前无活跃风险预警。</div>
            <div v-for="dim in riskDimensions" :key="dim.title" class="risk-dimension">
              <h3 class="risk-dimension-title" v-text="dim.title"></h3>
              <div v-for="(item, index) in dim.items" :key="index" class="risk-item">
                <span class="risk-level" v-text="displayText(item.level || '关注')"></span>
                <div><p class="risk-item-text" v-text="riskTitleText(item, dim.title)"></p><p v-if="item.impact" class="risk-item-impact" v-text="riskImpactText(item)"></p></div>
              </div>
            </div>
            <div v-for="alert in riskAlerts" :key="alert.message" class="risk-item"><span class="risk-level high" v-text="severityLabel(alert.severity)"></span><p class="risk-item-text" v-text="displayText(alert.message)"></p></div>
          </section>

          <section class="card card-accent-mint">
            <div class="card-header compact"><span class="card-kicker dark">策略研判</span><h2 class="card-title dark">专家研判报告</h2></div>
            <div class="analysis-summary" v-text="displayText(analysis.summary || '分析报告生成中...')"></div>
            <p class="analysis-body" v-text="displayText(analysis.trend_analysis || '')"></p>
            <div class="analysis-risks"><span v-for="risk in (analysis.risk_factors || [])" :key="risk" class="analysis-risk-tag" v-text="displayText(risk)"></span></div>
            <div v-if="advice.action || advice.reasoning || advice.confidence || advice.suggested_price_range || advice.timing" class="advice-box">
              <strong v-text="actionLabel(advice.action, '策略建议')"></strong>
              <span v-text="'置信度: ' + displayText(advice.confidence || '中')"></span>
              <p v-text="displayText(advice.reasoning || '')"></p>
            </div>
          </section>
        </div>

        <section v-if="false" class="kpi-row-removed">
          <template v-if="state.loading">
            <div v-for="i in 4" :key="i" class="kpi-card loading-skeleton"></div>
          </template>
          <div v-for="kpi in kpis" :key="kpi.title" class="kpi-card">
            <span class="kpi-label" v-text="kpi.title"></span>
            <strong class="kpi-value" v-text="kpi.value"></strong>
            <span class="kpi-change" :class="kpi.change_direction" v-text="(kpi.change || '') + ' ' + (kpi.unit || '')"></span>
          </div>
        </section>

        <div class="ensemble-info">综合预测策略：价格、方向与区间覆盖由后端集成模型校准。覆盖率 {{ formatNumber(data.ensemble_coverage, 1) }}%</div>
      </main>

      <main v-else-if="state.view === 'procurement'" class="view-container">
        <div class="procurement-grid">
          <section class="card card-accent-purple">
            <span class="card-kicker white">采购策略</span>
            <h2 class="strategy-action" v-text="actionLabel(advice.action, '持仓观望')"></h2>
            <p class="strategy-confidence" v-text="'置信度: ' + displayText(advice.confidence || '中')"></p>
            <p class="strategy-reasoning" v-text="displayText(advice.reasoning || '系统分析中...')"></p>
            <div class="strategy-meta"><span>建议价格区间</span><strong v-text="displayText(advice.suggested_price_range || 'N/A')"></strong><span>执行时机</span><strong v-text="displayText(advice.timing || 'N/A')"></strong></div>
          </section>

          <section class="card procurement-live-card">
            <div class="card-header compact"><span class="card-kicker">执行触发器</span><h2 class="card-title">采购动作看板</h2></div>
            <div class="procurement-live-price">
              <span>实时价格</span>
              <strong :key="currentPrice" class="animated-number">{{ formatNumber(currentPrice) }}</strong>
              <em>RMB/吨</em>
              <small :class="currentChange > 0 ? 'up' : currentChange < 0 ? 'down' : 'flat'">{{ currentChange >= 0 ? '+' : '' }}{{ formatNumber(currentChange) }} RMB/吨</small>
            </div>
            <div class="purchase-plan-list">
              <div v-for="item in purchasePlan" :key="item.step" class="purchase-plan-item">
                <span class="purchase-step" v-text="item.step"></span>
                <div>
                  <strong v-text="item.action"></strong>
                  <small v-text="item.note"></small>
                </div>
              </div>
            </div>
          </section>

          <section class="card card-wide">
            <div class="card-header compact"><span class="card-kicker">30天价格预测明细</span><h2 class="card-title">每日预测价格与预测区间</h2></div>
            <div class="table-container"><table class="data-table"><thead><tr><th>日期</th><th>预测级别</th><th>P50</th><th>P10</th><th>P90</th><th>区间宽度</th></tr></thead><tbody><tr v-for="(row, i) in predictionRows" :key="row.target_date || i"><td v-text="row.target_date || '-'"></td><td :class="tierClass(i)" v-text="tierLabel(i)"></td><td v-text="formatNumber(row.p50)"></td><td v-text="formatNumber(row.p10)"></td><td v-text="formatNumber(row.p90)"></td><td v-text="formatNumber(((row.p90-row.p10)/Math.max(row.p50,1))*100, 1) + '%'"></td></tr></tbody></table></div>
          </section>

          <section class="card">
            <div class="card-header compact"><span class="card-kicker">成本影响分析</span><h2 class="card-title">采购量价模拟</h2></div>
            <p class="muted-line">基于 100 吨采购量模拟</p>
            <div v-for="row in costRows" :key="row.label" class="cost-scenario"><span class="cost-scenario-label" v-text="row.label"></span><strong class="cost-scenario-value">{{ formatNumber(row.value / 10000, 1) }}万</strong><span class="cost-scenario-detail">{{ row.detail }} <b v-if="row.delta" :class="row.delta > 0 ? 'positive' : 'negative'">{{ row.delta > 0 ? '多支出' : '节省' }} {{ formatNumber(Math.abs(row.delta) / 10000, 2) }}万</b></span></div>
          </section>

          <section class="card"><div class="card-header compact"><span class="card-kicker">回测验证</span><h2 class="card-title">历史预测准确率走势</h2></div><ChartBox :option="backtestChartOption" height="300px" /></section>

          <section class="card card-wide">
            <div class="card-header compact"><span class="card-kicker">偏差追踪</span><h2 class="card-title">预测 vs 实际偏差分析</h2></div>
            <div class="table-container"><table class="data-table"><thead><tr><th>日期</th><th>实际价格</th><th>预测价格</th><th>偏差</th><th>偏差率</th></tr></thead><tbody><tr v-for="row in deviationRows" :key="row.date + row.predicted"><td v-text="row.date"></td><td v-text="formatNumber(row.actual)"></td><td v-text="formatNumber(row.predicted)"></td><td :class="row.dev > 0 ? 'positive' : 'negative'" v-text="(row.dev > 0 ? '+' : '') + formatNumber(row.dev)"></td><td :class="row.devPct > 0 ? 'positive' : 'negative'" v-text="(row.devPct > 0 ? '+' : '') + formatNumber(row.devPct, 2) + '%'"></td></tr></tbody></table></div>
          </section>

          <section class="card">
            <div class="card-header compact"><span class="card-kicker">模型对比</span><h2 class="card-title">预测模型性能对比</h2></div>
            <ChartBox :option="modelChartOption" height="280px" />
            <div v-if="fixedSplit.status === 'ready'" class="split-eval-note">
              <span>固定验证</span>
              <strong>{{ fixedSplit.train_window }} 训练，{{ fixedSplit.validation_window || 'N/A' }} 验证，{{ fixedSplit.test_window }} 测试</strong>
              <em>反过拟合校验：{{ fixedSplit.overfit_guard?.status_label || '待评估' }}，最大 MAPE 泛化差 {{ formatNumber(fixedSplit.overfit_guard?.max_mape_gap, 2) }}pct</em>
              <em>验证集最优价格模型：{{ fixedSplit.best_model }}</em>
            </div>
          </section>
          <section class="card"><div class="card-header compact"><span class="card-kicker">质量校验</span><h2 class="card-title">数据质量检查结果</h2></div><div class="qa-status-container"><div v-for="row in qaRows" :key="row.key" class="qa-check-item"><span class="qa-check-name" v-text="row.name"></span><span class="qa-check-status" :class="row.passed ? 'pass' : 'fail'" v-text="qaStatusText(row)"></span></div></div></section>

          <section class="card">
            <div class="card-header compact"><span class="card-kicker">模型评估</span><h2 class="card-title">量化指标矩阵</h2></div>
            <div class="table-container"><table class="data-table"><thead><tr><th>模型</th><th>状态</th><th>MAPE</th><th>RMSE</th><th>MAE</th><th>方向准确率</th><th>区间覆盖率</th></tr></thead><tbody><tr v-for="m in modelMetrics" :key="m.model_name"><td v-text="m.model_name"></td><td v-text="m.qa_auto_repaired ? '已自动修复' : (m.excluded_from_ensemble ? '已剔除' : '参与候选')"></td><td v-text="formatNumber(m.mape, 2) + '%'"></td><td v-text="formatNumber(m.rmse, 2)"></td><td v-text="formatNumber(m.mae, 2)"></td><td v-text="formatMetricPercent(m.directional_accuracy, m.directional_accuracy_applicable !== false, 1)"></td><td v-text="formatMetricPercent(m.coverage_rate, true, 1)"></td></tr></tbody></table></div>
          </section>

          <section class="card">
            <div class="card-header compact"><span class="card-kicker">数据质量</span><h2 class="card-title">数据源健康度</h2></div>
            <div class="quality-metric"><span>数据完整度</span><strong>{{ formatNumber(data.data_quality?.completeness, 1) }}%</strong></div>
            <div class="quality-bar"><div class="quality-bar-fill" :style="{width: Math.min(100, Math.max(0, data.data_quality?.completeness || 0)) + '%'}"></div></div>
            <div class="quality-metric"><span>总记录数</span><strong v-text="data.data_quality?.total_records || 0"></strong></div>
            <div class="quality-metric"><span>日期范围</span><strong>{{ data.data_quality?.date_range?.start || '-' }} 至 {{ data.data_quality?.date_range?.end || '-' }}</strong></div>
          </section>
        </div>
      </main>
      </template>

      <button
        v-if="state.view === 'procurement' && state.chatFabReady"
        class="chat-fab"
        :class="{dragging: state.chatDrag.active}"
        :style="chatFabStyle"
        type="button"
        title="采购问答"
        aria-label="打开采购问答"
        @pointerdown="onChatFabPointerDown"
        @pointermove="onChatFabPointerMove"
        @pointerup="onChatFabPointerUp"
        @pointercancel="onChatFabPointerCancel"
        @click="onChatFabClick"
      >问</button>
      <aside v-if="state.view === 'procurement' && state.chatOpen && state.chatFabReady" class="chat-panel" :style="chatPanelStyle">
        <div class="chat-panel-header"><span>采购问答</span><button class="chat-close" @click="state.chatOpen = false">x</button></div>
        <div class="chat-messages"><div v-for="(msg, i) in state.messages" :key="i" class="chat-msg" :class="msg.who" v-text="msg.who === 'bot' ? displayText(msg.text) : msg.text"></div><div v-if="state.chatSending" class="chat-msg bot">正在分析...</div></div>
        <form class="chat-input-row" @submit.prevent="sendChat"><input v-model="state.chatInput" class="chat-input" placeholder="输入问题..." /><button class="btn-primary chat-send">发送</button></form>
      </aside>

      <button
        v-if="state.showBackTop"
        class="back-top-btn"
        type="button"
        title="回到顶端"
        aria-label="回到顶端"
        @click="scrollToTop"
      >↑</button>

      <footer class="status-bar">
        <div class="status-left"><span class="status-dot" :class="{error: state.statusError}"></span><span class="status-text" v-text="state.statusError || '系统正常'"></span></div>
        <div class="status-right"><span>数据源: {{ data.data_source || '加载中' }}</span><span>模型服务: {{ llmModelLabel }}</span><span>{{ state.now.toLocaleString('zh-CN') }}</span></div>
      </footer>
    </section>
  `,
}).mount("#app");
