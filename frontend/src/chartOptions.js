(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  root.DashboardCharts = api;
})(typeof window !== "undefined" ? window : globalThis, function () {
  function asNumber(value, fallback = 0) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }

  function emptyChartOption(theme, text) {
    return {
      ...theme,
      title: { text, left: "center", top: "center", textStyle: { color: "#949494", fontSize: 13 } },
    };
  }

  function compactGrid(theme, right = 42) {
    return {
      ...(theme || {}),
      grid: { top: 52, right, bottom: 44, left: 68, containLabel: true },
    };
  }

  function tooltip() {
    return {
      trigger: "axis",
      confine: true,
      axisPointer: { type: "none" },
      backgroundColor: "#2d2d2d",
      borderColor: "#313131",
      textStyle: { color: "#e9e9e9", fontSize: 11 },
    };
  }

  function tooltipValue(point) {
    const data = point?.data;
    if (Array.isArray(data)) {
      if (Number.isFinite(Number(data[2]))) return Number(data[2]);
      if (Number.isFinite(Number(data[1]))) return Number(data[1]);
    }
    if (data && typeof data === "object") {
      if (Number.isFinite(Number(data.intervalUpper))) return Number(data.intervalUpper);
      if (Number.isFinite(Number(data.value))) return Number(data.value);
    }
    if (Array.isArray(point?.value)) {
      if (Number.isFinite(Number(point.value[2]))) return Number(point.value[2]);
      if (Number.isFinite(Number(point.value[1]))) return Number(point.value[1]);
    }
    if (point && point.value && typeof point.value === "object") {
      if (Number.isFinite(Number(point.value.intervalUpper))) return Number(point.value.intervalUpper);
      if (Number.isFinite(Number(point.value.value))) return Number(point.value.value);
    }
    if (point?.value == null) return null;
    return asNumber(point.value, null);
  }

  function tooltipDisplay(point, unit = "RMB/吨", digits = 0) {
    const data = point?.data && typeof point.data === "object" ? point.data : {};
    const valueObject = point?.value && typeof point.value === "object" && !Array.isArray(point.value) ? point.value : {};
    const dataArray = Array.isArray(point?.data) ? point.data : null;
    const valueArray = Array.isArray(point?.value) ? point.value : null;
    const lower = Number(data.intervalLower ?? valueObject.intervalLower ?? dataArray?.[1] ?? valueArray?.[1]);
    const upper = Number(data.intervalUpper ?? valueObject.intervalUpper ?? dataArray?.[2] ?? valueArray?.[2]);
    const suffix = unit === "%" ? "%" : ` ${unit}`;
    if (Number.isFinite(lower) && Number.isFinite(upper)) {
      return `${lower.toFixed(digits)}-${upper.toFixed(digits)}${suffix}`;
    }
    const value = tooltipValue(point);
    return value == null ? null : `${Number(value).toFixed(digits)}${suffix}`;
  }

  function axisTooltipFormatter(unit = "RMB/吨", digits = 0) {
    return (points) => {
      const list = Array.isArray(points) ? points : [points];
      const axisValue = list[0]?.axisValue || list[0]?.name || "";
      const rows = [`<div style="font-size:10px;color:#949494;margin-bottom:4px">${axisValue}</div>`];
      list.forEach((point) => {
        const display = tooltipDisplay(point, unit, digits);
        if (display != null && !String(point.seriesName).startsWith("_")) {
          rows.push(`<div style="display:flex;gap:6px;margin:2px 0"><span style="width:8px;height:8px;border-radius:50%;background:${point.color}"></span>${point.seriesName}: <b style="color:#fff">${display}</b></div>`);
        }
      });
      return rows.join("");
    };
  }

  function buildIntervalBandSeries(lowerData, spreadData, colors = {}) {
    const fill = colors.mintSoft || "rgba(43,198,178,0.34)";
    return [
      {
        name: "_预测区间下界",
        type: "line",
        data: lowerData,
        stack: "预测区间",
        symbol: "none",
        connectNulls: false,
        lineStyle: { opacity: 0 },
        itemStyle: { opacity: 0 },
        emphasis: { disabled: true },
        tooltip: { show: false },
        z: 1,
      },
      {
        name: "预测区间",
        type: "line",
        data: spreadData,
        stack: "预测区间",
        symbol: "none",
        connectNulls: false,
        lineStyle: { opacity: 0 },
        itemStyle: { color: fill, opacity: 0 },
        areaStyle: { color: fill, opacity: 1 },
        z: 1,
      },
    ];
  }

  function buildDateMarkerSeries(markers, markerY) {
    if (!Array.isArray(markers) || !markers.length || markerY == null) return [];
    return [{
      name: "_日期标记",
      type: "custom",
      coordinateSystem: "cartesian2d",
      silent: true,
      tooltip: { show: false },
      data: markers.map((marker) => ({
        value: [
          marker.date,
          markerY,
          marker.offset || 0,
          marker.color,
          marker.width || 1,
          marker.type || "solid",
          marker.label,
          marker.labelOffsetY || 16,
        ],
      })),
      renderItem(params, api) {
        const point = api.coord([api.value(0), api.value(1)]);
        const x = point[0] + Number(api.value(2) || 0);
        const top = params.coordSys.y;
        const bottom = params.coordSys.y + params.coordSys.height;
        const color = api.value(3) || "rgba(255,255,255,0.45)";
        const dash = api.value(5) === "dashed" ? [4, 4] : null;
        return {
          type: "group",
          children: [
            {
              type: "line",
              shape: { x1: x, y1: top, x2: x, y2: bottom },
              style: { stroke: color, lineWidth: Number(api.value(4) || 1), lineDash: dash },
            },
            {
              type: "text",
              style: {
                text: String(api.value(6) || ""),
                x: x + 6,
                y: top + Number(api.value(7) || 16),
                fill: color,
                font: "10px sans-serif",
                textAlign: "left",
                textVerticalAlign: "middle",
              },
            },
          ],
        };
      },
      z: 10,
    }];
  }

  function buildPriceChartOption({ history = [], predictionRows = [], range = 90, tier = "all", today = null, theme = {}, colors = {} }) {
    const hist = Array.isArray(history) ? history.slice(-range) : [];
    const maxDays = tier === "precise" ? 7 : tier === "standard" ? 14 : 30;
    const preds = (Array.isArray(predictionRows) ? predictionRows : []).slice(0, maxDays);
    if (!hist.length && !preds.length) return emptyChartOption(theme, "暂无价格数据");

    const hDates = hist.map((p) => p.date);
    const hPrices = hist.map((p) => asNumber(p.price, null));
    const pDates = preds.map((p, i) => p.target_date || `T+${i + 1}`);
    const dates = [...hDates, ...pDates];
    const historyPad = pDates.map(() => null);
    const forecastPad = hDates.map(() => null);

    const precise = [];
    const standard = [];
    const trend = [];
    preds.forEach((p, i) => {
      precise.push(i < 7 ? asNumber(p.p50, null) : null);
      standard.push(i >= 7 && i < 14 ? asNumber(p.p50, null) : null);
      trend.push(i >= 14 ? asNumber(p.p50, null) : null);
    });
    if (preds.length > 7) standard[6] = asNumber(preds[6].p50, null);
    if (preds.length > 14) trend[13] = asNumber(preds[13].p50, null);

    const padded = (values) => [...forecastPad, ...values];
    const preciseData = padded(precise);
    const standardData = padded(standard);
    const trendData = padded(trend);
    const lowerBoundaryData = dates.map(() => null);
    const intervalSpreadData = dates.map(() => null);
    if (hPrices.length && preds.length) {
      const lastIndex = hDates.length - 1;
      const lastPrice = hPrices[lastIndex];
      lowerBoundaryData[lastIndex] = lastPrice;
      intervalSpreadData[lastIndex] = { value: 0, intervalLower: lastPrice, intervalUpper: lastPrice };
    }
    preds.forEach((p, i) => {
      const p10 = asNumber(p.p10, null);
      const p90 = asNumber(p.p90, null);
      const index = hDates.length + i;
      if (p10 != null && p90 != null && p90 >= p10) {
        lowerBoundaryData[index] = p10;
        intervalSpreadData[index] = {
          value: p90 - p10,
          intervalLower: p10,
          intervalUpper: p90,
        };
      }
    });
    if (hPrices.length && preds.length) {
      const lastIndex = hDates.length - 1;
      const lastPrice = hPrices[lastIndex];
      preciseData[lastIndex] = lastPrice;
      if (preds.length > 7) standardData[hDates.length + 6] = asNumber(preds[6].p50, null);
      if (preds.length > 14) trendData[hDates.length + 13] = asNumber(preds[13].p50, null);
    }

    const markerValues = hPrices.concat(preds.flatMap((p) => [p.p10, p.p50, p.p90].map((v) => asNumber(v, null)))).filter((v) => Number.isFinite(v));
    const markerY = markerValues.length ? Math.min(...markerValues) : null;
    const dateMarkers = [];
    const forecastStartDate = hDates.length && pDates.length ? pDates[0] : null;
    const todayDate = today || new Date().toISOString().slice(0, 10);
    const markersShareDate = Boolean(forecastStartDate && todayDate === forecastStartDate);
    if (forecastStartDate) {
      dateMarkers.push({
        date: forecastStartDate,
        label: "预测起点",
        color: "rgba(255,255,255,0.58)",
        width: 1,
        type: "dashed",
        offset: markersShareDate ? -5 : 0,
        labelOffsetY: 16,
      });
    }
    if (todayDate && dates.includes(todayDate)) {
      dateMarkers.push({
        date: todayDate,
        label: "今日",
        color: colors.mint || "#2bc6b2",
        width: 1.4,
        type: "solid",
        offset: markersShareDate ? 5 : 0,
        labelOffsetY: markersShareDate ? 34 : 16,
      });
    }

    return {
      ...theme,
      tooltipUnit: "RMB/吨",
      tooltipDigits: 0,
      tooltip: {
        ...tooltip(),
        formatter: axisTooltipFormatter("RMB/吨", 0),
      },
      legend: { data: ["历史价格", "精确预测", "标准预测", "趋势参考", "预测区间"], textStyle: { color: "#949494", fontSize: 10 }, top: 0, right: 0 },
      xAxis: { type: "category", data: dates, axisLine: { lineStyle: { color: "#313131" } }, axisLabel: { color: "#949494", fontSize: 10, formatter: (v) => String(v).slice(5) } },
      yAxis: { type: "value", scale: true, axisLabel: { color: "#949494", fontSize: 10, formatter: (v) => Number(v).toFixed(0) }, splitLine: { lineStyle: { color: "rgba(49,49,49,0.55)", type: "dashed" } } },
      series: [
        ...buildIntervalBandSeries(lowerBoundaryData, intervalSpreadData, colors),
        {
          name: "历史价格",
          type: "line",
          data: [...hPrices, ...historyPad],
          smooth: true,
          symbol: "none",
          lineStyle: { color: colors.white, width: 2 },
          z: 5,
        },
        { name: "精确预测", type: "line", data: preciseData, smooth: true, connectNulls: true, symbolSize: 5, lineStyle: { color: colors.mint, width: 3 }, z: 4 },
        { name: "标准预测", type: "line", data: standardData, smooth: true, connectNulls: true, symbolSize: 5, lineStyle: { color: colors.mintMid, width: 2.5, type: "dashed" }, z: 4 },
        { name: "趋势参考", type: "line", data: trendData, smooth: true, connectNulls: true, symbolSize: 4, lineStyle: { color: colors.mintSoft, width: 2, type: "dotted" }, z: 4 },
        ...buildDateMarkerSeries(dateMarkers, markerY),
      ],
    };
  }

  function buildBacktestChartOption({ backtestResults = [], theme = {}, colors = {} }) {
    const rows = Array.isArray(backtestResults) ? backtestResults : [];
    if (!rows.length) return emptyChartOption(theme, "暂无回测数据");
    const dates = [];
    const actual = [];
    const predicted = [];
    rows.slice().reverse().forEach((period) => {
      (period.dates || []).forEach((date, i) => {
        const a = asNumber(period.actual?.[i], null);
        const p = asNumber(period.predicted?.[i], null);
        if (a != null && p != null) {
          dates.push(String(date).slice(5));
          actual.push(a);
          predicted.push(p);
        }
      });
    });
    return {
      ...compactGrid(theme, 36),
      tooltipUnit: "RMB/吨",
      tooltipDigits: 0,
      tooltip: { ...tooltip(), formatter: axisTooltipFormatter("RMB/吨", 0) },
      legend: { data: ["实际价格", "模型预测"], textStyle: { color: "#949494", fontSize: 10 }, top: 0 },
      xAxis: { type: "category", data: dates, axisLabel: { color: "#949494", fontSize: 10 }, axisLine: { lineStyle: { color: "#313131" } } },
      yAxis: { type: "value", scale: true, axisLabel: { color: "#949494", fontSize: 10, formatter: (v) => Number(v).toFixed(0) }, splitLine: { lineStyle: { color: "rgba(49,49,49,0.55)", type: "dashed" } } },
      series: [
        { name: "实际价格", type: "line", data: actual, symbol: "none", lineStyle: { color: colors.white, width: 2 } },
        { name: "模型预测", type: "line", data: predicted, symbol: "none", lineStyle: { color: colors.mint, width: 2, type: "dashed" } },
      ],
    };
  }

  function buildModelChartOption({ modelMetrics = [], theme = {}, colors = {} }) {
    const rows = (Array.isArray(modelMetrics) ? modelMetrics : []).filter((m) => m.model_name !== "ensemble");
    if (!rows.length) return emptyChartOption(theme, "暂无模型指标");
    const metricValue = (row, key) => {
      if (key === "directional_accuracy" && row.directional_accuracy_applicable === false) return null;
      if (row[key] == null) return null;
      return asNumber(row[key]);
    };
    const percentLabel = (params) => (params.value == null ? "" : `${params.value}%`);
    return {
      ...compactGrid(theme, 58),
      tooltipUnit: "%",
      tooltipDigits: 2,
      tooltip: { ...tooltip(), formatter: axisTooltipFormatter("%", 2) },
      legend: { data: ["价格准确率(%)", "方向准确率(%)", "区间覆盖率(%)"], textStyle: { color: "#949494", fontSize: 10 }, top: 0 },
      xAxis: { type: "category", data: rows.map((m) => m.model_name), axisLabel: { color: "#949494", fontSize: 10 }, axisLine: { lineStyle: { color: "#313131" } } },
      yAxis: [
        { type: "value", name: "准确率", min: 0, max: 100, axisLabel: { color: "#949494", fontSize: 10, formatter: "{value}%" }, splitLine: { lineStyle: { color: "rgba(49,49,49,0.55)", type: "dashed" } } },
        { type: "value", name: "准确率/覆盖率", min: 0, max: 100, axisLabel: { color: "#949494", fontSize: 10, formatter: "{value}%" }, splitLine: { show: false } },
      ],
      series: [
        { name: "价格准确率(%)", type: "bar", data: rows.map((m) => metricValue(m, "price_accuracy") ?? Math.max(0, 100 - asNumber(m.mape))), barWidth: 22, itemStyle: { color: colors.mint, borderRadius: [4, 4, 0, 0] }, label: { show: true, position: "top", color: "#949494", fontSize: 10, formatter: percentLabel } },
        { name: "方向准确率(%)", type: "line", yAxisIndex: 1, data: rows.map((m) => metricValue(m, "directional_accuracy")), symbol: "diamond", symbolSize: 9, lineStyle: { color: colors.purple, width: 2 }, itemStyle: { color: colors.purple }, label: { show: true, position: "top", color: colors.purple, fontSize: 10, formatter: percentLabel } },
        { name: "区间覆盖率(%)", type: "line", yAxisIndex: 1, data: rows.map((m) => metricValue(m, "coverage_rate")), symbol: "circle", symbolSize: 8, lineStyle: { color: colors.amber || "#ffa502", width: 2, type: "dashed" }, itemStyle: { color: colors.amber || "#ffa502" }, label: { show: true, position: "top", color: colors.amber || "#ffa502", fontSize: 10, formatter: percentLabel } },
      ],
    };
  }

  function buildPriceSummaryCards({ currentPrice = 0, currentChange = 0, predictionRows = [], modelMetrics = {}, newsSentiment = {} }) {
    const rows = Array.isArray(predictionRows) ? predictionRows : [];
    const avg = (items) => items.reduce((sum, row) => sum + asNumber(row.p50), 0) / Math.max(items.length, 1);
    const avg7 = avg(rows.slice(0, 7));
    const avg30 = avg(rows);
    const adjustmentPct = asNumber(newsSentiment.price_adjustment_pct, 0) * 100;
    const priceAccuracy = asNumber(modelMetrics.price_accuracy, Math.max(0, 100 - asNumber(modelMetrics.mape, 0)));
    const change = asNumber(currentChange, 0);
    const current = asNumber(currentPrice);
    const compare = (value) => current ? ((value - current) / current) * 100 : 0;
    const newsDirection = adjustmentPct > 0 ? "up" : adjustmentPct < 0 ? "down" : "flat";
    return [
      {
        label: "实时价格",
        value: `${current.toFixed(0)}`,
        unit: "RMB/吨",
        live: true,
        direction: change > 0 ? "up" : change < 0 ? "down" : "flat",
        detail: `${change >= 0 ? "+" : ""}${change.toFixed(0)} RMB/吨`,
      },
      {
        label: "7天预测均价",
        value: `${avg7.toFixed(0)}`,
        unit: "RMB/吨",
        accuracy: `准确率 ${priceAccuracy.toFixed(1)}%`,
        detail: `较当前 ${compare(avg7) >= 0 ? "+" : ""}${compare(avg7).toFixed(2)}%`,
      },
      {
        label: "30天预测均价",
        value: `${avg30.toFixed(0)}`,
        unit: "RMB/吨",
        accuracy: `准确率 ${priceAccuracy.toFixed(1)}%`,
        detail: `较当前 ${compare(avg30) >= 0 ? "+" : ""}${compare(avg30).toFixed(2)}%`,
      },
      {
        label: "新闻调优",
        value: `${adjustmentPct >= 0 ? "+" : ""}${adjustmentPct.toFixed(2)}%`,
        unit: "短期影响",
        direction: newsDirection,
        detail: adjustmentPct === 0 ? "暂无额外修正" : `情绪修正 ${adjustmentPct >= 0 ? "+" : ""}${adjustmentPct.toFixed(2)}%`,
      },
    ];
  }

  function buildProcurementTriggers({ currentPrice = 0, predictionRows = [], advice = {}, newsSentiment = {} }) {
    const rows = Array.isArray(predictionRows) ? predictionRows : [];
    const firstWeek = rows.slice(0, 7);
    const low = Math.min(...firstWeek.map((row) => asNumber(row.p10, Infinity)));
    const high = Math.max(...firstWeek.map((row) => asNumber(row.p90, -Infinity)));
    const confidence = advice.confidence || "中";
    const batch = confidence === "高" ? "50% / 30% / 20%" : confidence === "低" ? "20% / 40% / 40%" : "30% / 40% / 30%";
    const adjustmentPct = asNumber(newsSentiment.price_adjustment_pct, 0) * 100;
    return [
      { label: "低位触发", value: `≤ ${Number.isFinite(low) ? low.toFixed(0) : asNumber(currentPrice).toFixed(0)}`, note: "触达后启动首批采购" },
      { label: "分批节奏", value: batch, note: "首批 / 二批 / 尾批" },
      { label: "风险上沿", value: `≥ ${Number.isFinite(high) ? high.toFixed(0) : asNumber(currentPrice).toFixed(0)}`, note: "高于该位暂停追单" },
      { label: "新闻调优", value: `${adjustmentPct >= 0 ? "+" : ""}${adjustmentPct.toFixed(2)}%`, note: "用于复核执行窗口" },
    ];
  }

  return {
    buildPriceChartOption,
    buildBacktestChartOption,
    buildModelChartOption,
    buildPriceSummaryCards,
    buildProcurementTriggers,
  };
});
