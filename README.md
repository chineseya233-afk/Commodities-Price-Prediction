# 大宗商品采购价格预测 SaaS 系统用户手册

本项目是一个大宗商品采购价格预测 POC 看板，当前商品为 `0# 柴油`。系统由 FastAPI 后端、Vue 3 前端、ECharts 图表、多个预测模型和 AI 分析服务组成。后端启动后会加载价格数据、训练或运行模型、生成综合预测，并把前端页面托管在同一个服务中。

服务保持运行时，后台会按配置周期检查 EIA/FRED 数据和新闻情绪；数据更新后会自动刷新预测与报告。离线或 API 不可用时，系统会回退到模拟数据和模板报告。

## 功能概览

- 0# 柴油历史价格、预测区间和今日线/预测起点线展示。
- EIA/FRED 真实数据接入，失败时自动回退模拟数据。
- Naive、Prophet、XGBoost、TFT 和 Ensemble 多模型预测。
- 动态训练/验证/测试切分、滚动回测和模型 QA 校验。
- AI 专家研判、三维度风险报告、新闻情绪调优和采购问答。
- FastAPI 直接托管前端，当前不需要单独启动 npm/Vite。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `backend/` | FastAPI 后端、数据源、模型训练/预测、LLM 服务 |
| `frontend/` | Vue 3 CDN 前端看板、ECharts 图表配置和样式 |
| `tests/` | 后端单元测试和前端静态/逻辑测试 |
| `docs/` | 开发计划、规格说明和项目文档 |
| `backend/ml/trained_models/` | POC 轻量模型产物，保留进仓库便于直接运行 |
| `.env` | 公开占位配置模板；真实密钥只在本机替换，不要再次提交 |

## 快速启动

### 环境要求

- Python 3.10 或更高版本
- Chrome、Edge 或 Firefox 浏览器
- Windows、macOS、Linux 均可运行
- 可联网时可调用 AI API 和外部数据接口

### 安装依赖

先进入你解压或 `git clone` 得到的项目根目录，也就是包含 `README.md`、`backend/`、`frontend/` 的文件夹。下面的 `<你的项目根目录>` 请按你自己的电脑路径替换：

```powershell
cd "<你的项目根目录>"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

### 配置 `.env`

仓库中的 `.env` 是安全占位版配置。第一次运行前，请打开 `.env`，把需要的 key 和密钥替换成本机私有值。

最重要的配置：

- `JWT_SECRET_KEY`：必须换成你自己生成的随机字符串。
- `OPENAI_COMPATIBLE_API_KEY`：如果要使用 AI 报告/问答，填入 DeepSeek、OpenAI 或其他兼容 OpenAI Chat Completions 的供应商 key。
- `EIA_API_KEY`：默认 `demo` 可用但限制多；建议申请自己的 EIA key。
- `FRED_API_KEY`：用于宏观变量和汇率数据；不填时部分宏观数据不可用。

### 启动服务

```powershell
cd "<你的项目根目录>"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

前端是 Vue 3 CDN 版本，不需要单独启动 npm 或 Vite 服务，也不需要打开 3000 端口。FastAPI 会在 8000 端口直接托管 `frontend/index.html` 和 `frontend/src` 下的脚本、样式文件。

### 停止服务

在运行 uvicorn 的终端里按 `Ctrl + C`。

## 登录账号

POC 默认账号如下。部署到公网或多人共享前，请在本机 `.env` 中修改默认密码。

| 用户名 | 默认密码 | 角色 |
| --- | --- | --- |
| `admin` | `Admin123456@` | 管理员 |
| `executive` | `Exec123456@` | 决策用户 |
| `procurement` | `Proc123456@` | 采购用户 |

登录页支持“记住密码”和自动恢复登录态。会话过期、主动退出或关闭页面超过会话有效期后，需要重新登录。

## `.env` 配置说明和 key 获取方式


| 配置项 | 是否必须 | 获取或设置方式 |
| --- | --- | --- |
| `APP_NAME` | 否 | 应用名称，本地可保持默认值。 |
| `APP_ENV` | 否 | 运行环境，例如 `development`、`production`。 |
| `DEBUG` | 否 | 本地调试可设为 `true`，部署公网建议设为 `false`。 |
| `DATABASE_URL` | 否 | 默认使用本地 SQLite：`sqlite+aiosqlite:///./data/commodity_prediction.db`。生产可改为 PostgreSQL/MySQL 的 SQLAlchemy URL。 |
| `JWT_SECRET_KEY` | 是 | 自己生成，不从第三方网站获取。推荐命令：`python -c "import secrets; print(secrets.token_urlsafe(32))"`，把输出填入该项。 |
| `JWT_ALGORITHM` | 否 | 默认 `HS256` 即可。 |
| `JWT_EXPIRATION_MINUTES` | 否 | 登录 token 有效分钟数。 |
| `SESSION_IDLE_MINUTES` | 否 | 会话空闲超时分钟数。 |
| `REMEMBER_ME_DAYS` | 否 | 勾选记住登录后的有效天数。 |
| `SESSION_COOKIE_NAME` | 否 | 浏览器 cookie 名称。 |
| `COOKIE_SECURE` | 生产建议 | 本地 HTTP 用 `false`；公网 HTTPS 部署必须改为 `true`。 |
| `ADMIN_DEFAULT_PASSWORD` | 本地建议改 | 管理员默认密码。生产或共享环境必须改。 |
| `EXECUTIVE_DEFAULT_PASSWORD` | 本地建议改 | 决策用户默认密码。生产或共享环境必须改。 |
| `PROCUREMENT_DEFAULT_PASSWORD` | 本地建议改 | 采购用户默认密码。生产或共享环境必须改。 |
| `OPENAI_COMPATIBLE_API_KEY` | AI 功能需要 | DeepSeek 可在 `https://platform.deepseek.com/api_keys` 创建；OpenAI 可在 `https://platform.openai.com/api-keys` 创建；其他兼容供应商填对应平台的 key。 |
| `OPENAI_COMPATIBLE_BASE_URL` | AI 功能需要 | DeepSeek 默认 `https://api.deepseek.com/v1`；OpenAI 默认 `https://api.openai.com/v1`；其他供应商看其 OpenAI-compatible 文档。 |
| `OPENAI_COMPATIBLE_MODEL` | AI 功能需要 | 模型名，例如 `deepseek-v4-pro`、`deepseek-chat` 或供应商文档中可用的模型 ID。 |
| `OPENAI_COMPATIBLE_PROVIDER` | 否 | 前端展示用供应商标签，例如 `deepseek`、`openai`。 |
| `EIA_API_KEY` | 建议 | EIA 官方申请页：`https://www.eia.gov/opendata/register.php`。`demo` 可用于有限测试，但不适合稳定运行。 |
| `EIA_BASE_URL` | 否 | 默认 `https://api.eia.gov/v2`。 |
| `EIA_DIESEL_SERIES` | 否 | 当前柴油序列默认 `EER_EPD2DXL0_PF4_Y35NY_DPG`。更换商品时改为对应 EIA series。 |
| `FRED_API_KEY` | 建议 | FRED API key 页面：`https://fred.stlouisfed.org/docs/api/api_key.html`。需要登录 FRED 账号后申请或查看。 |
| `FRED_BASE_URL` | 否 | 默认 `https://api.stlouisfed.org/fred`。 |
| `MODEL_DIR` | 否 | 模型目录，默认 `backend/ml/trained_models`。 |
| `TRAINED_MODELS_DIR` | 否 | 训练模型目录，默认 `backend/ml/trained_models`。 |
| `PREDICTION_HORIZON` | 否 | 预测天数，默认 30。 |
| `LOOKBACK_WINDOW` | 否 | 特征回看窗口，默认 30。 |
| `DATA_REFRESH_SECONDS` | 否 | 后台刷新数据源的周期秒数，默认 300。 |
| `MARKET_DATA_START_DATE` | 否 | 真实数据拉取起点，默认 `2006-06-01`。 |
| `TEMP_DIR` | 否 | 临时目录，默认 `data/temp`。 |
| `BACKEND_PORT` | 否 | 后端端口配置，默认 8000；当前前端也由 FastAPI 在该端口托管。 |

## AI API 配置

推荐使用通用 OpenAI-compatible 配置：

```env
OPENAI_COMPATIBLE_API_KEY=你的模型API密钥
OPENAI_COMPATIBLE_BASE_URL=https://api.deepseek.com/v1
OPENAI_COMPATIBLE_MODEL=deepseek-v4-pro
OPENAI_COMPATIBLE_PROVIDER=deepseek
```

这组配置不绑定具体厂商。DeepSeek 只是默认示例；如果要接入 OpenAI 或其他兼容供应商，只需要替换 API key、base URL、model 和 provider。

AI 在系统中的作用：

- 生成“专家研判报告”和“三维度风险报告”。
- 根据 EIA 新闻、价格走势和模型指标生成“新闻调优”。
- 当模型 QA 异常时，生成修复建议。
- AI 输出不会直接绕过后端规则覆盖 `p10/p50/p90` 原始预测值。

## 数据源配置

当前启动链路在 `backend/main.py` 中：

1. 先尝试 EIA 数据源：`backend/data_providers/eia_provider.py`
2. 再尝试 FRED 宏观数据：`backend/data_providers/fred_provider.py`
3. 如果真实接口不可用，回退到中国柴油模拟器：`backend/data_providers/simulator.py`
4. 数据经过 `DataPreprocessor` 清洗，再进入特征工程和模型预测

当前真实数据处理规则：

- EIA 提供柴油美元/加仑价格。
- FRED 提供宏观变量，包括人民币兑美元汇率、Brent 油价和联邦基金利率等。
- 后端会用 FRED 汇率把 EIA 柴油价格换算为 RMB/吨，再用于训练、回测和前端展示。
- 官方数据源可能滞后几天，这属于数据源发布时间差异，不代表系统日期错误。
- `/api/health` 中的 `data_source=eia` 表示当前正在使用真实 EIA 数据；如果接口失败，会显示 `simulator` 并回退模拟数据。

如果要换成自己的价格数据，可以新增 provider，例如 `backend/data_providers/local_csv_provider.py`，返回包含 `date`、`price`、`high`、`low` 等字段的 DataFrame，再在 `backend/main.py` 的初始数据加载逻辑里接入。

建议数据格式：

| 字段 | 说明 |
| --- | --- |
| `date` | 日期，建议 `YYYY-MM-DD` |
| `price` | 柴油价格，单位 RMB/吨 |
| `high` | 当日高价，没有可用 `price` 代替 |
| `low` | 当日低价，没有可用 `price` 代替 |
| 其他字段 | 可加入库存、汇率、原油价格、政策事件标签等特征 |

## 训练、验证和测试集

系统会尽量使用 EIA/FRED 能拿到的全部历史数据，并按时间顺序切成三段：

1. 训练集：最早可用历史数据到验证集之前。
2. 验证集：训练集之后的一段最新历史窗口，用于选择最优模型。
3. 测试集：最后一段 holdout 数据，只用于报告最终泛化表现，不参与模型选择。

实现位置：

```text
backend/main.py::_evaluate_fixed_train_test_split()
```

切分规则：

- 测试集长度通常为最近 30-90 天，数据越多窗口越长，上限 90 天。
- 验证集长度通常也是 30-90 天。
- 数据足够时，训练集和验证/测试集之间会保留 1 条记录的 embargo 间隔，降低时间泄漏风险。
- `best_model` 由验证集 MAPE 选出。
- `oracle_test_best_model` 只是诊断项，表示如果偷看测试集会选哪个模型，不能作为真实选型依据。

如果要控制训练历史起点，可以修改：

```env
MARKET_DATA_START_DATE=2006-06-01
```

## 预测日期和今日线

预测数组和“最新真实价格日期”绑定，而不是和“今天”绑定。

- `p50[0]` 表示“最新真实价格日期的下一天”。
- 如果 EIA 最新价格只更新到 `2026-06-01`，那么 `p50[0]` 对应 `2026-06-02`。
- 前端会同时显示“预测起点线”和“今日线”。
- 如果预测起点和今日落在同一天，前端会画两条带像素偏移的日期标记线。
- 接口返回的预测点里包含 `source_index`，用于排查显示日期和模型输出下标的对应关系。

看板上的 30 天、60 天、90 天按钮只控制历史价格窗口长度；预测仍按 30 天输出，并按“精确预测 1-7 天、标准预测 8-14 天、趋势参考 15-30 天”分层展示。

## 模型说明

| 模型 | 用途 |
| --- | --- |
| Naive | 基线模型，用于判断其他模型是否超过“沿用最近价格”。 |
| Prophet | 趋势和季节性模型，适合观察中期趋势和均值回归。 |
| XGBoost | 使用滞后项、滚动统计、宏观变量等特征，适合捕捉非线性关系。 |
| TFT | 时间序列深度学习模型；依赖或模型不可用时会自动 fallback。 |
| Ensemble | 按预测区间分段选择表现最好的价格、方向和区间模型。 |

综合模型不是简单平均，而是按预测区间分段选模型：

- 1-7 天：短期价格和方向参考。
- 8-14 天：标准预测区间。
- 15-30 天：趋势参考区间。

区间覆盖不是越高越好。100% 覆盖率可能只是区间太宽，采购参考价值反而下降。系统会优先选择接近目标覆盖率且区间宽度合理的模型。

## QA 校验和自动修复

每个模型输出后会经过硬规则 QA：

- 预测值必须为正数。
- 价格必须处于合理范围。
- 单日涨跌幅不能异常。
- 7 日累计偏差不能异常。
- 预测值不能偏离历史波动范围。
- `p10 < p50 < p90` 的区间顺序必须正确。
- 区间宽度不能过窄或过宽。

如果模型失败，后端会先尝试确定性 guardrail 修复；修复后重新跑 QA；仍不通过的模型会被剔除出 ensemble。AI 建议只作为可审计的辅助依据，不会绕过规则校验。

## 页面和接口入口

常用页面：

- 决策大屏：登录后默认进入，适合管理层看整体方向和风险。
- 采购看板：适合查看模型指标、采购建议、预测明细和 AI 问答。
- 左上角菜单：用于切换商品，目前只实现 0# 柴油。

常用接口：

| 接口 | 说明 |
| --- | --- |
| `/api/health` | 系统健康检查 |
| `/api/auth/login` | 登录 |
| `/api/auth/session` | 恢复登录态 |
| `/api/dashboard/summary` | 看板总数据 |
| `/api/predictions/latest?model=ensemble` | 指定模型预测 |
| `/api/predictions/all-models` | 所有模型预测 |
| `/api/metrics/comparison` | 模型性能指标 |
| `/api/backtest/fixed-split` | 动态训练/验证/测试三段评估 |
| `/api/backtest/results` | 滚动回测结果 |
| `/api/chat` | AI 采购助手 |

## 测试命令

后端单元测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

前端图表配置测试：

```powershell
node tests\test_frontend_chart_options.js
```

只测试预测日期逻辑：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_prediction_dates -v
```

只测试模型选择、Naive 方向指标和 QA：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_metric_specialized_ensemble tests.test_model_evaluation tests.test_qa_service -v
```

## 常见问题

### 为什么今天和最新价格日期不一样？

这通常是 EIA 数据源发布时间滞后造成的。系统日期和最新真实价格日期是两个概念。预测从最新真实价格的下一天开始，今日线单独标记当前日期。

### 为什么 Prophet 区间覆盖率是 0？

表示最近验证窗口内实际价格没有落入 Prophet 的 p10/p90 区间。系统会保留这个真实指标，并选择覆盖率更合适的模型提供综合预测区间。

### 为什么 Naive 方向准确率是 N/A 或 0？

Naive 是平线基线，没有方向预测能力。后端会标记为不适用并从综合模型中剔除。它仍保留在模型对比中，用于衡量其他模型是否超过简单基线。

### 数据会不会持续更新？

会。只要后端服务不关闭，后台会按 `DATA_REFRESH_SECONDS` 周期检查 EIA/FRED 和新闻情绪。数据发生变化后会自动重建预测和报告。

## 参考链接

- DeepSeek API key：`https://platform.deepseek.com/api_keys`
- DeepSeek API 文档：`https://api-docs.deepseek.com/api/deepseek-api/`
- OpenAI API keys：`https://platform.openai.com/api-keys`
- EIA API key 注册：`https://www.eia.gov/opendata/register.php`
- FRED API key 文档：`https://fred.stlouisfed.org/docs/api/api_key.html`
