# Commodity Forecast POC

大宗商品采购价格预测 SaaS POC，目前聚焦 `0# 柴油` 的历史价格、模型预测、新闻情绪调优、采购策略和风险看板。

## 目录结构

- `backend/`：FastAPI 后端、模型训练/预测、数据源、LLM 分析服务。
- `frontend/`：Vue 3 CDN 版前端看板。
- `tests/`：后端和前端静态/逻辑测试。
- `docs/`：开发计划、规格说明和项目文档。
- `backend/ml/trained_models/`：POC 轻量模型产物，当前保留进仓库，便于迁移电脑后直接运行。

## 本地启动

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，填入本机真实的 `JWT_SECRET_KEY`、LLM API token 和需要使用的数据源 API key。不要提交 `.env`。

启动服务：

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node tests\test_frontend_chart_options.js
```

## Git 使用建议

首次初始化：

```powershell
git init
git add .
git status
git commit -m "Initial POC dashboard baseline"
```

后续开发：

```powershell
git checkout -b feature/your-change
git add .
git commit -m "Describe the change"
```

如果要推送到 GitHub/GitLab，先在远端创建空仓库，然后：

```powershell
git remote add origin <your-remote-url>
git branch -M main
git push -u origin main
```

## 不提交的内容

`.gitignore` 已排除以下内容：

- `.env` 和所有真实密钥配置。
- 虚拟环境、缓存、Python 字节码。
- 本地 SQLite 数据库。
- 训练日志、运行日志、Playwright 临时输出。
- 前端依赖和构建产物。
