---
title: Wild Animal Search
emoji: 🦁
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: mit
short_description: 上傳影片，AI 自動辨識其中出現的野生動物
---

# 野生動物偵測器 🦁

上傳一段影片，使用 Google Gemini AI 自動辨識影片中出現的野生動物。

## 功能

- 支援多種影片格式（MP4、MOV、AVI、MKV 等）
- 列出偵測到的野生動物名稱（中英文）、數量及行為描述
- 若無野生動物則明確告知

## 部署到 Hugging Face Spaces

1. 在 Space 的 **Settings → Secrets** 中新增：
   - `GOOGLE_API_KEY`：你的 Google AI Studio API Key

2. 推送程式碼：
   ```bash
   git remote add space https://huggingface.co/spaces/{你的帳號}/{space名稱}
   git push space main
   ```

## 本機開發

```bash
# 建立環境
uv sync

# 設定 API Key
cp .env.example .env
# 編輯 .env 填入 GOOGLE_API_KEY

# 啟動
uv run python app.py
```

## 技術架構

- **前端框架**：[Gradio](https://gradio.app/)
- **AI 模型**：Google Gemini 1.5 Flash（多模態影片分析）
- **套件管理**：[uv](https://docs.astral.sh/uv/)
