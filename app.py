import os
import time
import gradio as gr
from google import genai
from dotenv import load_dotenv

load_dotenv()

ANALYSIS_PROMPT = """請仔細分析這段影片，找出其中出現的所有野生動物（wild animals）。

判斷標準：
- 野生動物包含野外生活的哺乳類、鳥類、爬蟲類、兩棲類、魚類、昆蟲等
- 不包含家畜（狗、貓、牛、豬等寵物或農場動物）
- 不包含動物園中明顯被圈養的動物

如果影片中有野生動物，請用以下格式回答：

## 偵測到的野生動物

| 動物名稱 | 英文名稱 | 數量 | 行為描述 |
|--------|--------|------|---------|
| （填入） | （填入） | （填入） | （填入） |

**補充說明：**
（描述動物的出現時間點、棲息環境、互動情況等）

---

如果影片中完全沒有野生動物，請只回應：
沒有看到任何野生動物"""


def upload_and_wait(client: genai.Client, video_path: str):
    """上傳影片並等待 Gemini 處理完成"""
    print(f"正在上傳影片：{video_path}")
    video_file = client.files.upload(file=video_path)

    while video_file.state.name == "PROCESSING":
        print("影片處理中，請稍候...")
        time.sleep(3)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError("影片處理失敗，請確認影片格式是否正確")

    print(f"影片上傳完成：{video_file.name}")
    return video_file


def analyze_video(video_path: str, api_key: str) -> str:
    """分析影片中的野生動物"""
    if not video_path:
        return "請先上傳一段影片"

    resolved_key = os.getenv("GOOGLE_API_KEY") or api_key.strip()
    if not resolved_key:
        return "請提供 Google API Key（可在下方輸入，或設定環境變數 GOOGLE_API_KEY）"

    client = genai.Client(api_key=resolved_key)
    video_file = None
    try:
        video_file = upload_and_wait(client, video_path)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[video_file, ANALYSIS_PROMPT],
        )
        return response.text

    except Exception as e:
        err = str(e)
        if "limit: 0" in err or "free_tier" in err:
            return "Google API 免費配額已滿，請確認 Google Cloud 帳單已啟用"
        if "PERMISSION_DENIED" in err or "403" in err:
            return "API Key 權限不足，請確認帳單已啟用並重新產生 Key"
        return f"分析過程發生錯誤：{err}"

    finally:
        if video_file:
            try:
                client.files.delete(name=video_file.name)
                print(f"已清除暫存檔案：{video_file.name}")
            except Exception:
                pass


def build_ui() -> gr.Blocks:
    has_env_key = bool(os.getenv("GOOGLE_API_KEY"))

    with gr.Blocks(title="野生動物偵測器") as demo:
        gr.Markdown(
            """
# 🦁 野生動物偵測器
### 上傳一段影片，AI 將為你辨識其中出現的野生動物

**支援格式**：MP4、MOV、AVI、MKV 等常見影片格式
**使用模型**：Google Gemini 2.5 Flash
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="上傳影片",
                    sources=["upload"],
                )

                api_key_input = gr.Textbox(
                    label="Google API Key",
                    placeholder="貼上你的 Google AI Studio API Key",
                    type="password",
                    visible=not has_env_key,
                    value="",
                )

                if has_env_key:
                    gr.Markdown("✅ Google API Key 已從環境變數載入")

                analyze_btn = gr.Button(
                    "🔍 開始分析",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                result_output = gr.Markdown(
                    label="分析結果",
                    value="分析結果將顯示於此...",
                )

        analyze_btn.click(
            fn=analyze_video,
            inputs=[video_input, api_key_input],
            outputs=result_output,
        )

        gr.Markdown(
            "---\n使用 [Google Gemini API](https://ai.google.dev/) 提供影片分析能力"
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
