import os
import base64
import cv2
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

ANALYSIS_PROMPT = """以下是從一段影片中擷取的連續畫面，請仔細分析所有畫面，找出其中出現的野生動物（wild animals）。

判斷標準：
- 野生動物包含野外生活的哺乳類、鳥類、爬蟲類、兩棲類、魚類、昆蟲等
- 不包含家畜（狗、貓、牛、豬等寵物或農場動物）
- 不包含動物園中明顯被圈養的動物

如果有野生動物，請用以下格式回答：

## 偵測到的野生動物

| 動物名稱 | 英文名稱 | 數量 | 行為描述 |
|--------|--------|------|---------|
| （填入） | （填入） | （填入） | （填入） |

**補充說明：**
（描述動物的棲息環境、互動情況等）

---

如果所有畫面中完全沒有野生動物，請只回應：
沒有看到任何野生動物"""

MAX_FRAMES = 20
FRAME_INTERVAL_SEC = 2.0


def extract_frames(video_path: str) -> list[str]:
    """從影片中均勻擷取畫面，回傳 base64 編碼清單"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # 根據影片長度決定擷取間隔，最多 MAX_FRAMES 張
    interval = max(1, int(fps * FRAME_INTERVAL_SEC))
    if total_frames / interval > MAX_FRAMES:
        interval = max(1, total_frames // MAX_FRAMES)

    frames = []
    frame_idx = 0
    while cap.isOpened() and len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frames.append(base64.b64encode(buf).decode("utf-8"))
        frame_idx += 1

    cap.release()
    print(f"擷取到 {len(frames)} 張畫面（影片長度 {duration:.1f} 秒）")
    return frames


def analyze_video(video_path: str, api_key: str) -> str:
    """分析影片中的野生動物"""
    if not video_path:
        return "請先上傳一段影片"

    resolved_key = os.getenv("OPENAI_API_KEY") or api_key.strip()
    if not resolved_key:
        return "請提供 OpenAI API Key（可在下方輸入，或設定環境變數 OPENAI_API_KEY）"

    frames = extract_frames(video_path)
    if not frames:
        return "無法讀取影片，請確認影片格式是否正確（支援 MP4、MOV、AVI 等）"

    client = OpenAI(api_key=resolved_key)
    try:
        content = [{"type": "text", "text": ANALYSIS_PROMPT}]
        for frame_b64 in frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_b64}",
                    "detail": "low",
                },
            })

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            max_tokens=1500,
        )
        return response.choices[0].message.content

    except Exception as e:
        err = str(e)
        if "insufficient_quota" in err or "429" in err:
            return "OpenAI 配額不足，請確認帳號餘額：platform.openai.com/account/billing"
        if "invalid_api_key" in err or "401" in err:
            return "API Key 無效，請確認是否正確複製"
        return f"分析過程發生錯誤：{err}"


def build_ui() -> gr.Blocks:
    has_env_key = bool(os.getenv("OPENAI_API_KEY"))

    with gr.Blocks(title="野生動物偵測器") as demo:
        gr.Markdown(
            """
# 🦁 野生動物偵測器
### 上傳一段影片，AI 將為你辨識其中出現的野生動物

**支援格式**：MP4、MOV、AVI、MKV 等常見影片格式
**使用模型**：OpenAI GPT-4o（自動擷取最多 20 張關鍵畫面分析）
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="上傳影片",
                    sources=["upload"],
                )

                api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    placeholder="貼上你的 OpenAI API Key（sk-...）",
                    type="password",
                    visible=not has_env_key,
                    value="",
                )

                if has_env_key:
                    gr.Markdown("✅ OpenAI API Key 已從環境變數載入")

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
            "---\n使用 [OpenAI GPT-4o](https://openai.com) 提供影片分析能力"
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
