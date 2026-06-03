import os
import time
import base64
import cv2
import numpy as np
import gradio as gr
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

ANALYSIS_PROMPT = """請仔細分析這些影片畫面，找出其中出現的所有野生動物（wild animals）。

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

IR_HINT = "\n\n注意：這些是紅外線夜視影像，已經過對比增強與假彩色處理。動物通常呈現較亮的熱源區域，請根據輪廓和形態辨識。"

MAX_FRAMES = 24
FRAME_INTERVAL_SEC = 2.0


# ── 影像前處理 ──────────────────────────────────────────────

def enhance_ir_frame(frame: np.ndarray) -> np.ndarray:
    """紅外線影像增強：CLAHE + 銳化 + 假彩色"""
    # 轉灰階
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    # CLAHE 局部對比增強（最重要的步驟）
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 輕微銳化，強化動物輪廓
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # 假彩色：INFERNO 色板（冷→熱 = 黑→黃白）讓動物熱源區域更明顯
    colored = cv2.applyColorMap(sharpened, cv2.COLORMAP_INFERNO)
    return colored


def extract_frames(video_path: str, ir_mode: bool) -> list[str]:
    """擷取影片關鍵幀，IR 模式時套用增強處理"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    interval = max(1, int(fps * FRAME_INTERVAL_SEC))
    if total_frames / interval > MAX_FRAMES:
        interval = max(1, total_frames // MAX_FRAMES)

    frames_b64 = []
    idx = 0
    while cap.isOpened() and len(frames_b64) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            processed = enhance_ir_frame(frame) if ir_mode else frame
            _, buf = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames_b64.append(base64.b64encode(buf).decode("utf-8"))
        idx += 1

    cap.release()
    duration = total_frames / fps
    print(f"擷取 {len(frames_b64)} 幀（影片 {duration:.1f}s，IR 模式：{ir_mode}）")
    return frames_b64


# ── 分析邏輯 ────────────────────────────────────────────────

def analyze_video(video_path, api_key: str, ir_mode: bool) -> str:
    # Gradio 6 有時回傳 dict，取出實際路徑
    if isinstance(video_path, dict):
        video_path = video_path.get("video", video_path.get("name", ""))
    if not video_path:
        return "請先上傳一段影片"

    resolved_key = os.getenv("GOOGLE_API_KEY") or api_key.strip()
    if not resolved_key:
        return "請提供 Google API Key"

    client = genai.Client(api_key=resolved_key)

    if ir_mode:
        # IR 模式：提取幀 → 增強 → 以圖片清單送出（temperature=0 確保一致性）
        frames_b64 = extract_frames(video_path, ir_mode=True)
        if not frames_b64:
            return "無法讀取影片，請確認格式"

        prompt = ANALYSIS_PROMPT + IR_HINT
        parts = [types.Part(text=prompt)]
        for fb64 in frames_b64:
            parts.append(types.Part(
                inline_data=types.Blob(
                    mime_type="image/jpeg",
                    data=base64.b64decode(fb64),
                )
            ))

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=parts,
                config=types.GenerateContentConfig(temperature=0),
            )
            return response.text
        except Exception as e:
            return f"分析過程發生錯誤：{e}"

    else:
        # 一般模式：直接上傳影片（Gemini 原生支援）
        video_file = None
        try:
            print(f"上傳影片：{video_path}")
            video_file = client.files.upload(file=video_path)

            while video_file.state.name == "PROCESSING":
                print("影片處理中...")
                time.sleep(3)
                video_file = client.files.get(name=video_file.name)

            if video_file.state.name == "FAILED":
                return "影片處理失敗，請確認格式"

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[video_file, ANALYSIS_PROMPT],
                config=types.GenerateContentConfig(temperature=0),
            )
            return response.text

        except Exception as e:
            return f"分析過程發生錯誤：{e}"

        finally:
            if video_file:
                try:
                    client.files.delete(name=video_file.name)
                except Exception:
                    pass


# ── Gradio UI ───────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    has_env_key = bool(os.getenv("GOOGLE_API_KEY"))

    with gr.Blocks(title="野生動物偵測器") as demo:
        gr.Markdown(
            """
# 🦁 野生動物偵測器
### 上傳一段影片，AI 將為你辨識其中出現的野生動物

**支援格式**：MP4、MOV、AVI、MKV 等常見影片格式 ｜ **使用模型**：Google Gemini 2.5 Flash
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="上傳影片", sources=["upload"])

                ir_checkbox = gr.Checkbox(
                    label="🌡️ 紅外線 / 夜視模式",
                    value=False,
                    info="勾選後自動套用 CLAHE 對比增強 + 假彩色處理，改善夜視影像辨識準確度",
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

                analyze_btn = gr.Button("🔍 開始分析", variant="primary", size="lg")

            with gr.Column(scale=1):
                result_output = gr.Markdown(
                    label="分析結果",
                    value="分析結果將顯示於此...",
                )

        analyze_btn.click(
            fn=analyze_video,
            inputs=[video_input, api_key_input, ir_checkbox],
            outputs=result_output,
        )

        gr.Markdown(
            "---\n使用 [Google Gemini API](https://ai.google.dev/) 提供影片分析能力"
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
