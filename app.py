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
（描述動物的出現時間點、棲息環境等）

---

如果影片中完全沒有野生動物，請只回應：
沒有看到任何野生動物"""

IR_HINT = "\n\n注意：這些是紅外線夜視影像，已套用 CLAHE 對比增強與假彩色處理。"
BG_HINT = "\n\n注意：畫面中的【綠色方框】是系統透過背景減法自動標示的移動前景區域，可能是動物所在位置，請優先分析這些區域。"


def build_prompt(ir_mode: bool, has_bg: bool, location: str, species_hint: str) -> str:
    """組合完整 prompt，注入地點與物種情境"""
    parts = []

    # 地點情境
    if location.strip():
        parts.append(f"【拍攝地點】{location.strip()}")

    # 物種候選清單
    if species_hint.strip():
        parts.append(
            f"【該地區可能出現的物種】{species_hint.strip()}\n"
            "請優先對照上述物種清單進行辨識，若畫面模糊或特徵不明確，"
            "請根據地點、體型、行為推測最可能的物種並說明理由。"
        )

    context = "\n".join(parts)
    prompt = (f"{context}\n\n" if context else "") + ANALYSIS_PROMPT

    if ir_mode:
        prompt += IR_HINT
    if has_bg:
        prompt += BG_HINT
    return prompt

MAX_FRAMES = 24
FRAME_INTERVAL_SEC = 2.0
MIN_CONTOUR_AREA = 800   # 小於此面積的輪廓視為雜訊忽略


# ── 影像處理工具 ────────────────────────────────────────────

def enhance_ir_frame(frame: np.ndarray) -> np.ndarray:
    """紅外線影像增強：CLAHE + 銳化 + 假彩色"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = np.clip(cv2.filter2D(enhanced, -1, kernel), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(sharpened, cv2.COLORMAP_INFERNO)


def draw_fg_boxes(frame: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """在前景遮罩的輪廓外圍畫綠框"""
    result = frame.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) >= MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(result, "?", (x + 4, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return result


def build_bg_subtractor(bg_video_path: str) -> cv2.BackgroundSubtractor | None:
    """用空景影片訓練背景減法模型"""
    cap = cv2.VideoCapture(bg_video_path)
    if not cap.isOpened():
        return None
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=40, detectShadows=False
    )
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        subtractor.apply(frame)
        count += 1
    cap.release()
    print(f"背景模型已用 {count} 幀訓練完成")
    return subtractor


def extract_frames(
    video_path: str,
    ir_mode: bool,
    subtractor: cv2.BackgroundSubtractor | None = None,
) -> list[str]:
    """擷取並處理影片幀"""
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
            processed = frame.copy()

            # 1. 背景減法：畫綠框標出前景
            if subtractor is not None:
                fg_mask = subtractor.apply(frame)
                processed = draw_fg_boxes(processed, fg_mask)

            # 2. IR 增強：CLAHE + 假彩色
            if ir_mode:
                processed = enhance_ir_frame(processed)

            _, buf = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames_b64.append(base64.b64encode(buf).decode("utf-8"))
        idx += 1

    cap.release()
    print(f"擷取 {len(frames_b64)} 幀（{total_frames/fps:.1f}s，IR={ir_mode}，背景減法={subtractor is not None}）")
    return frames_b64


# ── 分析邏輯 ────────────────────────────────────────────────

def analyze_video(video_path, api_key: str, ir_mode: bool, bg_video_path,
                  location: str = "", species_hint: str = "") -> str:
    # Gradio 6 有時回傳 dict
    if isinstance(video_path, dict):
        video_path = video_path.get("video", video_path.get("name", ""))
    if isinstance(bg_video_path, dict):
        bg_video_path = bg_video_path.get("video", bg_video_path.get("name", ""))

    if not video_path:
        return "請先上傳一段影片"

    resolved_key = os.getenv("GOOGLE_API_KEY") or api_key.strip()
    if not resolved_key:
        return "請提供 Google API Key"

    # 建立背景模型（如有提供空景影片）
    subtractor = None
    if bg_video_path:
        subtractor = build_bg_subtractor(bg_video_path)
        if subtractor is None:
            return "空景影片無法讀取，請確認格式"

    use_frame_mode = ir_mode or (subtractor is not None)
    client = genai.Client(api_key=resolved_key)

    if use_frame_mode:
        # 幀模式：本機處理後送圖片清單
        frames_b64 = extract_frames(video_path, ir_mode=ir_mode, subtractor=subtractor)
        if not frames_b64:
            return "無法讀取影片，請確認格式"

        prompt = build_prompt(ir_mode, subtractor is not None, location, species_hint)

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
        # 一般模式：直接上傳影片
        video_file = None
        try:
            video_file = client.files.upload(file=video_path)
            while video_file.state.name == "PROCESSING":
                time.sleep(3)
                video_file = client.files.get(name=video_file.name)
            if video_file.state.name == "FAILED":
                return "影片處理失敗，請確認格式"
            normal_prompt = build_prompt(False, False, location, species_hint)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[video_file, normal_prompt],
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

**支援格式**：MP4、MOV、AVI、MKV 等 ｜ **使用模型**：Google Gemini 2.5 Flash
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="🎬 偵測影片", sources=["upload"])

                gr.Markdown("**🏔️ 空景參考影片**（選填）：上傳同一地點無動物的背景影片，系統用背景減法自動標示前景綠框，提升準確度")
                bg_video_input = gr.Video(
                    label="空景參考影片（選填）",
                    sources=["upload"],
                )

                location_input = gr.Textbox(
                    label="📍 拍攝地點（選填）",
                    placeholder="例：台灣南投縣蓮花池、非洲肯亞馬賽馬拉",
                )
                species_input = gr.Textbox(
                    label="🐾 可能出現的物種（選填）",
                    placeholder="例：台灣黑熊、山羌、石虎、藍腹鷴",
                    lines=2,
                )
                ir_checkbox = gr.Checkbox(
                    label="🌡️ 紅外線 / 夜視模式",
                    value=False,
                    info="套用 CLAHE 對比增強 + 假彩色處理，改善夜視影像辨識",
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
            inputs=[video_input, api_key_input, ir_checkbox, bg_video_input,
                    location_input, species_input],
            outputs=result_output,
        )

        gr.Markdown("""
---
**使用技巧**
- 單獨勾選「紅外線/夜視模式」：改善 IR 影像對比
- 單獨上傳「空景參考影片」：背景減法標出移動前景（綠框）
- 兩者同時啟用：先標出前景，再套用 IR 增強，效果最好

使用 [Google Gemini API](https://ai.google.dev/) 提供影片分析能力
""")

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
