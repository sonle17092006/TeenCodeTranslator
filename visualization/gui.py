"""
TeenCode Translator — Gradio real-time inference UI.
Model: sonleuid1/TeenCode-Translator-BARTpho (BARTpho fine-tuned).
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "sonleuid1/TeenCode-Translator-BARTpho")
MAX_LENGTH = 128
NUM_BEAMS = 5
FEEDBACK_PATH = Path(os.environ.get("FEEDBACK_PATH", "feedback_data.csv"))

# Demo examples (good for screen recording)
EXAMPLES = [
    "ck oi zk thik ik un tsua =))",
    "hqua t ms di test makeup look ny nhin chay pho z dk choi =]]]",
    "ao ny depppp qa mng oiiii, 10 diemmmm",
    "pass lai doi giay sz 39 gia 150 canh, ae nao chot ib le",
    "dm shop lam an ncc y, mua hang phi tienn vcl 1 sao!",
]

# ---------------------------------------------------------------------------
# Model (lazy load so import stays fast if needed)
# ---------------------------------------------------------------------------
_tokenizer = None
_model = None
_device = None


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_id: str = MODEL_ID):
    """Load tokenizer + model once; reuse across requests."""
    global _tokenizer, _model, _device
    if _model is not None:
        return _tokenizer, _model, _device

    _device = get_device()
    print(f"[TeenCode] Loading model: {model_id}")
    print(f"[TeenCode] Device: {_device}")
    _tokenizer = AutoTokenizer.from_pretrained(model_id)
    _model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    _model = _model.to(_device)
    _model.eval()
    print("[TeenCode] Model ready.")
    return _tokenizer, _model, _device


def translate_chunk(text: str, tokenizer, model, device) -> str:
    inputs = tokenizer(
        text.strip(),
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            num_beams=NUM_BEAMS,
            early_stopping=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def apply_may_tao_rule(text: str) -> str:
    """If output contains 'mày', prefer 'tao' over formal first-person pronouns."""
    if re.search(r"\bmày\b", text, flags=re.IGNORECASE):
        text = re.sub(r"\btôi\b|\bmình\b", "tao", text, flags=re.IGNORECASE)
        text = re.sub(r"\bTôi\b|\bMình\b", "Tao", text)
    return text


def process_long_text(input_text: str) -> str:
    """Chunk long input by punctuation, translate each part, rejoin."""
    if not input_text or not input_text.strip():
        return ""

    tokenizer, model, device = load_model()
    chunks = re.split(r"([.,;!?\n]+)", input_text)
    translated_chunks: list[str] = []

    for chunk in chunks:
        if re.match(r"^[.,;!?\n\s]+$", chunk):
            translated_chunks.append(chunk)
            continue
        if not chunk.strip():
            continue
        out = translate_chunk(chunk, tokenizer, model, device)
        out = apply_may_tao_rule(out)
        translated_chunks.append(out)

    return "".join(translated_chunks)


def save_feedback(original_text: str, corrected_text: str) -> str:
    """Append user correction for later active-learning / phase-2 fine-tune."""
    if not (original_text or "").strip() or not (corrected_text or "").strip():
        return "Please fill both the input and the corrected translation."

    file_exists = FEEDBACK_PATH.exists()
    with FEEDBACK_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["text", "target"])
        writer.writerow([original_text.strip(), corrected_text.strip()])

    with FEEDBACK_PATH.open(encoding="utf-8") as f:
        # header + data rows
        n = max(0, sum(1 for _ in f) - 1)
    return f"Saved. {n} feedback pair(s) ready for the next training phase."


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
body, .gradio-container, .main {
    background-color: #0f0524 !important;
    background-image: radial-gradient(circle at 20% 30%, #1a0b3c 0%, #0f0524 100%) !important;
    color: #00f2ff !important;
}
.gradio-container .form, .gradio-container .block, .gradio-container .contain {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#main-title {
    text-align: center;
    color: #ff00ff !important;
    font-size: 2.5em;
    font-weight: 900;
    text-shadow: 0 0 15px #ff00ff, 0 0 5px #ffffff;
    margin-bottom: 5px;
    letter-spacing: 3px;
}
.gradio-container textarea {
    background-color: #1a0b3c !important;
    color: #00f2ff !important;
    border: 2px solid #3d1a6d !important;
    border-radius: 12px !important;
    font-family: Consolas, monospace !important;
    box-shadow: inset 0 0 10px rgba(0,0,0,0.5) !important;
}
.gradio-container textarea:focus {
    border-color: #00f2ff !important;
    box-shadow: 0 0 15px rgba(0, 242, 255, 0.5) !important;
    outline: none !important;
}
.gradio-container label span {
    color: #bd93f9 !important;
    background-color: transparent !important;
    font-size: 14px !important;
    font-weight: bold !important;
    text-transform: uppercase;
}
.gradio-container button.primary {
    background: linear-gradient(45deg, #8e2de2, #4a00e0) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    font-weight: 800 !important;
    box-shadow: 0 4px 15px rgba(142, 45, 226, 0.4) !important;
    transition: 0.3s !important;
}
.gradio-container button.primary:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px rgba(142, 45, 226, 0.8) !important;
}
"""


def build_demo() -> gr.Blocks:
    """Build Gradio Blocks app (model loads on first inference or preloaded)."""
    with gr.Blocks(
        theme=gr.themes.Base(),
        css=CUSTOM_CSS,
        title="TeenCode Translator",
    ) as demo:
        gr.Markdown("<div id='main-title'>TEENCODE TRANSLATOR</div>")
        gr.Markdown(
            "<center><i style='color: #8b949e;'>"
            "GenZ slang / teencode → Standard Vietnamese · BARTpho · "
            "Lê Đan Sơn · IT-E10 K69 HUST"
            "</i></center>"
        )

        with gr.Row():
            with gr.Column(scale=2):
                input_box = gr.Textbox(
                    lines=8,
                    placeholder="Type teencode here… e.g. ck oi zk thik ik un tsua =))",
                    label="Input (teencode)",
                )
                output_box = gr.Textbox(
                    lines=8,
                    label="Output (standard Vietnamese)",
                    interactive=False,
                )
            with gr.Column(scale=1):
                gr.Markdown(
                    "<b style='color: #ff00ff;'>Active Learning feedback</b><br>"
                    "<span style='color:#8b949e;font-size:0.9em;'>"
                    "Correct a bad translation and save for the next fine-tune phase."
                    "</span>"
                )
                correction_box = gr.Textbox(
                    lines=4,
                    label="Corrected translation",
                    placeholder="Paste the correct Vietnamese here…",
                )
                btn_save = gr.Button("Save feedback", variant="primary")
                status_text = gr.Markdown("")

        gr.Examples(
            examples=EXAMPLES,
            inputs=input_box,
            label="Try an example",
        )

        # Real-time translate while typing (debounce via Gradio default change)
        input_box.change(
            fn=process_long_text,
            inputs=input_box,
            outputs=output_box,
        )
        btn_save.click(
            fn=save_feedback,
            inputs=[input_box, correction_box],
            outputs=status_text,
        )

    return demo


def launch(share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7860):
    """Load model then start Gradio server."""
    load_model()
    demo = build_demo()
    print(f"[TeenCode] Opening UI at http://{server_name}:{server_port}/")
    demo.launch(share=share, server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    launch()
