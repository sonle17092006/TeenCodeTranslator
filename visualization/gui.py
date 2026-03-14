import os
import re
import csv
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Cấu hình môi trường để tránh lỗi xung đột
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 2. KHỞI TẠO LÕI AI TỪ HUGGING FACE (Bypass hoàn toàn lỗi Pipeline)
repo_id = "sonleuid1/TeenCode-Translator-BARTpho"
print(f"🚀 Đang tải mô hình từ {repo_id}. Vui lòng đợi...")

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)

# Ép model chạy trên GPU (RTX 4060 Ti) nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval() # Bật chế độ suy luận để tiết kiệm RAM

# 3. HÀM XỬ LÝ: CHUNKING & DỊCH BẰNG Pytorch Native
def process_long_text(input_text):
    if not input_text.strip():
        return ""
    
    # Băm nhỏ đoạn văn dựa trên dấu câu
    chunks = re.split(r'([.,;!?\n]+)', input_text)
    translated_chunks = []
    
    for chunk in chunks:
        if re.match(r'^[.,;!?\n\s]+$', chunk):
            translated_chunks.append(chunk)
            continue
            
        # Mã hóa và đưa lên GPU
        inputs = tokenizer(
            chunk.strip(), 
            return_tensors="pt", 
            max_length=128, 
            truncation=True
        ).to(device)
        
        # Dịch thuật (Native Generate)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )
            
        # Giải mã kết quả
        ai_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # LUẬT "MÀY - TAO"
        if re.search(r'\bmày\b', ai_output, flags=re.IGNORECASE):
            ai_output = re.sub(r'\btôi\b|\bmình\b', 'tao', ai_output, flags=re.IGNORECASE)
            ai_output = re.sub(r'\bTôi\b|\bMình\b', 'Tao', ai_output)
            
        translated_chunks.append(ai_output)
        
    return "".join(translated_chunks)

# 4. HÀM THU THẬP DỮ LIỆU FEEDBACK
def save_feedback(original_text, corrected_text):
    if not original_text.strip() or not corrected_text.strip():
        return "⚠️ Vui lòng nhập đủ thông tin trước khi báo lỗi!"
    
    file_path = "feedback_data.csv"
    
    # Ghi nối vào file CSV (Format: text,target)
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([original_text, corrected_text])
        
    count = sum(1 for _ in open(file_path, encoding='utf-8'))
    return f"✅ Đã lưu thành công vào kho! (Hiện có {count} câu chờ train Phase sau)"

# 5. GIAO DIỆN CHUẨN DEV (GITHUB DARK MODE)
# 5. GIAO DIỆN CYBERPURPLE (NỀN TÍM SÂU - CHỮ SÁNG NEON)
custom_css = """
/* 1. ÉP NỀN TÍM VŨ TRỤ CHO TOÀN BỘ WEB */
body, .gradio-container, .main {
    background-color: #0f0524 !important; /* Tím than cực tối */
    background-image: radial-gradient(circle at 20% 30%, #1a0b3c 0%, #0f0524 100%) !important;
    color: #00f2ff !important; /* Chữ mặc định màu Cyan sáng */
}

/* 2. DỌN SẠCH CÁC KHUNG TRẮNG CỦA GRADIO */
.gradio-container .form, .gradio-container .block, .gradio-container .contain {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* 3. TIÊU ĐỀ PHÁT SÁNG MÀU TÍM HỒNG (MAGENTA) */
#main-title {
    text-align: center;
    /* Đổi màu chữ sang trắng bạc sáng */
    color: #ffffff !important; 
    font-size: 2.8em;
    font-weight: 900;
    /* Hiệu ứng tỏa sáng Neon Tím rực rỡ */
    text-shadow: 
        0 0 5px #fff, 
        0 0 10px #fff, 
        0 0 20px #ff00ff, 
        0 0 30px #ff00ff, 
        0 0 40px #ff00ff;
    margin-bottom: 10px;
    letter-spacing: 5px;
    text-transform: uppercase;
}

/* 4. Ô NHẬP LIỆU: NỀN TÍM ĐẬM, VIỀN CYAN PHÁT SÁNG */
.gradio-container textarea {
    background-color: #1a0b3c !important;
    color: #00f2ff !important;
    border: 2px solid #3d1a6d !important;
    border-radius: 12px !important;
    font-family: 'Consolas', monospace !important;
    box-shadow: inset 0 0 10px rgba(0,0,0,0.5) !important;
}

.gradio-container textarea:focus {
    border-color: #00f2ff !important;
    box-shadow: 0 0 15px rgba(0, 242, 255, 0.5) !important;
    outline: none !important;
}

/* 5. NHÃN (LABEL) MÀU TÍM SÁNG */
.gradio-container label span {
    color: #bd93f9 !important;
    background-color: transparent !important;
    font-size: 14px !important;
    font-weight: bold !important;
    text-transform: uppercase;
}

/* 6. NÚT BẤM TÍM NEON RỰC RỠ */
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

print("🚀 Đang khởi động giao diện CyberPurple...")
with gr.Blocks(theme=gr.themes.Base(), css=custom_css, js="() => document.body.classList.add('dark')") as demo:
    gr.Markdown("<div id='main-title'>TEENCODE TRANSLATOR</div>")
    gr.Markdown("<center><i style='color: #8b949e;'>Lê Đan Sơn IT-E10 K69 HUST</i></center>")
    
    with gr.Row():
        with gr.Column(scale=2):
            input_box = gr.Textbox(lines=8, placeholder="Nhập teencode...", label="📥 INPUT")
            output_box = gr.Textbox(lines=8, label="📤 OUTPUT", interactive=False)
            
        with gr.Column(scale=1):
            gr.Markdown("<b style='color: #ff00ff;'>🛠️ ACTIVE LEARNING</b>")
            correction_box = gr.Textbox(lines=4, label="BẢN DỊCH CHUẨN", placeholder="Sửa lỗi cho AI...")
            btn_save = gr.Button("💾 LƯU FEEDBACK", variant="primary")
            status_text = gr.Markdown("")

    input_box.change(fn=process_long_text, inputs=input_box, outputs=output_box)
    btn_save.click(fn=save_feedback, inputs=[input_box, correction_box], outputs=status_text)

if __name__ == "__main__":
    demo.launch(share=False)