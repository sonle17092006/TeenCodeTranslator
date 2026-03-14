# 🤖 TeenCode Translator BARTpho

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TeenCode Translator** là hệ thống AI chuẩn hóa ngôn ngữ mạng xã hội (Teencode GenZ, viết tắt, tiếng lóng trên Threads/TikTok) sang Tiếng Việt tiêu chuẩn. 

Mô hình được tinh chỉnh (fine-tuned) từ kiến trúc **BARTpho** với phương pháp **Active Learning** và lọc nhiễu dữ liệu bằng **Cross-Entropy Loss**, giúp dịch chuẩn xác các từ lóng phức tạp nhưng vẫn tuân thủ nghiêm ngặt quy tắc bảo lưu ngoại ngữ, dấu câu và từ ngữ nhạy cảm gốc.

👤 **Tác giả:** Lê Đan Sơn (IT-E10 K69 HUST)

---

## ✨ Tính năng nổi bật

- **⚡ Dịch thuật Real-time:** Giao diện Gradio Dark Mode cho phép gõ tới đâu dịch tới đó.
- **🧩 Xử lý văn bản siêu dài (Chunking):** Tự động băm nhỏ câu dựa trên dấu câu để vượt qua giới hạn độ dài token của mô hình, chống hiện tượng ảo giác (Hallucination).
- **📈 Thu thập Feedback:** Tích hợp tính năng lưu dữ liệu câu sai/câu sửa trực tiếp trên giao diện để phục vụ huấn luyện các phiên bản sau.

---

## 🚀 Cài đặt và Chạy thử (Local)

**Bước 1: Clone kho lưu trữ này về máy**
```bash
git clone [https://github.com/sonle17092006/TeenCodeTranslator.git](https://github.com/sonle17092006/TeenCodeTranslator.git)
cd TeenCodeTranslator
```

**Bước 2: Cài đặt các thư viện cần thiết**
```bash
pip install -r requirements.txt
```

**Bước 3: Khởi chạy phần mềm (Giao diện Web)**
```bash
python app.py
```
*Phần mềm sẽ mở ra tại địa chỉ `http://127.0.0.1:7860/` trên trình duyệt của bạn.*

---

## 💻 Hướng dẫn sử dụng qua Python Pipeline

Nếu bạn muốn tích hợp lõi AI này vào một ứng dụng khác, hãy sử dụng đoạn mã sau:

```python
from transformers import pipeline

translator = pipeline(
    "text2text-generation", 
    model="sonleuid1/TeenCode-Translator-BARTpho", # Thay bằng link HF của bạn
    device=0 # Bật GPU nếu có
)

# Text đầu vào (chứa teencode, emoji và lóng)
text = "ck oi zk thik ik ún tsua =))"

# Chạy inference
result = translator(text, max_length=64, num_beams=5, early_stopping=True)

print(result[0]['generated_text'])
# Output: "chồng ơi vợ thích đi uống trà sữa =))"
```

---

## 🧠 Kiến trúc và Huấn luyện (Training Methodology)

### 1. Dữ liệu (Dataset)
- **Nguồn:** 18.600 bình luận thực tế thu thập từ nền tảng Threads Việt Nam.
- **Inject_noise:** Dùng các quy tắc Heuristic để xác định các teencodes và mật độ của chúng, Inject các teencodes đó ngược lại vào dữ liệu gốc với trần là 70% teencodes trên 1 câu, đạt tỷ lệ trung bình 1 câu có 25% teencodes.
- **Lọc nhiễu (Loss-based Filtering):** - Đánh giá trên 9.000 samples ở Phase 1.
  - Loại bỏ dữ liệu rác/sai nhãn có `Max Word Loss > 13.0`.
  - Chọn lọc "Golden Dataset" với `0.1 < Sentence Loss < 2.0` (Vùng dữ liệu khó nhưng mô hình có thể học được pattern).

### 2. Quy trình huấn luyện (2 Phases)
- **Phase 1 (Full Fine-tune):** Huấn luyện bề rộng trên toàn bộ tập dữ liệu gốc (Learning Rate: 5e-5).
- **Phase 2 (Hard Examples Fine-tune / Active Learning):** Ép mô hình học sâu vào các câu khó và dữ liệu nhân tạo tự sinh (các cụm từ hay dịch sai như `cf`, `hnao`, `htrc`) với Learning Rate thấp (2e-5) để chống hội chứng quên (Catastrophic Forgetting).

### 3. Thông số hệ thống
- **Phần cứng:** 1x NVIDIA GeForce RTX 4060 Ti (16GB VRAM)
- **Đào tạo:** fp16 mixed precision, Batch Size 64, AdamW, 5 Epochs.
- **Kết quả:** Đạt Eval Loss ~0.200 tại Checkpoint 464, tốc độ Inference đạt ~512 samples/giây.

---

## ⚠️ Giới hạn & Cảnh báo
- **Không kiểm duyệt:** Hệ thống không tích hợp bộ lọc từ ngữ thô tục (Profanity Filter). Người tích hợp (Downstream users) cần lưu ý khi ứng dụng vào môi trường giáo dục/trẻ em.
- **Đa ngôn ngữ:** Mô hình chỉ hoạt động hiệu quả cho cặp Teencode -> Tiếng Việt chuẩn, không có khả năng dịch Anh-Việt.
