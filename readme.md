# TeenCode Translator (BARTpho)

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/sonleuid1/TeenCode-Translator-BARTpho)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://www.gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TeenCode Translator** is an NLP system that normalizes Vietnamese social-media language — GenZ teencode, abbreviations, and slang from Threads/TikTok — into **standard Vietnamese**.

It is built by fine-tuning **[BARTpho](https://github.com/VinAIResearch/BARTpho)** (`vinai/bartpho-syllable`) with a **two-phase training pipeline**, **loss-based noise filtering**, and **active-learning style hard-example refinement**. Inference is exposed through a **real-time Gradio** web app with sentence-level chunking and user-feedback collection for later retrain cycles.

| | |
|---|---|
| **Author** | Lê Đan Sơn (IT-E10 K69 HUST) |
| **Stack** | PyTorch · Hugging Face Transformers · Gradio |
| **Base model** | `vinai/bartpho-syllable` |
| **Published model** | [`sonleuid1/TeenCode-Translator-BARTpho`](https://huggingface.co/sonleuid1/TeenCode-Translator-BARTpho) |
| **Task** | Seq2Seq lexical / stylistic normalization (teencode → standard VN) |

---

## Demo video

Local demo recording of the Gradio UI (real-time translation + example inputs):

https://github.com/sonle17092006/TeenCodeTranslator/raw/data_handling/docs/demo.mp4

> File in repo: [`docs/demo.mp4`](docs/demo.mp4)  
> Original capture: `2026-08-05 15-08-05.mp4`

**What the demo shows**

1. Model load from Hugging Face and Gradio UI startup  
2. Clicking prepared teencode examples  
3. Real-time translation as text is entered  
4. Optional **Active Learning** feedback panel (save corrections for a future phase)

---

## Features

| Feature | Technical detail |
|--------|-------------------|
| **Real-time inference UI** | Gradio `Blocks` app; `input.change` triggers translation on every edit |
| **Long-text chunking** | Split on punctuation `[.,;!?\n]` so each segment stays within the model’s max length; reduces mid-sentence truncation and hallucination |
| **Native PyTorch generate** | `AutoModelForSeq2SeqLM.generate` with beam search (not a black-box pipeline-only path) |
| **Active Learning feedback** | User corrections appended to `feedback_data.csv` (`text`, `target`) for Phase-2 style retrain |
| **Device auto-select** | CUDA if available, else CPU (`MODEL_ID` overridable via env) |
| **Demo examples** | Built-in teencode samples for quick evaluation / recording |

---

## Quick start (run demo — no training)

```bash
cd TeenCodeTranslator

# optional virtualenv
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:7860/**

| Option | Command |
|--------|---------|
| Custom port | `python app.py --port 7861` |
| Public share link | `python app.py --share` |
| Custom model | `set MODEL_ID=sonleuid1/TeenCode-Translator-BARTpho` then `python app.py` |

**Dependencies of note**

- `sentencepiece` — required by **BartphoTokenizer** (will fail at load if missing)
- `torch` + `transformers` — model runtime  
- First run downloads weights from the Hugging Face Hub (network required)

---

## Architecture

### Problem formulation

Given a noisy Vietnamese social-media utterance \(x\) (teencode / slang / typos), the model produces a normalized sequence \(y\):

\[
x_{\text{teencode}} \;\xrightarrow{\;f_\theta\;}\; y_{\text{standard VN}}
\]

This is cast as **sequence-to-sequence** generation (encoder–decoder), not token classification. That choice preserves word order, punctuation, emojis, and foreign spans better than pure dictionary rewrite.

### Why BARTpho?

| Choice | Rationale |
|--------|-----------|
| **BARTpho syllable** | Pretrained Vietnamese BART; strong monolingual prior for VN morphology and informal orthography |
| **Syllable tokenization** | Vietnamese is typically segmented by syllables; matches social text better than English-centric WP-only setups for this domain |
| **Encoder–decoder** | Natural fit for “rewrite the whole sentence” normalization vs. mask-fill only |

**Inference model ID:** `sonleuid1/TeenCode-Translator-BARTpho`  
**Training base:** `vinai/bartpho-syllable`

### Inference pipeline (runtime)

```
User text
   │
   ▼
Chunk by punctuation  ── keep delimiters ──► [c1, p1, c2, p2, ...]
   │
   ▼  (per non-punctuation chunk)
Tokenizer (max_length=128, truncation)
   │
   ▼
model.generate(num_beams=5, early_stopping=True, max_length=128)
   │
   ▼
Decode (skip special tokens)
   │
   ▼
Post-rule: if output contains "mày", map formal "tôi"/"mình" → "tao"
   │
   ▼
Rejoin chunks → UI output
```

**Decode / search**

- **Beam search** (`num_beams=5`): better for short rewrites than pure greedy; reduces random rare-token slips  
- **`early_stopping`**: stop when all beams finish  
- **`max_length=128`**: cap per chunk; long posts rely on chunking instead of one huge generation

**Post-processing rule (“mày–tao”)**  
If the model emits the informal second-person *mày*, first-person *tôi/mình* in the same utterance are rewritten to *tao* so register stays consistent (common in VN chat).

---

## Data pipeline

### Sources

| Source | Role | Scale (this repo) |
|--------|------|-------------------|
| **Threads VN comments** | In-domain social text | ~**18,426** cleaned pairs in `process_thread_data/train_data.csv` / `final_cleaned_threads.csv` (CV / overview often rounds to **~18.6k**) |
| **Teencode lexicon** | Frequency list for injection / analysis | `process_thread_data/Teen_Code_Frequency.txt` |
| **Synthetic noise injection** | Increase teencode density for training | `teencode_0.7.csv` — inject up to **~70%** teencode tokens per sentence, target average density **~25%** |
| **Phase-2 hard set** | Difficult + augmented examples | `fine_tune/data_phase2.csv` (~4.1k) |
| **Loss audit tables** | Noise / hard-example mining | `fine_tune/high_loss.csv`, `fine_tune/bad_teencode.csv` |

Pair format used throughout training artifacts:

```text
text   → noisy / teencode input
target → standard Vietnamese reference
```

### Teencode injection (noise synthesis)

Heuristic pipeline (see `process_thread_data/`):

1. Detect standard forms that map to known teencode variants (frequency list).  
2. Replace a controlled fraction of tokens with teencode spellings.  
3. Cap density (e.g. 0.7 ceiling) so sentences remain learnable, not pure noise.  
4. Export parallel `text` / `target` for supervised seq2seq.

This expands coverage beyond naturally occurring teencode in raw Threads scrapes.

### Loss-based noise filtering

After Phase-1 (or an intermediate checkpoint), each sample can be scored with teacher-forced **cross-entropy**:

| Signal | Use |
|--------|-----|
| **Sentence loss** | Average CE over target tokens — overall difficulty / fit |
| **Per-token (word) loss** | Local spikes often mark bad labels, OCR-like garbage, or impossible targets |
| **Max word loss** | Drop if **> 13.0** (README/training policy) — treat as label noise |
| **Golden / hard band** | Prefer **0.1 < sentence loss < 2.0** — hard but learnable; not trivial and not pure noise |

Artifacts:

- `fine_tune/high_loss.csv` — samples with loss breakdowns  
- `fine_tune/bad_teencode.csv` — teencode tokens correlated with high loss  
- `fine_tune/loss_histograms_analysis.png` — distribution diagnostics  

**Idea:** high max-token loss often means “model and label disagree violently” (wrong target or rare junk), not “useful hard case.” Filtering those improves Phase-2 signal.

---

## Training methodology

### Phase 1 — full fine-tune (breadth)

| Hyperparameter | Value | Notes |
|----------------|-------|--------|
| Base | `vinai/bartpho-syllable` | HF checkpoint |
| Objective | Seq2Seq LM loss (CE on decoder) | Standard `Seq2SeqTrainer` |
| Learning rate | **5e-5** (also LR sweeps 2e-5 … 1e-4 in trials) | Logs under `fine_tune/logs/Trial_LR_*` |
| Optimizer | **AdamW** | HF Trainer default; weight decay 0.01 |
| Precision | **fp16** mixed precision | Lower VRAM, faster on GPU |
| Batch | `per_device=16` × `grad_accum=4` → **effective 64** | Memory-friendly on 16GB |
| Schedule | cosine + warmup ratio ~0.1 | Stable late training |
| Extra | gradient checkpointing, early stopping, label smoothing 0.1 | Regularization + VRAM |
| Hardware (local) | NVIDIA **RTX 4060 Ti 16GB** | TensorBoard events from local desktop runs |
| Cloud | **Vast.ai / iRender** (optional acceleration) | Same scripts/hyperparams on rented GPUs |

Checkpoints and trial logs live under `fine_tune/logs/` (e.g. `Trial_FullFT_LR_5e-05`, multiple LR trials).

### Phase 2 — hard examples / active learning (depth)

| Item | Detail |
|------|--------|
| Goal | Specialize on failures without catastrophic forgetting |
| LR | **2e-5** (lower than Phase 1) |
| Data | Hard band from loss filter + pattern mining (`cf`, `hnao`, `htrc`, …) + optional Gradio `feedback_data.csv` + `data_phase2.csv` |
| Log name | `Phase2_HardExamples_LR_2e-5` |

**Active Learning (practical loop used here)**

1. Deploy / run Phase-1 model.  
2. Mine high-loss / wrong-pattern samples; collect human corrections in the UI.  
3. Build a focused Phase-2 set.  
4. Fine-tune with lower LR so general Phase-1 knowledge is retained.

This is **hard-example + human-in-the-loop refinement**, not a full uncertainty-sampling research AL stack — but it matches the product loop (Gradio feedback → next train).

### Optimization summary (memory & speed)

| Technique | Effect |
|-----------|--------|
| **fp16** | ~½ activation memory vs fp32; higher throughput on Tensor Cores |
| **Gradient accumulation** | Large effective batch without OOM |
| **Gradient checkpointing** | Trade compute for VRAM |
| **AdamW** | Decoupled weight decay; standard for Transformers FT |
| **Chunked inference** | Keeps generation short and stable at serve time |

Reported training-side snapshot (project docs / runs): eval loss on the order of **~0.20** at strong checkpoints; local inference throughput depends on GPU vs CPU (demo may run on CPU if CUDA wheels are not installed).

---

## Project structure

```text
TeenCodeTranslator/
├── app.py                      # Entry: python app.py
├── requirements.txt
├── readme.md
├── docs/
│   └── demo.mp4                # Screen demo (Gradio UI)
├── visualization/
│   ├── gui.py                  # Gradio UI, chunking, feedback, generate()
│   └── __init__.py
├── process_thread_data/        # Threads cleaning, inject, train CSV artifacts
│   ├── train_data.csv
│   ├── final_cleaned_threads.csv
│   ├── teencode_0.7.csv
│   ├── Teen_Code_Frequency.txt
│   └── *.ipynb                 # historical processing notebooks
└── fine_tune/                  # Training notebooks, loss tables, TensorBoard logs
    ├── fine_tune_phase1.ipynb
    ├── fine_tune_phase2.ipynb
    ├── data_phase2.csv
    ├── high_loss.csv
    ├── bad_teencode.csv
    ├── loss_histograms_analysis.png
    └── logs/                   # Trial_* , Phase2_* event files
```

---

## Python API (embed without Gradio)

```python
from transformers import pipeline

translator = pipeline(
    "text2text-generation",
    model="sonleuid1/TeenCode-Translator-BARTpho",
    device=0,  # GPU; use -1 or omit for CPU depending on transformers version
)

text = "ck oi zk thik ik un tsua =))"
result = translator(text, max_length=64, num_beams=5, early_stopping=True)
print(result[0]["generated_text"])
```

For production parity with the demo app (chunking + mày–tao rule), prefer `visualization.gui.process_long_text` after the model is loaded.

---

## Limitations & safety

- **No profanity filter** — vulgar language may be preserved by design (faithful social-text normalization). Downstream apps (education, minors) should add their own moderation.  
- **Vietnamese teencode → standard VN only** — not a general EN↔VI translator.  
- **Domain shift** — best on short social comments; legal/medical formal text is out of scope.  
- **CPU demos** are correct but slower; use a CUDA build of PyTorch for interactive speed.  
- **Label noise** remains possible even after loss filtering; feedback loop is intended to keep improving hard cases.

---

## Citation / model card

If you use this project or the published weights:

- **Model:** https://huggingface.co/sonleuid1/TeenCode-Translator-BARTpho  
- **Base:** VinAI BARTpho — https://github.com/VinAIResearch/BARTpho  
- **Repo:** https://github.com/sonle17092006/TeenCodeTranslator  

---

## License

MIT (see badge). Respect Hugging Face / BARTpho upstream licenses when redistributing weights.
