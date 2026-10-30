> 贡献者：[SunMuzihang] ([2260031490])

<div align="center">

# Transformer From Scratch

**A from-scratch PyTorch reproduction of “Attention Is All You Need”.**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-2EA44F)](LICENSE)

</div>

---

## Overview

This repository is a week-by-week, from-scratch implementation of the original Transformer architecture (Vaswani et al., 2017). Rather than wrapping the built-in `torch.nn.Transformer`, every component is written and tested explicitly so the code can be read side by side with the paper.

## Current Progress

**Week 1 — complete**

Implemented the low-level building blocks:

- **Multi-Head Attention** (`model/attention.py`)
  - Scaled dot-product attention with `1 / sqrt(d_k)` scaling
  - Self-attention and encoder–decoder cross-attention
  - Padding masks and causal (look-ahead) masks
  - Optional attention-weight output for visualization
- **Positional Encoding** (`model/embedding.py`)
  - Fixed sinusoidal encoding from the paper
  - Learned positional embeddings (for the planned ablation)
  - Token embedding scaled by `sqrt(d_model)`

The implementation was verified locally with PyTorch 2.11 + CUDA 12.8.

## Project Structure

```text
.
├── model/
│   ├── __init__.py        # Public exports
│   ├── attention.py       # Scaled dot-product & multi-head attention
│   └── embedding.py       # Token embedding + sinusoidal/learned PE
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Environment Setup

```bash
git clone https://github.com/sunmu-length/transformer-from-scratch.git
cd transformer-from-scratch

python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

## Quick Verification

Run the smoke test from the repository root:

```bash
python -c "
import torch
from model import MultiHeadAttention, TransformerEmbedding, causal_mask

x = torch.randn(2, 5, 64)
mha = MultiHeadAttention(64, 8, dropout=0.0)
y, w = mha(x, x, x, attn_mask=causal_mask(5), need_weights=True)
print(y.shape, w.shape)

emb = TransformerEmbedding(1000, 64, positional_encoding='learned')
print(emb(torch.randint(0, 1000, (2, 5))).shape)
"
```

Expected output:

```text
torch.Size([2, 5, 64]) torch.Size([2, 5, 5])
torch.Size([2, 5, 64])
```

## Roadmap

- [x] **Week 1** — Multi-Head Attention and positional encoding
- [ ] **Week 2** — Encoder, Decoder, and the full-model forward pass
- [ ] **Week 3** — Training script, data pipeline, loss, and optimizer schedule
- [ ] **Week 4** — Ablation experiments, results, and documentation

## License

Released under the [MIT License](LICENSE).
