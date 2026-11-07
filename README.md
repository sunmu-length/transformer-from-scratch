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

**Week 2 — complete**

Assembled the full encoder–decoder model:

- **Feed-forward network** (`model/feed_forward.py`)
  - Position-wise two-layer MLP with ReLU and dropout
- **Shared building blocks** (`model/layers.py`)
  - Post-LN residual connections: `LayerNorm(x + dropout(sublayer(x)))`
  - Layer cloning and padding-mask utilities
- **Encoder** (`model/encoder.py`)
  - Self-attention sublayer followed by a feed-forward sublayer
- **Decoder** (`model/decoder.py`)
  - Masked self-attention, encoder–decoder cross-attention, and feed-forward sublayers
- **Full Transformer** (`model/transformer.py`)
  - Automatic causal and padding masks
  - Output projection tied to the target embedding
  - Optional source/target embedding sharing
- **Forward-pass test suite** (`tests/test_forward.py`)
  - 13 tests covering shapes, causality, padding invariance, NaN safety,
    gradient flow, and weight tying

Both weeks were verified locally with PyTorch 2.11 + CUDA 12.8.

## Project Structure

```text
.
├── model/
│   ├── __init__.py        # Public exports
│   ├── attention.py       # Scaled dot-product & multi-head attention
│   ├── embedding.py       # Token embedding + sinusoidal/learned PE
│   ├── feed_forward.py    # Position-wise feed-forward network
│   ├── layers.py          # Post-LN sublayers, cloning, padding masks
│   ├── encoder.py         # Encoder layer and stack
│   ├── decoder.py         # Decoder layer and stack
│   └── transformer.py     # Full encoder–decoder model
├── tests/
│   └── test_forward.py    # Forward-pass and invariant tests
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

Run the full-model smoke test from the repository root:

```bash
python -c "
import torch
from model import Transformer

model = Transformer(
    100, 100,
    d_model=64, num_heads=8,
    num_encoder_layers=2, num_decoder_layers=2,
    d_ff=128, dropout=0.0, pad_idx=0,
)
src = torch.randint(1, 100, (2, 7))
tgt = torch.randint(1, 100, (2, 5))
print(model(src, tgt).shape)
"
```

Expected output:

```text
torch.Size([2, 5, 100])
```

Then run the test suite:

```bash
python -m pytest -q
```

Expected output:

```text
13 passed
```

## Roadmap

- [x] **Week 1** — Multi-Head Attention and positional encoding
- [x] **Week 2** — Encoder, Decoder, and the full-model forward pass
- [ ] **Week 3** — Training script, data pipeline, loss, and optimizer schedule
- [ ] **Week 4** — Ablation experiments, results, and documentation

## License

Released under the [MIT License](LICENSE).
