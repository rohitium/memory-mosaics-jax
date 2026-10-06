# Memory Mosaics in JAX

[Memory Mosaics](https://arxiv.org/abs/2405.06394v3) (Zhang et al., ICLR 2025) and a GPT-2 baseline in
raw `jax.numpy`, written in the style of the [JAX training cookbook](https://docs.jax.dev/en/latest/the-training-cookbook.html),
with code to reproduce the 1-block panel of Fig. 7. The two models are laid out side by side:

```bash
git diff --no-index src/mm_jax/gpt2.py src/mm_jax/memory_mosaic.py
```

| GPT-2 | Memory Mosaic |
|---|---|
| causal self-attention | ContextMem: kernel regression over the sequence's own (key, value) pairs |
| q, k, v: projections of `x_t` | key: leaky average of past projections; value: mixes `x_t` and `x_{t+1}`; both unit-normed |
| query `t` sees keys `≤ t` | query `t` sees keys `< t`, so the value's peek can't leak |
| `1/√d` score scaling | learned per-head bandwidth |
| MLP | PersistentMem: the same retrieval over a learned key/value bank |
| positional embeddings | none |
| init std 0.02; logits `x·Eᵀ` | init std 1/√fan_in; logits `x·Eᵀ/√d` |

Models and training follow the code behind the paper's runs (`Library/` in
[facebookresearch/MemoryMosaics](https://github.com/facebookresearch/MemoryMosaics)); against it, logits
and AdamW updates match to <1e-6. `config.py` defaults are the paper's setup: BabiStories
(474.7M / 4.7M train / val tokens), d = 768, 12 heads, context 512, batch 512, AdamW (0.9, 0.95),
lr 5e-3, 2,000 warmup steps, cosine decay to 1e-4 over 80,000 steps, weight decay 0.1, dropout 0.05.

## Run

```bash
pip install -r requirements.txt
python tests/test_smoke.py
```

[`notebooks/memory_mosaics_colab.ipynb`](notebooks/memory_mosaics_colab.ipynb) runs in ~15 min on a Colab
v5e TPU or T4 GPU: the paper's model, data and optimizer, with batch 32 and as many steps as fit 10
minutes. That covers the start of training, while the paper's curves begin after ~1B tokens, so it shows
early trends, not Fig. 7's values. `PAPER_RUN = True` runs the full recipe (21B tokens per model).
