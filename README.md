# Memory Mosaics in JAX

A raw `jax.numpy` port of **Memory Mosaics** (Zhang et al., ICLR 2025,
[arXiv:2405.06394](https://arxiv.org/abs/2405.06394v3)), plus a GPT-2 baseline
and the code to reproduce a **Fig. 7 subplot**: a Memory Mosaic trains as well
as GPT-2 in a model as small as a single transformer block.

No flax/haiku: parameters are plain dicts, layers are pure functions. The two
models are written side by side, so a line-level diff shows exactly what Memory
Mosaics change:

```bash
diff src/mm_jax/gpt2.py src/mm_jax/memory_mosaic.py
```

## Quickstart

```bash
pip install -r requirements.txt
python tests/test_smoke.py
```

To reproduce the subplot, open
[`notebooks/memory_mosaics_colab.ipynb`](notebooks/memory_mosaics_colab.ipynb)
in Colab with a GPU runtime and run it top to bottom. It trains both 1-block
models with the same data and optimizer and saves `fig7_subplot.png`.

## Layout

```
src/mm_jax/
  nn.py              linear, layer norm, GELU, softmax, cross-entropy
  config.py          ModelConfig, MosaicConfig, TrainConfig
  gpt2.py            baseline: attention + MLP
  memory_mosaic.py   ContextMem + PersistentMem (equations and inits in the docstring)
  train.py           TextBatcher, make_trainer (AdamW, warmup-cosine, grad clip), eval_loss
tests/test_smoke.py  CPU smoke tests
notebooks/           Colab notebook for the Fig. 7 subplot
```

## GPT-2 vs Memory Mosaic

| GPT-2 (`gpt2.py`) | Memory Mosaic (`memory_mosaic.py`) |
|---|---|
| causal self-attention | **ContextMem**: kernel regression over the sequence's own (key, value) pairs |
| separate Q, K, V projections | one key projection, shared by query and key; values separate |
| key = projection of `x_t` | key = leaky average of past projections, unit-normed |
| value = projection of `x_t` | value mixes `x_{t+1}` and `x_t` (peeks one step ahead), unit-normed |
| causal mask `tril(k=0)` | **strictly** causal: query `t` sees keys `i < t`, so the peek can't leak |
| `1/sqrt(d)` score scaling | learned per-head bandwidth `key_scale` |
| MLP | **PersistentMem**: same retrieval over a learned key/value bank, no mask |
| learned positional embeddings | none |

Both models share the token embedding, final LayerNorm, LM head tied to the
embedding, GPT-2's scaled residual-projection init, and the training setup in
`train.py` (AdamW, warmup + cosine decay, grad clipping, weight decay on ≥2-D
tensors only). Any gap between the loss curves therefore comes from the
architecture.

The port follows the reference `nanoMosaics/mosaic_model.py` in
[facebookresearch/MemoryMosaics](https://github.com/facebookresearch/MemoryMosaics),
with three implementation differences that don't change the math:
- tensors are laid out `(B, T, nh, hs)` instead of `(B, nh, T, hs)`;
- the leaky average is a `lax.scan` recurrence (O(T) memory) instead of a
  materialized `T×T` matrix;
- dropout is omitted (it is 0 in every config used here).

## Smoke tests

These run on tiny configs with random tokens, on CPU in about 10 seconds:

| Test | Checks |
|---|---|
| `shapes_and_finite_grads` | logits are `(B, T, V)`, loss is a scalar, all gradients are finite |
| `loss_decreases` | `make_trainer` + `eval_loss` lower the loss on a memorizable sequence |
| `no_future_leak` | changing token `p` leaves logits before `p` unchanged; for the Mosaic this guards the strict mask |
| `context_mem_first_position_is_zero` | ContextMem returns 0 at position 0 (no past), as in the reference |
| `leaky_avg_matches_naive` | the scan equals `Σ_{i≤t} exp(-β(t-i))·k_i` |

## Scope and limitations

- **Data.** The paper trains on BabiStories, available from the paper repo. The
  notebook streams TinyStories from the Hugging Face Hub instead, which is
  synthetic small-text stories of the same kind. To use BabiStories, tokenize
  it and pass the ids to `TextBatcher`.
- **Scale.** The notebook is a miniature run: 128 dims, 4 heads, 400 steps,
  about 1M characters of text. The paper uses 768 dims at full training length.
  At this scale the claim being tested is qualitative: the two curves track
  each other. Every size is a config field.
- **No training results are checked in.** The repo has the code and the smoke tests only.

## Attribution

Independent JAX reimplementation of the PyTorch reference `nanoMosaics/` by the
Memory Mosaics authors (Meta Platforms, Inc.), itself derived from nanoGPT.
