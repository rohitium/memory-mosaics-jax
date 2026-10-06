# Memory Mosaics in JAX

A raw `jax.numpy` port of **Memory Mosaics** (Zhang et al., ICLR 2025,
[arXiv:2405.06394](https://arxiv.org/abs/2405.06394v3)), plus a GPT-2 baseline
and the code to reproduce the single-block panel of **Fig. 7**: a Memory Mosaic
trains as well as GPT-2 in a model as small as one transformer block.

No flax/haiku: parameters are plain dicts, layers are pure functions. The two
models are written side by side, so a line-level diff shows exactly what Memory
Mosaics change:

```bash
git diff --no-index src/mm_jax/gpt2.py src/mm_jax/memory_mosaic.py
```

(`diff -y -W 200` shows them side by side instead. Both exit with status 1
because the files differ.)

## Quickstart

```bash
pip install -r requirements.txt
python tests/test_smoke.py
```

Then open
[`notebooks/memory_mosaics_colab.ipynb`](notebooks/memory_mosaics_colab.ipynb)
in Colab on a v5e-1 TPU or T4 GPU and run it top to bottom (about 15 minutes). It downloads
BabiStories, trains both 1-block models, and plots training and validation
loss. See [Quick run vs. paper run](#quick-run-vs-paper-run) for what that plot
can and can't show.

## Layout

```
src/mm_jax/
  nn.py              layer norm, GELU, softmax, dropout, cross-entropy
  config.py          ModelConfig, MosaicConfig, TrainConfig (defaults = the paper's)
  gpt2.py            baseline: attention + MLP
  memory_mosaic.py   ContextMem + PersistentMem (equations and inits in the docstring)
  data.py            prepare_babistories (download + tokenize), TextBatcher
  train.py           make_trainer (AdamW, warmup-cosine, grad clip, grad accumulation), eval_loss
tests/test_smoke.py  CPU smoke tests
notebooks/           Colab notebook for the Fig. 7 panel
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
| init std 0.02, residual projections 0.02/√(2L) | init std 1/√fan_in, residual projections 1/√(d·L) |
| logits `x·Eᵀ` | logits `x·Eᵀ/√d` |

Both use pre-LayerNorm blocks without biases, an LM head tied to the token
embedding, and dropout 0.05 on attention weights and residual outputs.

## Paper setup

The defaults in `config.py` and `train.py` are the paper's BabiStories setup
(Sec. 7 and App. C; reference script `Library/scripts/train_babistories.sh`).
The paper tuned these for the transformer and reused them unchanged for the
Mosaic:

| | |
|---|---|
| data | BabiStories from the paper repo, GPT-2 BPE: 474,704,907 train / 4,749,107 val tokens |
| model | d = 768, 12 heads, vocab 50,304, context 512, `pmem_size` 2,688, 1 block |
| optimizer | AdamW, β = (0.9, 0.95), weight decay 0.1 on ≥2-D tensors, grad clip 1.0 |
| schedule | lr 5e-3, 2,000 warmup steps, cosine decay to 1e-4 over 80,000 steps |
| batch | 512 sequences (gradient accumulation over micro-batches of 16) |
| eval | every 2,000 steps on 640 validation sequences |

The models follow the code behind the paper's runs
(`Library/memory_mosaics/models/memory_mosaics.py` and
`Library/baseline/baselines.py` in
[facebookresearch/MemoryMosaics](https://github.com/facebookresearch/MemoryMosaics)),
not the alternate `nanoMosaics/` version. With the same weights and dropout
off, both reproduce the reference PyTorch logits to about 3e-7, and their
initial weight scales and parameter counts (45.7M Mosaic, 46.1M GPT-2) match.

Implementation differences that don't change the math:
- tensors are laid out `(B, T, nh, hs)` instead of `(B, nh, T, hs)`;
- training runs in float32 instead of float16 mixed precision;
- the batch of 512 is accumulated over micro-batches on one GPU instead of
  being split across 64 GPUs.

## Smoke tests

These run on tiny configs with random tokens, on CPU in under a minute:

| Test | Checks |
|---|---|
| `shapes_and_finite_grads` | logits are `(B, T, V)`, loss is a scalar, all gradients are finite (with dropout) |
| `loss_decreases` | `make_trainer` + `eval_loss` lower the loss on a memorizable sequence |
| `grad_accumulation_matches_full_batch` | micro-batching gives the same parameters as one full-batch step |
| `train_step_compiles_once` | the training step compiles once for a whole run, whatever the learning rate |
| `learning_rate_matches_optax_schedule` | warmup + cosine schedule equals `optax.warmup_cosine_decay_schedule` |
| `no_future_leak` | changing token `p` leaves logits before `p` unchanged; for the Mosaic this guards the strict mask |
| `context_mem_first_position_is_zero` | ContextMem returns 0 at position 0 (no past), as in the reference |
| `leaky_avg_matches_recurrence` | the leaky average equals `k̄_t = k_t + e^{-β}·k̄_{t-1}` |

## Quick run vs. paper run

The notebook's default **quick run** keeps the paper's model, data, optimizer and
dropout but uses batches of 32 sequences instead of 512, and sizes the schedule
to a 10-minute training budget: it times a few steps on the GPU it gets, then
picks the step count (warmup is 5% of it, and the cosine decay spans the whole
run). On a T4 that is roughly 5–10M tokens per model; a TPU gets several times more.

That is the very start of training. The paper's Fig. 7 starts at its first
evaluation, after 2,000 iterations of 512 sequences (about 1B tokens, two passes
over the data), and runs to 80,000 iterations (21B tokens). Its 1-block panel
spans losses of roughly 2.3 down to 2.0 (GPT-2) and 1.9 (Memory Mosaic), with the
Mosaic pulling ahead after about 30,000 iterations. A 15-minute run ends far
above that range, so it can show whether the two models train comparably at the
start, but it can't reproduce the paper's curve or its final gap.

`PAPER_RUN = True` in the notebook uses the full recipe (the `TrainConfig`
defaults). That is about 21B tokens per model and needs a long-lived
A100-class machine; the notebook doesn't checkpoint, so it isn't suited to
Colab sessions. No training results are checked in.

## Attribution

Independent JAX reimplementation of the PyTorch reference code by the Memory
Mosaics authors (Meta Platforms, Inc.), itself derived from nanoGPT. BabiStories
is downloaded from the authors' repository.
