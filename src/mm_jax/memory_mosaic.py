"""Memory Mosaic in raw jax.numpy: a port of the reference ``mosaic_model.py``
(facebookresearch/MemoryMosaics, nanoMosaics/).

Mirrors gpt2.py section by section, with two swaps:

  * ContextMem replaces causal self-attention. Retrieval is kernel regression
    over the sequence's own key/value pairs (paper eq. 7):
        y_T = sum_{i<T} softmax(k_T . k_i) * v_i
    Keys are leaky averages of the past; values peek one step ahead (v_T is
    built from x_{T+1}), so the mask must be STRICTLY lower-triangular.
  * PersistentMem replaces the MLP: the same retrieval, but over a learned
    key/value bank (P_k, P_v) instead of the sequence, so no causal mask.

There are no positional embeddings and no separate query projection.

Inits match the reference:
  * leaky decay beta: linspace(0.5, 5, n_head)    (stored /10, used x10)
  * key_scale 0.1     -> bandwidth exp(min(scale_pow * 10 * 0.1, 5))
  * val/out_scale -0.05 -> exp(-0.5)
  * value mix coef ~ Uniform(0, 1) per head
  * P_k, P_v ~ Normal(0, 1/sqrt(head_dim))

Shapes are (B, T, nh, hs); the reference uses (B, nh, T, hs).
"""

import jax
import jax.numpy as jnp

from .config import MosaicConfig
from .nn import (linear_init, linear_apply, layer_norm_init, layer_norm_apply,
                 softmax, cross_entropy_loss)


# ----------------------------------------------------------------------------
# Key / value featurizers
# ----------------------------------------------------------------------------

def leaky_avg_apply(beta, k):
    """out[t] = sum_{i<=t} exp(-beta * (t - i)) * k[i], per head.

    beta: (nh,), k: (B, T, nh, hs). The reference materializes a (T, T)
    coefficient matrix; this lax.scan runs the recurrence
    out[t] = k[t] + exp(-beta) * out[t-1] in O(T) memory.
    """
    decay = jnp.exp(-beta)[None, :, None]

    def step(carry, k_t):
        carry = k_t + decay * carry
        return carry, carry

    _, out = jax.lax.scan(step, jnp.zeros_like(k[:, 0]), jnp.swapaxes(k, 0, 1))
    return jnp.swapaxes(out, 0, 1)


def key_featurizer_init(key, cfg: MosaicConfig):
    return {
        "W_k": linear_init(key, cfg.n_embd, cfg.n_embd, use_bias=cfg.bias),
        "beta_raw": jnp.linspace(0.5, 5.0, cfg.n_head) / 10.0,
        "key_scale": jnp.full((cfg.n_head,), 0.1),
    }


def key_featurizer_apply(params, x, cfg: MosaicConfig, scale_pow):
    B, T, C = x.shape
    nh, hs = cfg.n_head, C // cfg.n_head
    k = linear_apply(params["W_k"], x).reshape(B, T, nh, hs)
    k = leaky_avg_apply(jnp.abs(params["beta_raw"]) * 10.0, k)
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-10)
    # learned per-head kernel bandwidth (replaces attention's 1/sqrt(hs))
    scale = jnp.exp(jnp.minimum(scale_pow * 10.0 * params["key_scale"], 5.0))
    return k * scale[None, None, :, None]


def val_featurizer_init(key, cfg: MosaicConfig):
    k1, k2 = jax.random.split(key)
    return {
        "W_v": linear_init(k1, cfg.n_embd, cfg.n_embd, use_bias=cfg.bias),
        "coef": jax.random.uniform(k2, (cfg.n_head,)),
        "val_scale": jnp.full((cfg.n_head,), -0.05),
    }


def val_featurizer_apply(params, x, cfg: MosaicConfig):
    B, T, C = x.shape
    nh, hs = cfg.n_head, C // cfg.n_head
    v = linear_apply(params["W_v"], x).reshape(B, T, nh, hs)
    # peek one step into the future: v_next[t] = v[t+1], last position = 0
    v_next = jnp.concatenate([v[:, 1:], jnp.zeros_like(v[:, :1])], axis=1)
    coef = params["coef"][None, None, :, None]
    v = (1.0 - coef) * v_next + coef * v
    v = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-10)
    return v * jnp.exp(10.0 * params["val_scale"])[None, None, :, None]


# ----------------------------------------------------------------------------
# ContextMem (token mixing; replaces attention)
# ----------------------------------------------------------------------------

def context_mem_init(key, cfg: MosaicConfig, proj_std: float):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "k_feat": key_featurizer_init(k1, cfg),
        "v_feat": val_featurizer_init(k2, cfg),
        "c_proj": linear_init(k3, cfg.n_embd, cfg.n_embd, std=proj_std,
                              use_bias=cfg.bias),
    }


def context_mem_apply(params, x, cfg: MosaicConfig):
    B, T, C = x.shape
    k = key_featurizer_apply(params["k_feat"], x, cfg, scale_pow=1)  # (B,T,nh,hs)
    v = val_featurizer_apply(params["v_feat"], x, cfg)               # (B,T,nh,hs)

    # Queries are positions 1..T-1 (position 0 has no past). Row q is
    # position q+1 and sees keys <= q, i.e. strictly earlier positions:
    # v_t contains x_{t+1}, so including the diagonal would leak the future.
    scores = jnp.einsum("bthd,bshd->bhts", k[:, 1:], k)              # (B,nh,T-1,T)
    mask = jnp.tril(jnp.ones((T - 1, T), dtype=bool))
    scores = jnp.where(mask[None, None], scores, -1e10)
    w = softmax(scores, axis=-1)
    y = jnp.einsum("bhts,bshd->bthd", w, v)                          # (B,T-1,nh,hs)
    y = jnp.concatenate([jnp.zeros_like(y[:, :1]), y], axis=1)       # position 0 -> 0

    y = y.reshape(B, T, C)
    return linear_apply(params["c_proj"], y)


# ----------------------------------------------------------------------------
# PersistentMem (channel mixing; replaces the MLP)
# ----------------------------------------------------------------------------

def persistent_mem_init(key, cfg: MosaicConfig, proj_std: float):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    nh, hs = cfg.n_head, cfg.n_embd // cfg.n_head
    bank_shape = (cfg.pmem_count, nh, cfg.pmem_size, hs)
    return {
        "k_feat": key_featurizer_init(k1, cfg),
        "P_k": jax.random.normal(k2, bank_shape) / jnp.sqrt(hs),
        "P_v": jax.random.normal(k3, bank_shape) / jnp.sqrt(hs),
        "out_scale": jnp.full((nh,), -0.05),
        "c_proj": linear_init(k4, cfg.n_embd, cfg.n_embd, std=proj_std,
                              use_bias=cfg.bias),
    }


def persistent_mem_apply(params, x, cfg: MosaicConfig):
    B, T, C = x.shape
    # scale_pow=2: the bank keys carry no scale of their own (as in the reference)
    k = key_featurizer_apply(params["k_feat"], x, cfg, scale_pow=2)  # (B,T,nh,hs)

    scores = jnp.einsum("bthd,nhsd->nbhts", k, params["P_k"])        # (n,B,nh,T,S)
    w = softmax(scores, axis=-1)
    y = jnp.einsum("nbhts,nhsd->bthd", w, params["P_v"]) / cfg.pmem_count
    y = y * jnp.exp(10.0 * params["out_scale"])[None, None, :, None]

    y = y.reshape(B, T, C)
    return linear_apply(params["c_proj"], y)


# ----------------------------------------------------------------------------
# Block + full model
# ----------------------------------------------------------------------------

def block_init(key, cfg: MosaicConfig, proj_std: float):
    k1, k2 = jax.random.split(key)
    return {
        "ln_1": layer_norm_init(cfg.n_embd),
        "ctx": context_mem_init(k1, cfg, proj_std),
        "ln_2": layer_norm_init(cfg.n_embd),
        "pmem": persistent_mem_init(k2, cfg, proj_std),
    }


def block_apply(params, x, cfg: MosaicConfig):
    x = x + context_mem_apply(params["ctx"], layer_norm_apply(params["ln_1"], x), cfg)
    x = x + persistent_mem_apply(params["pmem"], layer_norm_apply(params["ln_2"], x), cfg)
    return x


def init(key, cfg: MosaicConfig):
    proj_std = 0.02 / jnp.sqrt(2.0 * cfg.n_layer)  # GPT-2 residual-projection init
    k_wte, k_blocks = jax.random.split(key)
    block_keys = jax.random.split(k_blocks, cfg.n_layer)
    return {
        "wte": jax.random.normal(k_wte, (cfg.vocab_size, cfg.n_embd)) * 0.02,
        "blocks": [block_init(k, cfg, proj_std) for k in block_keys],
        "ln_f": layer_norm_init(cfg.n_embd),
        # lm_head tied to wte (applied as x @ wte.T).
    }


def apply(params, idx, cfg: MosaicConfig, targets=None):
    """Forward pass. idx: (B, T) int token ids. Returns (logits, loss|None)."""
    B, T = idx.shape
    assert T <= cfg.block_size, f"sequence length {T} > block_size {cfg.block_size}"

    x = params["wte"][idx]                      # token embeddings only

    for block in params["blocks"]:
        x = block_apply(block, x, cfg)

    x = layer_norm_apply(params["ln_f"], x)
    logits = x @ params["wte"].T                # tied weights

    loss = None
    if targets is not None:
        loss = cross_entropy_loss(logits, targets)
    return logits, loss
