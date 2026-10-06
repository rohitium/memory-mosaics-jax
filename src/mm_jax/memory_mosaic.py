"""Memory Mosaic in raw jax.numpy, matching the code behind the paper's BabiStories
runs (Library/memory_mosaics/models/memory_mosaics.py in facebookresearch/MemoryMosaics,
with k_fe_type=linearconv, k_kernel_size=1, v_fe_type=lowrlinearconv, v_shift=1).

Mirrors gpt2.py section by section (the key/value featurizers they use are
at the bottom of the file), with two swaps:

  * ContextMem replaces causal self-attention. Retrieval is kernel regression
    over the sequence's own key/value pairs (paper eq. 7):
        y_T = sum_{i<T} softmax(k_T . k_i) * v_i
    Keys are leaky averages of the past; values peek one step ahead (v_T is
    built from x_{T+1}), so the mask must be STRICTLY lower-triangular.
  * PersistentMem replaces the MLP: the same retrieval, but over a learned
    key/value bank (P_k, P_v) instead of the sequence, so no causal mask.

There are no positional embeddings and no separate query projection.

Learned scalars are stored divided by 10 and used multiplied by 10, as in the reference:
  * leaky decay beta: init linspace(0.5, 5, n_head)
  * key_scale 0.1     -> kernel bandwidth exp(min(10 * key_scale, 5))
  * val/out_scale -0.05 -> output scale exp(-0.5)
  * value mix coef ~ Uniform(0, 1) per head
  * P_k, P_v ~ Normal(0, 1/sqrt(head_dim))

Shapes are (B, T, nh, hs); the reference uses (B, nh, T, hs).
"""

import jax
import jax.numpy as jnp

from .config import MosaicConfig
from .nn import (normal, split_key, dropout, layer_norm, softmax,
                 cross_entropy_loss)


# ----------------------------------------------------------------------------
# ContextMem (token mixing; replaces attention)
# ----------------------------------------------------------------------------

def context_mem_init(key, cfg: MosaicConfig, proj_std):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "k_feat": key_featurizer_init(k1, cfg),
        "v_feat": val_featurizer_init(k2, cfg),
        "c_proj": normal(k3, (cfg.n_embd, cfg.n_embd), proj_std),
    }


def context_mem_apply(params, x, cfg: MosaicConfig, key):
    B, T, C = x.shape
    k_attn, k_resid = split_key(key, 2)
    k = key_featurizer_apply(params["k_feat"], x, cfg, scale_pow=1)  # (B,T,nh,hs)
    v = val_featurizer_apply(params["v_feat"], x, cfg)               # (B,T,nh,hs)

    # Queries are positions 1..T-1 (position 0 has no past). Row q is
    # position q+1 and sees keys <= q, i.e. strictly earlier positions:
    # v_t contains x_{t+1}, so including the diagonal would leak the future.
    scores = jnp.einsum("bthd,bshd->bhts", k[:, 1:], k)              # (B,nh,T-1,T)
    mask = jnp.tril(jnp.ones((T - 1, T), dtype=bool))
    scores = jnp.where(mask, scores, -jnp.inf)
    w = dropout(softmax(scores), cfg.dropout, k_attn)
    y = jnp.einsum("bhts,bshd->bthd", w, v)                          # (B,T-1,nh,hs)
    y = jnp.concatenate([jnp.zeros_like(y[:, :1]), y], axis=1)       # position 0 -> 0

    y = y.reshape(B, T, C)
    return dropout(y @ params["c_proj"], cfg.dropout, k_resid)


# ----------------------------------------------------------------------------
# PersistentMem (channel mixing; replaces the MLP)
# ----------------------------------------------------------------------------

def persistent_mem_init(key, cfg: MosaicConfig, proj_std):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    nh, hs = cfg.n_head, cfg.n_embd // cfg.n_head
    bank_shape = (cfg.pmem_count, nh, cfg.pmem_size, hs)
    return {
        "k_feat": key_featurizer_init(k1, cfg),
        "P_k": normal(k2, bank_shape, hs ** -0.5),
        "P_v": normal(k3, bank_shape, hs ** -0.5),
        "out_scale": jnp.full((nh,), -0.05),
        "c_proj": normal(k4, (cfg.n_embd, cfg.n_embd), proj_std),
    }


def persistent_mem_apply(params, x, cfg: MosaicConfig, key):
    B, T, C = x.shape
    k_attn, k_resid = split_key(key, 2)
    # scale_pow=2: the bank keys carry no scale of their own (as in the reference)
    k = key_featurizer_apply(params["k_feat"], x, cfg, scale_pow=2)  # (B,T,nh,hs)

    scores = jnp.einsum("bthd,nhsd->nbhts", k, params["P_k"])        # (n,B,nh,T,S)
    w = dropout(softmax(scores), cfg.dropout, k_attn)
    y = jnp.einsum("nbhts,nhsd->bthd", w, params["P_v"]) / cfg.pmem_count
    y = y * jnp.exp(10.0 * params["out_scale"])[None, None, :, None]

    y = y.reshape(B, T, C)
    return dropout(y @ params["c_proj"], cfg.dropout, k_resid)


# ----------------------------------------------------------------------------
# Block + full model
# ----------------------------------------------------------------------------

def block_init(key, cfg: MosaicConfig, proj_std):
    k1, k2 = jax.random.split(key)
    return {
        "ln_1": jnp.ones((cfg.n_embd,)),
        "ctx": context_mem_init(k1, cfg, proj_std),
        "ln_2": jnp.ones((cfg.n_embd,)),
        "pmem": persistent_mem_init(k2, cfg, proj_std),
    }


def block_apply(params, x, cfg: MosaicConfig, key):
    k1, k2 = split_key(key, 2)
    x = x + context_mem_apply(params["ctx"], layer_norm(x, params["ln_1"]), cfg, k1)
    x = x + persistent_mem_apply(params["pmem"], layer_norm(x, params["ln_2"]), cfg, k2)
    return x


def init(key, cfg: MosaicConfig):
    proj_std = (cfg.n_embd * cfg.n_layer) ** -0.5  # reference: 1/sqrt(fan_in)/sqrt(n_layer)
    k_wte, k_blocks = jax.random.split(key)
    return {
        # Tied with the LM head. The reference inits the shared tensor twice and the
        # head's 1/sqrt(n_embd) init runs last, so that is the effective std.
        "wte": normal(k_wte, (cfg.vocab_size, cfg.n_embd), cfg.n_embd ** -0.5),
        "blocks": [block_init(k, cfg, proj_std) for k in jax.random.split(k_blocks, cfg.n_layer)],
        "ln_f": jnp.ones((cfg.n_embd,)),
    }


def apply(params, idx, cfg: MosaicConfig, targets=None, key=None):
    """idx: (B, T) token ids. key: PRNG key for dropout, None to disable it.
    Returns (logits, loss or None)."""
    B, T = idx.shape
    assert T <= cfg.block_size, f"sequence length {T} > block_size {cfg.block_size}"
    block_keys = split_key(key, cfg.n_layer)

    x = params["wte"][idx]                      # token embeddings only

    for block, k in zip(params["blocks"], block_keys):
        x = block_apply(block, x, cfg, k)

    x = layer_norm(x, params["ln_f"])
    logits = x @ params["wte"].T / jnp.sqrt(cfg.n_embd)  # tied weights, scaled as in the reference

    loss = None
    if targets is not None:
        loss = cross_entropy_loss(logits, targets)
    return logits, loss


# ----------------------------------------------------------------------------
# Key / value featurizers
# ----------------------------------------------------------------------------

def leaky_avg_apply(beta, k):
    """out[t] = sum_{i<=t} exp(-beta * (t - i)) * k[i], per head. beta: (nh,), k: (B, T, nh, hs).

    The reference also multiplies by (1 - exp(-beta)); keys are unit-normed
    right after, so that per-head constant cancels and is omitted.
    """
    t = jnp.arange(k.shape[1])
    dist = t[:, None] - t[None, :]                                    # (T, T): t - i
    coef = jnp.where(dist >= 0, jnp.exp(-beta[:, None, None] * jnp.maximum(dist, 0)), 0.0)
    return jnp.einsum("hts,bshd->bthd", coef, k)


def key_featurizer_init(key, cfg: MosaicConfig):
    return {
        "W_k": normal(key, (cfg.n_embd, cfg.n_embd), cfg.n_embd ** -0.5),
        "beta_raw": jnp.linspace(0.5, 5.0, cfg.n_head) / 10.0,
        "key_scale": jnp.full((cfg.n_head,), 0.1),
    }


def key_featurizer_apply(params, x, cfg: MosaicConfig, scale_pow):
    B, T, C = x.shape
    nh, hs = cfg.n_head, C // cfg.n_head
    k = (x @ params["W_k"]).reshape(B, T, nh, hs)
    k = leaky_avg_apply(jnp.abs(params["beta_raw"]) * 10.0, k)
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-10)
    # learned per-head kernel bandwidth (replaces attention's 1/sqrt(hs))
    scale = jnp.exp(jnp.minimum(10.0 * params["key_scale"], 5.0)) ** scale_pow
    return k * scale[None, None, :, None]


def val_featurizer_init(key, cfg: MosaicConfig):
    k1, k2 = jax.random.split(key)
    return {
        "W_v": normal(k1, (cfg.n_embd, cfg.n_embd), cfg.n_embd ** -0.5),
        "coef": jax.random.uniform(k2, (cfg.n_head,)),
        "val_scale": jnp.full((cfg.n_head,), -0.05),
    }


def val_featurizer_apply(params, x, cfg: MosaicConfig):
    B, T, C = x.shape
    nh, hs = cfg.n_head, C // cfg.n_head
    v = (x @ params["W_v"]).reshape(B, T, nh, hs)
    # peek one step into the future: v_next[t] = v[t+1], last position = 0
    v_next = jnp.concatenate([v[:, 1:], jnp.zeros_like(v[:, :1])], axis=1)
    coef = params["coef"][None, None, :, None]
    v = (1.0 - coef) * v_next + coef * v
    v = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-10)
    return v * jnp.exp(10.0 * params["val_scale"])[None, None, :, None]
