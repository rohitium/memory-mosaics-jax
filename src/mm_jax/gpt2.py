"""GPT-2 baseline in raw jax.numpy: nanoGPT with bias=False, matching the paper's
baseline (Library/baseline/baselines.py in facebookresearch/MemoryMosaics).

Mirrors memory_mosaic.py section by section; `diff gpt2.py memory_mosaic.py`
shows exactly what Memory Mosaics change.
"""

import jax
import jax.numpy as jnp

from .config import ModelConfig
from .nn import (normal, split_key, dropout, layer_norm, gelu, softmax,
                 cross_entropy_loss)


# ----------------------------------------------------------------------------
# Causal self-attention (token mixing)
# ----------------------------------------------------------------------------

def attention_init(key, cfg: ModelConfig, proj_std):
    k1, k2 = jax.random.split(key)
    return {
        "c_attn": normal(k1, (cfg.n_embd, 3 * cfg.n_embd), 0.02),        # fused q, k, v
        "c_proj": normal(k2, (cfg.n_embd, cfg.n_embd), proj_std),
    }


def attention_apply(params, x, cfg: ModelConfig, key):
    B, T, C = x.shape
    nh, hs = cfg.n_head, C // cfg.n_head
    k_attn, k_resid = split_key(key, 2)
    q, k, v = jnp.split(x @ params["c_attn"], 3, axis=-1)
    q, k, v = (t.reshape(B, T, nh, hs) for t in (q, k, v))           # (B,T,nh,hs)

    scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(hs)      # (B,nh,T,T)
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))                     # query t sees keys <= t
    scores = jnp.where(mask, scores, -jnp.inf)
    w = dropout(softmax(scores), cfg.dropout, k_attn)
    y = jnp.einsum("bhts,bshd->bthd", w, v)                          # (B,T,nh,hs)

    y = y.reshape(B, T, C)
    return dropout(y @ params["c_proj"], cfg.dropout, k_resid)


# ----------------------------------------------------------------------------
# MLP (channel mixing)
# ----------------------------------------------------------------------------

def mlp_init(key, cfg: ModelConfig, proj_std):
    k1, k2 = jax.random.split(key)
    return {
        "c_fc": normal(k1, (cfg.n_embd, 4 * cfg.n_embd), 0.02),
        "c_proj": normal(k2, (4 * cfg.n_embd, cfg.n_embd), proj_std),
    }


def mlp_apply(params, x, cfg: ModelConfig, key):
    y = gelu(x @ params["c_fc"])
    return dropout(y @ params["c_proj"], cfg.dropout, key)


# ----------------------------------------------------------------------------
# Block + full model
# ----------------------------------------------------------------------------

def block_init(key, cfg: ModelConfig, proj_std):
    k1, k2 = jax.random.split(key)
    return {
        "ln_1": jnp.ones((cfg.n_embd,)),
        "attn": attention_init(k1, cfg, proj_std),
        "ln_2": jnp.ones((cfg.n_embd,)),
        "mlp": mlp_init(k2, cfg, proj_std),
    }


def block_apply(params, x, cfg: ModelConfig, key):
    k1, k2 = split_key(key, 2)
    x = x + attention_apply(params["attn"], layer_norm(x, params["ln_1"]), cfg, k1)
    x = x + mlp_apply(params["mlp"], layer_norm(x, params["ln_2"]), cfg, k2)
    return x


def init(key, cfg: ModelConfig):
    proj_std = 0.02 / jnp.sqrt(2.0 * cfg.n_layer)  # GPT-2 residual-projection init
    k_wte, k_wpe, k_blocks = jax.random.split(key, 3)
    return {
        "wte": normal(k_wte, (cfg.vocab_size, cfg.n_embd), 0.02),  # tied with the LM head
        "wpe": normal(k_wpe, (cfg.block_size, cfg.n_embd), 0.02),
        "blocks": [block_init(k, cfg, proj_std) for k in jax.random.split(k_blocks, cfg.n_layer)],
        "ln_f": jnp.ones((cfg.n_embd,)),
    }


def apply(params, idx, cfg: ModelConfig, targets=None, key=None):
    """idx: (B, T) token ids. key: PRNG key for dropout, None to disable it.
    Returns (logits, loss or None)."""
    B, T = idx.shape
    assert T <= cfg.block_size, f"sequence length {T} > block_size {cfg.block_size}"
    k_emb, *block_keys = split_key(key, cfg.n_layer + 1)

    x = params["wte"][idx] + params["wpe"][:T]  # token + positional embeddings
    x = dropout(x, cfg.dropout, k_emb)

    for block, k in zip(params["blocks"], block_keys):
        x = block_apply(block, x, cfg, k)

    x = layer_norm(x, params["ln_f"])
    logits = x @ params["wte"].T                # tied weights

    loss = None
    if targets is not None:
        loss = cross_entropy_loss(logits, targets)
    return logits, loss
