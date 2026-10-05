"""GPT-2 baseline in raw jax.numpy (nanoGPT architecture).

Mirrors memory_mosaic.py section by section; `diff gpt2.py memory_mosaic.py`
shows exactly what Memory Mosaics change.
"""

import jax
import jax.numpy as jnp

from .config import ModelConfig
from .nn import (linear_init, linear_apply, layer_norm_init, layer_norm_apply,
                 gelu, softmax, cross_entropy_loss)


# ----------------------------------------------------------------------------
# Causal self-attention (token mixing)
# ----------------------------------------------------------------------------

def attention_init(key, cfg: ModelConfig, proj_std: float):
    k1, k2 = jax.random.split(key)
    return {
        "c_attn": linear_init(k1, cfg.n_embd, 3 * cfg.n_embd, use_bias=cfg.bias),  # fused q, k, v
        "c_proj": linear_init(k2, cfg.n_embd, cfg.n_embd, std=proj_std,
                              use_bias=cfg.bias),
    }


def attention_apply(params, x, cfg: ModelConfig):
    B, T, C = x.shape
    nh, hs = cfg.n_head, C // cfg.n_head
    q, k, v = jnp.split(linear_apply(params["c_attn"], x), 3, axis=-1)
    q, k, v = (t.reshape(B, T, nh, hs) for t in (q, k, v))           # (B,T,nh,hs)

    scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(hs)      # (B,nh,T,T)
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))                     # query t sees keys <= t
    scores = jnp.where(mask[None, None], scores, -1e10)
    w = softmax(scores, axis=-1)
    y = jnp.einsum("bhts,bshd->bthd", w, v)                          # (B,T,nh,hs)

    y = y.reshape(B, T, C)
    return linear_apply(params["c_proj"], y)


# ----------------------------------------------------------------------------
# MLP (channel mixing)
# ----------------------------------------------------------------------------

def mlp_init(key, cfg: ModelConfig, proj_std: float):
    k1, k2 = jax.random.split(key)
    return {
        "c_fc": linear_init(k1, cfg.n_embd, 4 * cfg.n_embd, use_bias=cfg.bias),
        "c_proj": linear_init(k2, 4 * cfg.n_embd, cfg.n_embd, std=proj_std,
                              use_bias=cfg.bias),
    }


def mlp_apply(params, x, cfg: ModelConfig):
    y = gelu(linear_apply(params["c_fc"], x))
    return linear_apply(params["c_proj"], y)


# ----------------------------------------------------------------------------
# Block + full model
# ----------------------------------------------------------------------------

def block_init(key, cfg: ModelConfig, proj_std: float):
    k1, k2 = jax.random.split(key)
    return {
        "ln_1": layer_norm_init(cfg.n_embd),
        "attn": attention_init(k1, cfg, proj_std),
        "ln_2": layer_norm_init(cfg.n_embd),
        "mlp": mlp_init(k2, cfg, proj_std),
    }


def block_apply(params, x, cfg: ModelConfig):
    x = x + attention_apply(params["attn"], layer_norm_apply(params["ln_1"], x), cfg)
    x = x + mlp_apply(params["mlp"], layer_norm_apply(params["ln_2"], x), cfg)
    return x


def init(key, cfg: ModelConfig):
    proj_std = 0.02 / jnp.sqrt(2.0 * cfg.n_layer)  # GPT-2 residual-projection init
    k_wte, k_wpe, k_blocks = jax.random.split(key, 3)
    block_keys = jax.random.split(k_blocks, cfg.n_layer)
    return {
        "wte": jax.random.normal(k_wte, (cfg.vocab_size, cfg.n_embd)) * 0.02,
        "wpe": jax.random.normal(k_wpe, (cfg.block_size, cfg.n_embd)) * 0.02,
        "blocks": [block_init(k, cfg, proj_std) for k in block_keys],
        "ln_f": layer_norm_init(cfg.n_embd),
        # lm_head tied to wte (applied as x @ wte.T).
    }


def apply(params, idx, cfg: ModelConfig, targets=None):
    """Forward pass. idx: (B, T) int token ids. Returns (logits, loss|None)."""
    B, T = idx.shape
    assert T <= cfg.block_size, f"sequence length {T} > block_size {cfg.block_size}"

    x = params["wte"][idx] + params["wpe"][:T]  # token + positional embeddings

    for block in params["blocks"]:
        x = block_apply(block, x, cfg)

    x = layer_norm_apply(params["ln_f"], x)
    logits = x @ params["wte"].T                # tied weights

    loss = None
    if targets is not None:
        loss = cross_entropy_loss(logits, targets)
    return logits, loss
