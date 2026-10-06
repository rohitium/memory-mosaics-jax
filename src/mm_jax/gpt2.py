"""GPT-2 baseline (Library/baseline/baselines.py in facebookresearch/MemoryMosaics).
Laid out like memory_mosaic.py, so a diff of the two shows what Memory Mosaics change."""

import jax
import jax.numpy as jnp

from .nn import key_iter, normal, dropout, layer_norm, cross_entropy


# Causal self-attention (token mixing)

def attention_init(keys, cfg, proj_std):
    return {
        "c_attn": normal(keys, (cfg.n_embd, 3 * cfg.n_embd), 0.02),  # q, k, v
        "c_proj": normal(keys, (cfg.n_embd, cfg.n_embd), proj_std),
    }


def attention_apply(params, x, cfg, keys):
    B, T, C = x.shape
    q, k, v = (t.reshape(B, T, cfg.n_head, -1) for t in jnp.split(x @ params["c_attn"], 3, -1))
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(q.shape[-1])
    mask = jnp.tril(jnp.ones((T, T), bool))  # query t sees keys <= t
    w = dropout(jax.nn.softmax(jnp.where(mask, scores, -jnp.inf)), cfg.dropout, keys)
    y = jnp.einsum("bhts,bshd->bthd", w, v).reshape(B, T, C)
    return dropout(y @ params["c_proj"], cfg.dropout, keys)


# MLP (channel mixing)

def mlp_init(keys, cfg, proj_std):
    return {
        "c_fc": normal(keys, (cfg.n_embd, 4 * cfg.n_embd), 0.02),
        "c_proj": normal(keys, (4 * cfg.n_embd, cfg.n_embd), proj_std),
    }


def mlp_apply(params, x, cfg, keys):
    y = jax.nn.gelu(x @ params["c_fc"])
    return dropout(y @ params["c_proj"], cfg.dropout, keys)


# Block and model

def block_init(keys, cfg, proj_std):
    return {
        "ln_1": jnp.ones(cfg.n_embd),
        "attn": attention_init(keys, cfg, proj_std),
        "ln_2": jnp.ones(cfg.n_embd),
        "mlp": mlp_init(keys, cfg, proj_std),
    }


def block_apply(params, x, cfg, keys):
    x = x + attention_apply(params["attn"], layer_norm(x, params["ln_1"]), cfg, keys)
    return x + mlp_apply(params["mlp"], layer_norm(x, params["ln_2"]), cfg, keys)


def init(key, cfg):
    keys, proj_std = key_iter(key), 0.02 / (2 * cfg.n_layer) ** 0.5
    return {
        "wte": normal(keys, (cfg.vocab_size, cfg.n_embd), 0.02),  # tied with the output layer
        "wpe": normal(keys, (cfg.block_size, cfg.n_embd), 0.02),
        "blocks": [block_init(keys, cfg, proj_std) for _ in range(cfg.n_layer)],
        "ln_f": jnp.ones(cfg.n_embd),
    }


def apply(params, idx, cfg, targets=None, key=None):
    """idx: (B, T) token ids; key=None disables dropout. Returns (logits, loss or None)."""
    keys = None if key is None else key_iter(key)
    x = dropout(params["wte"][idx] + params["wpe"][:idx.shape[1]], cfg.dropout, keys)
    for block in params["blocks"]:
        x = block_apply(block, x, cfg, keys)
    logits = layer_norm(x, params["ln_f"]) @ params["wte"].T
    return logits, None if targets is None else cross_entropy(logits, targets)
