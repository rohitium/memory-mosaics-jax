"""Memory Mosaic (Library/memory_mosaics/models/memory_mosaics.py in facebookresearch/MemoryMosaics).
Laid out like gpt2.py; the key/value featurizers (paper eq. 6) are at the bottom."""

import jax
import jax.numpy as jnp

from .nn import key_iter, normal, dropout, layer_norm, cross_entropy


# ContextMem (token mixing; replaces attention): paper eq. 7

def context_mem_init(keys, cfg, proj_std):
    return {
        "k_feat": key_featurizer_init(keys, cfg),
        "v_feat": val_featurizer_init(keys, cfg),
        "c_proj": normal(keys, (cfg.n_embd, cfg.n_embd), proj_std),
    }


def context_mem_apply(params, x, cfg, keys):
    B, T, C = x.shape
    k, v = key_featurizer_apply(params["k_feat"], x, cfg, 1), val_featurizer_apply(params["v_feat"], x, cfg)
    scores = jnp.einsum("bthd,bshd->bhts", k[:, 1:], k)  # queries 1..T-1; position 0 has no past
    mask = jnp.tril(jnp.ones((T - 1, T), bool))  # query t sees keys < t: v[t] contains x[t+1]
    w = dropout(jax.nn.softmax(jnp.where(mask, scores, -jnp.inf)), cfg.dropout, keys)
    y = jnp.einsum("bhts,bshd->bthd", w, v)
    y = jnp.concatenate([jnp.zeros_like(y[:, :1]), y], 1).reshape(B, T, C)
    return dropout(y @ params["c_proj"], cfg.dropout, keys)


# PersistentMem (channel mixing; replaces the MLP): the same retrieval over a learned bank

def persistent_mem_init(keys, cfg, proj_std):
    hs = cfg.n_embd // cfg.n_head
    return {
        "k_feat": key_featurizer_init(keys, cfg),
        "P_k": normal(keys, (cfg.n_head, cfg.pmem_size, hs), hs ** -0.5),
        "P_v": normal(keys, (cfg.n_head, cfg.pmem_size, hs), hs ** -0.5),
        "out_scale": jnp.full(cfg.n_head, -0.05, jnp.float32),
        "c_proj": normal(keys, (cfg.n_embd, cfg.n_embd), proj_std),
    }


def persistent_mem_apply(params, x, cfg, keys):
    B, T, C = x.shape
    k = key_featurizer_apply(params["k_feat"], x, cfg, 2)  # bandwidth squared: P_k has no scale
    w = dropout(jax.nn.softmax(jnp.einsum("bthd,hsd->bhts", k, params["P_k"])), cfg.dropout, keys)
    y = jnp.einsum("bhts,hsd->bthd", w, params["P_v"]) * jnp.exp(10 * params["out_scale"])[:, None]
    return dropout(y.reshape(B, T, C) @ params["c_proj"], cfg.dropout, keys)


# Block and model

def block_init(keys, cfg, proj_std):
    return {
        "ln_1": jnp.ones(cfg.n_embd),
        "ctx": context_mem_init(keys, cfg, proj_std),
        "ln_2": jnp.ones(cfg.n_embd),
        "pmem": persistent_mem_init(keys, cfg, proj_std),
    }


def block_apply(params, x, cfg, keys):
    x = x + context_mem_apply(params["ctx"], layer_norm(x, params["ln_1"]), cfg, keys)
    return x + persistent_mem_apply(params["pmem"], layer_norm(x, params["ln_2"]), cfg, keys)


def init(key, cfg):
    keys, proj_std = key_iter(key), (cfg.n_embd * cfg.n_layer) ** -0.5
    return {
        "wte": normal(keys, (cfg.vocab_size, cfg.n_embd), cfg.n_embd ** -0.5),  # tied; std of the reference's output-layer init
        "blocks": [block_init(keys, cfg, proj_std) for _ in range(cfg.n_layer)],
        "ln_f": jnp.ones(cfg.n_embd),
    }


def apply(params, idx, cfg, targets=None, key=None):
    """idx: (B, T) token ids; key=None disables dropout. Returns (logits, loss or None)."""
    keys = None if key is None else key_iter(key)
    x = params["wte"][idx]  # no positional embeddings
    for block in params["blocks"]:
        x = block_apply(block, x, cfg, keys)
    logits = layer_norm(x, params["ln_f"]) @ params["wte"].T / cfg.n_embd ** 0.5
    return logits, None if targets is None else cross_entropy(logits, targets)


# Key and value featurizers. Learned scalars are stored /10 and used x10, as in the reference.

def leaky_avg(beta, k):
    """k̄[t] = sum_{i<=t} exp(-beta (t-i)) k[i], per head. The reference's extra
    (1 - exp(-beta)) factor cancels in the normalization that follows."""
    t = jnp.arange(k.shape[1])
    dist = jnp.maximum(t[:, None] - t, 0)  # clamped: exp overflow above the diagonal gives NaN gradients
    coef = jnp.where(t[:, None] >= t, jnp.exp(-beta[:, None, None] * dist), 0)
    return jnp.einsum("hts,bshd->bthd", coef, k)


def key_featurizer_init(keys, cfg):
    return {
        "W_k": normal(keys, (cfg.n_embd, cfg.n_embd), cfg.n_embd ** -0.5),
        "beta": jnp.linspace(0.05, 0.5, cfg.n_head),  # leaky-average decay
        "scale": jnp.full(cfg.n_head, 0.1, jnp.float32),  # kernel bandwidth, capped at e^5
    }


def key_featurizer_apply(params, x, cfg, scale_pow):
    k = (x @ params["W_k"]).reshape(*x.shape[:2], cfg.n_head, -1)
    k = leaky_avg(10 * jnp.abs(params["beta"]), k)
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    return k * jnp.exp(jnp.minimum(10 * params["scale"], 5))[:, None] ** scale_pow


def val_featurizer_init(keys, cfg):
    return {
        "W_v": normal(keys, (cfg.n_embd, cfg.n_embd), cfg.n_embd ** -0.5),
        "coef": jax.random.uniform(next(keys), (cfg.n_head,)),
        "scale": jnp.full(cfg.n_head, -0.05, jnp.float32),
    }


def val_featurizer_apply(params, x, cfg):
    v = (x @ params["W_v"]).reshape(*x.shape[:2], cfg.n_head, -1)
    v_next = jnp.concatenate([v[:, 1:], jnp.zeros_like(v[:, :1])], 1)  # peek one step ahead
    c = params["coef"][:, None]
    v = (1 - c) * v_next + c * v
    return v / jnp.linalg.norm(v, axis=-1, keepdims=True) * jnp.exp(10 * params["scale"])[:, None]
