"""Raw jax.numpy building blocks: parameters are plain arrays in dicts, layers are pure functions."""

import jax
import jax.numpy as jnp


def normal(key, shape, std):
    return std * jax.random.normal(key, shape)


def split_key(key, n):
    """jax.random.split that passes key=None (dropout off) through."""
    return [None] * n if key is None else list(jax.random.split(key, n))


def dropout(x, rate, key):
    if key is None or rate == 0.0:
        return x
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)


def layer_norm(x, weight, eps=1e-5):
    """LayerNorm without bias, as in the reference (bias=False)."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * weight


def gelu(x):
    """Tanh approximation of GELU, as in GPT-2."""
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x ** 3)))


def softmax(x, axis=-1):
    e = jnp.exp(x - jnp.max(x, axis=axis, keepdims=True))
    return e / jnp.sum(e, axis=axis, keepdims=True)


def cross_entropy_loss(logits, targets):
    """Mean next-token cross-entropy. logits: (B, T, V), targets: (B, T) ints."""
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(logp, targets[..., None], axis=-1))


def count_params(params):
    return sum(p.size for p in jax.tree_util.tree_leaves(params))
