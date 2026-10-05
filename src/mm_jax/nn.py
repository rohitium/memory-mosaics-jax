"""Raw jax.numpy layers: parameters are plain dicts, layers are pure functions."""

import jax
import jax.numpy as jnp


def linear_init(key, in_dim, out_dim, std=0.02, use_bias=True):
    params = {"w": jax.random.normal(key, (in_dim, out_dim)) * std}
    if use_bias:
        params["b"] = jnp.zeros((out_dim,))
    return params


def linear_apply(params, x):
    y = x @ params["w"]
    if "b" in params:
        y = y + params["b"]
    return y


def layer_norm_init(dim):
    return {"weight": jnp.ones((dim,)), "bias": jnp.zeros((dim,))}


def layer_norm_apply(params, x, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * params["weight"] + params["bias"]


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
