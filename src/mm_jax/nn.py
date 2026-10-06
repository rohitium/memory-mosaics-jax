"""Helpers shared by gpt2.py and memory_mosaic.py."""

import itertools
from functools import partial

import jax
import jax.numpy as jnp


def key_iter(key):  # endless fresh keys, as in the JAX training cookbook
    return map(partial(jax.random.fold_in, key), itertools.count())


def normal(keys, shape, std):
    return std * jax.random.normal(next(keys), shape)


def dropout(x, rate, keys):  # keys=None: no dropout
    if keys is None:
        return x
    return jnp.where(jax.random.bernoulli(next(keys), 1 - rate, x.shape), x / (1 - rate), 0)


def layer_norm(x, weight):  # no bias, as in the reference
    return jax.nn.standardize(x, epsilon=1e-5) * weight


def cross_entropy(logits, targets):
    return -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[..., None], -1).mean()
