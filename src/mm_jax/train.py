"""Training shared by both models, in the style of the JAX training cookbook
(https://docs.jax.dev/en/latest/the-training-cookbook.html), with the reference trainer's recipe:
AdamW as torch.optim.AdamW, global-norm gradient clipping, weight decay on >=2-D tensors, and
linear warmup then cosine decay. Each batch is split into micro-batches whose gradients are
averaged (the reference splits it across GPUs)."""

import math
import os
import pickle
from functools import partial

import jax
import jax.numpy as jnp


def learning_rate(step, cfg):
    """Plain Python, so it never interrupts the accelerator; as the reference get_cosine_lr."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps
    progress = min(1, (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps))
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.learning_rate - cfg.min_lr)


def make_trainer(model, model_cfg, cfg):
    """Returns init_state(key), train_step(state, batch, lr) -> (state, loss) and
    eval_step(params, batch) -> loss (no dropout). train_step reuses the state's buffers."""
    n_micro = cfg.batch_size // cfg.micro_batch_size

    def loss_fn(params, batch, key):
        return model.apply(params, batch[0], model_cfg, batch[1], key)[1]

    @jax.jit
    def init_state(key):
        params = model.init(key, model_cfg)
        zeros = lambda: jax.tree.map(jnp.zeros_like, params)
        return {"params": params, "mu": zeros(), "nu": zeros(), "step": jnp.zeros((), jnp.int32)}

    @partial(jax.jit, donate_argnums=0)
    def train_step(state, batch, lr):
        def accumulate(grads, xs):
            loss, g = jax.value_and_grad(loss_fn)(state["params"], *xs)
            return jax.tree.map(jnp.add, grads, g), loss

        micro = jax.tree.map(lambda a: a.reshape(n_micro, -1, a.shape[-1]), batch)
        keys = jax.random.split(jax.random.fold_in(jax.random.key(1), state["step"]), n_micro)  # dropout
        grads, losses = jax.lax.scan(accumulate, jax.tree.map(jnp.zeros_like, state["params"]), (micro, keys))
        norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads))) / n_micro
        grads = jax.tree.map(lambda g: g / n_micro * jnp.minimum(1, cfg.grad_clip / (norm + 1e-6)), grads)
        step = state["step"] + 1
        mu = jax.tree.map(lambda m, g: cfg.beta1 * m + (1 - cfg.beta1) * g, state["mu"], grads)
        nu = jax.tree.map(lambda v, g: cfg.beta2 * v + (1 - cfg.beta2) * g * g, state["nu"], grads)

        def update(p, m, v):
            adam = m / (1 - cfg.beta1 ** step) / (jnp.sqrt(v / (1 - cfg.beta2 ** step)) + 1e-8)
            return p - lr * (adam + (cfg.weight_decay if p.ndim >= 2 else 0) * p)

        params = jax.tree.map(update, state["params"], mu, nu)
        return {**state, "params": params, "mu": mu, "nu": nu, "step": step}, losses.mean()

    eval_step = jax.jit(lambda params, batch: loss_fn(params, batch, None))
    return init_state, train_step, eval_step


def save(path, tree):
    """Atomically pickles a pytree (e.g. the train state and loss log) to path."""
    with open(path + ".tmp", "wb") as f:
        pickle.dump(jax.device_get(tree), f)
    os.replace(path + ".tmp", path)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)
