"""Training harness shared by both models, matching the reference trainer
(Library/train_memory_mosaics.py, train_baselines.py): AdamW with linear warmup
and cosine decay to min_lr, global-norm gradient clipping, weight decay on
>=2-D tensors only. The batch is split into micro-batches whose gradients are
averaged, which equals one full-batch step (the reference spreads it over GPUs).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax


def learning_rate(step, train_cfg):
    """Learning rate for the step-th update (0-based), as the reference get_cosine_lr:
    linear warmup from 0, then cosine decay to min_lr at max_steps."""
    lr, min_lr = train_cfg.learning_rate, train_cfg.min_lr
    if step < train_cfg.warmup_steps:
        return lr * step / train_cfg.warmup_steps
    if step >= train_cfg.max_steps:
        return min_lr
    progress = (step - train_cfg.warmup_steps) / (train_cfg.max_steps - train_cfg.warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (lr - min_lr)


def make_trainer(model_apply, params, model_cfg, train_cfg):
    """Returns (train_step, eval_step, state):
        train_step(state, batch, key, lr) -> (state, mean loss)   batch: (batch_size, T) arrays
        eval_step(params, batch) -> loss                          dropout off
    The learning rate is an argument (see learning_rate), so changing the
    schedule doesn't recompile the step.
    """
    tx = optax.chain(  # AdamW without the learning-rate step, which train_step applies
        optax.clip_by_global_norm(train_cfg.grad_clip),
        optax.scale_by_adam(b1=train_cfg.beta1, b2=train_cfg.beta2),
        optax.add_decayed_weights(train_cfg.weight_decay,
                                  # no decay on norms or learned scales
                                  mask=jax.tree_util.tree_map(lambda p: p.ndim >= 2, params)),
    )
    n_micro = train_cfg.batch_size // train_cfg.micro_batch_size
    assert n_micro * train_cfg.micro_batch_size == train_cfg.batch_size

    def loss_fn(params, batch, key):
        idx, targets = batch
        return model_apply(params, idx, model_cfg, targets, key)[1]

    @jax.jit
    def train_step(state, batch, key, lr):
        micro_batches = jax.tree_util.tree_map(
            lambda a: a.reshape(n_micro, train_cfg.micro_batch_size, -1), batch)

        def accumulate(grad_sum, xs):
            loss, grads = jax.value_and_grad(loss_fn)(state["params"], *xs)
            return jax.tree_util.tree_map(jnp.add, grad_sum, grads), loss

        zeros = jax.tree_util.tree_map(jnp.zeros_like, state["params"])
        grad_sum, losses = jax.lax.scan(accumulate, zeros,
                                        (micro_batches, jax.random.split(key, n_micro)))
        grads = jax.tree_util.tree_map(lambda g: g / n_micro, grad_sum)
        updates, opt_state = tx.update(grads, state["opt_state"], state["params"])
        params = jax.tree_util.tree_map(lambda p, u: p - lr * u, state["params"], updates)
        return {"params": params, "opt_state": opt_state}, losses.mean()

    @jax.jit
    def eval_step(params, batch):
        return loss_fn(params, batch, None)

    return train_step, eval_step, {"params": params, "opt_state": tx.init(params)}


def eval_loss(eval_step, params, batcher, train_cfg, seq_len):
    """Mean loss over eval_batches micro-batches."""
    batches = (batcher.next_batch(train_cfg.micro_batch_size, seq_len)
               for _ in range(train_cfg.eval_batches))
    return float(np.mean([eval_step(params, b) for b in batches]))
