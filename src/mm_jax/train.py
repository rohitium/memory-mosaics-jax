"""Training harness shared by both models, matching the reference trainer
(Library/train_memory_mosaics.py, train_baselines.py): AdamW with linear warmup
and cosine decay to min_lr, global-norm gradient clipping, weight decay on
>=2-D tensors only. The batch is split into micro-batches whose gradients are
averaged, which equals one full-batch step (the reference spreads it over GPUs).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax


def make_trainer(model_apply, params, model_cfg, train_cfg):
    """Returns (train_step, eval_step, state):
        train_step(state, batch, key) -> (state, mean loss)   batch: (batch_size, T) arrays
        eval_step(params, batch) -> loss                      dropout off
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=train_cfg.learning_rate,
        warmup_steps=train_cfg.warmup_steps,
        decay_steps=train_cfg.max_steps,
        end_value=train_cfg.min_lr,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(train_cfg.grad_clip),
        optax.adamw(schedule, b1=train_cfg.beta1, b2=train_cfg.beta2,
                    weight_decay=train_cfg.weight_decay,
                    # no decay on norms or learned scales
                    mask=jax.tree_util.tree_map(lambda p: p.ndim >= 2, params)),
    )
    n_micro = train_cfg.batch_size // train_cfg.micro_batch_size
    assert n_micro * train_cfg.micro_batch_size == train_cfg.batch_size

    def loss_fn(params, batch, key):
        idx, targets = batch
        return model_apply(params, idx, model_cfg, targets, key)[1]

    @jax.jit
    def train_step(state, batch, key):
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
        params = optax.apply_updates(state["params"], updates)
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
