"""Shared data batcher and training harness for both models.

Same data, optimizer and schedule for GPT-2 and the Mosaic, so loss-curve
differences come from the architecture alone. Mirrors nanoGPT/nanoMosaics:
AdamW, linear warmup + cosine decay, global-norm gradient clipping, weight
decay on >=2-D tensors only.
"""

import jax
import numpy as np
import optax


class TextBatcher:
    """Random (x, y) next-token crops from a 1-D array of token ids."""

    def __init__(self, ids, seed=0):
        self.ids = np.asarray(ids, dtype=np.int32)
        self.rng = np.random.default_rng(seed)

    def next_batch(self, batch_size, seq_len):
        start = self.rng.integers(0, len(self.ids) - seq_len, size=batch_size)
        chunk = self.ids[start[:, None] + np.arange(seq_len + 1)]
        return chunk[:, :-1], chunk[:, 1:]


def make_trainer(model_apply, params, model_cfg, train_cfg):
    """Returns (train_step, eval_step, state):
        train_step(state, batch) -> (state, loss)
        eval_step(params, batch) -> loss
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=train_cfg.learning_rate,
        warmup_steps=train_cfg.warmup_steps,
        decay_steps=train_cfg.max_steps,
        end_value=train_cfg.learning_rate * train_cfg.min_lr_frac,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(train_cfg.grad_clip),
        optax.adamw(schedule, weight_decay=train_cfg.weight_decay,
                    # no decay on biases, norms or learned scales
                    mask=jax.tree_util.tree_map(lambda p: p.ndim >= 2, params)),
    )

    @jax.jit
    def eval_step(params, batch):
        idx, targets = batch
        return model_apply(params, idx, model_cfg, targets)[1]

    @jax.jit
    def train_step(state, batch):
        loss, grads = jax.value_and_grad(eval_step)(state["params"], batch)
        updates, opt_state = tx.update(grads, state["opt_state"], state["params"])
        params = optax.apply_updates(state["params"], updates)
        return {"params": params, "opt_state": opt_state}, loss

    return train_step, eval_step, {"params": params, "opt_state": tx.init(params)}


def eval_loss(eval_step, params, batcher, train_cfg):
    """Mean loss over ``eval_batches`` batches."""
    batches = (batcher.next_batch(train_cfg.batch_size, train_cfg.block_size)
               for _ in range(train_cfg.eval_batches))
    return float(np.mean([eval_step(params, b) for b in batches]))
