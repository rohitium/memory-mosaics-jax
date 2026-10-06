"""Smoke tests on tiny configs and random tokens (CPU, under a minute).

Run with ``python tests/test_smoke.py`` or ``pytest tests``.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mm_jax import gpt2, memory_mosaic, train
from mm_jax.config import ModelConfig, MosaicConfig, TrainConfig
from mm_jax.data import TextBatcher

V, T = 64, 16
TINY = dict(vocab_size=V, block_size=T, n_layer=1, n_head=2, n_embd=16)
MODELS = [("gpt2", gpt2, ModelConfig(**TINY)),
          ("mosaic", memory_mosaic, MosaicConfig(**TINY, pmem_size=16))]
TINY_TRAIN = dict(batch_size=8, micro_batch_size=4, learning_rate=3e-3, min_lr=3e-4,
                  warmup_steps=5, max_steps=60, eval_batches=4)


def random_ids(n, seed):
    return np.random.default_rng(seed).integers(0, V, size=n)


def test_shapes_and_finite_grads():
    idx, targets = TextBatcher(random_ids(256, 0)).next_batch(4, T)
    key = jax.random.PRNGKey(0)
    for name, mod, cfg in MODELS:
        params = mod.init(key, cfg)
        logits, loss = mod.apply(params, idx, cfg, targets, key)
        assert logits.shape == (4, T, V) and loss.shape == (), name
        grads = jax.grad(lambda p: mod.apply(p, idx, cfg, targets, key)[1])(params)
        assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)), name


def test_loss_decreases():
    """Train (with dropout) on crops of one short random sequence, which the model can memorize."""
    ids = random_ids(256, 1)
    t_cfg = TrainConfig(**TINY_TRAIN)
    for name, mod, cfg in MODELS:
        train_step, eval_step, state = train.make_trainer(
            mod.apply, mod.init(jax.random.PRNGKey(1), cfg), cfg, t_cfg)
        before = train.eval_loss(eval_step, state["params"], TextBatcher(ids, 9), t_cfg, T)
        batcher, key = TextBatcher(ids), jax.random.PRNGKey(2)
        for step in range(t_cfg.max_steps):
            key, k = jax.random.split(key)
            lr = train.learning_rate(step, t_cfg)
            state, _ = train_step(state, batcher.next_batch(t_cfg.batch_size, T), k, lr)
        after = train.eval_loss(eval_step, state["params"], TextBatcher(ids, 9), t_cfg, T)
        assert after < before, f"{name}: {before:.3f} -> {after:.3f}"
        print(f"  {name}: loss {before:.3f} -> {after:.3f}")


def test_grad_accumulation_matches_full_batch():
    batch = TextBatcher(random_ids(256, 3)).next_batch(8, T)
    for name, mod, cfg in MODELS:
        cfg = type(cfg)(**{**cfg.__dict__, "dropout": 0.0})
        params = mod.init(jax.random.PRNGKey(3), cfg)
        results = []
        for micro in (8, 2):
            t_cfg = TrainConfig(**{**TINY_TRAIN, "micro_batch_size": micro})
            train_step, _, state = train.make_trainer(mod.apply, params, cfg, t_cfg)
            for step in range(3):
                state, loss = train_step(state, batch, jax.random.PRNGKey(0), train.learning_rate(step, t_cfg))
            results.append(state["params"])
        for a, b in zip(*map(jax.tree_util.tree_leaves, results)):
            assert jnp.allclose(a, b, atol=1e-5), name


def test_train_step_compiles_once():
    """Parameters must keep their exact types across steps (e.g. no weak-typed
    initial arrays), and the learning rate is an argument, so one compile serves
    the whole run."""
    batch = TextBatcher(random_ids(256, 7)).next_batch(8, T)
    for name, mod, cfg in MODELS:
        train_step, _, state = train.make_trainer(mod.apply, mod.init(jax.random.PRNGKey(7), cfg), cfg,
                                                  TrainConfig(**TINY_TRAIN))
        for step in range(3):
            state, _ = train_step(state, batch, jax.random.PRNGKey(step), 1e-3 * step)
        assert train_step._cache_size() == 1, name


def test_learning_rate_matches_optax_schedule():
    t_cfg = TrainConfig(**TINY_TRAIN)
    schedule = optax.warmup_cosine_decay_schedule(0.0, t_cfg.learning_rate, t_cfg.warmup_steps,
                                                  t_cfg.max_steps, t_cfg.min_lr)
    for step in range(t_cfg.max_steps + 5):
        assert np.isclose(train.learning_rate(step, t_cfg), schedule(step), rtol=1e-5), step


def test_no_future_leak():
    """Changing token p must not change logits before p (for the Mosaic this
    checks the strictly lower-triangular ContextMem mask)."""
    p = 5
    idx = jnp.asarray(random_ids(2 * T, 4).reshape(2, T))
    perturbed = idx.at[:, p].set((idx[:, p] + 17) % V)
    for name, mod, cfg in MODELS:
        params = mod.init(jax.random.PRNGKey(4), cfg)
        before, _ = mod.apply(params, idx, cfg)
        after, _ = mod.apply(params, perturbed, cfg)
        assert jnp.allclose(before[:, :p], after[:, :p], atol=1e-5), f"{name}: future leak"
        assert not jnp.allclose(before[:, p], after[:, p], atol=1e-5), f"{name}: input ignored"


def test_context_mem_first_position_is_zero():
    cfg = MODELS[1][2]
    k1, k2 = jax.random.split(jax.random.PRNGKey(5))
    params = memory_mosaic.context_mem_init(k1, cfg, proj_std=0.02)
    y = memory_mosaic.context_mem_apply(params, jax.random.normal(k2, (2, 8, cfg.n_embd)), cfg, None)
    assert jnp.all(y[:, 0] == 0.0)


def test_leaky_avg_matches_recurrence():
    k = jax.random.normal(jax.random.PRNGKey(6), (2, 9, 3, 5))
    beta = jnp.array([0.7, 2.0, 4.5])
    expected, acc = [], jnp.zeros_like(k[:, 0])
    for t in range(k.shape[1]):            # out[t] = k[t] + exp(-beta) * out[t-1]
        acc = k[:, t] + jnp.exp(-beta)[None, :, None] * acc
        expected.append(acc)
    assert jnp.allclose(memory_mosaic.leaky_avg_apply(beta, k), jnp.stack(expected, 1), atol=1e-5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("All smoke tests passed.")
