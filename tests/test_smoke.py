"""Smoke tests on tiny configs and random tokens (CPU, under a minute).

Run with ``python tests/test_smoke.py`` or ``pytest tests``.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mm_jax import gpt2, memory_mosaic, train
from mm_jax.config import ModelConfig, MosaicConfig, TrainConfig

V, T = 64, 16
TINY = dict(vocab_size=V, block_size=T, n_layer=1, n_head=2, n_embd=16)
MODELS = [("gpt2", gpt2, ModelConfig(**TINY)),
          ("mosaic", memory_mosaic, MosaicConfig(**TINY, pmem_size=16))]


def random_ids(n, seed):
    return np.random.default_rng(seed).integers(0, V, size=n)


def test_shapes_and_finite_grads():
    idx, targets = train.TextBatcher(random_ids(256, 0)).next_batch(4, T)
    for name, mod, cfg in MODELS:
        params = mod.init(jax.random.PRNGKey(0), cfg)
        logits, loss = mod.apply(params, idx, cfg, targets)
        assert logits.shape == (4, T, V) and loss.shape == (), name
        grads = jax.grad(lambda p: mod.apply(p, idx, cfg, targets)[1])(params)
        assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)), name


def test_loss_decreases():
    """Train on crops of one short random sequence; the model can memorize it."""
    ids = random_ids(256, 1)
    t_cfg = TrainConfig(batch_size=8, block_size=T, learning_rate=3e-3,
                        warmup_steps=5, max_steps=60, eval_batches=4)
    for name, mod, cfg in MODELS:
        train_step, eval_step, state = train.make_trainer(
            mod.apply, mod.init(jax.random.PRNGKey(1), cfg), cfg, t_cfg)
        before = train.eval_loss(eval_step, state["params"], train.TextBatcher(ids, 9), t_cfg)
        batcher = train.TextBatcher(ids)
        for _ in range(t_cfg.max_steps):
            state, _ = train_step(state, batcher.next_batch(t_cfg.batch_size, T))
        after = train.eval_loss(eval_step, state["params"], train.TextBatcher(ids, 9), t_cfg)
        assert after < before, f"{name}: {before:.3f} -> {after:.3f}"
        print(f"  {name}: loss {before:.3f} -> {after:.3f}")


def test_no_future_leak():
    """Changing token p must not change logits before p (for the Mosaic this
    checks the strictly lower-triangular ContextMem mask)."""
    p = 5
    idx = jnp.asarray(random_ids(2 * T, 2).reshape(2, T))
    perturbed = idx.at[:, p].set((idx[:, p] + 17) % V)
    for name, mod, cfg in MODELS:
        params = mod.init(jax.random.PRNGKey(2), cfg)
        before, _ = mod.apply(params, idx, cfg)
        after, _ = mod.apply(params, perturbed, cfg)
        assert jnp.allclose(before[:, :p], after[:, :p], atol=1e-5), f"{name}: future leak"
        assert not jnp.allclose(before[:, p], after[:, p], atol=1e-5), f"{name}: input ignored"


def test_context_mem_first_position_is_zero():
    cfg = MODELS[1][2]
    k1, k2 = jax.random.split(jax.random.PRNGKey(3))
    params = memory_mosaic.context_mem_init(k1, cfg, proj_std=0.02)  # c_proj bias inits to 0
    y = memory_mosaic.context_mem_apply(params, jax.random.normal(k2, (2, 8, cfg.n_embd)), cfg)
    assert jnp.all(y[:, 0] == 0.0)


def test_leaky_avg_matches_naive():
    k = jax.random.normal(jax.random.PRNGKey(4), (2, 9, 3, 5))
    beta = jnp.array([0.7, 2.0, 4.5])
    t = jnp.arange(9)
    dist = (t[:, None] - t[None, :])[..., None]                       # (T, T, 1)
    weights = jnp.where(dist >= 0, jnp.exp(-beta * dist), 0.0)        # (T, T, nh)
    expected = jnp.einsum("tsh,bshd->bthd", weights, k)
    assert jnp.allclose(memory_mosaic.leaky_avg_apply(beta, k), expected, atol=1e-5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("All smoke tests passed.")
