"""Smoke tests on tiny models (CPU, under a minute): python tests/test_smoke.py"""

import os
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mm_jax import gpt2, memory_mosaic, train
from mm_jax.config import ModelConfig, MosaicConfig, TrainConfig
from mm_jax.data import batches

V, T = 64, 16
TINY = dict(vocab_size=V, block_size=T, n_head=2, n_embd=16)
MODELS = [(gpt2, ModelConfig(**TINY)), (memory_mosaic, MosaicConfig(**TINY, pmem_size=16))]
CFG = TrainConfig(batch_size=8, micro_batch_size=4, learning_rate=3e-3, min_lr=3e-4, warmup_steps=5, max_steps=60)
IDS = np.random.default_rng(0).integers(0, V, 256)  # one short sequence the models can memorize


def test_training():
    """Loss falls, and the step compiles once although the learning rate changes."""
    for mod, cfg in MODELS:
        init_state, train_step, eval_step = train.make_trainer(mod, cfg, CFG)
        state, val, data = init_state(jax.random.key(0)), next(batches(IDS, 32, T, 1)), batches(IDS, 8, T)
        before = eval_step(state["params"], val)
        for step in range(CFG.max_steps):
            state, _ = train_step(state, next(data), train.learning_rate(step, CFG))
        assert eval_step(state["params"], val) < before and train_step._cache_size() == 1


def test_grad_accumulation_matches_full_batch():
    batch = next(batches(IDS, 8, T))
    for mod, cfg in MODELS:
        cfg = type(cfg)(**{**cfg.__dict__, "dropout": 0.0})
        results = []
        for micro in 8, 2:
            t_cfg = TrainConfig(**{**CFG.__dict__, "micro_batch_size": micro})
            init_state, train_step, _ = train.make_trainer(mod, cfg, t_cfg)
            state = init_state(jax.random.key(0))
            for _ in range(3):
                state, _ = train_step(state, batch, 1e-3)
            results.append(jax.tree.leaves(state["params"]))
        assert all(jnp.allclose(a, b, atol=1e-5) for a, b in zip(*results))


def test_checkpoint_resume_is_exact():
    mod, cfg = MODELS[1]
    init_state, train_step, _ = train.make_trainer(mod, cfg, CFG)
    data = [next(batches(IDS, 8, T, seed)) for seed in range(4)]
    path = os.path.join(tempfile.gettempdir(), "mm_jax_test.pkl")
    a, b = init_state(jax.random.key(0)), init_state(jax.random.key(0))
    for batch in data:
        a, _ = train_step(a, batch, 1e-3)
    for i, batch in enumerate(data):
        if i == 2:  # interrupt b halfway, through a checkpoint
            train.save(path, b)
            b = jax.device_put(train.load(path))
        b, _ = train_step(b, batch, 1e-3)
    assert all(np.array_equal(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))
    assert train_step._cache_size() == 1


def test_no_future_leak():
    """Changing token 5 changes logits at 5 but not before (for the Mosaic: the strict mask)."""
    idx = jnp.asarray(IDS[:2 * T].reshape(2, T))
    for mod, cfg in MODELS:
        params = mod.init(jax.random.key(0), cfg)
        a, b = (mod.apply(params, i, cfg)[0] for i in (idx, idx.at[:, 5].add(1) % V))
        assert jnp.allclose(a[:, :5], b[:, :5], atol=1e-5) and not jnp.allclose(a[:, 5], b[:, 5], atol=1e-5)


def test_leaky_avg_matches_recurrence():
    k, beta = jax.random.normal(jax.random.key(0), (2, 9, 3, 5)), jnp.array([0.7, 2.0, 4.5])
    acc, expected = 0, []
    for t in range(9):  # k̄[t] = k[t] + exp(-beta) k̄[t-1]
        acc = k[:, t] + jnp.exp(-beta)[:, None] * acc
        expected.append(acc)
    assert jnp.allclose(memory_mosaic.leaky_avg(beta, k), jnp.stack(expected, 1), atol=1e-5)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
