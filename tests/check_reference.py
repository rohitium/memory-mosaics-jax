"""Checks the JAX models and training step against the authors' PyTorch code (Library/ in
facebookresearch/MemoryMosaics, downloaded at a pinned commit): same weights give the same
logits, 5 training steps give the same parameters, and full-size parameter counts agree.
Needs `pip install torch`. Run: python tests/check_reference.py"""

import contextlib
import io
import os
import sys
import tempfile
import types
import urllib.request

import jax
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mm_jax import gpt2, memory_mosaic, train
from mm_jax.config import ModelConfig, MosaicConfig, TrainConfig

COMMIT = "191158129c8dab3637769988e336009a9bc5d4f2"
ref_dir = os.path.join(tempfile.mkdtemp(), "ref")
os.makedirs(ref_dir)
open(os.path.join(ref_dir, "__init__.py"), "w").close()
for path in "memory_mosaics/models/memory_mosaics.py", "memory_mosaics/models/memory.py", "baseline/baselines.py":
    url = f"https://raw.githubusercontent.com/facebookresearch/MemoryMosaics/{COMMIT}/Library/{path}"
    urllib.request.urlretrieve(url, os.path.join(ref_dir, os.path.basename(path)))
sys.path.insert(0, os.path.dirname(ref_dir))

from ref.baselines import GPT  # noqa: E402
from ref.memory_mosaics import StackAssoMem  # noqa: E402

quiet = contextlib.redirect_stdout(io.StringIO())  # the reference prints parameter counts


def gpt2_ref(cfg):
    return GPT(types.SimpleNamespace(**cfg.__dict__, bias=False))


def mosaic_ref(cfg):
    return StackAssoMem(cfg.vocab_size, cfg.n_head, cfg.n_embd, cfg.n_layer, block_size=cfg.block_size,
                        ic_dropout=cfg.dropout, hd_dropout=cfg.dropout, pmem_size=cfg.pmem_size,
                        config=types.SimpleNamespace(skip_tokens=0))


def gpt2_pairs(ref, p):  # (torch parameter, JAX array, layout)
    t = ref.transformer
    pairs = [(t.wte.weight, p["wte"], ""), (t.wpe.weight, p["wpe"], ""), (t.ln_f.weight, p["ln_f"], "")]
    for b, bp in zip(t.h, p["blocks"]):
        pairs += [(b.ln_1.weight, bp["ln_1"], ""), (b.ln_2.weight, bp["ln_2"], ""),
                  (b.attn.c_attn.weight, bp["attn"]["c_attn"], "T"), (b.attn.c_proj.weight, bp["attn"]["c_proj"], "T"),
                  (b.mlp.c_fc.weight, bp["mlp"]["c_fc"], "T"), (b.mlp.c_proj.weight, bp["mlp"]["c_proj"], "T")]
    return pairs


def mosaic_pairs(ref, p):
    pairs = [(ref.emb.weight, p["wte"], ""), (ref.ln_out.weight, p["ln_f"], "")]
    for b, bp in zip(ref.blocks, p["blocks"]):
        a, q, c, m = b.attn, b.pmem, bp["ctx"], bp["pmem"]
        pairs += [(b.ln1.weight, bp["ln_1"], ""), (b.ln2.weight, bp["ln_2"], ""), (a.c_proj.weight, c["c_proj"], "T"),
                  (a.k_featurizer.conv.weight, c["k_feat"]["W_k"], "T"), (a.leaky_key_beta, c["k_feat"]["beta"], ""),
                  (a.kernel_beta, c["k_feat"]["scale"], ""), (a.v_featurizer.linear.weight, c["v_feat"]["W_v"], "T"),
                  (a.v_featurizer.coef, c["v_feat"]["coef"], ""), (a.value_beta, c["v_feat"]["scale"], ""),
                  (q.k_featurizer.conv.weight, m["k_feat"]["W_k"], "T"), (q.leaky_key_beta, m["k_feat"]["beta"], ""),
                  (q.kernel_beta, m["k_feat"]["scale"], ""), (q.pmem.M_k0, m["P_k"], ""),
                  (q.pmem.M_v0, m["P_v"], ""), (q.value_beta, m["out_scale"], ""), (q.c_proj.weight, m["c_proj"], "T")]
    return pairs


def to_jax(t, layout, shape):  # torch stores Linear weights as (out, in) and Conv1d as (out, in, 1)
    a = t.detach().numpy()
    return (a.reshape(a.shape[0], -1).T if layout == "T" else a).reshape(shape)


def to_torch(a, layout, t):
    a = np.array(a)
    return torch.from_numpy(a.T.copy() if layout == "T" else a).reshape(t.shape)


def check(name, module, ref_fn, pairs_fn, cfg):
    t_cfg = TrainConfig(batch_size=8, micro_batch_size=2, warmup_steps=2, max_steps=6, grad_clip=0.5)  # clips every step
    init_state, train_step, _ = train.make_trainer(module, cfg, t_cfg)
    state = init_state(jax.random.key(0))
    # move the learned scalars away from their initial values, so every parameter matters
    state["params"] = jax.tree.map(lambda a: a + 0.03 * jax.random.normal(jax.random.key(1), a.shape), state["params"])
    with quiet:
        ref = ref_fn(cfg)
        opt = ref.configure_optimizers(t_cfg.weight_decay, 0.0, (t_cfg.beta1, t_cfg.beta2), "cpu")
    pairs = pairs_fn(ref, state["params"])
    assert len({id(t) for t, _, _ in pairs}) == len(list(ref.parameters())), "unmapped reference parameters"
    with torch.no_grad():
        for t, a, layout in pairs:
            t.copy_(torch.from_numpy(np.asarray(a)).reshape(t.shape) if layout in ("", "")
                    else torch.from_numpy(np.asarray(a).T.copy()).reshape(t.shape))

    rng = np.random.default_rng(0)
    data = [rng.integers(0, cfg.vocab_size, (8, cfg.block_size + 1)).astype(np.int32) for _ in range(5)]
    idx, targets = data[0][:, :-1], data[0][:, 1:]
    logits = np.asarray(module.apply(state["params"], idx, cfg)[0])
    ref_logits = ref.eval()(torch.from_numpy(idx).long(), torch.from_numpy(targets).long())[0].detach().numpy()
    logit_diff = np.abs(logits - ref_logits).max()

    ref.train()
    for step, chunk in enumerate(data):
        idx, targets = chunk[:, :-1], chunk[:, 1:]
        lr = train.learning_rate(step, t_cfg)
        state, _ = train_step(state, (idx, targets), lr)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad()
        ref(torch.from_numpy(idx).long(), torch.from_numpy(targets).long())[1].backward()
        torch.nn.utils.clip_grad_norm_(ref.parameters(), t_cfg.grad_clip)
        opt.step()
    param_diff = max(np.abs(to_jax(t, layout, a.shape) - np.asarray(a)).max()
                     for t, a, layout in pairs_fn(ref, state["params"]))

    with quiet:
        full = ref_fn(type(cfg)())
    counts = sum(p.numel() for p in full.parameters()), sum(a.size for a in jax.tree.leaves(module.init(jax.random.key(0), type(cfg)())))
    print(f"{name}: max |logit diff| {logit_diff:.1e}, max |param diff| after 5 steps {param_diff:.1e}, "
          f"full-size params {counts[0]:,} (reference) vs {counts[1]:,}")
    assert logit_diff < 1e-5 and param_diff < 1e-5 and counts[0] == counts[1]  # float32 rounding: ~1e-7 to 1e-6


if __name__ == "__main__":
    small = dict(vocab_size=101, block_size=24, n_layer=2, n_head=4, n_embd=32, dropout=0.0)
    check("GPT-2", gpt2, gpt2_ref, gpt2_pairs, ModelConfig(**small))
    check("Memory Mosaic", memory_mosaic, mosaic_ref, mosaic_pairs, MosaicConfig(**small, pmem_size=40))
