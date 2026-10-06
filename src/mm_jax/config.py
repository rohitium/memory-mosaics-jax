"""Configs. Defaults are the paper's BabiStories setup (arXiv:2405.06394v3, Sec. 7 and
App. C; reference script Library/scripts/train_babistories.sh), with n_layer=1 for
the single-block Fig. 7 curves. The transformer's hyperparameters are reused verbatim
for the Memory Mosaic, as in the paper.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 50304   # GPT-2 BPE (50257) padded to a multiple of 64, as in the reference
    block_size: int = 512     # context length
    n_layer: int = 1
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.05     # on attention weights and residual-branch outputs


@dataclass
class MosaicConfig(ModelConfig):
    pmem_size: int = 2688     # slots per head: P_k + P_v = 4.1M params vs 4.7M in a GPT-2 MLP
    pmem_count: int = 1


@dataclass
class TrainConfig:
    batch_size: int = 512        # sequences per optimizer step
    micro_batch_size: int = 16   # gradient-accumulation chunk; fit to GPU memory, doesn't change the math
    learning_rate: float = 5e-3
    min_lr: float = 1e-4
    warmup_steps: int = 2000
    max_steps: int = 80000       # cosine decay runs over the full run
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every: int = 2000
    eval_batches: int = 40       # micro-batches per evaluation (640 sequences, as in the reference)
