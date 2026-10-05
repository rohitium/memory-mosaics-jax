"""Hyperparameter configs (plain dataclasses)."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 50257   # GPT-2 BPE vocabulary
    block_size: int = 128     # context length
    n_layer: int = 1          # Fig. 7 subplot: a single block
    n_head: int = 4
    n_embd: int = 128
    bias: bool = True         # bias in Linear layers, like GPT-2


@dataclass
class MosaicConfig(ModelConfig):
    # Persistent-memory bank: pmem_size slots per head, pmem_count banks.
    # pmem_size ~= 3.5 * n_embd matches a GPT-2 block's parameter count
    # (paper: 2688 slots at 768 dims).
    pmem_size: int = 448
    pmem_count: int = 1


@dataclass
class TrainConfig:
    batch_size: int = 16
    block_size: int = 128
    learning_rate: float = 3e-4
    min_lr_frac: float = 0.1  # cosine decay floor, as a fraction of peak LR
    weight_decay: float = 0.1
    warmup_steps: int = 50
    max_steps: int = 400
    grad_clip: float = 1.0
    eval_every: int = 50
    eval_batches: int = 10
