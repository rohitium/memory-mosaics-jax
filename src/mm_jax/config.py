"""Defaults are the paper's BabiStories setup (arXiv:2405.06394v3 Sec. 7, App. C;
Library/scripts/train_babistories.sh in facebookresearch/MemoryMosaics), with one block."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 50304  # GPT-2 BPE, padded to a multiple of 64
    block_size: int = 512
    n_layer: int = 1
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.05


@dataclass
class MosaicConfig(ModelConfig):
    pmem_size: int = 2688  # persistent-memory slots per head


@dataclass
class TrainConfig:
    batch_size: int = 512
    micro_batch_size: int = 16  # gradient accumulation; only affects memory use
    learning_rate: float = 5e-3
    min_lr: float = 1e-4
    warmup_steps: int = 2000
    max_steps: int = 80000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every: int = 2000
    eval_batches: int = 40  # x micro_batch_size = 640 sequences, as in the reference
