"""BabiStories, tokenized as the reference prepare.py does, and random-crop batches."""

import itertools
import json
import os
import urllib.request

import numpy as np

URL = "https://github.com/facebookresearch/MemoryMosaics/raw/main/BabiStories/data/babistories-dataset.7z.{:03d}"


def prepare_babistories(data_dir, max_train_tokens=None):
    """Downloads and tokenizes BabiStories into uint16 token files; returns their paths.
    max_train_tokens stops the training split early (at a 10k-story boundary)."""
    import multivolumefile
    import py7zr
    import tiktoken

    os.makedirs(data_dir, exist_ok=True)
    bins = {"train": f"{data_dir}/train{max_train_tokens or ''}.bin", "val": f"{data_dir}/val.bin"}
    if all(map(os.path.exists, bins.values())):
        return bins
    archive = f"{data_dir}/babistories-dataset.7z"
    for i in range(1, 6):
        urllib.request.urlretrieve(URL.format(i), f"{archive}.{i:03d}")
    with multivolumefile.open(archive, "rb") as f, py7zr.SevenZipFile(f) as z:
        z.extractall(data_dir)
    enc = tiktoken.get_encoding("gpt2")
    for split, limit in ("val", None), ("train", max_train_tokens):
        n = 0
        with open(f"{data_dir}/{split}dataset.txt") as f, open(bins[split] + ".tmp", "wb") as out:
            while (limit is None or n < limit) and (lines := list(itertools.islice(f, 10_000))):
                stories = enc.encode_ordinary_batch([json.loads(line) for line in lines])
                ids = np.concatenate([np.array(s + [enc.eot_token], np.uint16) for s in stories])
                ids.tofile(out)
                n += len(ids)
        os.rename(bins[split] + ".tmp", bins[split])
    return bins


def batches(ids, batch_size, seq_len, seed=0):
    """Endless random (inputs, targets) crops from a token array."""
    rng = np.random.default_rng(seed)
    while True:
        start = rng.integers(0, len(ids) - seq_len, (batch_size, 1))
        chunk = ids[start + np.arange(seq_len + 1)].astype(np.int32)
        yield chunk[:, :-1], chunk[:, 1:]
