"""BabiStories, tokenized as the reference prepare.py does, and random-crop batches."""

import itertools
import json
import os
import urllib.request

import numpy as np

URL = "https://github.com/facebookresearch/MemoryMosaics/raw/main/BabiStories/data/babistories-dataset.7z.{:03d}"


def prepare_babistories(data_dir):
    """Downloads and tokenizes BabiStories into uint16 token files; returns their paths."""
    import multivolumefile
    import py7zr
    import tiktoken

    os.makedirs(data_dir, exist_ok=True)
    bins = {split: f"{data_dir}/{split}.bin" for split in ("train", "val")}
    if all(map(os.path.exists, bins.values())):
        return bins
    archive = f"{data_dir}/babistories-dataset.7z"
    for i in range(1, 6):
        urllib.request.urlretrieve(URL.format(i), f"{archive}.{i:03d}")
    with multivolumefile.open(archive, "rb") as f, py7zr.SevenZipFile(f) as z:
        z.extractall(data_dir)
    enc = tiktoken.get_encoding("gpt2")
    for split, path in bins.items():
        with open(f"{data_dir}/{split}dataset.txt") as f, open(path + ".tmp", "wb") as out:
            while lines := list(itertools.islice(f, 10_000)):
                stories = enc.encode_ordinary_batch([json.loads(line) for line in lines])
                np.concatenate([np.array(s + [enc.eot_token], np.uint16) for s in stories]).tofile(out)
        os.rename(path + ".tmp", path)
    return bins


def batches(ids, batch_size, seq_len, seed=0):
    """Endless random (inputs, targets) crops from a token array."""
    rng = np.random.default_rng(seed)
    while True:
        start = rng.integers(0, len(ids) - seq_len, (batch_size, 1))
        chunk = ids[start + np.arange(seq_len + 1)].astype(np.int32)
        yield chunk[:, :-1], chunk[:, 1:]
