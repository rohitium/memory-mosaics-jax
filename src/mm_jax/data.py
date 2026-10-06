"""BabiStories (the paper's dataset) and a random-crop batcher."""

import itertools
import json
import os
import urllib.request

import numpy as np

ARCHIVE_URL = ("https://github.com/facebookresearch/MemoryMosaics/raw/main/"
               "BabiStories/data/babistories-dataset.7z.{:03d}")


def prepare_babistories(data_dir, max_train_tokens=None):
    """Download BabiStories into data_dir and tokenize it to {train,val}.bin.

    Matches the reference Library/memory_mosaics/data/BabiStories/prepare.py:
    one JSON-encoded story per line, GPT-2 BPE (tiktoken encode_ordinary) plus
    <|endoftext|> after each story, stored as uint16. Skips work already done.
    max_train_tokens: stop tokenizing the training split after about this many
    tokens (written to train_<max_train_tokens>.bin instead).
    Needs `pip install tiktoken py7zr multivolumefile`. Returns {split: path}.
    """
    import multivolumefile
    import py7zr
    import tiktoken

    os.makedirs(data_dir, exist_ok=True)
    train_name = "train" if max_train_tokens is None else f"train_{max_train_tokens}"
    bins = {"train": os.path.join(data_dir, f"{train_name}.bin"),
            "val": os.path.join(data_dir, "val.bin")}
    if all(os.path.exists(p) for p in bins.values()):
        return bins

    archive = os.path.join(data_dir, "babistories-dataset.7z")
    if not os.path.exists(os.path.join(data_dir, "traindataset.txt")):
        for i in range(1, 6):  # ~470 MB in five parts
            part = f"{archive}.{i:03d}"
            if not os.path.exists(part):
                urllib.request.urlretrieve(ARCHIVE_URL.format(i), part + ".tmp")
                os.rename(part + ".tmp", part)
        with multivolumefile.open(archive, mode="rb") as f, py7zr.SevenZipFile(f) as z:
            z.extractall(data_dir)  # traindataset.txt (2.0 GB), valdataset.txt (20 MB)

    enc = tiktoken.get_encoding("gpt2")
    for split, limit in (("val", None), ("train", max_train_tokens)):
        if os.path.exists(bins[split]):
            continue
        n_tokens = 0
        with open(os.path.join(data_dir, f"{split}dataset.txt")) as f, \
                open(bins[split] + ".tmp", "wb") as out:
            while (limit is None or n_tokens < limit) and (lines := list(itertools.islice(f, 10_000))):
                stories = enc.encode_ordinary_batch([json.loads(line) for line in lines])
                ids = np.concatenate([np.array(s + [enc.eot_token], np.uint16) for s in stories])
                ids.tofile(out)
                n_tokens += len(ids)
        os.rename(bins[split] + ".tmp", bins[split])
    return bins


class TextBatcher:
    """Random (x, y) next-token crops from a 1-D token array, e.g. a .bin file
    opened with np.memmap(path, dtype=np.uint16, mode="r")."""

    def __init__(self, ids, seed=0):
        self.ids = ids
        self.rng = np.random.default_rng(seed)

    def next_batch(self, batch_size, seq_len):
        start = self.rng.integers(0, len(self.ids) - seq_len, size=batch_size)
        chunk = self.ids[start[:, None] + np.arange(seq_len + 1)].astype(np.int32)
        return chunk[:, :-1], chunk[:, 1:]
