"""Weight management for LocalDreamEnv.

The DOOM Dungeon HG checkpoint is public:
https://huggingface.co/alakazamworld/doom-dungeon-hg (weights CC BY-NC-SA).
Files are downloaded once into a local cache and sha256-verified against the
pins below (which match the model card / HOSTING.md). A mismatch is a hard
error — never train against unverified weights.

Resolution order:
  1. explicit `weights_dir=` argument
  2. env var ALAKAZAM_GYM_WEIGHTS_DIR (must already contain the files)
  3. cache ~/.cache/alakazam-gym/doom-dungeon-hg (downloaded on first use)
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request

HF_BASE = "https://huggingface.co/alakazamworld/doom-dungeon-hg/resolve/main"
FILES = {
    # name: (bytes, sha256) — pins from the published model card
    "denoiser.onnx": (
        279689999,
        "62417a400befccde771317c4e8c9788ade237b6ba8f8840dccb8301af4b3cc37"),
    "init_state.json": (
        886772,
        "066162c3e96f49232c59ea35a6455854c5674acbf80c55190ee9dfaa49370500"),
    "model_meta.json": (
        639,
        "2fa60c8d00bc0a2c00ec8b2c87b1c1efbfab9ec87e200b190f46cafd3a78e941"),
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(dirpath: str, strict_size: bool = True) -> None:
    for name, (size, sha) in FILES.items():
        p = os.path.join(dirpath, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing weight file: {p}")
        if strict_size and os.path.getsize(p) != size:
            raise ValueError(f"{p}: size {os.path.getsize(p)} != pinned {size}")
        got = _sha256(p)
        if got != sha:
            raise ValueError(f"{p}: sha256 {got} != pinned {sha} — refusing")


def ensure_weights(weights_dir: str | None = None) -> str:
    """Return a directory containing verified weights, downloading if needed."""
    if weights_dir:
        _verify(weights_dir)
        return weights_dir
    envdir = os.environ.get("ALAKAZAM_GYM_WEIGHTS_DIR")
    if envdir:
        _verify(envdir)
        return envdir
    cache = os.path.join(os.path.expanduser("~"), ".cache", "alakazam-gym",
                         "doom-dungeon-hg")
    os.makedirs(cache, exist_ok=True)
    for name, (size, sha) in FILES.items():
        p = os.path.join(cache, name)
        if os.path.exists(p) and _sha256(p) == sha:
            continue
        url = f"{HF_BASE}/{name}"
        print(f"[alakazam-gym] downloading {name} ({size/1e6:.0f} MB) from "
              f"huggingface.co/alakazamworld/doom-dungeon-hg ...",
              file=sys.stderr, flush=True)
        tmp = p + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, p)
    _verify(cache)
    return cache
