"""Atomic publication of P5-produced NPZs without changing the shared contract.

Stage in the destination directory, flush, verify every array, then publish.
The default uses an exclusive hard link so a concurrently created output wins.
Explicit replacement uses os.replace; failures before publication keep the old
artifact intact. Filesystems without hard-link support fail without a fallback
that could expose a partial NPZ or overwrite an existing output.
"""
import os
from pathlib import Path
import tempfile

import numpy as np

from mtah import contract


def _verify(path, name, expected):
    with np.load(path, allow_pickle=False) as archive:
        actual = {key: archive[key] for key in archive.files}
    contract.check(name, actual)
    if set(actual) != set(expected):
        raise ValueError("staged P5 NPZ field verification failed")
    for key, value in expected.items():
        same = np.array_equal(actual[key], value, equal_nan=value.dtype.kind in "fc")
        if actual[key].dtype != value.dtype or not same:
            raise ValueError("staged P5 NPZ array verification failed")


def save_npz(name, replace=False, **arrays):
    """Match contract dtypes/checks, publishing only after a verified round trip."""
    if name not in ("p_dead", "tag_prior"):
        raise ValueError("P5 writer only supports p_dead and tag_prior")
    prepared = {key: np.asarray(value) for key, value in arrays.items()}
    for field, kind in contract.SPEC[name]["fields"].items():
        if field in prepared and kind != "bool":
            prepared[field] = prepared[field].astype(np.float32)
    contract.check(name, prepared)
    output = Path(contract.path(name))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=str(output.parent),
                                         prefix=".p5-", suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez(handle, **prepared)
            handle.flush()
            os.fsync(handle.fileno())
        _verify(temporary, name, prepared)
        if replace:
            os.replace(str(temporary), str(output))
        else:
            os.link(str(temporary), str(output))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return str(output)
