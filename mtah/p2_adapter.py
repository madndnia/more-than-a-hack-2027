"""Read-only P2 CSV -> contract-compatible p_dead.npz, with source SHA-256.

    python -m mtah.p2_adapter --csv data/derived/p_dead.csv

The CSV is never changed. A different existing NPZ requires --replace-npz.
"""
import argparse
import csv
import hashlib
import io
import os
from pathlib import Path

import numpy as np

from mtah import contract


def read_csv(path):
    """Strict frame alignment, exact column names, finite probabilities; no repairs."""
    content = Path(path).read_bytes()
    reader = csv.reader(io.StringIO(content.decode("utf-8-sig")), strict=True)
    if next(reader, None) != ["frame", "p_dead"]:
        raise ValueError("P2 CSV must have exactly the header frame,p_dead")
    result = np.empty(contract.N_FRAMES, dtype=np.float64)
    count = 0
    for row in reader:
        if count >= contract.N_FRAMES or len(row) != 2:
            raise ValueError("P2 CSV has extra rows or malformed fields")
        try:
            frame, probability = int(row[0]), float(row[1])
        except ValueError:
            raise ValueError("P2 CSV contains a noninteger frame or nonnumeric probability") from None
        if frame != count:
            raise ValueError("P2 CSV frames must be exactly 0..N_FRAMES-1 in order")
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("P2 CSV probabilities must be finite and in [0, 1]")
        result[count] = probability
        count += 1
    if count != contract.N_FRAMES:
        raise ValueError("P2 CSV has the wrong number of frames")
    return result, hashlib.sha256(content).hexdigest()


def adapt(csv_path, replace=False):
    source, output = Path(csv_path), Path(contract.path("p_dead"))
    if (source.resolve() == output.resolve()
            or (output.exists() and os.path.samefile(str(source), str(output)))):
        raise ValueError("P2 CSV source and NPZ destination must be different files")
    values, digest = read_csv(source)
    converted = values.astype(np.float32)  # contract.save uses float32 for probabilities
    contract.check("p_dead", {"p_dead": converted})
    if output.exists() and not replace:
        previous = contract.load("p_dead")
        if (np.asarray(previous.get("source_sha256", "")).shape == ()
                and str(previous.get("source_sha256", "")) == digest
                and np.array_equal(previous["p_dead"], converted)):
            return str(output)
        raise FileExistsError("p_dead.npz differs or has no provenance; use --replace-npz explicitly")
    result = contract.save("p_dead", p_dead=converted, source_sha256=np.array(digest))
    verified = contract.load("p_dead")
    if (not np.array_equal(verified["p_dead"], converted)
            or str(verified["source_sha256"]) != digest):
        raise ValueError("P2 NPZ round-trip verification failed")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", required=True)
    parser.add_argument("--replace-npz", action="store_true")
    args = parser.parse_args(argv)
    try:
        adapt(args.csv, replace=args.replace_npz)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, "P2 adapter failed: {}\n".format(error))
    print("P2 adapter OK: {} frames; CSV unchanged; NPZ verified".format(contract.N_FRAMES))


if __name__ == "__main__":
    main()
