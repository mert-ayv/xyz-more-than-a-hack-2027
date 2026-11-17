"""Run one approach on BAR-ATH and write its submission file.

    python common/run.py <approach>      # -> evaluation/results/<approach>.csv

Reads data/ locally. Prints only aggregate counts (possessions, mean duration, label shares) --
safe to show an agent. Never prints frames or positions.
"""

import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.match import load_bar_ath  # noqa: E402

RESULTS = os.path.join(ROOT, "evaluation", "results")


def load_approach(name):
    path = os.path.join(ROOT, "approaches", name, "approach.py")
    spec = importlib.util.spec_from_file_location("approach_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary(labels, periods, fps):
    """Aggregate counts only: possessions, mean duration, label shares over the graded periods."""
    n_poss, frames_poss = 0, 0
    graded = np.concatenate([labels[s:e] for s, e in periods])
    for s, e in periods:
        seg = labels[s:e]
        filled = seg.copy()
        last = None
        for k, v in enumerate(filled):  # same forward-fill the scorer applies to X
            if v == "X" and last is not None:
                filled[k] = last
            elif v != "X":
                last = v
        change = np.r_[True, filled[1:] != filled[:-1]]
        starts = np.flatnonzero(change)
        n_poss += int(np.isin(filled[starts], ["H", "A"]).sum())
        frames_poss += int(np.isin(filled, ["H", "A"]).sum())
    shares = {k: float((graded == k).mean()) for k in "HADX"}
    return {"possessions": n_poss,
            "mean_possession_s": frames_poss / max(n_poss, 1) / fps,
            "shares": shares}


def write(path, labels):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, v) for i, v in enumerate(labels))


def main():
    name = sys.argv[1]
    match = load_bar_ath()
    labels = np.asarray(load_approach(name).predict(match))
    assert len(labels) == match.n_frames and set(np.unique(labels)) <= set("HADX")
    path = os.path.join(RESULTS, name + ".csv")
    write(path, labels)
    s = summary(labels, match.periods, match.fps)
    print("wrote {}: {} possessions, mean {:.1f} s, shares {}".format(
        os.path.relpath(path, ROOT), s["possessions"], s["mean_possession_s"],
        {k: round(v, 3) for k, v in s["shares"].items()}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        # A message from the loader could quote a raw data line; print only the error type so the
        # output stays safe to show an agent. Re-run with MTAH_DEBUG=1 to see the full traceback.
        if os.environ.get("MTAH_DEBUG"):
            raise
        print("FAILED: {} (set MTAH_DEBUG=1 for details)".format(type(exc).__name__))
        sys.exit(1)
