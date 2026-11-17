"""BAR-ATH target CSV from several final models: average owner and calibrated evidence, decoder (8,8,1).

    python approaches/dense_control/ens_target.py --works ../results/dense_long/target ../results/dense_steps2/target \
        ../results/dense_w128/target --target ../results/final/target/remote1 --csv ../results/run_ens/target/Ens.csv

Each --works dir holds smooth_target.py outputs (dense.npz + its pooled inner-fold calibration.json). Selection:
results/run_ens/Ens.json (dev LOGO, same averaging). No target label is read.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import dense_control as dense  # noqa: E402
from common.run import write  # noqa: E402
from approaches.dense_control.smooth_target import SETTING, sha  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--works", type=Path, nargs="+", required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    args = ap.parse_args(); args.csv.parent.mkdir(parents=True, exist_ok=True)
    owners, evid = [], []
    for w in args.works:
        cal = json.loads((w/"calibration.json").read_text())
        with np.load(w/"dense.npz") as d:
            owners.append(d["owner"]); evid.append(dense.calibrated_evidence(d["transition"], cal))
    periods = dense.read_item(args.target/"cache")["periods"]
    pred = dense.decode_grid(np.mean(owners, 0), np.mean(evid, 0), periods, [SETTING])[:, 0]
    write(str(args.csv), np.array(list("HADX"))[pred])
    used = [p for w in args.works for p in (w/"dense.npz", w/"calibration.json")]
    dense.dump(args.csv.with_suffix(".receipt.json"), dict(csv_sha256=sha(args.csv), rows=len(pred), setting=SETTING,
        inputs={str(p): sha(p) for p in used}, selection="../results/run_ens/Ens.json"))
    print("CSV", sha(args.csv), len(pred), flush=True)


if __name__ == "__main__":
    main()
