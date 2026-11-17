"""BAR-ATH target outputs from the smooth-ball final model (results/dense_smooth/final).

    python approaches/dense_control/smooth_target.py --dense ../results/dense_smooth --work ../results/dense_smooth/target \
        --target ../results/final/target/remote1 --csv ../Submission/Xyz.csv --weight 1

1. Pooled transition calibration on the seven smooth inner-fold seed-104 predictions (final.calibrate_fit recipe).
2. Four-view inference of the final model on the real target cache (already smooth; never re-smoothed).
3. Same decision step as redecoded.py: weight·dense + (1-weight)·frozen champion spans, decoder (8,8,1).
No target label is read. At --weight 1 a missing --target cache is built from the supplied tactical files, and a
calibration.json already in --work is used as is (public recipe: model.pt + calibration.json + supplied data only).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import dense_control as dense  # noqa: E402
from common.run import write  # noqa: E402

SETTING = (8, 8, 1.)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def calibrate(torch, args):
    path = args.work/"calibration.json"
    if path.exists(): return json.loads(path.read_text())
    from approaches.dense_control.final import INNER  # refit only: needs the dev caches
    xs, ys = [], []
    for game, outer in INNER.items():
        it = dense.read_item(args.dense/"cache"/f"{game}-104")
        with np.load(args.dense/"probabilities"/f"{outer}-inner-{game}-104.npz") as z:
            assert str(z["evaluation_game"]) == game and str(z["role"]) == "inner"
            transition = z["transition"]
        valid = dense.transition_valid(it["gt"], it["periods"])
        xs.append(transition[valid]); ys.append(dense.directed_targets(it["gt"], it["periods"], dense.CONFIG["boundary_radius"])[valid])
    x = torch.from_numpy(np.concatenate(xs)); target = np.concatenate(ys); y = torch.from_numpy(target.astype(np.float32))
    raw = torch.nn.Parameter(torch.full((6,), .5413248546)); bias = torch.nn.Parameter(torch.full((6,), -np.log(50), dtype=torch.float32))
    opt = torch.optim.LBFGS([raw, bias], lr=.5, max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); slope = torch.nn.functional.softplus(raw)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x*slope+bias, y)
        loss = loss + .001*((slope-1)**2).mean() + .00001*(bias**2).mean(); loss.backward(); return loss
    opt.step(closure)
    cal = dict(slope=torch.nn.functional.softplus(raw).detach().tolist(), bias=bias.detach().tolist(),
               prior=target.mean(0).tolist(), valid_frames=len(target), inner_map=INNER)
    dense.dump(path, cal)
    return cal


def build_target(cache):
    """Dense target cache straight from the supplied tactical files (final.py build without champion/seal gates)."""
    if (cache/"meta.json").exists(): return
    from common.match import load_bar_ath
    match = load_bar_ath(cache=False); assert match.fps == 25
    G, P, M, meta = dense.features(match)
    cache.mkdir(parents=True, exist_ok=True)
    for name, value in dict(G=G, P=P, M=M, gt=np.full(match.n_frames, 3, np.uint8), periods=np.array(match.periods)).items():
        dense.save_npy(cache/f"{name}.npy", value)
    dense.dump(cache/"meta.json", dict(meta, game="target", target_labels_loaded=False))
    print("built target cache", match.n_frames, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dense", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--weight", type=float, default=.75)
    args = ap.parse_args(); args.work.mkdir(parents=True, exist_ok=True)
    torch = dense.torch_setup(4)
    if args.weight == 1: build_target(args.target/"cache")
    cal = calibrate(torch, args)
    out = args.work/"dense.npz"
    if not out.exists():
        ck = torch.load(args.dense/"final/model.pt", map_location="cpu", weights_only=False)
        assert ck["role"] == "final" and ck["train_games"] == list(dense.GAMES)
        model = dense.make_net(torch, ck["config"]); model.load_state_dict(ck["model"]); model.eval()
        it = dense.read_item(args.target/"cache")
        owner, transition = dense.infer(torch, model, it, torch.device("cpu"), ck["config"]["halo"])
        assert np.isfinite(owner).all() and np.isfinite(transition).all()
        np.savez_compressed(out, owner=owner, transition=transition, model_sha256=sha(args.dense/"final/model.pt"))
    periods = dense.read_item(args.target/"cache")["periods"]  # same halves as champion.npz
    with np.load(out) as d:
        n = len(d["owner"]); emission = d["owner"]
        if args.weight < 1:  # weight 1 = deliverable, dense-only: no champion pipeline needed (0·x is exact)
            from approaches.dense_control.ensemble import painted
            with np.load(args.target/"champion.npz") as c:
                emission = (1-args.weight)*painted(n, c["S"], c["E"], c["P"]) + args.weight*emission
        pred = dense.decode_grid(emission, dense.calibrated_evidence(d["transition"], cal), periods, [SETTING])[:, 0]
    write(str(args.csv), np.array(list("HADX"))[pred])
    used = [out, args.work/"calibration.json"] + ([args.target/"champion.npz"] if args.weight < 1 else [])
    dense.dump(args.csv.with_suffix(".receipt.json"), dict(csv_sha256=sha(args.csv), rows=n, weight=args.weight, setting=SETTING,
        inputs={str(p): sha(p) for p in used},
        selection=str(args.dense/"Eval.json")))
    print("CSV", sha(args.csv), n, flush=True)


if __name__ == "__main__":
    main()
