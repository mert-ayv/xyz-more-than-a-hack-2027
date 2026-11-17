"""K = smooth dense + auxiliary contact head (results/objective arm K) on the σ=8 ball, without touching dense_control.py.

    python approaches/dense_control/contact.py train|predict|calibrate --out ../results/dense_contact ...   (same CLI as smooth.py)
    python approaches/dense_control/contact.py final freeze|train --out ../results/dense_contact/final --cache ../results/dense_contact/cache ...
    python approaches/dense_control/contact.py targets --out ../results/dense_contact [--metrica] [--kin] [--wide]   (writes <out>/contact/<game>.npz)
    python approaches/dense_control/contact.py run approaches/dense_control/smooth_target.py --dense ../results/dense_contact ...

Targets: <out>/contact/<game>.npz, the clean-DFL contact-geometry proxy of objective_trial.contact_proxy (unblurred ball;
Metrica valid=0): ball-accel local max ≥40 m/s² within 1.5 m of the nearest player → ±3-frame Gaussian on that team.
Loss = dense loss + .5 * BCE(contact, pos_weight 50) over valid core frames, exactly as objective_trial.losses arm K.
The head's gradient enters through a forward hook on the owner conv, so dense.train/final_train run unchanged;
models carry extra contact.* weights and must be predicted through this entry point.
--kin (marker <out>/contact/KIN): second training-only head regresses clean-ball speed/10, log1p(accel)/3 and turn cosine
on every valid frame of all 7 games (smooth L1, weight .5); the σ=8 input has lost exactly these kinematics.
--wide (marker <out>/contact/WIDE): contact Gaussian σ=4 frames (±12) instead of σ=1 (±3), pos_weight 50/4 so the positive
mass per contact stays K's. K's head peaks only 49–66% within ±3 frames of the proxy and its output decorrelates over 12–27
frames: the σ=8 input cannot time a contact to one frame, only which team touched it (92–94% right, out of fold).
<out>/contact/STEPS (text, e.g. 3000): OneCycle steps for the final model (LONG); folds take theirs from <out>/plan.json.
<out>/contact/WIDTH (text, e.g. 128): trunk width for the final model, same rule (W128).
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import dense_control as dense  # noqa: E402

WEIGHT, POS_WEIGHT, KIN_WEIGHT, WIDE_SIGMA = .5, 50., .5, 4
_make_net, _prepare, _batch = dense.make_net, dense.prepare, dense.batch
_pending = {}  # contact targets of the batch being trained; empty at inference


def contact_dir():
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    return out/"contact" if (out/"contact").exists() else out.parent/"contact"  # final model: --out <run>/final


def marker_path(name):  # --out for train/predict/final, --dense for smooth_target
    for flag in ("--out", "--dense"):
        if flag in sys.argv:
            d = Path(sys.argv[sys.argv.index(flag) + 1])
            return next((p for p in (d/"contact"/name, d.parent/"contact"/name) if p.exists()), None)
    return None


def kin_on(marker="KIN"): return marker_path(marker) is not None


if marker_path("STEPS"): dense.CONFIG["steps"] = int(marker_path("STEPS").read_text())  # LONG: final_train freezes CONFIG; folds read plan.json
if marker_path("WIDTH"): dense.CONFIG["width"] = int(marker_path("WIDTH").read_text())  # W128: same for the trunk width


def make_net(torch, config):
    model = _make_net(torch, config)  # shared parameters initialised first, as in arm K
    model.contact = torch.nn.Conv1d(config["width"], 2, 1)
    kin = kin_on()
    pos_weight = POS_WEIGHT / WIDE_SIGMA if kin_on("WIDE") else POS_WEIGHT
    if kin: model.kin = torch.nn.Conv1d(config["width"], 3, 1)
    halo = config["halo"]

    class AddLoss(torch.autograd.Function):  # value = owner, backward also sends d(aux)=1 → aux joins the loss
        @staticmethod
        def forward(ctx, x, aux): return x.clone()
        @staticmethod
        def backward(ctx, grad): return grad, torch.ones((), device=grad.device)

    def hook(module, inputs, output):
        if not (model.training and _pending): return None
        CT, CV = (torch.from_numpy(_pending.pop(k)).to(output.device) for k in ("CT", "CV"))
        logits = model.contact(inputs[0]).transpose(1, 2)[:, halo:-halo]
        CT, CV = CT[:, halo:-halo], CV[:, halo:-halo].float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, CT,
            pos_weight=torch.tensor(pos_weight, device=output.device), reduction="none")
        loss = WEIGHT * (loss * CV[..., None]).sum() / (CV.sum() * 2).clamp(min=1)
        if kin:
            KT, KV = (torch.from_numpy(_pending.pop(k)).to(output.device) for k in ("KT", "KV"))
            pred = model.kin(inputs[0]).transpose(1, 2)[:, halo:-halo]
            KT, KV = KT[:, halo:-halo], KV[:, halo:-halo].float()
            k = torch.nn.functional.smooth_l1_loss(pred, KT, reduction="none")
            loss = loss + KIN_WEIGHT * (k * KV[..., None]).sum() / (KV.sum() * 3).clamp(min=1)
        return AddLoss.apply(output, loss)

    model.owner.register_forward_hook(hook)
    return model


def prepare(items):
    _prepare(items)
    for it in items:
        with np.load(contact_dir() / f"{it['meta']['game']}.npz") as z:
            assert len(z["valid"]) == len(it["gt"])
            it.update(contact=z["target"].astype(np.float32), contact_valid=z["valid"].copy())
            if "kin" in z: it.update(kin=z["kin"].astype(np.float32), kin_valid=z["kin_valid"].copy())


def finite_runs(valid, periods, minimum=1):
    for a, b in periods:
        edges = np.diff(np.r_[0, valid[a:b].astype(np.int8), 0])
        for s, e in zip(np.flatnonzero(edges == 1)+a, np.flatnonzero(edges == -1)+a):
            if e-s >= minimum: yield int(s), int(e)


def contact_proxy(match, sigma=1):
    """objective_trial.contact_proxy for DFL; no owner labels. Validity covers the full ±3 accel context."""
    n = match.n_frames; target = np.zeros((n, 2), np.float32); valid = np.zeros(n, bool)
    ball = match.ball * np.asarray(match.pitch); players = match.players * np.asarray(match.pitch)
    accel = np.full(n, np.nan)
    for s, e in finite_runs(np.isfinite(ball).all(1), match.periods, 9):
        velocity = np.gradient(ball[s:e], axis=0)*match.fps
        accel[s+2:e-2] = (np.linalg.norm(np.gradient(velocity, axis=0), axis=1)*match.fps)[2:-2]
    distances = np.stack([np.where(np.isfinite(d), d, np.inf).min(1) for d in
                          (np.linalg.norm(players[:, match.team == team]-ball[:, None], axis=2) for team in (0, 1))], 1)
    for s, e in finite_runs(np.isfinite(accel) & np.isfinite(distances).all(1), match.periods, 7): valid[s+3:e-3] = True
    localmax, safe = np.ones(n, bool), np.nan_to_num(accel, nan=-np.inf)
    for offset in range(-3, 4):
        shifted = np.full(n, -np.inf); lo, hi = max(0, -offset), min(n, n-offset)
        shifted[lo:hi] = safe[lo+offset:hi+offset]; localmax &= safe >= shifted
    team = distances.argmin(1)
    for t in np.flatnonzero(valid & localmax & (accel >= 40) & (distances.min(1) <= 1.5)):
        idx = np.arange(max(0, t-3*sigma), min(n, t+3*sigma+1))
        target[idx, team[t]] = np.maximum(target[idx, team[t]], np.exp(-.5*((idx-t)/sigma)**2))
    target[~valid] = 0
    return target, valid


def kinematics(match):
    """Clean-ball speed/10, log1p(accel)/3, cos(v[t-2], v[t+2]) (1 below 1 m/s); valid where all are defined."""
    n = match.n_frames; kin = np.zeros((n, 3), np.float32); valid = np.zeros(n, bool)
    ball = match.ball * np.asarray(match.pitch)
    for s, e in finite_runs(np.isfinite(ball).all(1), match.periods, 9):
        v = np.gradient(ball[s:e], axis=0)*match.fps; speed = np.linalg.norm(v, axis=1)
        accel = np.linalg.norm(np.gradient(v, axis=0), axis=1)*match.fps
        a, b = v[:-4], v[4:]
        cos = (a*b).sum(1) / np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-9)
        turn = np.where(np.minimum(speed[:-4], speed[4:]) > 1, cos, 1.)
        kin[s+2:e-2] = np.stack([speed[2:-2]/10, np.log1p(accel[2:-2])/3, turn], 1)
        valid[s+2:e-2] = speed[2:-2] <= 40  # clean tracks teleport at restarts (up to 660 m/s); no real ball is faster
    return kin, valid


def write_targets(out, metrica=False, kin=False, wide=False):
    from common import dfl_data, tuning_data
    (out/"contact").mkdir(parents=True, exist_ok=True)
    if kin: (out/"contact"/"KIN").touch()
    if wide: (out/"contact"/"WIDE").touch()
    sigma = WIDE_SIGMA if wide else 1
    for game in dense.GAMES:
        extra = {}
        if kin:
            match = (dfl_data.load_game(game[4:]) if game.startswith("dfl_") else tuning_data.load_game(game[8:]))[0]
            k, kv = kinematics(match); extra = dict(kin=k, kin_valid=kv)
            assert len(k) == len(dense.read_item(out/"cache"/f"{game}-0")["gt"])
            print(game, "kin", int(kv.sum()), np.round(k[kv].mean(0), 3).tolist(), flush=True)
        if game.startswith("dfl_"): target, valid = contact_proxy(dfl_data.load_game(game[4:])[0], sigma)
        elif metrica:  # Metrica clean ball: 83–86% of on-ball events have a proxy contact ±3 frames (results/metrica_contact)
            target, valid = contact_proxy(tuning_data.load_game(game[8:])[0], sigma)
            assert len(target) == len(dense.read_item(out/"cache"/f"{game}-0")["gt"])
        else: n = len(dense.read_item(out/"cache"/f"{game}-0")["gt"]); target, valid = np.zeros((n, 2), np.float32), np.zeros(n, bool)
        np.savez_compressed(out/"contact"/f"{game}.npz", target=target, valid=valid, **extra)
        print(game, int(valid.sum()), int((target == 1).sum()), round(float(target.sum()), 1), flush=True)


class Capture:  # records the window slices dense.batch takes from G
    def __init__(self, values, item, seen): self.values, self.item, self.seen = values, item, seen
    def __getitem__(self, key): self.seen.append((self.item, key)); return self.values[key]
    def __len__(self): return len(self.values)


class RNG:  # records draws so the final xf flip can be replayed on the targets
    def __init__(self, rng): self.rng, self.ints = rng, []
    def random(self): return self.rng.random()
    def integers(self, *a, **k): v = self.rng.integers(*a, **k); self.ints.append(int(v)); return v


def batch(items, rng, config):
    seen, original, wrapped = [], [it["G"] for it in items], RNG(rng)
    try:
        for it in items: it["G"] = Capture(it["G"], it, seen)
        arrays = _batch(items, wrapped, config)
    finally:
        for it, G in zip(items, original): it["G"] = G
    CT = np.stack([it["contact"][sl] for it, sl in seen]); CV = np.stack([it["contact_valid"][sl] for it, sl in seen])
    if wrapped.ints[-2]: CT = CT[:, :, ::-1].copy()  # xf swaps teams
    _pending.update(CT=CT, CV=CV)
    if "kin" in items[0]:  # mirror-invariant scalars: no flip
        _pending.update(KT=np.stack([it["kin"][sl] for it, sl in seen]), KV=np.stack([it["kin_valid"][sl] for it, sl in seen]))
    return arrays


dense.make_net, dense.prepare, dense.batch = make_net, prepare, batch

if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:  # numpy-RNG stream and arrays identical to dense.batch; targets follow the flip
        rng_items = []
        for n, game in ((3000, "a"), (2500, "b")):
            gt = np.random.default_rng(1).integers(0, 3, n).astype(np.uint8)
            it = dict(G=np.random.default_rng(2).normal(size=(n, 20)).astype(np.float32), P=np.zeros((n, 2, 3, 18), np.float32),
                      M=np.ones((n, 2, 3), bool), gt=gt, periods=np.array([[0, n]]), meta=dict(game=game))
            it["G"][:, 0] = np.arange(n); it["contact"] = np.stack([np.arange(n), -np.ones(n)], 1).astype(np.float32)
            it["contact_valid"] = np.ones(n, bool); it["kin"] = np.tile(it["G"][:, :1], 3); it["kin_valid"] = np.ones(n, bool)
            rng_items.append(it)
        _prepare(rng_items); cfg = dict(window=256, batch=8); flips = set()
        for seed in range(6):
            a = _batch(rng_items, np.random.default_rng(seed), cfg); b = batch(rng_items, np.random.default_rng(seed), cfg)
            assert all((x == y).all() for x, y in zip(a, b))
            flip = bool((_pending["CT"][:, :, 0] < 0).all()); flips.add(flip)
            assert (_pending["CT"][:, :, int(flip)] == b[0][:, :, 0]).all()
            assert (_pending["KT"][:, :, 0] == b[0][:, :, 0]).all()  # kin follows the same window, unflipped
        assert flips == {False, True}
        print("ok")
    elif sys.argv[1:2] == ["targets"]:
        write_targets(Path(sys.argv[sys.argv.index("--out") + 1]), "--metrica" in sys.argv, "--kin" in sys.argv, "--wide" in sys.argv)
    elif sys.argv[1:2] in (["final"], ["run"]):  # final_train.py freeze|train …, or run <script> …, with the patched net
        import runpy
        script = str(Path(__file__).with_name("final_train.py")) if sys.argv[1] == "final" else sys.argv.pop(2)
        sys.argv = [script] + sys.argv[2:]
        runpy.run_path(script, run_name="__main__")
    else:
        dense.main()
