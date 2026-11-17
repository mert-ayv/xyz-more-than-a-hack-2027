"""Fixed all-development dense training. No calibration, reserved or target reads."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import dense_control as dense


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sources():
    return {str(p.relative_to(ROOT)): sha(p) for p in (
        Path(__file__), ROOT / "common/dense_control.py", ROOT / "common/tuning_data.py",
        ROOT / "common/dfl_data.py", ROOT / "evaluation/possession_metrics.py")}


def finite(torch, value):
    if torch.is_tensor(value):
        assert torch.isfinite(value).all(), "nonfinite checkpoint tensor"
    elif isinstance(value, dict):
        for v in value.values(): finite(torch, v)
    elif isinstance(value, (list, tuple)):
        for v in value: finite(torch, v)
    elif isinstance(value, float):
        assert np.isfinite(value), "nonfinite checkpoint scalar"


def restore_rng(torch, ck, rng, device):
    rng.bit_generator.state = ck["numpy_rng"]
    torch.set_rng_state(ck["torch_rng"].cpu())
    if device.type == "cuda":
        assert ck["cuda_rng"] is not None
        torch.cuda.set_rng_state_all([state.cpu() for state in ck["cuda_rng"]])


def check(args):
    torch = dense.torch_setup(args.threads)
    device = torch.device(args.device)
    rng = np.random.default_rng(7)
    ck = dict(numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state().to(device),
              cuda_rng=[state.to(device) for state in torch.cuda.get_rng_state_all()] if device.type == "cuda" else None)
    expected = (rng.random(5), torch.rand(5), torch.rand(5, device=device))
    restore_rng(torch, ck, rng, device)
    actual = (rng.random(5), torch.rand(5), torch.rand(5, device=device))
    assert np.array_equal(expected[0], actual[0])
    assert all(torch.equal(a, b) for a, b in zip(expected[1:], actual[1:]))
    print(json.dumps(dict(check="CPU/CUDA RNG restoration exact", device=str(device))))


def freeze(args):
    assert not (args.out / "model.pt").exists() and not (args.out / "checkpoint.pt").exists()
    # ponytail: one explicit input manifest; reuse verified transfer hashes rather than a new cache builder.
    expected = json.loads(args.expected_manifest.read_text())
    assert expected["complete"] and expected["mismatches"] == []
    identity = dense.source_identity()
    rows = []
    for game in dense.GAMES:
        for seed in dense.TRAIN_SEEDS:
            name = f"{game}-{seed}"
            path = args.cache / name
            files = {}
            for filename in ("G.npy", "P.npy", "M.npy", "gt.npy", "periods.npy", "meta.json"):
                key = f"results/dense/cache/{name}/{filename}"
                actual = dict(bytes=(path / filename).stat().st_size, sha256=sha(path / filename))
                assert actual == expected["hashes"][key], key
                files[filename] = actual
            it = dense.read_item(path)
            meta = it["meta"]
            assert meta["game"] == game and meta["seed"] == seed and meta["source_identity"] == identity
            assert hashlib.sha256(it["gt"].tobytes()).hexdigest() == meta["original_gt_sha256"]
            n = len(it["gt"])
            assert it["G"].shape == (n, 20) and it["P"].shape == (n, 2, meta["player_capacity"], 18)
            assert it["M"].shape == it["P"].shape[:-1] and it["M"].dtype == bool
            previous = 0
            for a, b in it["periods"]:
                assert previous <= a < b <= n and b - a >= dense.CONFIG["window"]
                previous = b
            rows.append(dict(item=name, meta=meta, files=files, periods=it["periods"].tolist(), frames=n))
    assert len(rows) == 28 and {r["meta"]["game"] for r in rows} == set(dense.GAMES)
    src = sources()
    config = json.loads(json.dumps(dense.CONFIG))
    recipe = dict(config=config, seed=7, train_games=list(dense.GAMES), train_seeds=list(dense.TRAIN_SEEDS),
                  weight_decay=.001, clip_grad_norm=5., role="final", source_identity=identity,
                  training_code_sha256=sha(__file__), dense_code_sha256=sha(dense.__file__), sources=src,
                  training="Exact frozen dense primitives and training objective; fresh model, all seven development games.",
                  exclusions="No seed0/1/2/104, reserved or target inputs. No evaluation/model selection in this entrypoint.")
    manifest = dict(complete=True, items=rows, count=28, expected_transfer_sha256=sha(args.expected_manifest))
    for name, value in (("recipe.json", recipe), ("inputs.json", manifest)):
        path = args.out / name
        if path.exists(): assert json.loads(path.read_text()) == value, f"frozen {name} changed"
        else: dense.dump(path, value)
    for rel in src:
        dest = args.out / "sources" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists(): assert sha(dest) == src[rel]
        else: shutil.copyfile(ROOT / rel, dest)
    torch = dense.torch_setup(args.threads)
    record = dict(complete=True, frozen_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  recipe_sha256=sha(args.out / "recipe.json"), input_manifest_sha256=sha(args.out / "inputs.json"),
                  environment=dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
                                   cuda=torch.version.cuda, device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
    if (args.out / "freeze.json").exists():
        old = json.loads((args.out / "freeze.json").read_text())
        assert all(old[k] == record[k] for k in ("recipe_sha256", "input_manifest_sha256", "environment"))
    else: dense.dump(args.out / "freeze.json", record)
    print(json.dumps(dict(event="frozen", **record)), flush=True)


def identities(out):
    frozen = json.loads((out / "freeze.json").read_text())
    assert frozen["complete"]
    assert sha(out / "recipe.json") == frozen["recipe_sha256"]
    assert sha(out / "inputs.json") == frozen["input_manifest_sha256"]
    recipe = json.loads((out / "recipe.json").read_text())
    assert recipe["sources"] == sources() and recipe["source_identity"] == dense.source_identity()
    assert recipe["config"] == json.loads(json.dumps(dense.CONFIG))
    assert recipe["train_games"] == list(dense.GAMES) and recipe["train_seeds"] == list(dense.TRAIN_SEEDS)
    return dict(config=recipe["config"], source_identity=recipe["source_identity"], role="final", seed=7,
                train_games=recipe["train_games"], train_seeds=recipe["train_seeds"],
                training_code_sha256=recipe["training_code_sha256"], dense_code_sha256=recipe["dense_code_sha256"],
                recipe_sha256=frozen["recipe_sha256"], input_manifest_sha256=frozen["input_manifest_sha256"])


def verify_identity(ck, identity):
    assert all(ck[k] == v for k, v in identity.items()), "checkpoint identity mismatch"


def audit(args):
    torch = dense.torch_setup(args.threads)
    identity = identities(args.out)
    ck = torch.load(args.out / "checkpoint.pt", map_location="cpu", weights_only=False)
    model = torch.load(args.out / "model.pt", map_location="cpu", weights_only=False)
    for state in (ck, model):
        verify_identity(state, identity)
        assert state["step"] == dense.CONFIG["steps"]
        finite(torch, state)
    assert ck["scheduler"]["last_epoch"] == dense.CONFIG["steps"]
    assert all(torch.equal(ck["model"][k], v) for k, v in model["model"].items())
    progress = json.loads((args.out / "progress.json").read_text())
    assert progress["complete"] and [r["step"] for r in progress["history"]] == list(range(10, dense.CONFIG["steps"] + 1, 10))
    assert ck["torch_rng"].dtype == torch.uint8 and all(s.dtype == torch.uint8 for s in ck["cuda_rng"] or [])
    report = dict(complete=True, step=dense.CONFIG["steps"], identity=identity, finite_model_optimizer_scheduler=True,
                  complete_progress=True, identical_checkpoint_model=True,
                  files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in
                         (args.out / "model.pt", args.out / "checkpoint.pt", args.out / "progress.json")})
    dense.dump(args.out / "validation.json", report)
    print(json.dumps(report), flush=True)


def train(args):
    identity = identities(args.out)
    config = identity["config"]
    torch = dense.torch_setup(args.threads)
    device = torch.device(args.device)
    checkpoint = args.out / "checkpoint.pt"
    if (args.out / "model.pt").exists():
        audit(args)
        return
    manifest = json.loads((args.out / "inputs.json").read_text())
    items = []
    for row in manifest["items"]:
        path = args.cache / row["item"]
        assert all(sha(path / filename) == saved["sha256"] for filename, saved in row["files"].items())
        items.append(dense.read_item(path))
    dense.prepare(items)
    model = dense.make_net(torch, config).to(device)
    for key, value in dense.normalisation(items).items():
        getattr(model, key).copy_(torch.from_numpy(value).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=.001)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, config["lr"], total_steps=config["steps"])
    rng = np.random.default_rng(7)
    start = 0
    if checkpoint.exists():
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        verify_identity(ck, identity)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        start = ck["step"]
        restore_rng(torch, ck, rng, device)
    halo = config["halo"]
    tick = time.monotonic()
    progress = args.out / "progress.json"
    history = json.loads(progress.read_text()).get("history", []) if start and progress.exists() else []
    history = [row for row in history if row["step"] <= start]
    previous_elapsed = history[-1]["elapsed"] if history else 0
    print(json.dumps(dict(event="start", device=str(device), torch=str(torch.__version__), **identity,
                          parameters=sum(p.numel() for p in model.parameters()), resume_step=start)), flush=True)
    for step in range(start, config["steps"]):
        arrays = dense.batch(items, rng, config)
        G, P, M, Y, T, V = [torch.from_numpy(v).to(device) for v in arrays]
        owner, transition = model(G, P, M)
        owner, transition, Y, T, V = owner[:, halo:-halo], transition[:, halo:-halo], Y[:, halo:-halo], T[:, halo:-halo], V[:, halo:-halo]
        valid = Y < 3
        targets = Y.masked_fill(~valid, -100)
        owner_loss = torch.nn.functional.cross_entropy(owner.reshape(-1, 3), targets.reshape(-1), ignore_index=-100)
        transition_loss = torch.nn.functional.binary_cross_entropy_with_logits(transition, T,
            pos_weight=torch.tensor(config["positive_weight"], device=device), reduction="none")
        transition_loss = (transition_loss * V[..., None]).sum() / (V.sum() * 6).clamp(min=1)
        loss = owner_loss + config["transition_weight"] * transition_loss
        assert torch.isfinite(loss), "nonfinite loss"
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        opt.step(); sched.step()
        if (step + 1) % 10 == 0:
            row = dict(step=step + 1, loss=float(loss.item()), owner=float(owner_loss.item()), transition=float(transition_loss.item()),
                       elapsed=previous_elapsed + time.monotonic() - tick, seconds_per_step=(time.monotonic() - tick) / (step + 1 - start))
            history.append(row); print(json.dumps(row), flush=True)
            dense.dump(progress, dict(config=config, last=row, complete=False, history=history))
        if (step + 1) % 100 == 0 or step + 1 == config["steps"]:
            dense.atomic_torch_save(torch, checkpoint, dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                step=step + 1, numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None, **identity))
    dense.atomic_torch_save(torch, args.out / "model.pt", dict(model=model.state_dict(), step=config["steps"], **identity))
    dense.dump(progress, dict(config=config, complete=True, history=history,
                             elapsed=previous_elapsed + time.monotonic() - tick, torch=str(torch.__version__), device=str(device)))
    audit(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("freeze", "train", "audit", "check"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--expected-manifest", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    globals()[args.stage](args)


if __name__ == "__main__": main()
