"""Rank every approach on the external tuning games. This, not the public 5 minutes, picks winners.

    python common/compare.py                          # all approaches, 10 seeds, vs the reference
    python common/compare.py A B --ref B --seeds 20   # A against B
    python common/compare.py --panel                  # also the robustness panel (slower)
    python common/compare.py --games all              # metrica | dfl | all (default metrica)
    python common/compare.py --games dfl --holdout    # + the 2 DFL holdout games (final pick only)

Scored with the official evaluation/possession_metrics.py. A difference counts as real only if
(1) its 90% paired block-bootstrap interval excludes 0, (2) it has the same sign on every tuning
game, and (3) with --panel, it keeps its sign on every robustness scenario.

The public BAR-ATH window appears only as a validity check of evaluation/results/<approach>.csv.
It is never scored here: EVALUATION.md forbids using it to choose between versions.
"""

import argparse
import copy
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from common import dfl_data, tuning_data  # noqa: E402
from common.run import load_approach  # noqa: E402

PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
RESULTS = os.path.join(ROOT, "evaluation", "results")
BLOCK_S = 300          # bootstrap block: 5 minutes of play
BOOT_N = 200
DEFAULT_REF = "hold_gaps_hysteresis"


def all_approaches():
    base = os.path.join(ROOT, "approaches")
    return sorted(d for d in os.listdir(base)
                  if os.path.exists(os.path.join(base, d, "approach.py")))


def spell_f1(gt, pred):
    runs = pm.valid_runs(gt)
    return pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)["mean_f1"]


def full_score(gt, pred):
    r = pm.evaluate(gt, pred)
    return {"spell": r["spell_f1_mean"], "macro": r["macro_f1"], "n_pred": r["n_pred_spells"],
            "n_gt": r["n_gt_spells"], "home_err": r["pred_home_share"] - r["gt_home_share"]}


def public_check(name):
    """Validity of evaluation/results/<name>.csv only. No score against the public ground truth."""
    path = os.path.join(RESULTS, name + ".csv")
    if not os.path.exists(path):
        return "no file"
    reason = pm.validate_submission(pm.read_label_csv(path), len(pm.read_label_csv(PUBLIC_GT)))
    return "INVALID: " + reason if reason else "valid"


def blocks(gt, fps):
    """Frame-index arrays of ~5 min blocks of graded play."""
    out = []
    for s, e in pm.valid_runs(gt):
        edges = list(range(s, e, BLOCK_S * fps)) + [e]
        out += [np.arange(a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a > fps * 60]
    return out


def paired_ci(gts, preds_a, preds_b, fps, rng):
    """90% interval of mean spellF1(a) - spellF1(b), resampling 5-min blocks within each game.

    gts: {game: gt}; preds_*: {game: [pred per seed]}. The same resampled blocks are used for
    both approaches and every seed, so the interval is for the paired difference."""
    game_blocks = {g: blocks(gt, fps) for g, gt in gts.items()}
    diffs = []
    for _ in range(BOOT_N):
        d = []
        for g, gt in gts.items():
            bl = game_blocks[g]
            pick = rng.integers(0, len(bl), len(bl))
            idx = np.concatenate([bl[i] for i in pick])
            sub_gt = gt[idx].copy()
            starts = np.cumsum([0] + [len(bl[i]) for i in pick[:-1]])
            sub_gt[starts] = pm.X  # each block is its own run: no spell spans two blocks
            for pa, pb in zip(preds_a[g], preds_b[g]):
                d.append(spell_f1(sub_gt, pa[idx]) - spell_f1(sub_gt, pb[idx]))
        diffs.append(np.mean(d))
    return np.percentile(diffs, [5, 95])


def load_games(which, holdout=False):
    """{game name: (clean Match, gt)} for --games metrica | dfl | all."""
    out = {}
    if which in ("metrica", "all"):
        out.update({"metrica_{}".format(g): tuning_data.load_game(g) for g in tuning_data.GAMES})
    if which in ("dfl", "all"):
        ids = dfl_data.GAMES if holdout else dfl_data.TUNING
        out.update({"dfl_" + g: dfl_data.load_game(g) for g in ids})
    return out


def run_all(names, games, seeds, scenarios):
    """{scenario: {name: {game: [pred codes per seed]}}}. One degraded copy in memory at a time."""
    modules = {name: load_approach(name) for name in names}
    out = {sc: {name: {g: [] for g in games} for name in names} for sc in scenarios}
    for sc, params in scenarios.items():
        for g, (m, _) in games.items():
            for seed in range(seeds):
                d = tuning_data.degrade(m, seed, params)
                for name in names:
                    out[sc][name][g].append(
                        pm.label_codes(np.asarray(modules[name].predict(copy.deepcopy(d)))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("approaches", nargs="*")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--panel", action="store_true")
    ap.add_argument("--games", choices=("metrica", "dfl", "all"), default="metrica")
    ap.add_argument("--holdout", action="store_true", help="include the DFL holdout games")
    args = ap.parse_args()
    names = args.approaches or all_approaches()
    if args.ref not in names:
        names.append(args.ref)

    games = load_games(args.games, args.holdout)
    gts = {g: gt for g, (_, gt) in games.items()}
    scenarios = {"base": {}}
    if args.panel:
        scenarios.update(tuning_data.PANEL)
    preds = run_all(names, games, args.seeds, scenarios)
    rng = np.random.default_rng(0)

    rows = []
    for name in names:
        p = preds["base"][name]
        per = [full_score(gts[g], x) for g in games for x in p[g]]
        mean = lambda k: float(np.mean([r[k] for r in per]))  # noqa: E731
        by_game = {g: np.mean([spell_f1(gts[g], a) - spell_f1(gts[g], b)
                               for a, b in zip(p[g], preds["base"][args.ref][g])]) for g in games}
        ci = paired_ci(gts, p, preds["base"][args.ref], 25, rng) if name != args.ref else (0, 0)
        panel = {}
        for sc in scenarios:
            if sc == "base":
                continue
            panel[sc] = np.mean([spell_f1(gts[g], a) - spell_f1(gts[g], b)
                                 for g in games for a, b in zip(preds[sc][name][g],
                                                                preds[sc][args.ref][g])])
        spell_game = {g: np.mean([spell_f1(gts[g], x) for x in p[g]]) for g in games}
        rows.append((mean("spell"), name, mean("macro"), mean("n_pred"), mean("n_gt"),
                     mean("home_err"), ci, by_game, panel, spell_game))

    rows.sort(key=lambda r: -r[0])
    ref_spell = next(r[0] for r in rows if r[1] == args.ref)
    print("tuning games {} x {} seeds; diffs vs {}; 90% paired block-bootstrap CI\n".format(
        list(games), args.seeds, args.ref))
    head = "{:<26} {:>8} {:>8} {:>23} {:>15} {:>10} {:>8}".format(
        "approach", "spellF1", "macroF1", "diff [90% CI]", "diff per game", "poss/true", "homeErr")
    head += "  {:>8}".format("spell/g")
    if args.panel:
        head += "  {:>26}".format("worst panel diff")
    print(head + "  public file")
    for spell, name, macro, npred, ngt, herr, ci, by_game, panel, spell_game in rows:
        is_ref = name == args.ref
        diff_s = "ref" if is_ref else "{:+.3f} [{:+.3f},{:+.3f}]".format(spell - ref_spell, *ci)
        games_s = "-" if is_ref else " ".join("{:+.3f}".format(v) for v in by_game.values())
        line = "{:<26} {:>8.4f} {:>8.4f} {:>23} {:>15} {:>10} {:>+8.3f}".format(
            name, spell, macro, diff_s, games_s, "{:.0f}/{:.0f}".format(npred, ngt), herr)
        line += "  " + " ".join("{:.3f}".format(v) for v in spell_game.values())
        if args.panel:
            worst = None if is_ref or not panel else min(panel.items(), key=lambda kv: kv[1])
            line += "  {:>26}".format("-" if worst is None else "{:+.3f} {}".format(worst[1], worst[0]))
        print(line + "  " + public_check(name))


if __name__ == "__main__":
    main()
