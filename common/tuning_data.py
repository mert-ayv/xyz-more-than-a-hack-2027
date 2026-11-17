"""External tuning games: Metrica's public sample data, degraded to look like our tracking.

Why this exists: the rules forbid tuning on BAR-ATH's ground truth (the public 5 minutes included),
so approaches are ranked here instead. Source: https://github.com/metrica-sports/sample-data
(anonymised, public; its README asks users to be responsible and acknowledge the source).

What the sample games look like (measured): full player identities, all 22 players always present,
and the ball is never missing during play -- it is NaN only while the ball is dead. Ours has 46
fragmenting track slots, and its ball is missing on 41% of graded frames in long blocks that
mostly look like stoppages, plus ~5 s gaps in play. `degrade()` reproduces that; the DEGRADE rates
come from `common/atd_stats.py`, aggregates of BAR-ATH's *tracking* only, never its ground truth.
Run `python common/atd_stats.py` to see both side by side.

Ground truth here is built the way the competition's is described: alive = the clean tracking has
the ball, team = the team of the last on-ball event (pass, carry, recovery, shot, set piece, ball
lost). It is event-derived, so its notion of possession is close to, not identical to, ours.
"""

import os
import sys
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.match import AWAY, HOME, UNKNOWN, Match  # noqa: E402

# Params
DATA_DIR = os.environ.get("MTAH_TUNING_DIR", os.path.join(ROOT, "external", "metrica_sample"))
BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
GAMES = (1, 2)  # game 3 changes its slot layout mid-match (substitutions); not loaded yet
PITCH = (105.0, 68.0)
FPS = 25
GT_FILL_GAP = 25  # ball NaN runs shorter than this inside live play are tracking blips, not stoppages

POSSESSION_EVENTS = {"PASS", "CARRY", "RECOVERY", "SHOT", "SET PIECE", "BALL LOST"}

DEGRADE = {
    # Measured on BAR-ATH's tracking with common/atd_stats.py (aggregates only, never ground truth):
    # ball missing 40.7% of graded frames in 150 blocks, 79 of them >10 s with players nearly
    # still (1.0 m/s, the same as dead balls in the clean sample games), 71 shorter ones of
    # 5-9 s. So the tracker mostly loses the ball when it is dead, plus ~5 s gaps in play.
    "ball_dead_hidden": 0.95,     # share of dead stretches where the ball is not seen
    "ball_dead_early_s": 1.0,     # the gap may start up to this long before the ball is dead
    "ball_dead_tail_s": 3.0,      # ... and the ball may reappear up to this long before the restart
    "live_gaps_per_min": 0.7,     # short gaps during play (most of the 71 short gaps in ~106 min)
    "live_gap_frames": (120, 230),  # their length: p25-p90 of BAR-ATH's short gaps
    "ball_crowd_weight": 1.0,     # live gaps start more often with each player within 3 m ...
    "ball_fast_weight": 2.0,      # ... and when the ball moves faster than 15 m/s (long passes)
    "ball_noise_m": 1.0,          # BAR-ATH ball-to-nearest-player p25/p50 = 0.9/1.9 m (atd_stats)
    "min_gap_frames": 120,        # BAR-ATH has no ball gap under ~5 s: the tracker bridges them
    "player_missing": 0.068,      # 20.5 of 22 players tracked per frame
    "player_gap_mean": 200,       # with 11.5 slot fragments/min overall (~0.5 per player-minute)
    "player_noise_m": 0.1,
    "n_slots": (24, 22),          # BAR-ATH: 24 home slots, 22 away
    "restart_drop": 0.4,          # Smart Tagging has 59 restarts; a match has ~100
    "restart_unknown": 0.4,       # 32 of 59 tagged restarts have no team: all throw-ins + some
    "restart_offset_s": -9.5,     # BAR-ATH tags sit ~5 s before the ball reappears (atd_stats)
    "restart_false_per_min": 0.0, # tags with no real restart behind them (panel only)
    "restart_jitter_s": 3.5,      # ... with a wide spread (p25-p75 of that offset spans ~7 s)
}


# Robustness panel: ways our match could differ from the calibrated tuning games. An approach's
# gain over the reference should keep its sign under every one (common/compare.py --panel).
PANEL = {
    "gaps_misaligned": {"ball_dead_early_s": 4.0, "ball_dead_tail_s": 6.0},
    "dead_ball_visible": {"ball_dead_hidden": 0.75},
    "live_long_gaps": {"live_gaps_per_min": 1.0, "live_gap_frames": (120, 400)},
    "restarts_late": {"restart_offset_s": -2.0},
    "false_restarts": {"restart_false_per_min": 0.2},
    "more_noise": {"ball_noise_m": 1.4, "player_noise_m": 0.3},
}


# -- download ----------------------------------------------------------------------------------

def _files(game):
    stem = "Sample_Game_{0}/Sample_Game_{0}_".format(game)
    return [stem + "RawEventsData.csv", stem + "RawTrackingData_Home_Team.csv",
            stem + "RawTrackingData_Away_Team.csv"]


def download(games=GAMES, data_dir=DATA_DIR):
    for game in games:
        for rel in _files(game):
            path = os.path.join(data_dir, rel)
            if os.path.exists(path):
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print("downloading", rel)
            urllib.request.urlretrieve(BASE_URL + "/" + rel, path + ".part")
            os.replace(path + ".part", path)


# -- loading -----------------------------------------------------------------------------------

def _read_tracking(path):
    """(period, ball (N,2), players (N,P,2)) from one team's Metrica CSV."""
    t = pd.read_csv(path, skiprows=2)
    cols = list(t.columns)
    xs = [i for i, c in enumerate(cols) if c.startswith("Player")]
    players = np.stack([t.iloc[:, [i, i + 1]].to_numpy(np.float32) for i in xs], axis=1)
    b = cols.index("Ball")
    ball = t.iloc[:, [b, b + 1]].to_numpy(np.float32)
    return t["Period"].to_numpy(), ball, players


def _runs(mask):
    """[(start, stop)] of True runs."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def load_game(game, data_dir=DATA_DIR):
    """(clean Match, gt_codes) for one sample game. gt_codes use the scorer's 0..3 = H,A,D,X."""
    download((game,), data_dir)
    ev_path, home_path, away_path = [os.path.join(data_dir, f) for f in _files(game)]
    period, ball, home = _read_tracking(home_path)
    _, _, away = _read_tracking(away_path)
    players = np.concatenate([home, away], axis=1)
    team = np.r_[np.full(home.shape[1], HOME), np.full(away.shape[1], AWAY)].astype(np.int8)
    n = len(ball)

    periods = [(int(np.flatnonzero(period == p)[0]), int(np.flatnonzero(period == p)[-1]) + 1)
               for p in np.unique(period) if p > 0]
    # The halves are back to back in the file. Leave one ungraded frame between them so the scorer
    # (valid_runs) sees two runs and never carries a label across half-time.
    periods = [periods[0]] + [(s + 1, e) for s, e in periods[1:]]

    events = pd.read_csv(ev_path)
    events["frame"] = events["Start Frame"].astype(int) - 1
    events["team"] = np.where(events["Team"] == "Home", HOME, AWAY)

    # Owner: team of the most recent possession event (forward-filled); D before the first one.
    owner = np.full(n, -1, dtype=np.int8)
    poss = events[events["Type"].isin(POSSESSION_EVENTS)].sort_values("frame", kind="stable")
    owner[poss["frame"].to_numpy()] = poss["team"].to_numpy()
    idx = np.maximum.accumulate(np.where(owner >= 0, np.arange(n), 0))
    owner = owner[idx]

    alive = ~np.isnan(ball[:, 0])
    for s, e in _runs(~alive):
        if e - s < GT_FILL_GAP and s > 0 and e < n:
            alive[s:e] = True

    gt = np.full(n, 3, dtype=np.int8)  # X
    for s, e in periods:
        seg = slice(s, e)
        gt[seg] = np.where(alive[seg], owner[seg], 2)
    gt[(gt < 2) & (owner < 0)] = 2

    restarts = []
    sp = events[(events["Type"] == "SET PIECE") |
                ((events["Type"] == "PASS") & events["Subtype"].fillna("").str.contains("GOAL KICK"))]
    for _, r in sp.iterrows():
        kind = "goal kick" if r["Type"] == "PASS" else str(r["Subtype"]).split("-")[0].strip().lower()
        restarts.append({"frame": int(r["frame"]), "team": int(r["team"]),
                         "kind": kind.replace("throw in", "throw-in")})

    match = Match(name="metrica_sample_{}".format(game), fps=FPS, pitch=PITCH, ball=ball,
                  players=players, team=team, periods=periods, restarts=restarts)
    return match, gt


# -- degradation -------------------------------------------------------------------------------

def _gap_mask(n, start_prob, mean_len, rng):
    """Boolean mask of gaps: each frame starts a gap with start_prob[t], length ~ geometric."""
    starts = np.flatnonzero(rng.random(n) < start_prob)
    lengths = rng.geometric(1.0 / mean_len, size=len(starts))
    mask = np.zeros(n, bool)
    for s, l in zip(starts, lengths):
        mask[s:s + l] = True
    return mask


def _smooth_noise(shape, sigma, width, rng):
    """Temporally smooth noise (tracking error drifts; it does not jump every frame)."""
    raw = rng.normal(0, 1, shape).astype(np.float32)
    c = np.cumsum(np.concatenate([np.zeros((1,) + shape[1:], np.float32), raw]), axis=0)
    sm = (c[width:] - c[:-width]) / np.sqrt(width)
    pad = np.repeat(sm[-1:], shape[0] - len(sm), axis=0)
    return np.concatenate([sm, pad]) * sigma


def degrade(match, seed=0, params=None):
    """A copy of a clean Match that looks like our automatic tracking. Deterministic per seed."""
    p = dict(DEGRADE, **(params or {}))
    rng = np.random.default_rng(seed)
    n = match.n_frames
    scale = np.asarray(match.pitch, dtype=np.float32)

    # Ball. Clean data has the ball NaN exactly when it is dead; ours loses it on most, not all,
    # dead stretches, plus some ~5 s gaps in play. Fill dead stretches first (the ball being
    # fetched for the restart) so the dead stretches we keep visible have a position.
    ball = match.ball.copy()
    dead = np.isnan(ball[:, 0])
    seen = np.flatnonzero(~dead)
    for k in range(2):
        ball[:, k] = np.interp(np.arange(n), seen, ball[seen, k])
    fps = match.fps
    hide = np.zeros(n, bool)
    for s, e in _runs(dead):
        if rng.random() < p["ball_dead_hidden"]:
            early = int(rng.uniform(0, p["ball_dead_early_s"]) * fps)
            tail = int(rng.uniform(0, p["ball_dead_tail_s"]) * fps)
            hide[max(0, s - early):max(s, e - tail)] = True

    # Live-play gaps, more likely in crowds and on fast balls -- where possession actually changes.
    bm = ball * scale
    near = np.nansum(np.linalg.norm(match.players * scale - bm[:, None, :], axis=2) < 3.0, axis=1)
    speed = np.r_[0, np.linalg.norm(np.diff(bm, axis=0), axis=1) * fps]
    weight = 1 + p["ball_crowd_weight"] * near + p["ball_fast_weight"] * (speed > 15)
    weight = np.where(dead, 0, weight)
    start = weight / weight[~dead].mean() * p["live_gaps_per_min"] / (60 * fps)
    lo, hi = p["live_gap_frames"]
    for s in np.flatnonzero(rng.random(n) < start):
        hide[s:s + rng.integers(lo, hi + 1)] = True
    d = np.diff(np.r_[0, hide.astype(np.int8), 0])
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if e - s < p["min_gap_frames"]:
            hide[s:e] = False  # the tracker bridges short gaps (position already interpolated)
    ball[hide] = np.nan
    ball += _smooth_noise(ball.shape, p["ball_noise_m"], 50, rng) / scale

    # Players: drop in runs, then re-assign every surviving fragment to a free slot of its team, so
    # slots are tracklets rather than people -- the way our 46 slots behave.
    n_home, n_away = p["n_slots"]
    out_players = np.full((n, n_home + n_away, 2), np.nan, dtype=np.float32)
    out_team = np.r_[np.full(n_home, HOME), np.full(n_away, AWAY)].astype(np.int8)
    busy = np.full(n_home + n_away, -1)
    fragments = []
    for j in range(match.players.shape[1]):
        present = ~np.isnan(match.players[:, j, 0])
        if not present.any():
            continue
        missing = _gap_mask(n, np.full(n, p["player_missing"] / p["player_gap_mean"]),
                            p["player_gap_mean"], rng)
        fragments += [(s, e, j) for s, e in _runs(present & ~missing)]
    for s, e, j in sorted(fragments):
        pool = np.flatnonzero((out_team == match.team[j]) & (busy < s))
        if not len(pool):
            continue  # more simultaneous fragments than slots; drop it, as a tracker would
        slot = rng.choice(pool)
        busy[slot] = e
        out_players[s:e, slot] = match.players[s:e, j]
    out_players += _smooth_noise(out_players.shape, p["player_noise_m"], 100, rng) / scale

    # Restarts: drop some, jitter the rest, and lose the team on throw-ins like Smart Tagging does.
    restarts = []
    for r in match.restarts:
        if rng.random() < p["restart_drop"]:
            continue
        jitter = int(round((p["restart_offset_s"] + rng.normal(0, p["restart_jitter_s"])) * match.fps))
        restarts.append({"frame": int(np.clip(r["frame"] + jitter, 0, n - 1)),
                         "team": UNKNOWN if r["kind"] == "throw-in" or
                         rng.random() < p["restart_unknown"] else r["team"],
                         "kind": r["kind"]})

    graded_frames = sum(e - s for s, e in match.periods)
    for _ in range(rng.poisson(p["restart_false_per_min"] * graded_frames / (60 * match.fps))):
        s, e = match.periods[rng.integers(len(match.periods))]
        restarts.append({"frame": int(rng.integers(s, e)), "team": int(rng.integers(-1, 2)),
                         "kind": "free kick"})

    return Match(name=match.name, fps=match.fps, pitch=match.pitch, ball=ball,
                 players=out_players, team=out_team, periods=match.periods, restarts=restarts)


if __name__ == "__main__":
    for g in GAMES:
        m, gt = load_game(g)
        d = degrade(m)
        graded = gt < 3
        alive = graded & (gt < 2)
        print("{}: {} frames, graded {}, ball missing on graded {:.1%} (alive {:.1%}), "
              "players/frame {:.1f}, restarts {}".format(
                  m.name, m.n_frames, graded.sum(),
                  np.isnan(d.ball[graded, 0]).mean(), np.isnan(d.ball[alive, 0]).mean(),
                  (~np.isnan(d.players[graded, :, 0])).sum(1).mean(), len(d.restarts)))
