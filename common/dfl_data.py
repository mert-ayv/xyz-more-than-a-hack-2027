"""External tuning games, part 2: the 7 DFL/IDSSE Bundesliga open games, in the tuning-game shape.

Source: Bassek, Rein, Weber et al., "An integrated dataset of spatiotemporal and event data in
elite soccer", Sci Data 12, 195 (2025), https://doi.org/10.1038/s41597-025-04505-y -- CC-BY 4.0,
mirrored at huggingface.co/datasets/pysport/idsse-data (the URLs kloppy's
`sportec.load_open_tracking_data` uses). 25 fps optical tracking of all players and the ball.

Why these games: unlike Metrica's (event-derived labels), the DFL position feed carries the
tracking's own per-frame BallStatus (alive/dead) and BallPossession (1 = home, 2 = away) on the
ball. That is exactly how the competition describes its ground truth: D where the tracking marks the
ball dead, otherwise the team the tracking names as owning it. So labels here are mechanical:
    gt = D if BallStatus == 0 else (H if BallPossession == 1 else A)
No "controlled possession" filter is applied (see `python common/dfl_data.py` for the counts).

The clean Match follows the Metrica convention `tuning_data.degrade()` expects: the ball is NaN
exactly where it is dead, so the same degradation applies unchanged.

kloppy's own sportec deserializer keeps every frame as a dict (many GB per game); this file streams
the XML with lxml instead and converts each game once to external/dfl/<id>.npz (~20 MB).
"""

import json
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.match import AWAY, HOME, UNKNOWN, Match  # noqa: E402

# Params
DATA_DIR = os.environ.get("MTAH_DFL_DIR", os.path.join(ROOT, "external", "dfl"))
BASE_URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
FPS = 25
COMPETITION = {"J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001", "J03WOH": "DFL-COM-000002",
               "J03WOY": "DFL-COM-000002", "J03WPY": "DFL-COM-000002", "J03WQQ": "DFL-COM-000002",
               "J03WR9": "DFL-COM-000002"}
GAMES = tuple(sorted(COMPETITION))
# Holdout: fixed before any score was looked at -- the last two ids in sorted order. Excluded from
# compare.py unless --holdout; meant for the final pick at code freeze, not for tuning.
HOLDOUT = GAMES[-2:]
TUNING = tuple(g for g in GAMES if g not in HOLDOUT)
RESTARTS = {"ThrowIn": "throw-in", "FreeKick": "free kick", "CornerKick": "corner kick",
            "GoalKick": "goal kick", "KickOff": "kick off", "Penalty": "penalty"}
SECTIONS = ("firstHalf", "secondHalf")


def _url(match_id, kind):
    stem = {"meta": "DFL_02_01_matchinformation", "event": "DFL_03_02_events_raw",
            "tracking": "DFL_04_03_positions_raw_observed"}[kind]
    return BASE_URL + "{}_{}_DFL-MAT-{}.xml".format(stem, COMPETITION[match_id], match_id)


def _fetch(match_id, kind, data_dir):
    path = os.path.join(data_dir, "raw", "{}_{}.xml".format(match_id, kind))
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("downloading", match_id, kind)
        urllib.request.urlretrieve(_url(match_id, kind), path + ".part")
        os.replace(path + ".part", path)
    return path


def _time(s):
    return datetime.fromisoformat(s).timestamp()


def _parse_positions(path):
    """Stream the position XML: {section: {obj: (N, X, Y[, status, possession])}}, team per obj."""
    from lxml import etree  # kloppy's dependency, already installed

    out = {s: {} for s in SECTIONS}
    obj_team, t0, pitch = {}, {}, (105.0, 68.0)
    cur = None
    for event, el in etree.iterparse(path, events=("start", "end"), huge_tree=True):
        if event == "start":
            if el.tag == "FrameSet":
                sec, obj = el.get("GameSection"), el.get("PersonId")
                is_ball = el.get("TeamId") == "BALL"
                obj = "ball" if is_ball else obj
                obj_team.setdefault(obj, el.get("TeamId"))
                cur = (sec, obj, is_ball, [])
            continue
        if el.tag == "Frame" and cur is not None:
            a = el.attrib
            row = (int(a["N"]), float(a["X"]), float(a["Y"]))
            if cur[2]:
                row += (int(a["BallStatus"]), int(a["BallPossession"]))
                if cur[0] not in t0:
                    t0[cur[0]] = (row[0], _time(a["T"]))
            cur[3].append(row)
        elif el.tag == "FrameSet" and cur is not None:
            if cur[0] in out:
                out[cur[0]][cur[1]] = np.array(cur[3], dtype=np.float64)
            cur = None
        elif el.tag == "PitchSize":
            pitch = (float(el.get("X")), float(el.get("Y")))
        el.clear()
        while el.getprevious() is not None:
            del el.getparent()[0]
    return out, obj_team, t0, pitch


def _parse_events(path, home_id):
    """[(unix time, team, kind)] of restarts."""
    out = []
    for ev in ET.parse(path).getroot().iter("Event"):
        for child in ev:
            if child.tag not in RESTARTS:
                continue
            team_id = child.get("Team") or next((d.get("Team") for d in child.iter()
                                                 if d.get("Team")), None)
            team = UNKNOWN if team_id is None else HOME if team_id == home_id else AWAY
            out.append((_time(ev.get("EventTime")), team, RESTARTS[child.tag]))
    return out


def convert(match_id, data_dir=DATA_DIR):
    """Download one game and write <id>.npz. One game in memory at a time; raw position XML deleted."""
    npz = os.path.join(data_dir, match_id + ".npz")
    if os.path.exists(npz):
        return npz
    meta = ET.parse(_fetch(match_id, "meta", data_dir)).getroot().find(".//General")
    home_id = meta.get("HomeTeamId")
    pos_path = _fetch(match_id, "tracking", data_dir)
    print("converting", match_id)
    pos, obj_team, t0, pitch = _parse_positions(pos_path)

    # Contiguous frame axis: half 1, one ungraded frame, half 2 (the scorer's valid_runs then
    # never carries a label across half-time, as in tuning_data).
    offsets, periods, start = {}, [], 0
    for sec in SECTIONS:
        n_sec = pos[sec]["ball"][:, 0]
        first, last = int(n_sec.min()), int(n_sec.max())
        offsets[sec] = start - first
        periods.append((start, start + last - first + 1))
        start = periods[-1][1] + 1
    n = periods[-1][1]
    # Players of the two clubs only: the Bundesliga games also track the referees.
    people = sorted(o for o in obj_team if obj_team[o] in (home_id, meta.get("GuestTeamId")))
    team = np.array([HOME if obj_team[o] == home_id else AWAY for o in people], dtype=np.int8)
    L, W = pitch
    ball = np.full((n, 2), np.nan, np.float32)
    players = np.full((n, len(people), 2), np.nan, np.float32)
    status = np.zeros(n, np.int8)       # 0 dead / outside the halves
    possession = np.zeros(n, np.int8)   # 1 home, 2 away (DFL convention)
    for sec in SECTIONS:
        for obj, arr in pos[sec].items():
            f = arr[:, 0].astype(np.int64) + offsets[sec]
            xy = np.c_[(arr[:, 1] + L / 2) / L, (arr[:, 2] + W / 2) / W].astype(np.float32)
            if obj == "ball":
                ball[f] = xy
                status[f], possession[f] = arr[:, 3], arr[:, 4]
            elif obj in people:
                players[f, people.index(obj)] = xy

    restarts = []
    for t, tm, kind in _parse_events(_fetch(match_id, "event", data_dir), home_id):
        for sec, (s, e) in zip(SECTIONS, periods):
            n0, tt = t0[sec]
            f = int(round((t - tt) * FPS)) + n0 + offsets[sec]
            if s - 5 * FPS <= f < e:
                restarts.append({"frame": int(np.clip(f, s, e - 1)), "team": tm, "kind": kind})
                break
    np.savez_compressed(npz, ball=ball, players=players, team=team, status=status,
                        possession=possession, periods=np.array(periods), pitch=np.array(pitch),
                        restarts=json.dumps(restarts))
    os.remove(pos_path)
    return npz


def load_game(match_id, data_dir=DATA_DIR):
    """(clean Match, gt_codes) for one DFL game. gt_codes use the scorer's 0..3 = H,A,D,X."""
    z = np.load(convert(match_id, data_dir))
    periods = [tuple(int(v) for v in p) for p in z["periods"]]
    status, possession = z["status"], z["possession"]
    alive = status == 1
    gt = np.full(len(status), 3, np.int8)  # X outside the halves
    for s, e in periods:
        gt[s:e] = np.where(alive[s:e], np.where(possession[s:e] == 1, 0, 1), 2)
    gt[(gt < 2) & ~np.isin(possession, (1, 2))] = 2  # alive with no owner (never seen, checked)
    ball = z["ball"].copy()
    ball[~alive] = np.nan  # tuning_data convention: clean ball is NaN exactly when dead
    match = Match(name="dfl_" + match_id, fps=FPS, pitch=tuple(float(v) for v in z["pitch"]),
                  ball=ball, players=z["players"], team=z["team"], periods=periods,
                  restarts=json.loads(str(z["restarts"])))
    return match, gt


def _spells(gt):
    """Possession spells (maximal H or A runs) inside graded frames: [(start, stop, code)]."""
    d = np.flatnonzero(np.diff(np.r_[-1, gt, -1]) != 0)
    return [(a, b, gt[a]) for a, b in zip(d[:-1], d[1:]) if gt[a] < 2]


if __name__ == "__main__":
    for g in GAMES:
        m, gt = load_game(g)
        graded = gt < 3
        sp = _spells(gt)
        dur = np.array([b - a for a, b, _ in sp]) / FPS
        print("{}{}: graded {:.1f} min, H/A/D {:.1%}/{:.1%}/{:.1%}, possessions {} ({:.1f}/min), "
              "median {:.1f} s, <2 s {:.0%}, restarts {}".format(
                  g, " (holdout)" if g in HOLDOUT else "", graded.sum() / FPS / 60,
                  (gt == 0).sum() / graded.sum(), (gt == 1).sum() / graded.sum(),
                  (gt == 2).sum() / graded.sum(), len(sp), len(sp) / (graded.sum() / FPS / 60),
                  np.median(dur), (dur < 2).mean(), len(m.restarts)))
        on_pitch = (~np.isnan(m.players[graded, :, 0])).sum(1).mean()
        assert 20 <= on_pitch <= 22.01, (g, on_pitch)  # 22 players, fewer after a red card
