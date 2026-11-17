"""One in-memory shape for every match we work on: BAR-ATH and the external tuning games.

Every approach takes a `Match` and returns one label per frame. Because the tuning games are
loaded into exactly the same shape, an approach never knows (or cares) which one it is running on,
and `common/compare.py` can rank approaches on the tuning games before anyone runs them on ours.

Coordinates are normalised 0-1 like the ATD; multiply by `pitch` for metres.
"""

import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

HOME, AWAY, UNKNOWN = 0, 1, -1


@dataclass
class Match:
    name: str
    fps: int
    pitch: tuple                  # (length, width) in metres
    ball: np.ndarray              # (N, 2) float32, NaN where the ball is not seen
    players: np.ndarray           # (N, S, 2) float32, NaN where a slot is empty
    team: np.ndarray              # (S,) int8, HOME or AWAY per slot
    periods: list                 # [(start, stop)] half-open frame ranges that are graded
    restarts: list = field(default_factory=list)   # [{frame, team, kind}] team may be UNKNOWN
    phases: list = field(default_factory=list)     # Smart Tagging phases; empty on tuning games

    @property
    def n_frames(self):
        return len(self.ball)

    def metres(self, xy):
        return xy * np.asarray(self.pitch, dtype=np.float32)


# -- BAR-ATH -----------------------------------------------------------------------------------

def _slot_teams(meta_path):
    """HOME/AWAY per track slot; first <Team> in the EPTS metadata is home (see starter/baseline.py)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home = team_ids[0]
    return np.array([HOME if p.get("teamId") == home else AWAY for p in root.iter("Player")],
                    dtype=np.int8)


def _parse_atd(raw_path, n_frames, n_slots):
    """Raw ATD lines '<frame>:<x,y,s;x,y,s;...>:<x,y,s>' into ball (N,2) and players (N,S,2)."""
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1  # ATD frame_id counts from 1, video frames from 0
            b = parts[2].split(",")
            if b[0][:1] != "N":
                ball[f] = (float(b[0]), float(b[1]))
            for s, group in enumerate(parts[1].split(";")[:n_slots]):
                if group[:1] == "N" or not group:
                    continue
                g = group.split(",")
                players[f, s] = (float(g[0]), float(g[1]))
    return ball, players


def _bar_ath_restarts(game, data_dir):
    """Set-piece restarts and phases from Smart Tagging (automatic, so evidence, not answers)."""
    sys.path.insert(0, os.path.join(ROOT, "starter"))
    import load_smart_tagging  # noqa: E402

    names = {"barcelona": HOME, "atletic club": AWAY, "athletic club": AWAY}
    restarts, phases = [], []
    for inst in load_smart_tagging.load(game, data_dir):
        team = names.get((inst["tags"].get("Team") or "").strip().lower(), UNKNOWN)
        item = {"start": inst["start_frame"], "end": inst["end_frame"],
                "code": inst["code"].strip().upper(), "team": team}
        phases.append(item)
        if item["code"] == "SET PIECES":
            restarts.append({"frame": inst["start_frame"], "team": team,
                             "kind": (inst["tags"].get("Type") or "").strip().lower()})
    return restarts, phases


def load_bar_ath(game=GAME, data_dir=None, cache=True):
    """Load the competition match. Reads data/ locally; nothing here prints any of it."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    team = _slot_teams(meta_path)

    cache_path = raw_path + ".npz"  # lives next to the data, inside gitignored data/
    if cache and os.path.exists(cache_path) and \
            os.path.getmtime(cache_path) > os.path.getmtime(raw_path):
        z = np.load(cache_path)
        ball, players = z["ball"], z["players"]
    else:
        ball, players = _parse_atd(raw_path, n_frames, len(team))
        if cache:
            np.savez(cache_path, ball=ball, players=players)

    periods = [(start, stop + 1) for _, start, stop in half_video_blocks(game)]
    restarts, phases = _bar_ath_restarts(game, data_dir)
    return Match(name=game, fps=GAMES[game]["fps"], pitch=tuple(GAMES[game]["pitch"]),
                 ball=ball, players=players, team=team, periods=periods,
                 restarts=restarts, phases=phases)
