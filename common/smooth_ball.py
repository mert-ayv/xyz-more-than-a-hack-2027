"""Barça-like ball: BAR-ATH's tracker low-passes the ball; clean sources keep frame jitter.

sigma=8 frames matches BAR-ATH's unlabelled ball statistics (median accel 0.04, turn_cos 0.97;
dev standard degrade gives 0.76 / 0.70). Kept apart from tuning_data.py so the frozen pipeline's
source identity, and hence the exact reproduction of earlier files, is unchanged.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from common.tuning_data import _runs, degrade

SIGMA = 8


def switch(match, rate, seed=0):
    """Cut slot fragments at Poisson ID switches (rate per second) and move each piece to a free slot of its team.
    rate 0.05 gives BAR-ATH's fragment p10/p50 2/14 s (dev 15/80 s, results/player_match)."""
    rng = np.random.default_rng(20_000 + seed)
    p = match.players
    pieces = []
    for j in range(p.shape[1]):
        for s, e in _runs(~np.isnan(p[:, j, 0])):
            cuts = np.flatnonzero(rng.random(e - s) < rate / match.fps) + s
            b = np.r_[s, cuts[cuts > s], e]
            pieces += [(a, c, j) for a, c in zip(b[:-1], b[1:]) if c > a]
    out = np.full_like(p, np.nan); busy = np.full(p.shape[1], -1)
    for s, e, j in sorted(pieces):
        pool = np.flatnonzero((match.team == match.team[j]) & (busy < s))
        slot = rng.choice(pool) if len(pool) else j  # ponytail: no free slot → keep own (never seen with 24/22 slots)
        busy[slot] = e; out[s:e, slot] = p[s:e, j]
    match.players = out
    return match


def smooth(match, sigma=SIGMA, noise=0., seed=0, psigma=0., rate=0.):
    """noise: white position jitter (metres) added after smoothing; BAR-ATH's flat high-frequency floor (Queue 4).
    psigma: also low-pass each player slot fragment (BAR-ATH player accel p50 0.73 vs dev 5.6, results/player_match)."""
    if psigma:
        players = match.players.copy()
        for j in range(players.shape[1]):
            for s, e in _runs(~np.isnan(players[:, j, 0])):
                players[s:e, j] = gaussian_filter1d(players[s:e, j], psigma, axis=0, mode="nearest")
        match.players = players
    ball = match.ball.copy()
    for s, e in _runs(~np.isnan(ball[:, 0])):
        ball[s:e] = gaussian_filter1d(ball[s:e], sigma, axis=0, mode="nearest")
    if noise:
        ball += np.random.default_rng(10_000 + seed).normal(0, noise, ball.shape) / np.asarray(match.pitch, ball.dtype)
    match.ball = ball
    return switch(match, rate, seed) if rate else match


def degrade_smooth(clean, seed, sigma=SIGMA, noise=0., psigma=0., rate=0., params=None):
    return smooth(degrade(clean, seed, params), sigma, noise, seed, psigma, rate)


if __name__ == "__main__":  # self-check: ID switches keep every position, its team, and fragment the slots
    from common.match import Match
    rng = np.random.default_rng(0)
    n = 25 * 600
    players = np.full((n, 46, 2), np.nan, np.float32); players[:, [0, 1, 24, 25]] = rng.random((n, 4, 2))
    team = np.r_[np.zeros(24), np.ones(22)].astype(np.int8)
    m = switch(Match("t", 25, (105, 68), np.zeros((n, 2), np.float32), players.copy(), team, [(0, n)]), 0.05)
    for t in (0, 1):
        assert np.allclose(np.sort(np.nan_to_num(m.players[:, team == t], nan=9), 1), np.sort(np.nan_to_num(players[:, team == t], nan=9), 1))
    assert sum(len(_runs(~np.isnan(m.players[:, j, 0]))) for j in range(46)) > 4 * 20
    print("ok")
