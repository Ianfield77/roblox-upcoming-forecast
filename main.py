#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming top 10.

make.fun competition roblox-up-and-coming-forecast-01.

Model: Monte Carlo rank diffusion.
  - Every game on (or just off) the chart gets a rank and a drift.
  - Drift is the game's recent rank velocity (ranks/hour), estimated from
    consecutive reads, time-decayed and shrunk toward zero.
  - Each simulated hour a game's rank moves by drift + Gaussian noise
    (noise scale grows with rank band) minus a weak mean-reversion pull
    toward the middle of the chart. Ranks are clamped to [1, 130].
  - Games leave the chart 28 days after creation when the history's
    details give a creation date; their probability mass flows to the
    games that remain.
  - The simulated top-10 occupancy of every hour inside the scored
    window is averaged into one probability per (game, place).
  - A game is named at a place only when its simulated probability beats
    the unnamed share it would otherwise get; the rest stays unnamed,
    which covers tail games and brand-new entrants.

Stdlib only, no network, fully deterministic (fixed seed, own Box-Muller
normal from random.random(), deterministic sort keys and JSON output).
"""
import json
import math
import os
import random
import sys
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------- tunables
# Tuned by backtest on the true practice window (34 archive reads -> the 10
# reads of 8-10 Nov 2026) plus two held-out ~72h splits. The chart churns
# fast over 72h, so the winning point spreads probability widely: drift is
# shrunk hard, diffusion is large, mean reversion is strong.
SEED = 20261113
N_PATHS = 2500            # Monte Carlo paths
TAU_HOURS = 36.0          # age decay (hours) for velocity estimation
SHRINK_K = 8.0            # drift shrinkage (effective segments)
DRIFT_CLAMP = 1.0         # max |drift|, ranks/hour
KAPPA = 0.008             # mean reversion per hour toward RANK_EQ
RANK_EQ = 50.0
OFF_CHART_RANK = 100.0    # starting rank for games seen recently but off chart
OFF_CHART_RECENT_HOURS = 48.0
MIN_P_NAME = 0.0002       # never name a game below this probability
MAX_ROWS_PER_PLACE = 100
MAX_RANK = 130.0
CHART_DAYS = 28           # games leave the chart this long after creation

# Per-hour diffusion (rank sd) by rank band, from the practice archive's
# segment-velocity sd * sqrt(mean segment gap), scaled x1.6 by backtest.
SIGMA_BANDS = ((10, 1.44), (30, 3.04), (60, 4.8), (10 ** 9, 5.44))


def sigma_for(rank):
    for top, sig in SIGMA_BANDS:
        if rank <= top:
            return sig
    return SIGMA_BANDS[-1][1]


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def make_gauss(rng):
    """N(0, 1) sampler built only on rng.random() (stable across versions)."""
    cache = []

    def gauss():
        if cache:
            return cache.pop()
        u1 = rng.random()
        u2 = rng.random()
        if u1 <= 0.0:
            u1 = 1e-12
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        cache.append(r * math.sin(th))
        return r * math.cos(th)

    return gauss


def load_history(input_dir):
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    reads = []
    seen_ts = set()
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not r.get("ok", True):
                continue
            chart = r.get("up_and_coming")
            if not chart or not r.get("at"):
                continue
            try:
                ts = parse_ts(r["at"])
            except ValueError:
                continue
            key = ts.timestamp()
            if key in seen_ts:  # recorder retries can duplicate a timestamp
                continue
            seen_ts.add(key)
            reads.append(r)
    reads.sort(key=lambda r: parse_ts(r["at"]))
    return task, reads


def estimate_drift(obs, t_last, shrink_k=SHRINK_K):
    """Shrunk, time-decayed rank velocity (ranks/hour) from (ts, rank) pairs."""
    num = 0.0
    den = 0.0
    w2 = 0.0
    for (t0, r0), (t1, r1) in zip(obs, obs[1:]):
        dt = (t1 - t0).total_seconds() / 3600.0
        if dt <= 0:
            continue
        age = (t_last - t1).total_seconds() / 3600.0
        w = math.sqrt(dt) * math.exp(-age / TAU_HOURS)
        num += w * (r1 - r0) / dt
        den += w
        w2 += w * w
    if den <= 0:
        return 0.0
    raw = num / den
    n_eff = den * den / w2 if w2 > 0 else 0.0
    shrunk = raw * n_eff / (n_eff + shrink_k)
    return max(-DRIFT_CLAMP, min(DRIFT_CLAMP, shrunk))


def build_forecast(reads, task, n_paths=N_PATHS, seed=SEED,
                   kappa=KAPPA, sigma_scale=1.0, shrink_k=SHRINK_K):
    """Simulate the window and return the forecast dict."""
    places = int(task.get("places", 10))
    window = task.get("window") or {}
    last = reads[-1]
    t_last = parse_ts(last["at"])
    if window.get("from"):
        t_from = parse_ts(window["from"])
    else:
        t_from = t_last
    if window.get("until"):
        t_until = parse_ts(window["until"])
    else:
        t_until = t_from + timedelta(hours=72)
    if t_until <= t_from:
        t_until = t_from + timedelta(hours=72)

    # ---- per-game observations and creation dates
    obs = {}       # uid -> [(ts, rank), ...]
    created = {}   # uid -> datetime
    for r in reads:
        ts = parse_ts(r["at"])
        for g in r.get("up_and_coming") or []:
            uid = g["universe_id"]
            obs.setdefault(uid, []).append((ts, g["rank"]))
        for uid_s, d in (r.get("details") or {}).items():
            if d and d.get("created"):
                try:
                    created[int(uid_s)] = parse_ts(d["created"])
                except ValueError:
                    pass

    last_chart = sorted(last["up_and_coming"], key=lambda g: g["rank"])
    last_rank = {g["universe_id"]: g["rank"] for g in last_chart}
    n_chart = len(last_chart)

    # ---- state: games on the last chart + games seen in the last 48h
    uids, rank0, drift, sigma, exit_step = [], [], [], [], []
    t0 = min(t_last, t_from)
    horizon_h = int(math.ceil((t_until - t0).total_seconds() / 3600.0))
    for uid, pts in obs.items():
        last_seen = pts[-1][0]
        if uid in last_rank:
            r0 = float(last_rank[uid])
        else:
            if (t_last - last_seen).total_seconds() / 3600.0 > OFF_CHART_RECENT_HOURS:
                continue
            r0 = OFF_CHART_RANK
        d = estimate_drift(pts, t_last, shrink_k) if len(pts) >= 2 else 0.0
        uids.append(uid)
        rank0.append(r0)
        drift.append(d)
        sigma.append(sigma_for(r0) * sigma_scale)
        c = created.get(uid)
        if c is not None:
            ex = int(math.floor((c + timedelta(days=CHART_DAYS) - t0).total_seconds() / 3600.0))
            exit_step.append(ex if ex > 0 else 0)
        else:
            exit_step.append(None)

    n_games = len(uids)
    if n_games == 0:
        raise ValueError("no games in history")

    # ---- Monte Carlo
    counts = [dict() for _ in range(places)]  # counts[place-1][uid] = hits
    n_hours = 0
    for path in range(n_paths):
        rng = random.Random(seed + path)
        gauss = make_gauss(rng)
        ranks = list(rank0)
        for step in range(1, horizon_h + 1):
            t = t0 + timedelta(hours=step)
            for i in range(n_games):
                ex = exit_step[i]
                if ex is not None and step >= ex:
                    ranks[i] = 1e9
                    continue
                r = ranks[i]
                r += drift[i] + gauss() * sigma[i] - kappa * (r - RANK_EQ)
                if r < 1.0:
                    r = 1.0
                elif r > MAX_RANK:
                    r = MAX_RANK
                ranks[i] = r
            if t < t_from:
                continue
            n_hours += 1 if path == 0 else 0
            order = sorted(range(n_games), key=lambda i: (ranks[i], uids[i]))
            for k in range(min(places, n_games)):
                uid = uids[order[k]]
                d = counts[k]
                d[uid] = d.get(uid, 0) + 1
    total = n_paths * max(n_hours, 1)
    probs = []  # (place, uid, p)
    for k in range(places):
        for uid, c in counts[k].items():
            probs.append((k + 1, uid, c / total))

    # ---- build the forecast rows
    probabilities = []
    for k in range(1, places + 1):
        cand = sorted(((p, uid) for (pl, uid, p) in probs if pl == k),
                      key=lambda x: (-x[0], x[1]))
        rows = []
        s = 0.0
        for p, uid in cand:
            if len(rows) >= MAX_ROWS_PER_PLACE:
                break
            if p < MIN_P_NAME:
                break
            # name it only while it beats the unnamed share it would get
            unnamed = (1.0 - s) / max(n_chart - len(rows), 1)
            if p <= unnamed:
                break
            rows.append([uid, p])
            s += p
        # round to 6 decimals; trim the smallest row if rounding pushed over 1
        rows = [[uid, round(p, 6)] for uid, p in rows]
        while rows and sum(p for _, p in rows) > 1.0:
            rows.pop()
        for uid, p in rows:
            probabilities.append({"universe_id": uid, "place": k, "p": p})

    # ---- top10: greedy argmax per place, no repeats
    prob_by_place = {}
    for pl, uid, p in probs:
        prob_by_place.setdefault(pl, []).append((p, uid))
    used = set()
    top10 = []
    for k in range(1, places + 1):
        best = None
        for p, uid in prob_by_place.get(k, []):
            if uid in used:
                continue
            if best is None or p > best[0]:
                best = (p, uid)
        if best is not None:
            used.add(best[1])
            top10.append(best[1])
    for g in last_chart:  # fill any gap from the last chart's order
        if len(top10) >= places:
            break
        if g["universe_id"] not in used:
            used.add(g["universe_id"])
            top10.append(g["universe_id"])
    return {"probabilities": probabilities, "top10": top10[:places]}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, reads = load_history(input_dir)
    if len(reads) < 2:
        # Fallback: persistence forecast from the single read.
        last = reads[-1] if reads else {"up_and_coming": []}
        chart = sorted(last["up_and_coming"], key=lambda g: g["rank"])
        chart = [g["universe_id"] for g in chart]
        probs = []
        for k in range(1, int(task.get("places", 10)) + 1):
            if k - 1 < len(chart):
                probs.append({"universe_id": chart[k - 1], "place": k, "p": 0.5})
        forecast = {"probabilities": probs, "top10": chart[:10]}
    else:
        forecast = build_forecast(reads, task)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
