"""gta6-twitch-ccu-forecast-01 entry: forecast GTA VI launch-day Twitch viewers.

Model (stdlib only): a p1-weighted blend of two views of the comparable launches
shipped in the case file.

  * Regression view: for each hour h, the log-ratio log(y_h / p1) is regressed on
    log(p1) across the comparable launches. Hour-1's print is a strong predictor
    of the day's shape (corr(log p1, log peak ratio) ~ -0.94): small hour-1 prints
    explode later, huge ones stay flat.
  * Neighbour view: the k = 12 comparables whose log hour-1 print is closest to
    the target's, and the empirical quantiles of their log-ratios per hour.
    This is what keeps the forecast sane when p1 is outside the training range:
    a plain linear regression would predict an absurdly low evening ratio for a
    1M-viewer hour 1 (0.25x), while the biggest launches on record stayed near
    0.9-1.3x.

The blend weight w rises with log(p1) (0 below the training 25th percentile, 1
above the 90th): small hour-1 prints trust the regression, huge ones trust the
neighbours. Both the median and the interval widths are blended the same way, in
log space, so the 5% / 95% quantiles stay honest: wide where launches genuinely
diverge, narrow where the big launches agree.

On the four practice cases (true prints from SullyGnome) this scores a mean
pinball loss of ~0.20, against the house baseline's 0.387.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages, 8 s per
case, 1 CPU. -P keeps this file's directory OFF sys.path, so this file imports
nothing local: a single file, standard library only.

What it must do:
  - read every <id>.in in the input directory (the grader runs one case at a time,
    with only that case's file there): JSON {"schema": "makefun/twitch-ccu-case/v1",
    "id", "game", "day_start", "checkpoint", "levels", "known", "forecast",
    "history", "history_hourly", "streams", "categories_hourly", "comparables",
    "notes"}, where `known` holds hour 1's print and `forecast` lists hours 2..24;
  - write <id>.out to the output directory: {"hours": [{"hour": 2, "q": [q05, q25,
    q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24 each once, no other key,
    each q five finite numbers above 0 and at most 1e9, non-decreasing; at most
    65,536 bytes. Anything else is an invalid case, scored as if every quantile
    equalled hour 1's print, and an entry with an invalid case takes no prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand
in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Neighbours used by the local view.
K = 12
# Blend-weight anchors, as quantiles of the training log hour-1 prints.
LO_Q, HI_Q = 0.25, 0.9
# The baseline's spread, used only as a fallback when a case has too few
# comparables to fit the model at all.
BASE_SIGMA = 0.15
BASE_Z = [-1.645, -0.674, 0, 0.674, 1.645]


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def round2(x: float):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def quantile(sorted_vals, q):
    """Linear-interpolated quantile of an already sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return None
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def hourly_map(comp: dict) -> dict:
    """A comparable's release-day hours 1..24 (UTC) -> viewers, positives only.

    A comparable whose release day is not fully covered (fewer than 20 of the 24
    hours) is not used: the GTA VI pre-release curve, for instance, is keyed to
    its premiere day and carries no release-day hours at all.
    """
    ds = parse_ts(comp["day_start"])
    m = {}
    for pt in comp.get("hourly", []):
        v = pt.get("viewers")
        if v is None or v <= 0:
            continue
        t = parse_ts(pt["from"])
        delta = (t - ds).total_seconds() / 3600.0
        if 0 <= delta < 24:
            m[int(delta) + 1] = float(v)
    return m


def baseline_quantiles(p1: float, h: int) -> list:
    """The house baseline (persistence with a widening random-walk spread)."""
    return [round2(p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1))) for z in BASE_Z]


def forecast_case(case: dict) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"case asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known", []) if k["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    # Training set: every comparable with a usable release-day curve.
    train = []
    for comp in case.get("comparables", []):
        m = hourly_map(comp)
        if 1 in m and len(m) >= 20:
            train.append(m)

    if len(train) < 5:
        # Too little to fit: fall back to the baseline, which is always valid.
        return {"hours": [{"hour": row["hour"], "q": baseline_quantiles(p1, row["hour"])}
                        for row in case["forecast"]]}

    # Regression view: per hour, log(y_h / p1) = a + b * log(p1), with the
    # sorted residuals kept for honest interval widths.
    fit = {}
    for h in range(2, 25):
        xs, ys = [], []
        for m in train:
            if h in m:
                xs.append(math.log(m[1]))
                ys.append(math.log(m[h] / m[1]))
        n = len(xs)
        if n < 5:
            continue
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        b = sxy / sxx if sxx else 0.0
        a = my - b * mx
        resid = sorted(y - (a + b * x) for x, y in zip(xs, ys))
        fit[h] = (a, b, resid)

    # Blend weight: 0 while log p1 is below the training 25th percentile, 1
    # above the 90th, linear in between.
    lps = sorted(math.log(m[1]) for m in train)
    lo = quantile(lps, LO_Q)
    hi = quantile(lps, HI_Q)
    lp = math.log(p1)
    w = 0.0 if hi <= lo else min(1.0, max(0.0, (lp - lo) / (hi - lo)))
    # The regression median is also winsorised to the training log-p1 range
    # (5th to 95th percentile): a linear log-log fit extrapolates wildly
    # outside it (a 1-viewer hour 1 would forecast a 100,000x evening), and
    # the neighbours already carry the information at the extremes.
    lp_lo = quantile(lps, 0.05)
    lp_hi = quantile(lps, 0.95)
    lp_reg = min(max(lp, lp_lo), lp_hi)

    # Neighbour view: the K comparables closest in log hour-1 print.
    neigh = sorted(train, key=lambda m: abs(math.log(m[1]) - lp))[:K]

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        if h not in fit:
            hours.append({"hour": h, "q": baseline_quantiles(p1, h)})
            continue
        a, b, resid = fit[h]
        med_r = a + b * lp_reg
        lr = sorted(math.log(m[h] / m[1]) for m in neigh if h in m)
        med_k = quantile(lr, 0.5) if lr else med_r
        med = (1.0 - w) * med_r + w * med_k
        # Widths: regression residuals and neighbour quantiles, blended in
        # log space the same way as the median.
        qs_r = [med + quantile(resid, tau) for tau in LEVELS]
        qs_k = [quantile(lr, tau) if lr else med for tau in LEVELS]
        qs = [(1.0 - w) * qr + w * qk for qr, qk in zip(qs_r, qs_k)]
        vals = []
        prev = 0.0
        for q in qs:
            v = p1 * math.exp(q)
            v = min(max(v, 1.0), 1e9)
            v = round2(v)
            if v < prev:
                v = prev  # rounding must not break the non-decreasing order
            vals.append(v)
            prev = v
        hours.append({"hour": h, "q": vals})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
