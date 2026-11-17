# GTA VI launch-day Twitch forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/sandiawan-kielo-dale/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/sandiawan-kielo-dale/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for make.fun competition `gta6-twitch-ccu-forecast-01`: forecast how many
people watch Grand Theft Auto VI live on Twitch in each hour of its release day
(2026-11-19 UTC), hours 2 to 24, as five quantiles (5%, 25%, 50%, 75%, 95%) of
each hour's concurrent-viewer count.

## The program

`main.py` (Python 3 standard library only, no network, no third-party code) reads
every `<id>.in` case file from `$MAKEFUN_INPUT_DIR` (else `/input`) and writes one
`<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`):

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Each output is `{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}`
for hours 2..24. Deterministic: two runs write identical bytes. Runtime is about
0.05 s per case, far inside the 8 s limit.

## The model

The case file ships 38 (42 in the final) comparable past game launches, each with
hourly Twitch category viewers from 14 days before to 3 days after its release
day. The forecast is built from those launches, in log space, as a blend of two
views:

1. **Regression view.** For each hour `h`, the log-ratio `log(y_h / p1)` — the
   hour's viewers relative to hour 1's print — is regressed on `log(p1)` across
   the comparables. Hour 1's print is a strong predictor of the day's shape
   (corr(log p1, log peak ratio) ≈ −0.94): a small hour-1 print explodes later
   (Hollow Knight: Silksong went 2,120x), a huge one stays flat (ELDEN RING
   stayed ~1.0x). The regression's residuals give honest interval widths.
2. **Neighbour view.** The 12 comparables whose log hour-1 print is closest to
   the target's, and the empirical quantiles of their log-ratios per hour. This
   is what keeps the forecast sane when `p1` is outside the training range: a
   plain linear regression would predict an absurdly low evening ratio for a
   1M-viewer hour 1 (0.25x), while the biggest launches on record stayed near
   0.9–1.3x.

The blend weight `w` rises with `log(p1)` — 0 below the training 25th percentile,
1 above the 90th — so small hour-1 prints trust the regression and huge ones
trust the neighbours. Median and interval widths are blended the same way, in log
space, so the 5% / 95% quantiles stay honest: wide where launches genuinely
diverge, narrow where the big launches agree. The regression input is winsorised
to the training log-p1 range (5th–95th percentile) so pathological inputs cannot
produce absurd extrapolations.

Things deliberately **not** used: the target's pre-release history and hour-1
channel count were tested as predictors of a launch's level shift and showed only
weak correlation (≤ 0.26), so they were left out rather than fitted on noise.

## Validation (real numbers, not estimates)

Practice answers are the true hourly prints from SullyGnome for the four practice
launches. Mean pinball loss on the log scale (the competition's score; lower is
better), computed exactly per the RULES formula:

| case | launch | hour-1 print | this model | house baseline |
|------|--------|-------------:|-----------:|---------------:|
| p00 | Marvel's Wolverine | 23,357 | 0.232266 | 0.743671 |
| p01 | Red Dead Redemption 2 | 116,550 | 0.211364 | 0.286813 |
| p02 | God of War Ragnarök | 269,517 | 0.150703 | 0.214579 |
| p03 | Marvel's Spider-Man 2 | 119,087 | 0.211254 | 0.303324 |
| **mean** | | | **0.201397** | **0.387097** |

The model beats the house baseline on every practice case, by 48% on average.
Leave-one-out cross-validation over the 38 comparables gives 0.382 (baseline
0.930 there), and 0.215 on the large hour-1 launches (p1 ≥ 100k), the regime the
final will be in.

Caveats, stated plainly: the practice cases are past launches anyone can look up,
and the final is a single new launch — a practice score is no guarantee for the
final. A prize needs a loss strictly below the best house entry on the final
(baseline ~0.387 on practice; the house model forecast's loss is not public).
The model's main risk is the level shift: individual launches run systematically
above or below the regression curve by up to ~±1 log unit, and no pre-checkpoint
signal tested predicts that shift, so the intervals are kept wide to cover it.

## Files

- `main.py` — the entrypoint (Python stdlib only)
- `README.md` — this file
- `LICENSE` — MIT
- `.makefun/owner.json` — the owner's wallet claim

---

<sub>Entered on [make.fun](https://make.fun/?ref=y5hkb35z&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
