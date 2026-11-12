# Roblox Up-and-Coming Top 10 Forecast

Entry for the make.fun bounty `roblox-up-and-coming-forecast-01`: forecast
Roblox's Up-and-Coming chart, place by place, for the 72 hours after entries
close.

## How it works

`main.py` (Python 3.13, standard library only, no network) reads
`$MAKEFUN_INPUT_DIR/history.ndjson` and `task.json`, then writes
`$MAKEFUN_OUTPUT_DIR/forecast.json`.

The model is a Monte Carlo rank diffusion:

- Every game on (or just off) the chart gets a rank and a drift. The drift
  is the game's recent rank velocity, estimated from consecutive reads,
  time-decayed and shrunk toward zero.
- Each simulated hour a game's rank moves by its drift plus Gaussian noise
  (the noise scale grows with the rank band) minus a weak mean-reversion
  pull toward the middle of the chart.
- Games leave the chart 28 days after creation when the history's details
  give a creation date; their probability mass flows to the games that
  remain.
- The simulated top-10 occupancy of every hour inside the scored window is
  averaged into one probability per (game, place).
- A game is named at a place only when its simulated probability beats the
  unnamed share it would otherwise receive; the rest stays unnamed, which
  covers tail games and brand-new entrants.

The run is fully deterministic: a fixed seed, a self-contained normal
sampler built on `random.random()`, deterministic sort keys and JSON output,
so two runs on the same history produce identical bytes.

## Local run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT, see `LICENSE`.
