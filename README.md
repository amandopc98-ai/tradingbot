# Alpaca Multi-Strategy Trading Bot

A Python trading bot that connects to the [Alpaca Markets API](https://alpaca.markets/)
and trades five instruments simultaneously — **SPY**, **QQQ**, **BTC/USD**, **GLD**,
and **USO** — each running a different, independently configurable strategy, under a
shared ATR-based risk model.

> **Not financial advice.** This is a template for algorithmic trading infrastructure,
> not a proven profitable strategy. Defaults to Alpaca's **paper trading** endpoint —
> review every strategy parameter and test thoroughly on paper before ever pointing
> this at a live account.

## Strategies

| Symbol | Strategy | Timeframe | Logic | Exit |
|---|---|---|---|---|
| SPY | Mean reversion | 15-min | Long when price < 1.5σ below 20-period SMA; short when > 1.5σ above | Price returns to the mean |
| QQQ | Mean reversion | 15-min | Same as SPY, with a wider 1.8σ threshold | Price returns to the mean |
| BTC/USD | Momentum breakout | 1-hour | Long on a close above the 20-period high with volume ≥ 1.5x the 20-period average (long-only — see note below) | 20-period low breakdown, or 2x ATR trailing stop |
| GLD | Trend following | 4-hour | Long when EMA(50) crosses above EMA(200); short/exit on the reverse cross | Opposite EMA cross, or 3x ATR trailing stop |
| USO | Trend following | 4-hour | Same as GLD | Opposite EMA cross, or 3x ATR trailing stop |

**Note on BTC/USD:** Alpaca's crypto trading is spot-only and does not support short
selling. The breakout strategy's "short" signal is therefore only used to *exit* an
existing long — the bot never opens a new short position in BTC/USD.

**A sixth strategy, `opening_range_breakout`, exists but is backtest-only** — it is
not wired into any symbol's live mapping above. It trades the 15:29-16:00
Europe/Berlin opening range: waits for a close beyond the range, then a retest of
the broken level, then enters on a close back through the breakout bar's extreme,
with a fixed stop at the opposite range boundary and a take-profit at a configurable
R-multiple (default 1.8x risk). See **Backtesting** below for how to run it.

## Risk management

- **ATR-based position sizing.** Each symbol's 14-period ATR is computed on every
  entry. Position size is set so that a 1x ATR adverse move equals exactly
  `RISK_PCT_PER_TRADE` (default 1%) of account equity — a quiet instrument gets a
  larger position, a volatile one gets a smaller one, so dollar risk per trade stays
  constant across all five symbols.
- **Hard stop-loss, no exceptions.** Every position gets an initial stop exactly 1x
  ATR from its entry price, which by construction caps the loss on that trade at
  `RISK_PCT_PER_TRADE` of equity. This stop is checked against the latest trade price
  on *every* poll cycle, not just on bar closes, so it can react intra-bar.
- **Trailing stops.** BTC/USD and GLD/USO additionally trail a wider stop (2x / 3x ATR
  from the extreme price reached) as a trade moves favorably. The trailing stop only
  ever tightens toward price — it never loosens past the original hard stop, so
  worst-case risk per trade never exceeds the initial 1%.
- **Correlation filter.** If SPY and QQQ are both already long, new long entries on
  BTC/USD are blocked, to avoid stacking correlated risk-on exposure.
- **Position caps.** `MAX_ALLOCATION_PCT` caps how much of current buying power a
  single symbol's entry can use, regardless of what the ATR sizing formula computes.

## Architecture

```
config.py                  Loads .env, defines per-symbol strategy/timeframe/risk config
bot/
  alpaca_client.py         Thin wrapper over alpaca-py: account, bars, quotes, orders
  indicators.py            Pure functions: SMA, rolling std, EMA, ATR, rolling high/low
  portfolio.py             Local position/stop state, persisted to portfolio_state.json
  risk_manager.py          Position sizing, stop-loss/trailing-stop math, correlation filter
  strategies/              One Strategy subclass per approach (mean reversion, momentum
                           breakout, trend following, opening-range breakout); pluggable
                           via STRATEGY_REGISTRY
  main.py                  Poll loop: checks stops every tick, evaluates strategy signals
                           only when a new bar has closed for that symbol's timeframe
  backtest.py              Replays the same strategies/risk model over historical bars
                           (see Backtesting below); never imported by or changes main.py
  tests/                   Unit tests for indicators/strategies/risk/backtest using
                           synthetic data (no network or API keys required)
```

Each strategy only looks at price history and the current position side, and returns
a `Signal` (`LONG` / `SHORT` / `EXIT` / `HOLD`). Sizing, stops, and order execution are
all handled centrally in `bot/main.py` / `bot/risk_manager.py`, so the same risk model
applies uniformly no matter which strategy is trading. `config.py` lives at the repo
root (outside the `bot` package) since it's the one file you're expected to open and
edit for local setup.

## Setup

1. **Get Alpaca API keys.** Sign up at [alpaca.markets](https://alpaca.markets/) and
   generate paper trading API keys from the dashboard (use live keys only once you're
   confident in the strategy).

2. **Install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure:**
   ```bash
   cp .env.example .env
   # edit .env and fill in ALPACA_API_KEY / ALPACA_SECRET_KEY
   ```

4. **Run the tests** (no API keys needed — pure logic only):
   ```bash
   python -m pytest bot/tests/ -v
   ```

5. **Run the bot** (from the repo root, as a module so `bot/` can import the root-level `config.py`):
   ```bash
   python -m bot.main
   ```
   Logs go to both the console and `logs/tradingbot.log` (rotating, 3x5MB). Position
   state persists to `portfolio_state.json` so a restart doesn't lose trailing-stop
   tracking; on startup the bot reconciles that file against Alpaca's actual open
   positions.

## Backtesting

`bot/backtest.py` replays the live strategies over historical Alpaca bar data. It
reuses the exact same `bot/strategies/*` classes, `bot/risk_manager.py` functions, and
`bot/portfolio.py` position bookkeeping the live bot uses (even importing
`build_strategies()` / `shorting_allowed()` from `bot/main.py` directly) so a backtest
can't silently drift out of sync with live behavior. It never modifies `bot/main.py`
or writes to `portfolio_state.json`.

```bash
python -m bot.backtest --symbol SPY --start 2023-01-01 --end 2024-01-01
```

- Omit `--symbol` to backtest all 5 configured symbols; pass `--symbol` multiple times
  to backtest several together (e.g. `--symbol SPY --symbol QQQ --symbol BTC/USD`) —
  symbols backtested together share one portfolio, so the SPY/QQQ-blocks-BTC
  correlation filter is only meaningful when they're run in the same command.
- `--symbol` isn't limited to the 5 configured symbols — any symbol works as long as
  `--strategy` is given for it too (e.g. `--symbol XLE --strategy mean_reversion`),
  since there's otherwise no way to know which strategy/timeframe to use for it. A
  symbol that's neither in `config.py` nor given a `--strategy` gets a clear error
  instead of running.
- `--equity` sets the starting simulated equity per symbol (default 100,000).
- `--output-dir` sets where equity-curve CSVs are written (default `backtests/`, which
  is gitignored).
- `--strategy` overrides which strategy runs, for every `--symbol` in that invocation
  — e.g. `--symbol GLD --strategy opening_range_breakout` backtests GLD with the
  opening-range strategy instead of its live trend_following mapping, without
  changing that live mapping in `config.py` at all. Strategies whose stop-loss/
  take-profit aren't ATR-based (currently only `opening_range_breakout`) manage their
  own exits internally instead of the shared ATR stop — see the `self_managed_exits`
  note in `bot/backtest.py`'s module docstring.
- `--trend-filter` / `--trend-filter-period MINUTES` configure `opening_range_breakout`'s
  optional trend filter for this run (off by default, matching the strategy's own
  default): when enabled, an entry is only taken if price is above (long) or below
  (short) the SMA/EMA of the last `trend_filter_period` 1-minute bars (default 120);
  otherwise that day's setup is discarded. Only has an effect together with
  `--strategy opening_range_breakout`. Example:
  `--symbol GLD --strategy opening_range_breakout --trend-filter --trend-filter-period 60`.

For each symbol it prints total return, win rate, number of trades, and max drawdown,
and writes a timestamp/equity CSV you can plot yourself. See the module docstring in
`bot/backtest.py` for the documented fidelity trade-offs (e.g. stops are checked once
per closed bar using that bar's close price, since only OHLC bar data is available —
the live bot checks on every poll tick using the latest trade price).

## Configuration reference (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | — | Your Alpaca API credentials |
| `ALPACA_PAPER` | `true` | `true` = paper trading endpoint, `false` = live |
| `POLL_INTERVAL_SECONDS` | `60` | Main loop tick interval |
| `RISK_PCT_PER_TRADE` | `0.01` | Fraction of equity risked per 1x ATR move |
| `MAX_ALLOCATION_PCT` | `0.30` | Max fraction of buying power per symbol at entry |
| `ALLOW_SHORTING` | `true` | Enables short entries on equities (requires a margin-enabled account); crypto is always long-only |

## Limitations / things to review before going live

- **No pattern day trading (PDT) handling.** If your account is under $25k and these
  strategies round-trip same-day, you can trip PDT restrictions on the equity side.
- **No slippage/partial-fill handling beyond what Alpaca's market orders do
  natively.** All orders are simple market orders.
- **Shorting equities requires a margin account** enabled for the specific symbol;
  Alpaca will reject the order otherwise — the bot logs and skips rather than
  crashing.
- **Single-process, in-memory + JSON-file state.** This is not built for multi-instance
  or high-availability deployment as-is.
- Strategy parameters (periods, thresholds, ATR multiples) were specified as given and
  have not been backtested here — validate them against historical data before
  committing real capital.
