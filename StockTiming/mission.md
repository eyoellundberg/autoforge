# Mission — StockTiming

## Job
Optimize moving average crossover parameters for a stock timing strategy across
different market regimes (trending, ranging, volatile, event-driven).

## What good looks like
- Positive PnL in trending and event regimes
- Near-zero or positive PnL in ranging and volatile regimes (capital preservation)
- Strategy parameters differ meaningfully by regime — no single set dominates all

## Abstain when
- Regime is unclassifiable (insufficient price history)
- Volatility is extreme (> 3x normal) — risk of whipsaw exceeds expected reward

## Failure
- Negative PnL in trending regimes (should be the easiest to profit from)
- Single parameter set winning > 80% of all rounds regardless of regime

---
*This file is the human-readable contract for the specialist.
simulation.py is the technical implementation of it.*
