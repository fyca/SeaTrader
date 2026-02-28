# Indicator Caching Contract

To keep backtests fast, indicator math is centralized in:

- `src/tradebot/indicators/service.py`

## Rule

When adding a new indicator:

1. Implement it in `IndicatorService`
2. Expose a method via `indicator_service`
3. Use that service from strategies/risk code (do **not** call `rolling/ewm/pct_change` directly there)

## Guardrail

`tests/test_indicator_guardrails.py` enforces this by scanning `src/tradebot/strategies` and `src/tradebot/signals`.

If that test fails, move indicator math into `IndicatorService` and call it from there.
