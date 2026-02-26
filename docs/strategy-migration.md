# Strategy Migration (Legacy -> Per-Asset Entry/Exit)

## Summary
SeaTrader now supports per-asset strategy selection for both live and backtesting:

- `strategies.stocks.entry_strategy`
- `strategies.stocks.exit_strategy`
- `strategies.stocks.exit_enabled`
- `strategies.crypto.entry_strategy`
- `strategies.crypto.exit_strategy`
- `strategies.crypto.exit_enabled`

Legacy `strategy_id` is still supported as fallback.

## Backward compatibility
If per-asset strategy refs are unset:
- Entry selection falls back to `strategy_id`.
- Exit rule evaluation falls back to legacy behavior (stop loss / trend break / existing risk logic).

## Recommended migration
1. Keep existing `strategy_id` in config.
2. Set per-asset entry strategies first.
3. Set per-asset exit strategies.
4. Enable exits per asset only after validating behavior in backtests.
5. Keep versions pinned once stable.

## Validation behavior
Runtime and API now fail fast when:
- a configured strategy id does not exist
- exit is enabled but exit strategy is missing

## Artifacts / observability
Live artifacts now include `strategy_snapshot` metadata so each run records which strategies were used.
