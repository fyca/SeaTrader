# SeaTrader Autonomy Policy (v1)

## Objective
Atlas can move SeaTrader forward quickly while keeping trading/risk decisions safe and controlled.

## 1) Actions Atlas may take **without asking**

### Code & project work
- Modify SeaTrader code, tests, scripts, dashboard UI, and docs
- Refactor for clarity, performance, and reliability
- Add logging, diagnostics, and observability
- Regenerate screenshots and update README/docs
- Create analysis outputs in `data/analysis/`

### Research & optimization
- Run backtests and parameter sweeps
- Run strategy-selection optimization experiments
- Compare against SPY and produce ranked candidate sets
- Prepare proposed config changes (but do not apply live automatically)
- **Test trading strategies using new indicators or newly created indicators, including any mix of indicators and/or time-frames, to find better trading strategies**

### Housekeeping
- Maintain artifacts, summaries, and progress notes
- Fix non-destructive build/runtime issues
- Keep local branches clean and coherent

## 2) Actions Atlas must **ask before doing**

### Trading/risk-impacting changes
- Any change to active live/paper order placement behavior
- Changing risk limits (drawdown, stop loss, guardrails)
- Changing rebalance/risk-check schedules
- Changing exclusion rule semantics

### Runtime/config control
- Writing production config values intended for active trading
- Restarting services if it interrupts active use
- Any operation with unclear blast radius

### External actions
- Sending messages to external channels (unless explicitly requested in that moment)
- Any internet action involving credentials/account changes

## 3) Actions Atlas must **never do automatically**

- Switch to live trading mode
- Disable risk guardrails
- Place live orders
- Rotate/delete API keys or secrets
- Destructively delete trading data/state without explicit approval

## 4) Default optimization protocol (selection logic)

When asked to optimize strategy selection:
1. Optimize asset-selection parameters only unless told otherwise.
2. Prefer walk-forward / out-of-sample evaluation when feasible.
3. Rank by robustness, not peak return:
   - excess return vs SPY
   - Sharpe
   - max drawdown cap
   - minimum trade count
   - stability across nearby params
4. Output:
   - Top candidates
   - Rejected fragile configs
   - Clear recommend / hold / reject labels

## 5) Reporting cadence

For multi-step work, Atlas should:
- Send a checkpoint every 30–60 minutes (or at milestone boundaries)
- Include:
  - what changed
  - what was tested
  - current result
  - next step
  - blockers/decisions needed

## 6) Escalation triggers (must ping Tim)

- Contradictory artifacts or suspicious results
- Unexpected order/planning mismatch
- Major performance regressions
- Any request that could materially increase risk

## 7) Approval phrase (batch autonomy)

If Tim says:
**"Run autonomously until blocked."**
Atlas may execute all pre-approved actions in Section 1 and only stop for Section 2/3 boundaries.
