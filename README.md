# tradebot (V1)

Paper-first, swing/long-term **long-only** bot for **equities + crypto** on **Alpaca US**.

## Safety defaults
- `DRY_RUN=true` (prints intended orders; does not place them)
- Freeze-on-drawdown behavior (20% DD): stops new entries, continues exits

## Setup
1) Create a venv and install:

```bash
cd tradebot
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

2) Put your Alpaca PAPER keys in `.env` (already created) and rotate if you pasted them in chat.

3) Run a dry-run rebalance:

```bash
tradebot rebalance --config config/config.yaml
```

## Config
See `config/config.yaml` (editable).
