from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any

from tradebot.strategies.registry import get_strategy, list_strategies
from tradebot.strategies.user_store import load_user_strategy
from tradebot.util.config import BotConfig, StrategyRef


AssetClass = Literal["stocks", "crypto"]
PolicyType = Literal["entry", "exit"]


@dataclass
class ResolvedPolicy:
    asset_class: AssetClass
    policy_type: PolicyType
    source: Literal["legacy", "builder"]
    strategy_id: str
    version: int | None
    payload: Any


def _ref_for(cfg: BotConfig, asset_class: AssetClass, policy_type: PolicyType) -> StrategyRef | None:
    aset = cfg.strategies.stocks if asset_class == "stocks" else cfg.strategies.crypto
    if policy_type == "entry":
        return aset.entry_strategy
    return aset.exit_strategy


def resolve_policy(cfg: BotConfig, *, asset_class: AssetClass, policy_type: PolicyType) -> ResolvedPolicy:
    """Resolve policy with backward-compatible fallback.

    - If per-asset builder ref is configured, load user strategy JSON as payload.
    - Otherwise use legacy registry strategy object from cfg.strategy_id.
    """
    ref = _ref_for(cfg, asset_class, policy_type)
    if ref and ref.id:
        spec = load_user_strategy(ref.id)
        return ResolvedPolicy(
            asset_class=asset_class,
            policy_type=policy_type,
            source="builder",
            strategy_id=ref.id,
            version=ref.version,
            payload=spec,
        )

    # legacy fallback
    strat = get_strategy(cfg.strategy_id)
    return ResolvedPolicy(
        asset_class=asset_class,
        policy_type=policy_type,
        source="legacy",
        strategy_id=cfg.strategy_id,
        version=None,
        payload=strat,
    )


def resolve_for_rebalance(cfg: BotConfig) -> dict[str, ResolvedPolicy]:
    return {
        "stocks_entry": resolve_policy(cfg, asset_class="stocks", policy_type="entry"),
        "crypto_entry": resolve_policy(cfg, asset_class="crypto", policy_type="entry"),
    }


def resolve_for_risk_check(cfg: BotConfig) -> dict[str, ResolvedPolicy]:
    return {
        "stocks_exit": resolve_policy(cfg, asset_class="stocks", policy_type="exit"),
        "crypto_exit": resolve_policy(cfg, asset_class="crypto", policy_type="exit"),
    }


def strategy_snapshot(cfg: BotConfig) -> dict:
    reb = resolve_for_rebalance(cfg)
    ex = resolve_for_risk_check(cfg)
    return {
        "stocks": {
            "entry": {"id": reb["stocks_entry"].strategy_id, "version": (cfg.strategies.stocks.entry_strategy.version if cfg.strategies.stocks.entry_strategy else None)},
            "exit": {"id": ex["stocks_exit"].strategy_id, "version": (cfg.strategies.stocks.exit_strategy.version if cfg.strategies.stocks.exit_strategy else None)},
            "exit_enabled": bool(cfg.strategies.stocks.exit_enabled),
        },
        "crypto": {
            "entry": {"id": reb["crypto_entry"].strategy_id, "version": (cfg.strategies.crypto.entry_strategy.version if cfg.strategies.crypto.entry_strategy else None)},
            "exit": {"id": ex["crypto_exit"].strategy_id, "version": (cfg.strategies.crypto.exit_strategy.version if cfg.strategies.crypto.exit_strategy else None)},
            "exit_enabled": bool(cfg.strategies.crypto.exit_enabled),
        },
    }


def validate_strategy_refs(cfg: BotConfig) -> list[str]:
    errs: list[str] = []
    items = list_strategies()
    by_id = {str(x.get("id") or ""): x for x in items}
    for asset, aset in (("stocks", cfg.strategies.stocks), ("crypto", cfg.strategies.crypto)):
        for kind, ref in (("entry", aset.entry_strategy), ("exit", aset.exit_strategy)):
            if ref is None:
                continue
            sid = str(ref.id or "").strip()
            if not sid:
                errs.append(f"{asset}.{kind}_strategy id is empty")
                continue
            meta = by_id.get(sid)
            if meta is None:
                errs.append(f"{asset}.{kind}_strategy '{sid}' not found")
                continue
            if not bool(meta.get("legacy_untyped")):
                mtype = str(meta.get("type") or "")
                mac = str(meta.get("asset_class") or "")
                if mtype and mtype != kind:
                    errs.append(f"{asset}.{kind}_strategy '{sid}' has type={mtype}")
                if mac and mac != asset:
                    errs.append(f"{asset}.{kind}_strategy '{sid}' has asset_class={mac}")
        if bool(aset.exit_enabled) and aset.exit_strategy is None:
            errs.append(f"{asset}.exit_enabled=true requires {asset}.exit_strategy")
    return errs
