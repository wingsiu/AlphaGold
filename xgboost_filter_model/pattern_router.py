"""Rule-based pattern matching for specialist model routing."""

from __future__ import annotations

import os

import pandas as pd

from config.v14_patterns import PATTERN_REGISTRY


_OPS = {
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _eval_rule(series: pd.Series, rule: dict) -> pd.Series:
    feat = rule["feat"]
    op = rule["op"]
    val = rule["val"]
    if feat not in series.index if isinstance(series, pd.Series) and not hasattr(series, "columns") else feat not in series:
        # series is a row (Series) or df column access handled by caller
        pass
    fn = _OPS[op]
    return fn(series[feat], val)


def _rule_threshold(pattern_name: str | None, rule: dict) -> float:
    """Env override for sweeps: V14_{PATTERN}_{FEAT}_MAX e.g. V14_REVERSAL_FVG_LONG_TIME_FROM_FVG_BULL_MAX."""
    val = rule["val"]
    if not pattern_name:
        return val
    feat = str(rule["feat"]).upper()
    key = f"V14_{pattern_name.upper()}_{feat}_MAX"
    raw = os.environ.get(key, "").strip()
    if raw:
        return float(raw)
    return val


def eval_rules_df(
    df: pd.DataFrame, rules: list[dict], *, pattern_name: str | None = None
) -> pd.Series:
    """True where all rules pass."""
    if not rules:
        return pd.Series(True, index=df.index)
    mask = pd.Series(True, index=df.index)
    for rule in rules:
        feat = rule["feat"]
        if feat not in df.columns:
            return pd.Series(False, index=df.index)
        col = df[feat]
        val = _rule_threshold(pattern_name, rule)
        op = rule["op"]
        fn = _OPS[op]
        mask &= fn(col, val)
    return mask


def _active_pattern_rules(spec: dict, *, training: bool) -> list[dict]:
    if training:
        return spec.get("pattern", [])
    return spec.get("router", spec.get("pattern", []))


def pattern_mask(df: pd.DataFrame, pattern_name: str, *, training: bool = False) -> pd.Series:
    """Boolean mask for one pattern (context + pattern/router rules, minus excludes)."""
    spec = PATTERN_REGISTRY[pattern_name]
    rules = _active_pattern_rules(spec, training=training)
    mask = eval_rules_df(df, spec.get("context", []))
    mask &= eval_rules_df(
        df,
        rules,
        pattern_name=pattern_name if not training else None,
    )
    exclude = spec.get("exclude", [])
    if exclude:
        mask &= ~eval_rules_df(df, exclude)
    return mask


def assign_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign highest-priority matching pattern per row.
    Adds columns: pattern_name (str or NA), pattern_id (int or -1).
    """
    out = df.copy()
    names = sorted(PATTERN_REGISTRY.keys(), key=lambda k: PATTERN_REGISTRY[k]["priority"])
    pattern_name = pd.Series(pd.NA, index=out.index, dtype="object")
    pattern_id = pd.Series(-1, index=out.index, dtype=int)

    for i, name in enumerate(names):
        hit = pattern_mask(out, name)
        # only assign if not already taken by higher priority
        unassigned = pattern_name.isna()
        take = hit & unassigned
        pattern_name.loc[take] = name
        pattern_id.loc[take] = i

    out["pattern_name"] = pattern_name
    out["pattern_id"] = pattern_id
    return out


def route_pattern_row(row: pd.Series) -> str | None:
    """Single-bar routing for live bot (priority order)."""
    names = sorted(PATTERN_REGISTRY.keys(), key=lambda k: PATTERN_REGISTRY[k]["priority"])
    for name in names:
        spec = PATTERN_REGISTRY[name]
        if not _row_passes(row, spec.get("context", []), pattern_name=None):
            continue
        if not _row_passes(
            row, _active_pattern_rules(spec, training=False), pattern_name=name
        ):
            continue
        if _row_passes(row, spec.get("exclude", []), pattern_name=None):
            continue
        return name
    return None


def _row_passes(row: pd.Series, rules: list[dict], *, pattern_name: str | None) -> bool:
    if not rules:
        return True
    for rule in rules:
        feat = rule["feat"]
        if feat not in row.index or pd.isna(row[feat]):
            return False
        fn = _OPS[rule["op"]]
        if not fn(row[feat], _rule_threshold(pattern_name, rule)):
            return False
    return True


def count_pattern_samples(df: pd.DataFrame) -> dict[str, int]:
    return {name: int(pattern_mask(df, name).sum()) for name in PATTERN_REGISTRY}
