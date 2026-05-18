import argparse
import copy
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import qlib
from qlib.config import C
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord


DEFAULT_PROVIDER_URI = "~/.qlib/qlib_data/cn_data"

DEFAULT_MODEL_CONFIG = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        "loss": "mse",
        "colsample_bytree": 0.8879,
        "learning_rate": 0.2,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 20,
    },
}
QUANTAALPHA_MODEL_CONFIG = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        "loss": "mse",
        "learning_rate": 0.1,
        "max_depth": 8,
        "num_leaves": 210,
        "colsample_bytree": 0.8879,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "num_threads": 20,
        "seed": 42,
        "random_state": 42,
        "early_stopping_round": 50,
        "num_boost_round": 500,
        "min_child_samples": 100,
        "feature_fraction_bynode": 0.8,
    },
}
DEFAULT_LABEL_EXPRESSION = "Ref($close, -2) / Ref($close, -1) - 1"


@dataclass(frozen=True)
class SequenceToken:
    length: int


def _broadcast_row_stat(x: pd.DataFrame, stat: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        np.repeat(stat.to_numpy()[:, None], x.shape[1], axis=1),
        index=x.index,
        columns=x.columns,
    )


def ABS(x):
    return x.abs()


def SIGN(x):
    return np.sign(x)


def LOG(x):
    return np.log(x)


def MAX(x, y=None):
    if y is None:
        return _broadcast_row_stat(x, x.max(axis=1))
    return np.maximum(x, y)


def MIN(x, y=None):
    if y is None:
        return _broadcast_row_stat(x, x.min(axis=1))
    return np.minimum(x, y)


def EXP(x):
    return np.exp(x)


def INV(x):
    return 1.0 / x


def DELAY(x, n):
    return x.shift(int(n))


def DELTA(x, n):
    n = int(n)
    return x - x.shift(n)


def TS_SUM(x, n):
    return x.rolling(int(n), min_periods=1).sum()


def TS_MEAN(x, n):
    return x.rolling(int(n), min_periods=1).mean()


def TS_STD(x, n):
    return x.rolling(int(n), min_periods=1).std()


def TS_VAR(x, n):
    return x.rolling(int(n), min_periods=1).var()


def TS_MAX(x, n):
    return x.rolling(int(n), min_periods=1).max()


def TS_MIN(x, n):
    return x.rolling(int(n), min_periods=1).min()


def TS_MEDIAN(x, n):
    return x.rolling(int(n), min_periods=1).median()


def TS_MAD(x, n):
    def _mad(arr):
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return np.nan
        return np.mean(np.abs(arr - arr.mean()))

    return x.rolling(int(n), min_periods=1).apply(_mad, raw=True)


def TS_ZSCORE(x, n):
    mean = TS_MEAN(x, n)
    std = TS_STD(x, n)
    return (x - mean) / (std + 1e-12)


def TS_RANK(x, n):
    def _rank(arr):
        if np.isnan(arr[-1]):
            return np.nan
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return np.nan
        return pd.Series(valid).rank(pct=True).iloc[-1]

    return x.rolling(int(n), min_periods=1).apply(_rank, raw=True)


def TS_PCTCHANGE(x, n):
    return x.pct_change(int(n), fill_method=None)


def TS_COVARIANCE(x, y, n):
    n = int(n)
    if isinstance(x, SequenceToken):
        x = _sequence_like(y, x.length)
    if isinstance(y, SequenceToken):
        y = _sequence_like(x, y.length)
    mean_x = TS_MEAN(x, n)
    mean_y = TS_MEAN(y, n)
    return TS_MEAN(x * y, n) - mean_x * mean_y


def TS_CORR(x, y, n):
    if isinstance(x, SequenceToken):
        x = _sequence_like(y, x.length)
    if isinstance(y, SequenceToken):
        y = _sequence_like(x, y.length)
    cov = TS_COVARIANCE(x, y, n)
    std_x = TS_STD(x, n)
    std_y = TS_STD(y, n)
    return cov / (std_x * std_y + 1e-12)


def COUNT(cond, n):
    return cond.astype(float).rolling(int(n), min_periods=1).sum()


def SUMIF(x, n, cond):
    return x.where(cond, 0.0).rolling(int(n), min_periods=1).sum()


def DECAYLINEAR(x, n):
    n = int(n)

    def _weighted_mean(arr):
        valid = ~np.isnan(arr)
        if not valid.any():
            return np.nan
        weights = np.arange(1, len(arr) + 1, dtype=float)
        weights = weights[valid]
        vals = arr[valid]
        return np.sum(vals * weights) / np.sum(weights)

    return x.rolling(n, min_periods=1).apply(_weighted_mean, raw=True)


def EMA(x, n):
    return x.ewm(span=int(n), adjust=False, min_periods=1).mean()


def SMA(x, m, n=None):
    m = int(m)
    if n is None:
        return TS_MEAN(x, m)
    return x.ewm(alpha=float(n) / float(m), adjust=True, min_periods=1).mean()


def MACD(x, short_n, long_n):
    return EMA(x, short_n) - EMA(x, long_n)


def RSI(x, n):
    delta = x.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    rs = TS_MEAN(up, n) / (TS_MEAN(down, n) + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def HIGHDAY(x, n):
    def _highday(arr):
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return np.nan
        return len(arr) - 1 - int(np.nanargmax(arr))

    return x.rolling(int(n), min_periods=1).apply(_highday, raw=True)


def TS_ARGMAX(x, n):
    def _argmax(arr):
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return np.nan
        return len(arr) - int(np.nanargmax(arr))

    return x.rolling(int(n), min_periods=1).apply(_argmax, raw=True)


def PERCENTILE(x, q, n):
    return x.rolling(int(n), min_periods=1).quantile(float(q))


def SUMAC(x, n):
    return TS_SUM(x, n)


def SEQUENCE(n):
    return SequenceToken(int(n))


def _sequence_like(x: pd.DataFrame, n: int) -> pd.DataFrame:
    data = np.tile(np.arange(1, len(x) + 1, dtype=float)[:, None], (1, x.shape[1]))
    return pd.DataFrame(data, index=x.index, columns=x.columns)


def REGBETA(y, x, n):
    n = int(n)
    if isinstance(x, SequenceToken):
        x = _sequence_like(y, x.length)
    cov = TS_COVARIANCE(y, x, n)
    var = TS_VAR(x, n)
    return cov / (var + 1e-12)


def REGRESI(y, x, n):
    n = int(n)
    if isinstance(x, SequenceToken):
        x = _sequence_like(y, x.length)
    beta = REGBETA(y, x, n)
    alpha = TS_MEAN(y, n) - beta * TS_MEAN(x, n)
    return y - (alpha + beta * x)


def RANK(x):
    return x.rank(axis=1, pct=True)


def ZSCORE(x):
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    return (x - mean.to_numpy()[:, None]) / (std.to_numpy()[:, None] + 1e-12)


def MEAN(x):
    return _broadcast_row_stat(x, x.mean(axis=1))


def STD(x):
    return _broadcast_row_stat(x, x.std(axis=1))


def MEDIAN(x):
    return _broadcast_row_stat(x, x.median(axis=1))


def IF(cond, left, right):
    return left.where(cond, right)


def _is_window_arg(value) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


def _reference_panel(*values) -> Optional[pd.DataFrame]:
    for value in values:
        if isinstance(value, pd.DataFrame):
            return value
    return None


def _broadcast_like(value, ref: pd.DataFrame):
    if isinstance(value, pd.DataFrame):
        return value
    return pd.DataFrame(value, index=ref.index, columns=ref.columns)


def Neg(x):
    return -x


def Add(x, y):
    return x + y


def Sub(x, y):
    return x - y


def Mul(x, y):
    return x * y


def Div(x, y):
    return x / y


def Abs(x):
    return ABS(x)


def Sign(x):
    return SIGN(x)


def Log(x):
    return LOG(x)


def CsRank(x):
    return RANK(x)


def TsRank(x, n):
    return TS_RANK(x, n)


def TsMax(x, n):
    return TS_MAX(x, n)


def TsMin(x, n):
    return TS_MIN(x, n)


def TsArgMax(x, n):
    return TS_ARGMAX(x, n)


def Mean(x, n=None):
    if n is None:
        return MEAN(x)
    return TS_MEAN(x, n)


def Std(x, n=None):
    if n is None:
        return STD(x)
    return TS_STD(x, n)


def Med(x, n=None):
    if n is None:
        return MEDIAN(x)
    return TS_MEDIAN(x, n)


def Max(x, y=None):
    if y is None:
        return MAX(x)
    if _is_window_arg(y):
        return TS_MAX(x, y)
    return np.maximum(x, y)


def Min(x, y=None):
    if y is None:
        return MIN(x)
    if _is_window_arg(y):
        return TS_MIN(x, y)
    return np.minimum(x, y)


def Max2(x, y):
    return np.maximum(x, y)


def Min2(x, y):
    return np.minimum(x, y)


def Corr(x, y, n):
    return TS_CORR(x, y, n)


def Cov(x, y, n):
    return TS_COVARIANCE(x, y, n)


def Skew(x, n=None):
    if n is None:
        return _broadcast_row_stat(x, x.skew(axis=1))
    return x.rolling(int(n), min_periods=min(3, int(n))).skew()


def Kurt(x, n=None):
    if n is None:
        return _broadcast_row_stat(x, x.kurt(axis=1))
    return x.rolling(int(n), min_periods=min(4, int(n))).kurt()


def WMA(x, n):
    n = int(n)
    weights = np.array([0.9**i for i in range(n)][::-1], dtype=float)

    def _weighted_mean(arr):
        valid = ~np.isnan(arr)
        if not valid.any():
            return np.nan
        w = weights[: len(arr)][valid]
        vals = arr[valid]
        return float(np.sum(vals * w) / np.sum(w))

    return x.rolling(n, min_periods=1).apply(_weighted_mean, raw=True)


def Slope(x, n):
    n = int(n)
    return REGBETA(x, SEQUENCE(n), n)


def Resi(x, n):
    n = int(n)
    return REGRESI(x, SEQUENCE(n), n)


def Rsquare(x, n):
    n = int(n)
    corr = TS_CORR(x, SEQUENCE(n), n)
    return corr * corr


def SignedPower(x, power):
    abs_x = ABS(x)
    return SIGN(x) * np.power(abs_x, float(power))


def RatioMul(x, y):
    return x * y


def Greater(x, y):
    return x > y


def Less(x, y):
    return x < y


def Eq(x, y):
    return x == y


def And(x, y):
    ref = _reference_panel(x, y)
    if ref is None:
        return bool(x) and bool(y)
    left = _broadcast_like(x, ref).fillna(False).astype(bool)
    right = _broadcast_like(y, ref).fillna(False).astype(bool)
    return left & right


def Or(x, y):
    ref = _reference_panel(x, y)
    if ref is None:
        return bool(x) or bool(y)
    left = _broadcast_like(x, ref).fillna(False).astype(bool)
    right = _broadcast_like(y, ref).fillna(False).astype(bool)
    return left | right


def IfElse(cond, left, right):
    ref = _reference_panel(cond, left, right)
    if ref is None:
        return left if cond else right
    cond_panel = _broadcast_like(cond, ref).fillna(False).astype(bool)
    left_panel = _broadcast_like(left, ref)
    right_panel = _broadcast_like(right, ref)
    return left_panel.where(cond_panel, right_panel)


def _normalize_expr(expr: str) -> str:
    expr = _convert_ternary(expr)
    expr = _convert_logical_expr(expr)
    return re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", lambda m: f'FIELDS["{m.group(1)}"]', expr)


def _convert_logical_expr(expr: str) -> str:
    expr = expr.strip()
    expr = _convert_logical_in_parentheses(expr)
    parts = _split_top_level_commas(expr)
    if len(parts) > 1:
        return ", ".join(_parenthesize_logical_current_level(part.strip()) for part in parts)
    return _parenthesize_logical_current_level(expr)


def _convert_logical_in_parentheses(expr: str) -> str:
    result = []
    idx = 0
    while idx < len(expr):
        ch = expr[idx]
        if ch != "(":
            result.append(ch)
            idx += 1
            continue

        end_idx = _find_matching_right_paren(expr, idx)
        inner = expr[idx + 1 : end_idx]
        result.append(f"({_convert_logical_expr(inner)})")
        idx = end_idx + 1
    return "".join(result)


def _parenthesize_logical_current_level(expr: str) -> str:
    pos = _find_top_level_operator(expr, "||")
    if pos != -1:
        left = _parenthesize_logical_current_level(expr[:pos])
        right = _parenthesize_logical_current_level(expr[pos + 2 :])
        return f"(({left.strip()}) | ({right.strip()}))"

    pos = _find_top_level_operator(expr, "&&")
    if pos != -1:
        left = _parenthesize_logical_current_level(expr[:pos])
        right = _parenthesize_logical_current_level(expr[pos + 2 :])
        return f"(({left.strip()}) & ({right.strip()}))"

    return expr


def _convert_ternary(expr: str) -> str:
    expr = _convert_ternary_in_parentheses(expr)
    parts = _split_top_level_commas(expr)
    if len(parts) > 1:
        return ", ".join(_convert_ternary(part.strip()) for part in parts)
    q_pos = _find_top_level_char(expr, "?")
    if q_pos == -1:
        return expr
    c_pos = _find_matching_colon(expr, q_pos)
    cond = _convert_ternary(expr[:q_pos].strip())
    left = _convert_ternary(expr[q_pos + 1 : c_pos].strip())
    right = _convert_ternary(expr[c_pos + 1 :].strip())
    return f"IF({cond}, {left}, {right})"


def _convert_ternary_in_parentheses(expr: str) -> str:
    result = []
    idx = 0
    while idx < len(expr):
        ch = expr[idx]
        if ch != "(":
            result.append(ch)
            idx += 1
            continue

        end_idx = _find_matching_right_paren(expr, idx)
        inner = expr[idx + 1 : end_idx]
        result.append(f"({_convert_ternary(inner)})")
        idx = end_idx + 1
    return "".join(result)


def _find_top_level_char(expr: str, target: str) -> int:
    depth = 0
    for idx, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == target and depth == 0:
            return idx
    return -1


def _find_matching_right_paren(expr: str, left_idx: int) -> int:
    depth = 0
    for idx in range(left_idx, len(expr)):
        if expr[idx] == "(":
            depth += 1
        elif expr[idx] == ")":
            depth -= 1
            if depth == 0:
                return idx
    raise ValueError(f"Unmatched parentheses in expression: {expr}")


def _find_top_level_operator(expr: str, operator: str) -> int:
    depth = 0
    idx = 0
    while idx <= len(expr) - len(operator):
        ch = expr[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and expr[idx : idx + len(operator)] == operator:
            return idx
        idx += 1
    return -1


def _split_top_level_commas(expr: str) -> list[str]:
    depth = 0
    start = 0
    parts = []
    for idx, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(expr[start:idx])
            start = idx + 1
    parts.append(expr[start:])
    return parts


def _find_matching_colon(expr: str, q_pos: int) -> int:
    depth = 0
    nested_q = 0
    for idx in range(q_pos + 1, len(expr)):
        ch = expr[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "?" and depth == 0:
            nested_q += 1
        elif ch == ":" and depth == 0:
            if nested_q == 0:
                return idx
            nested_q -= 1
    raise ValueError(f"Invalid ternary expression: {expr}")


def _to_datetime_instrument_index(df: pd.DataFrame) -> pd.DataFrame:
    if list(df.index.names) == ["datetime", "instrument"]:
        return df.sort_index()
    if list(df.index.names) == ["instrument", "datetime"]:
        return df.swaplevel().sort_index()
    raise ValueError(f"Unexpected index names: {df.index.names}")


def _stack_panel(panel: pd.DataFrame, name: str) -> pd.Series:
    panel = panel.copy()
    panel.index.name = "datetime"
    panel.columns.name = "instrument"
    try:
        return panel.stack(future_stack=True).rename(name)
    except TypeError:
        return panel.stack(dropna=False).rename(name)


def _load_factor_library(path: Path) -> List[dict]:
    payload = json.loads(path.read_text())
    factors = []
    for factor_id, factor_info in payload["factors"].items():
        factor_payload = dict(factor_info)
        factor_payload.setdefault("factor_id", factor_id)
        factors.append(factor_payload)
    return factors


def _load_base_fields(instruments: str, start_time: str, end_time: str) -> Dict[str, pd.DataFrame]:
    instruments = D.instruments(instruments)
    raw = D.features(
        instruments,
        ["$open", "$close", "$high", "$low", "$volume", "$vwap", "$amount"],
        start_time,
        end_time,
    )
    raw = _to_datetime_instrument_index(raw)

    fields = {}
    for col in raw.columns:
        panel = raw[col].unstack("instrument").sort_index()
        fields[col[1:]] = panel
    fields["return"] = fields["close"] / fields["close"].shift(1) - 1.0
    fields["returns"] = fields["return"]
    fields["amt"] = fields["amount"]
    return fields


def _build_legacy_feature_label_frames(
    factors: List[dict],
    field_panels: Dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    env = {
        "__builtins__": {},
        "FIELDS": field_panels,
        "ABS": ABS,
        "Abs": Abs,
        "Add": Add,
        "And": And,
        "COUNT": COUNT,
        "Corr": Corr,
        "Cov": Cov,
        "CsRank": CsRank,
        "DECAYLINEAR": DECAYLINEAR,
        "DELAY": DELAY,
        "Delay": DELAY,
        "DELTA": DELTA,
        "Delta": DELTA,
        "Div": Div,
        "EMA": EMA,
        "Eq": Eq,
        "EXP": EXP,
        "Greater": Greater,
        "HIGHDAY": HIGHDAY,
        "IF": IF,
        "IfElse": IfElse,
        "INV": INV,
        "Kurt": Kurt,
        "Less": Less,
        "LOG": LOG,
        "Log": Log,
        "MAX": MAX,
        "MACD": MACD,
        "MEAN": MEAN,
        "MEDIAN": MEDIAN,
        "MIN": MIN,
        "Max": Max,
        "Max2": Max2,
        "Mean": Mean,
        "Med": Med,
        "Min": Min,
        "Min2": Min2,
        "MACD": MACD,
        "Mul": Mul,
        "Neg": Neg,
        "Or": Or,
        "PERCENTILE": PERCENTILE,
        "RANK": RANK,
        "RatioMul": RatioMul,
        "REGBETA": REGBETA,
        "REGRESI": REGRESI,
        "Resi": Resi,
        "RSI": RSI,
        "Rsquare": Rsquare,
        "SEQUENCE": SEQUENCE,
        "SIGN": SIGN,
        "SMA": SMA,
        "STD": STD,
        "SUMAC": SUMAC,
        "SUMIF": SUMIF,
        "Sign": Sign,
        "SignedPower": SignedPower,
        "Skew": Skew,
        "Slope": Slope,
        "Std": Std,
        "Sub": Sub,
        "SMA": SMA,
        "TS_ARGMAX": TS_ARGMAX,
        "TS_CORR": TS_CORR,
        "TS_COVARIANCE": TS_COVARIANCE,
        "TS_MAD": TS_MAD,
        "TS_MAX": TS_MAX,
        "TS_MEAN": TS_MEAN,
        "TS_MEDIAN": TS_MEDIAN,
        "TS_MIN": TS_MIN,
        "TS_PCTCHANGE": TS_PCTCHANGE,
        "TS_RANK": TS_RANK,
        "TS_STD": TS_STD,
        "TS_SUM": TS_SUM,
        "TS_VAR": TS_VAR,
        "TS_ZSCORE": TS_ZSCORE,
        "TsArgMax": TsArgMax,
        "TsMax": TsMax,
        "TsMin": TsMin,
        "TsRank": TsRank,
        "WMA": WMA,
        "ZSCORE": ZSCORE,
    }

    feature_series = []
    for idx, factor in enumerate(factors, start=1):
        expr = _normalize_expr(factor["factor_expression"])
        try:
            result = eval(expr, env, {})
        except Exception as exc:
            print(f"failed factor: {factor['factor_name']}", flush=True)
            print(f"raw expr: {factor['factor_expression']}", flush=True)
            print(f"normalized expr: {expr}", flush=True)
            raise
        if not isinstance(result, pd.DataFrame):
            raise TypeError(f"{factor['factor_name']} did not evaluate to DataFrame")
        feature_series.append(_stack_panel(result, factor["factor_name"]))
        if idx % 10 == 0 or idx == len(factors):
            print(f"evaluated {idx}/{len(factors)} factors", flush=True)

    feature_df = pd.concat(feature_series, axis=1).sort_index()
    label_panel = field_panels["close"].shift(-2) / field_panels["close"].shift(-1) - 1.0
    label_df = _stack_panel(label_panel, "LABEL0").to_frame().sort_index()
    return feature_df, label_df


def _default_quantaalpha_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_quantaalpha_root(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _import_quantaalpha_calculator(_quantaalpha_root: Path | None = None):
    try:
        from qlib.contrib.quantaalpha_compat import CustomFactorCalculator
    except Exception as exc:
        raise ImportError(
            "Failed to import vendored QA-compatible CustomFactorCalculator from qlib"
        ) from exc

    return CustomFactorCalculator


def _resolve_factor_cache_dir(args, _quantaalpha_root: Path | None = None) -> Path:
    if args.factor_cache_dir:
        return Path(args.factor_cache_dir).expanduser().resolve()
    return (Path(__file__).resolve().parents[3] / "outputs/factor_cache").resolve()


def _load_env_assignments(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value[:1] == value[-1:] and value[:1] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _resolve_provider_uri(args) -> Tuple[str, str]:
    explicit_provider = os.path.expanduser(args.provider_uri)
    if args.feature_pipeline != "quantaalpha":
        return explicit_provider, "cli/default"

    if args.provider_uri and args.provider_uri != DEFAULT_PROVIDER_URI:
        return explicit_provider, "cli"

    env_provider = os.environ.get("QLIB_DATA_DIR") or os.environ.get("QLIB_PROVIDER_URI")
    if env_provider:
        return os.path.expanduser(env_provider), "environment"

    return explicit_provider, "cli/default"


def _build_quantaalpha_config(args) -> dict:
    config = {
        "data": {
            "provider_uri": os.path.expanduser(args.provider_uri),
            "region": args.region,
            "market": args.market,
            "start_time": args.start_time,
            "end_time": args.end_time,
        },
        "llm": {
            "cache_dir": str(_resolve_factor_cache_dir(args)),
        },
        "factor_calculation": {
            "recompute_backend": "thread",
            "recompute_n_jobs": args.num_threads,
        },
    }
    return config


def _load_label_frame(instruments: str, start_time: str, end_time: str) -> pd.DataFrame:
    stock_list = D.instruments(instruments)
    label_df = D.features(
        stock_list,
        [DEFAULT_LABEL_EXPRESSION],
        start_time,
        end_time,
        freq="day",
    )
    label_df.columns = ["LABEL0"]
    return label_df


def _normalize_multiindex(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        raise TypeError(f"{df_name} index must be MultiIndex, got {type(df.index)}")

    names = list(df.index.names)
    new_names = list(names)
    for idx, name in enumerate(names):
        level_values = df.index.get_level_values(idx)
        if name in {"datetime", "date"}:
            new_names[idx] = "datetime"
        elif name in {"instrument", "stock"}:
            new_names[idx] = "instrument"
        elif name is None:
            if pd.api.types.is_datetime64_any_dtype(level_values):
                new_names[idx] = "datetime"
            elif level_values.dtype == object or pd.api.types.is_string_dtype(level_values):
                new_names[idx] = "instrument"

    if new_names != names:
        df = df.copy()
        df.index = df.index.set_names(new_names)

    actual_names = list(df.index.names)
    if actual_names == ["instrument", "datetime"]:
        return df.swaplevel().sort_index()
    if actual_names == ["datetime", "instrument"]:
        return df.sort_index()

    raise ValueError(f"{df_name} index names must be datetime/instrument, got {actual_names}")


def _align_feature_and_label(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    feature_df = _normalize_multiindex(feature_df.copy(), "feature")
    label_df = _normalize_multiindex(label_df.copy(), "label")
    metadata = {
        "feature_rows_before_align": len(feature_df),
        "label_rows_before_align": len(label_df),
        "used_merge_fallback": False,
    }

    common_index = feature_df.index.intersection(label_df.index)
    if len(common_index) == 0 and len(feature_df) > 0 and len(label_df) > 0:
        feature_dt = feature_df.index.get_level_values("datetime")
        label_dt = label_df.index.get_level_values("datetime")

        if not pd.api.types.is_datetime64_any_dtype(feature_dt):
            feature_df.index = pd.MultiIndex.from_arrays(
                [pd.to_datetime(feature_dt), feature_df.index.get_level_values("instrument")],
                names=["datetime", "instrument"],
            )
        if not pd.api.types.is_datetime64_any_dtype(label_dt):
            label_df.index = pd.MultiIndex.from_arrays(
                [pd.to_datetime(label_dt), label_df.index.get_level_values("instrument")],
                names=["datetime", "instrument"],
            )
        common_index = feature_df.index.intersection(label_df.index)

    if len(common_index) == 0:
        metadata["used_merge_fallback"] = True
        feature_reset = feature_df.reset_index()
        label_reset = label_df.reset_index()
        merged = pd.merge(feature_reset, label_reset, on=["datetime", "instrument"], how="inner")
        if merged.empty:
            raise ValueError(
                "Feature and label indices do not overlap after normalization: "
                f"feature_rows={len(feature_df)}, label_rows={len(label_df)}"
            )
        merged = merged.set_index(["datetime", "instrument"]).sort_index()
        feature_cols = list(feature_df.columns)
        label_cols = list(label_df.columns)
        feature_df = merged[feature_cols]
        label_df = merged[label_cols]
    else:
        feature_df = feature_df.loc[common_index].sort_index()
        label_df = label_df.loc[common_index].sort_index()

    metadata["aligned_rows"] = len(feature_df)
    return feature_df, label_df, metadata


def _apply_quantaalpha_preprocessing(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined_df = pd.concat([feature_df.copy(), label_df.copy()], axis=1)
    feature_cols = list(feature_df.columns)
    label_cols = list(label_df.columns)

    combined_df[feature_cols] = combined_df[feature_cols].fillna(0)
    combined_df[feature_cols] = combined_df[feature_cols].replace([np.inf, -np.inf], 0)
    for col in feature_cols:
        combined_df[col] = combined_df.groupby(level="datetime")[col].transform(
            lambda x: (x.rank(pct=True) - 0.5) if len(x) > 1 else 0
        )

    valid_rows = combined_df[label_cols].notna().all(axis=1)
    combined_df = combined_df.loc[valid_rows].sort_index()
    for col in label_cols:
        combined_df[col] = combined_df.groupby(level="datetime")[col].transform(
            lambda x: (x.rank(pct=True) - 0.5) if len(x) > 1 else 0
        )

    return combined_df[feature_cols].copy(), combined_df[label_cols].copy()


def _build_quantaalpha_feature_label_frames(
    factors: List[dict],
    args,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    CustomFactorCalculator = _import_quantaalpha_calculator()
    qa_config = _build_quantaalpha_config(args)
    cache_dir = Path(qa_config["llm"]["cache_dir"])

    calculator = CustomFactorCalculator(
        data_df=None,
        cache_dir=cache_dir,
        auto_extract_cache=True,
        config=qa_config,
    )
    feature_df, diagnostics = calculator.calculate_factors_batch(
        factors,
        use_cache=not args.disable_factor_cache,
        persist_cache=args.persist_factor_cache,
        return_diagnostics=True,
    )
    if feature_df is None or feature_df.empty:
        raise ValueError("QuantaAlpha factor calculator returned no features")

    label_df = _load_label_frame(args.market, args.start_time, args.end_time)
    aligned_feature, aligned_label, align_metadata = _align_feature_and_label(feature_df, label_df)
    post_feature, post_label = _apply_quantaalpha_preprocessing(aligned_feature, aligned_label)

    factor_source_counts = dict(sorted(Counter(diagnostics.factor_sources.values()).items()))
    metadata = {
        "feature_pipeline": "quantaalpha",
        "label_pipeline": "quantaalpha",
        "quantaalpha_root": "vendored_in_qlib",
        "factor_cache_dir": str(cache_dir),
        "factor_count_requested": len(factors),
        "factor_count_loaded": len(post_feature.columns),
        "feature_rows": len(post_feature),
        "label_rows": len(post_label),
        "factor_source_counts": factor_source_counts,
        "cache_location_hit_count": diagnostics.cache_location_hit_count,
        "cache_hit_count": diagnostics.cache_hit_count,
        "compute_count": diagnostics.compute_count,
        "fail_count": diagnostics.fail_count,
        "skipped_count": diagnostics.skipped_count,
        "failed_factors": diagnostics.failed_factors,
        "missing_cache_locations": diagnostics.missing_cache_locations,
        **align_metadata,
    }

    print(
        "quantaalpha feature pipeline: "
        f"rows={len(post_feature)}, factors={len(post_feature.columns)}, "
        f"sources={factor_source_counts}",
        flush=True,
    )
    return post_feature, post_label, metadata


def _build_feature_label_frames(
    factors: List[dict],
    args,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if args.feature_pipeline == "legacy":
        field_panels = _load_base_fields(args.market, args.start_time, args.end_time)
        feature_df, label_df = _build_legacy_feature_label_frames(factors, field_panels)
        metadata = {
            "feature_pipeline": "legacy",
            "factor_count_requested": len(factors),
            "factor_count_loaded": len(feature_df.columns),
            "feature_rows": len(feature_df),
            "label_rows": len(label_df),
        }
        return feature_df, label_df, metadata

    return _build_quantaalpha_feature_label_frames(factors, args)


def _build_dataset_config(feature_df: pd.DataFrame, label_df: pd.DataFrame, args) -> dict:
    if args.feature_pipeline == "quantaalpha":
        learn_processors = []
        infer_processors = []
    else:
        learn_processors = [
            {"class": "DropnaLabel"},
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
        ]
        infer_processors = []

    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "DataHandlerLP",
                "module_path": "qlib.data.dataset.handler",
                "kwargs": {
                    "start_time": args.start_time,
                    "end_time": args.end_time,
                    "data_loader": {
                        "class": "StaticDataLoader",
                        "module_path": "qlib.data.dataset.loader",
                        "kwargs": {
                            "config": {
                                "feature": feature_df,
                                "label": label_df,
                            }
                        },
                    },
                    "learn_processors": learn_processors,
                    "infer_processors": infer_processors,
                },
            },
            "segments": {
                "train": [args.train_start, args.train_end],
                "valid": [args.valid_start, args.valid_end],
                "test": [args.test_start, args.test_end],
            },
        },
    }


def _build_quantaalpha_dataset(feature_df: pd.DataFrame, label_df: pd.DataFrame, args):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandler

    combined_df = pd.concat({"feature": feature_df, "label": label_df}, axis=1).sort_index()
    segments = {
        "train": (args.train_start, args.train_end),
        "valid": (args.valid_start, args.valid_end),
        "test": (args.test_start, args.test_end),
    }

    class PrecomputedDataHandler(DataHandler):
        def __init__(self, data_df, segment_map):
            self._data = data_df
            self._segments = segment_map

        @property
        def data_loader(self):
            return None

        @property
        def instruments(self):
            try:
                return list(self._data.index.get_level_values("instrument").unique())
            except KeyError:
                return list(self._data.index.get_level_values(1).unique())

        def fetch(
            self,
            selector=None,
            level="datetime",
            col_set="feature",
            data_key=None,
            squeeze=False,
            proc_func=None,
        ):
            if col_set in ("feature", "label"):
                result = self._data[col_set].copy()
            elif col_set == "__all" or col_set is None:
                result = self._data.copy()
            else:
                if isinstance(col_set, (list, tuple)):
                    result = self._data[list(col_set)].copy()
                else:
                    result = self._data.copy()

            if selector is not None:
                try:
                    dates = result.index.get_level_values("datetime")
                except KeyError:
                    dates = result.index.get_level_values(0)

                if isinstance(selector, (list, tuple)) and len(selector) == 2:
                    start, end = selector
                    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
                    result = result.loc[mask]
                elif isinstance(selector, slice):
                    start = selector.start
                    end = selector.stop
                    if start is not None and end is not None:
                        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
                        result = result.loc[mask]

            if squeeze and result.shape[1] == 1:
                result = result.iloc[:, 0]

            return result

        def get_cols(self, col_set="feature"):
            if col_set in self._data.columns.get_level_values(0):
                return list(self._data[col_set].columns)
            return list(self._data.columns.get_level_values(1))

        def setup_data(self, **kwargs):
            pass

        def config(self, **kwargs):
            pass

    handler = PrecomputedDataHandler(combined_df, segments)
    return DatasetH(handler=handler, segments=segments)


def _build_dataset(feature_df: pd.DataFrame, label_df: pd.DataFrame, args):
    if args.feature_pipeline == "quantaalpha":
        return _build_quantaalpha_dataset(feature_df, label_df, args)

    dataset_config = _build_dataset_config(feature_df, label_df, args)
    return init_instance_by_config(dataset_config)


def _build_model_config(args) -> dict:
    if args.feature_pipeline == "quantaalpha":
        model_config = copy.deepcopy(QUANTAALPHA_MODEL_CONFIG)
    else:
        model_config = copy.deepcopy(DEFAULT_MODEL_CONFIG)
    model_config["kwargs"]["num_threads"] = args.num_threads
    return model_config


def _build_port_analysis_config(args, model, dataset) -> dict:
    deal_price = args.deal_price or ("open" if args.feature_pipeline == "quantaalpha" else "close")
    return {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": args.topk,
                "n_drop": args.n_drop,
            },
        },
        "backtest": {
            "start_time": args.test_start,
            "end_time": args.test_end,
            "account": args.account,
            "benchmark": args.benchmark,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": deal_price,
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }


def _extract_signal_metrics(recorder, sig_analysis) -> dict:
    metrics = {}
    ic_series = sig_analysis.get("ic.pkl") if isinstance(sig_analysis, dict) else None
    ric_series = sig_analysis.get("ric.pkl") if isinstance(sig_analysis, dict) else None

    if ic_series is None or ric_series is None:
        try:
            ic_series = recorder.load_object("sig_analysis/ic.pkl")
            ric_series = recorder.load_object("sig_analysis/ric.pkl")
        except Exception:
            pass

    if isinstance(ic_series, pd.Series) and len(ic_series) > 0:
        ic_std = ic_series.std()
        metrics["IC"] = float(ic_series.mean())
        metrics["ICIR"] = float(ic_series.mean() / ic_std) if ic_std > 0 else 0.0

    if isinstance(ric_series, pd.Series) and len(ric_series) > 0:
        ric_std = ric_series.std()
        metrics["Rank IC"] = float(ric_series.mean())
        metrics["Rank ICIR"] = float(ric_series.mean() / ric_std) if ric_std > 0 else 0.0

    return metrics


def _filter_invalid_price_signals(
    pred: pd.Series,
    stock_list: List[str],
    start_time: str,
    end_time: str,
) -> Tuple[pd.Series, dict]:
    try:
        price_data = D.features(
            stock_list,
            ["$close"],
            start_time=start_time,
            end_time=end_time,
            freq="day",
        )
        price_data = _normalize_multiindex(price_data, "price")
        pred_df = _normalize_multiindex(pred.to_frame("score"), "prediction")
        invalid_mask = (price_data["$close"] == 0) | (price_data["$close"].isna())
        invalid_index = price_data.index[invalid_mask]
        filtered_index = pred_df.index.intersection(invalid_index)
        pred_df.loc[filtered_index, "score"] = np.nan
        return pred_df["score"], {
            "invalid_price_rows": int(invalid_mask.sum()),
            "filtered_signal_rows": int(len(filtered_index)),
        }
    except Exception as exc:
        return pred, {"price_filter_error": str(exc)}


def _run_quantaalpha_backtest(args, pred: pd.Series) -> Tuple[dict, dict]:
    from qlib.backtest import backtest as qlib_backtest
    from qlib.contrib.evaluate import risk_analysis

    market = D.instruments(args.market)
    stock_list = D.list_instruments(
        market,
        start_time=args.test_start,
        end_time=args.test_end,
        as_list=True,
    )
    filtered_pred, diagnostics = _filter_invalid_price_signals(
        pred,
        stock_list,
        args.test_start,
        args.test_end,
    )

    portfolio_metric_dict, _ = qlib_backtest(
        executor={
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
                "verbose": False,
                "indicator_config": {"show_indicator": False},
            },
        },
        strategy={
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {
                "signal": filtered_pred,
                "topk": args.topk,
                "n_drop": args.n_drop,
            },
        },
        start_time=args.test_start,
        end_time=args.test_end,
        account=args.account,
        benchmark=args.benchmark,
        exchange_kwargs={
            "codes": stock_list,
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": args.deal_price or "open",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    )

    metrics: dict[str, float] = {}
    artifacts = {}
    if portfolio_metric_dict and "1day" in portfolio_metric_dict:
        report_df, positions_df = portfolio_metric_dict["1day"]
        artifacts = {
            "report_normal_1day.pkl": report_df,
            "positions_normal_1day.pkl": positions_df,
        }
        if isinstance(report_df, pd.DataFrame) and "return" in report_df.columns:
            portfolio_return = report_df["return"].replace([np.inf, -np.inf], np.nan).fillna(0)
            bench_return = (
                report_df["bench"].replace([np.inf, -np.inf], np.nan).fillna(0)
                if "bench" in report_df.columns
                else 0
            )
            cost = (
                report_df["cost"].replace([np.inf, -np.inf], np.nan).fillna(0)
                if "cost" in report_df.columns
                else 0
            )
            excess_return_with_cost = (portfolio_return - bench_return - cost).dropna()
            if len(excess_return_with_cost) > 0:
                analysis = risk_analysis(excess_return_with_cost)
                if isinstance(analysis, pd.DataFrame):
                    analysis = analysis["risk"] if "risk" in analysis.columns else analysis.iloc[:, 0]

                ann_ret = float(analysis.get("annualized_return", 0))
                info_ratio = float(analysis.get("information_ratio", 0))
                max_dd = float(analysis.get("max_drawdown", 0))

                metrics["annualized_return"] = ann_ret
                metrics["information_ratio"] = info_ratio
                metrics["max_drawdown"] = max_dd
                metrics["1day.excess_return_with_cost.annualized_return"] = ann_ret
                metrics["1day.excess_return_with_cost.information_ratio"] = info_ratio
                metrics["1day.excess_return_with_cost.max_drawdown"] = max_dd

                if max_dd != 0 and not np.isnan(ann_ret) and not np.isinf(ann_ret):
                    metrics["calmar_ratio"] = ann_ret / abs(max_dd)

    return metrics, {
        **diagnostics,
        "stock_count": len(stock_list),
        "artifacts": list(artifacts.keys()),
        "artifact_payloads": artifacts,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train and backtest a LightGBM model with all factors library.")
    parser.add_argument(
        "--factor-json",
        default=str(Path(__file__).resolve().parents[3] / "all_factors_library.json"),
        help="Path to all_factors_library.json",
    )
    parser.add_argument(
        "--provider-uri",
        default=DEFAULT_PROVIDER_URI,
        help="Qlib provider URI. For `quantaalpha` pipeline, leaving the default lets the workflow auto-resolve "
        "QuantaAlpha's `.env` / local data path first.",
    )
    parser.add_argument("--region", default="cn")
    parser.add_argument("--market", default="csi300")
    parser.add_argument("--benchmark", default="SH000300")
    parser.add_argument(
        "--deal-price",
        choices=["open", "close"],
        default="",
        help="Override backtest deal price. By default, QuantaAlpha pipeline uses `open` and legacy uses `close`.",
    )
    parser.add_argument("--start-time", default="2016-01-01")
    parser.add_argument("--end-time", default="2025-12-26")
    parser.add_argument("--train-start", default="2016-01-01")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--valid-start", default="2021-01-01")
    parser.add_argument("--valid-end", default="2021-12-31")
    parser.add_argument("--test-start", default="2022-01-01")
    parser.add_argument("--test-end", default="2025-12-26")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--n-drop", type=int, default=5)
    parser.add_argument("--account", type=float, default=100000000.0)
    parser.add_argument("--exp-name", default="all_factors_library_lgbm")
    parser.add_argument("--uri-folder", default="mlruns")
    parser.add_argument("--num-threads", type=int, default=20)
    parser.add_argument(
        "--feature-pipeline",
        choices=["quantaalpha", "legacy"],
        default="quantaalpha",
        help="Feature generation path. `quantaalpha` uses the vendored QA-compatible factor pipeline inside qlib.",
    )
    parser.add_argument(
        "--quantaalpha-root",
        default=str(_default_quantaalpha_root()),
        help="Deprecated compatibility arg; the QA-compatible implementation is now vendored inside qlib.",
    )
    parser.add_argument(
        "--factor-cache-dir",
        default="",
        help="Optional factor MD5 cache directory. Leave empty to use qlib's vendored cache path.",
    )
    parser.add_argument(
        "--disable-factor-cache",
        action="store_true",
        help="Force recomputing factors instead of reading vendored factor caches.",
    )
    parser.add_argument(
        "--persist-factor-cache",
        action="store_true",
        help="Persist newly recomputed factors back into the vendored MD5 cache.",
    )
    parser.add_argument(
        "--feature-cache",
        default="",
        help="Optional pickle path for caching the merged feature/label dataframe",
    )
    parser.add_argument(
        "--output",
        default="all_factors_library_metrics.json",
        help="Path to save summary metrics json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output).resolve()
    deal_price = args.deal_price or ("open" if args.feature_pipeline == "quantaalpha" else "close")
    provider_uri, provider_source = _resolve_provider_uri(args)
    args.provider_uri = provider_uri

    exp_manager = copy.deepcopy(C["exp_manager"])
    exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / args.uri_folder)
    qlib.init(provider_uri=provider_uri, region=args.region, exp_manager=exp_manager)
    print(f"using provider_uri={provider_uri} (source={provider_source})", flush=True)

    factor_path = Path(args.factor_json).resolve()
    factors = _load_factor_library(factor_path)
    print(f"loaded {len(factors)} factors from {factor_path}", flush=True)

    feature_df, label_df, feature_metadata = _build_feature_label_frames(factors, args)

    if args.feature_cache:
        cache_path = Path(args.feature_cache).resolve()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat({"feature": feature_df, "label": label_df}, axis=1).to_pickle(cache_path)
        print(f"saved feature cache to {cache_path}", flush=True)

    model_config = _build_model_config(args)
    model = init_instance_by_config(model_config)
    dataset = _build_dataset(feature_df, label_df, args)

    port_analysis_config = _build_port_analysis_config(args, model, dataset)

    with R.start(experiment_name=args.exp_name):
        R.log_params(
            factor_count=len(factors),
            factor_json=str(factor_path),
            market=args.market,
            benchmark=args.benchmark,
            train_segment=f"{args.train_start}:{args.train_end}",
            valid_segment=f"{args.valid_start}:{args.valid_end}",
            test_segment=f"{args.test_start}:{args.test_end}",
            topk=args.topk,
            n_drop=args.n_drop,
            feature_pipeline=args.feature_pipeline,
            deal_price=deal_price,
            feature_rows=len(feature_df),
            feature_count=len(feature_df.columns),
        )
        model.fit(dataset)
        R.save_objects(**{"model.pkl": model})

        recorder = R.get_recorder()
        SignalRecord(model, dataset, recorder).generate()
        sig_analysis = SigAnaRecord(recorder, ana_long_short=False, ann_scaler=252).generate()

        backtest_diagnostics = {}
        manual_metrics = {}
        signal_metrics = _extract_signal_metrics(recorder, sig_analysis)
        if signal_metrics:
            recorder.log_metrics(**signal_metrics)
            manual_metrics.update(signal_metrics)

        if args.feature_pipeline == "quantaalpha":
            pred = model.predict(dataset)
            qa_metrics, backtest_diagnostics = _run_quantaalpha_backtest(args, pred)
            artifact_payloads = backtest_diagnostics.pop("artifact_payloads", {})
            if artifact_payloads:
                recorder.save_objects(**artifact_payloads)
            if qa_metrics:
                recorder.log_metrics(**qa_metrics)
                manual_metrics.update(qa_metrics)
        else:
            PortAnaRecord(recorder, port_analysis_config, "day").generate()

        metrics = recorder.list_metrics()
        merged_metrics = {**metrics, **manual_metrics}
        params = recorder.list_params()
        summary = {
            "run_id": recorder.id,
            "experiment_id": recorder.experiment_id,
            "factor_json": str(factor_path),
            "factor_count": len(factors),
            "feature_metadata": feature_metadata,
            "provider_uri": provider_uri,
            "provider_source": provider_source,
            "deal_price": deal_price,
            "model_config": model_config["kwargs"],
            "backtest_diagnostics": backtest_diagnostics,
            "segments": {
                "train": [args.train_start, args.train_end],
                "valid": [args.valid_start, args.valid_end],
                "test": [args.test_start, args.test_end],
            },
            "metrics": {
                "IC": merged_metrics.get("IC"),
                "ICIR": merged_metrics.get("ICIR"),
                "Rank IC": merged_metrics.get("Rank IC"),
                "Rank ICIR": merged_metrics.get("Rank ICIR"),
                "annualized_return": merged_metrics.get("annualized_return"),
                "information_ratio": merged_metrics.get("information_ratio"),
                "max_drawdown": merged_metrics.get("max_drawdown"),
                "1day.excess_return_with_cost.annualized_return": merged_metrics.get(
                    "1day.excess_return_with_cost.annualized_return"
                ),
                "1day.excess_return_with_cost.information_ratio": merged_metrics.get(
                    "1day.excess_return_with_cost.information_ratio"
                ),
                "1day.excess_return_with_cost.max_drawdown": merged_metrics.get(
                    "1day.excess_return_with_cost.max_drawdown"
                ),
            },
            "all_metrics": merged_metrics,
            "params": params,
            "artifacts": recorder.list_artifacts(),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

        print(f"saved metrics to {output_path}")
        print(f"run_id={recorder.id}")
        print(f"IC={summary['metrics']['IC']}")
        print(f"Rank IC={summary['metrics']['Rank IC']}")


if __name__ == "__main__":
    main()
