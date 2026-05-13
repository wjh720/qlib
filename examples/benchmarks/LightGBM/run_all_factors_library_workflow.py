import argparse
import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import qlib
from qlib.config import C
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord


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
    mean_x = TS_MEAN(x, n)
    mean_y = TS_MEAN(y, n)
    return TS_MEAN(x * y, n) - mean_x * mean_y


def TS_CORR(x, y, n):
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


def SMA(x, n):
    return TS_MEAN(x, n)


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


def IF(cond, left, right):
    return left.where(cond, right)


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
    factors = list(payload["factors"].values())
    factors.sort(key=lambda item: item["factor_name"])
    return factors


def _load_base_fields(instruments: str, start_time: str, end_time: str) -> Dict[str, pd.DataFrame]:
    instruments = D.instruments(instruments)
    raw = D.features(instruments, ["$open", "$close", "$high", "$low", "$volume"], start_time, end_time)
    raw = _to_datetime_instrument_index(raw)

    fields = {}
    for col in raw.columns:
        panel = raw[col].unstack("instrument").sort_index()
        fields[col[1:]] = panel
    fields["return"] = fields["close"] / fields["close"].shift(1) - 1.0
    return fields


def _build_feature_label_frames(
    factors: List[dict],
    field_panels: Dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    env = {
        "__builtins__": {},
        "FIELDS": field_panels,
        "ABS": ABS,
        "COUNT": COUNT,
        "DECAYLINEAR": DECAYLINEAR,
        "DELAY": DELAY,
        "DELTA": DELTA,
        "EMA": EMA,
        "EXP": EXP,
        "HIGHDAY": HIGHDAY,
        "IF": IF,
        "INV": INV,
        "LOG": LOG,
        "MACD": MACD,
        "MEAN": MEAN,
        "PERCENTILE": PERCENTILE,
        "RANK": RANK,
        "REGBETA": REGBETA,
        "REGRESI": REGRESI,
        "RSI": RSI,
        "SEQUENCE": SEQUENCE,
        "SIGN": SIGN,
        "SMA": SMA,
        "STD": STD,
        "SUMAC": SUMAC,
        "SUMIF": SUMIF,
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


def _build_dataset_config(feature_df: pd.DataFrame, label_df: pd.DataFrame, args) -> dict:
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
                    "learn_processors": [
                        {"class": "DropnaLabel"},
                        {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
                    ],
                },
            },
            "segments": {
                "train": [args.train_start, args.train_end],
                "valid": [args.valid_start, args.valid_end],
                "test": [args.test_start, args.test_end],
            },
        },
    }


def _build_port_analysis_config(args, model, dataset) -> dict:
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
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train and backtest a LightGBM model with all factors library.")
    parser.add_argument(
        "--factor-json",
        default=str(Path(__file__).resolve().parents[3] / "all_factors_library.json"),
        help="Path to all_factors_library.json",
    )
    parser.add_argument("--provider-uri", default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--region", default="cn")
    parser.add_argument("--market", default="csi300")
    parser.add_argument("--benchmark", default="SH000300")
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

    exp_manager = copy.deepcopy(C["exp_manager"])
    exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / args.uri_folder)
    qlib.init(provider_uri=args.provider_uri, region=args.region, exp_manager=exp_manager)

    factor_path = Path(args.factor_json).resolve()
    factors = _load_factor_library(factor_path)
    print(f"loaded {len(factors)} factors from {factor_path}", flush=True)

    field_panels = _load_base_fields(args.market, args.start_time, args.end_time)
    feature_df, label_df = _build_feature_label_frames(factors, field_panels)

    if args.feature_cache:
        cache_path = Path(args.feature_cache).resolve()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat({"feature": feature_df, "label": label_df}, axis=1).to_pickle(cache_path)
        print(f"saved feature cache to {cache_path}", flush=True)

    model_config = copy.deepcopy(DEFAULT_MODEL_CONFIG)
    model_config["kwargs"]["num_threads"] = args.num_threads

    dataset_config = _build_dataset_config(feature_df, label_df, args)
    model = init_instance_by_config(model_config)
    dataset = init_instance_by_config(dataset_config)

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
        )
        model.fit(dataset)
        R.save_objects(**{"model.pkl": model})

        recorder = R.get_recorder()
        SignalRecord(model, dataset, recorder).generate()
        SigAnaRecord(recorder, ana_long_short=False, ann_scaler=252).generate()
        PortAnaRecord(recorder, port_analysis_config, "day").generate()

        metrics = recorder.list_metrics()
        params = recorder.list_params()
        summary = {
            "run_id": recorder.id,
            "experiment_id": recorder.experiment_id,
            "factor_json": str(factor_path),
            "factor_count": len(factors),
            "segments": {
                "train": [args.train_start, args.train_end],
                "valid": [args.valid_start, args.valid_end],
                "test": [args.test_start, args.test_end],
            },
            "metrics": {
                "IC": metrics.get("IC"),
                "ICIR": metrics.get("ICIR"),
                "Rank IC": metrics.get("Rank IC"),
                "Rank ICIR": metrics.get("Rank ICIR"),
                "1day.excess_return_with_cost.annualized_return": metrics.get(
                    "1day.excess_return_with_cost.annualized_return"
                ),
                "1day.excess_return_with_cost.information_ratio": metrics.get(
                    "1day.excess_return_with_cost.information_ratio"
                ),
                "1day.excess_return_with_cost.max_drawdown": metrics.get(
                    "1day.excess_return_with_cost.max_drawdown"
                ),
            },
            "all_metrics": metrics,
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
