import argparse
import copy
import gc
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import qlib
from qlib.backtest import backtest as qlib_backtest
from qlib.backtest.profit_attribution import get_stock_weight_df
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.eva.alpha import calc_ic
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.utils.data import zscore
from qlib.utils import init_instance_by_config


DEFAULT_PROVIDER_URI = "~/.qlib/qlib_data/cn_data"
DEFAULT_LABEL_PRICE = "close"
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
class RunSpec:
    family: str
    feature_qa: bool
    label_qa: bool
    deal_price: str

    @property
    def run_name(self) -> str:
        return (
            f"{self.family}"
            f"__feature_{'qa' if self.feature_qa else 'raw'}"
            f"__label_{'qa' if self.label_qa else 'default'}"
            f"__deal_{self.deal_price}"
        )


class PrecomputedDataHandler(DataHandler):
    def __init__(self, data_df: pd.DataFrame, segments: Dict[str, tuple[str, str]]):
        self._data = data_df
        self._segments = segments

    @property
    def data_loader(self):
        return None

    @property
    def instruments(self):
        return list(self._data.index.get_level_values("instrument").unique())

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
            dates = result.index.get_level_values("datetime")
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run 2^3 QA trick ablation on Alpha158 and 177-factor library.")
    parser.add_argument("--provider-uri", default="", help="Override provider URI. Leave empty to auto-resolve.")
    parser.add_argument("--region", default="cn")
    parser.add_argument("--market", default="csi300")
    parser.add_argument("--benchmark", default="SH000300")
    parser.add_argument("--start-time", default="2016-01-01")
    parser.add_argument("--train-start", default="2016-01-01")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--valid-start", default="2021-01-01")
    parser.add_argument("--valid-end", default="2021-12-31")
    parser.add_argument("--test-start", default="2022-01-01")
    parser.add_argument(
        "--test-end",
        default="",
        help="Leave empty to auto-truncate to the latest date shared by the cached 177-factor feature matrices.",
    )
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--n-drop", type=int, default=5)
    parser.add_argument("--account", type=float, default=100000000.0)
    parser.add_argument("--num-threads", type=int, default=20)
    parser.add_argument(
        "--legacy-cache",
        default="outputs/my_all_factors_run/features.pkl",
        help="Combined legacy 177-factor feature/label cache.",
    )
    parser.add_argument(
        "--qa-feature-cache",
        default="outputs/all_factors_library_quantaalpha_feature.pkl",
        help="Combined QA-feature/raw-label 177-factor cache.",
    )
    parser.add_argument(
        "--provider-metrics-json",
        default="outputs/all_factors_library_quantaalpha_feature_metrics.json",
        help="JSON used to discover the provider URI from previous 177-factor QA runs.",
    )
    parser.add_argument(
        "--output",
        default="outputs/qa_trick_ablation_summary.json",
        help="Path to save the ablation summary JSON.",
    )
    parser.add_argument(
        "--label-follows-deal-price",
        action="store_true",
        help="When enabled, open/close runs also switch the raw label from close-to-close to open-to-open.",
    )
    return parser.parse_args()


def resolve_provider_uri(args) -> tuple[str, str]:
    if args.provider_uri:
        return os.path.expanduser(args.provider_uri), "cli"

    candidate_jsons = [
        Path(args.provider_metrics_json),
        Path("outputs/all_factors_library_quantaalpha_full_align_metrics.json"),
        Path("outputs/qlib_qa_full_strict_metrics.json"),
        Path("outputs/qlib_qa_split_backtest_metrics.json"),
    ]
    for metrics_path in candidate_jsons:
        if metrics_path.exists():
            try:
                payload = json.loads(metrics_path.read_text())
                provider_uri = payload.get("provider_uri")
                provider_source = payload.get("provider_source", str(metrics_path))
                if provider_uri:
                    return os.path.expanduser(provider_uri), provider_source
            except Exception:
                pass

    qa_bundled = Path("/mnt/data2/wjh/code/QuantaAlpha/data/qlib/cn_data")
    if qa_bundled.exists():
        return str(qa_bundled), "quantaalpha_repo"

    return os.path.expanduser(DEFAULT_PROVIDER_URI), "default"


def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        raise TypeError(f"expected MultiIndex, got {type(df.index)}")
    if df.index.nlevels != 2:
        raise TypeError(f"expected 2-level MultiIndex, got {df.index.nlevels}")

    level0 = df.index.get_level_values(0)
    level1 = df.index.get_level_values(1)
    level0_is_dt = pd.api.types.is_datetime64_any_dtype(level0)
    level1_is_dt = pd.api.types.is_datetime64_any_dtype(level1)

    if (not level0_is_dt) and level1_is_dt:
        df = df.copy()
        df.index = df.index.swaplevel(0, 1)

    df = df.copy()
    df.index = df.index.set_names(["datetime", "instrument"])
    return df.sort_index()


def to_float32(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(np.float32, copy=False)


def centered_rank(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.groupby(level="datetime", group_keys=False).rank(pct=True)
    ranked = ranked - 0.5
    return to_float32(ranked)


def cs_zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.groupby(level="datetime", group_keys=False).apply(zscore)
    return to_float32(normalized)


def preprocess_label(label_raw: pd.DataFrame, use_qa_rank: bool) -> pd.DataFrame:
    label = normalize_index(label_raw)
    label = label.loc[label.notna().all(axis=1)].copy()
    if use_qa_rank:
        return centered_rank(label)
    return cs_zscore_df(label)


def build_label_expression(price_field: str) -> str:
    return f"Ref(${price_field}, -2)/Ref(${price_field}, -1) - 1"


def preprocess_alpha158_feature(feature_raw: pd.DataFrame, use_qa: bool) -> pd.DataFrame:
    feature = normalize_index(feature_raw)
    if not use_qa:
        return to_float32(feature.copy())

    feature = feature.copy()
    feature = feature.replace([np.inf, -np.inf], 0)
    feature = feature.fillna(0)
    feature = centered_rank(feature)
    return to_float32(feature)


def preprocess_177_feature(feature_df: pd.DataFrame) -> pd.DataFrame:
    return to_float32(normalize_index(feature_df.copy()))


def align_feature_label(feature_df: pd.DataFrame, label_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_df = normalize_index(feature_df)
    label_df = normalize_index(label_df)
    common_index = feature_df.index.intersection(label_df.index)
    feature_df = feature_df.loc[common_index].sort_index()
    label_df = label_df.loc[common_index].sort_index()
    return to_float32(feature_df), to_float32(label_df)


def slice_dates(df: pd.DataFrame, start_time: str, end_time: str) -> pd.DataFrame:
    dates = df.index.get_level_values("datetime")
    mask = (dates >= pd.Timestamp(start_time)) & (dates <= pd.Timestamp(end_time))
    return df.loc[mask].sort_index()


def build_dataset(feature_df: pd.DataFrame, label_df: pd.DataFrame, args) -> DatasetH:
    combined_df = pd.concat({"feature": feature_df, "label": label_df}, axis=1).sort_index()
    segments = {
        "train": (args.train_start, args.train_end),
        "valid": (args.valid_start, args.valid_end),
        "test": (args.test_start, args.test_end),
    }
    return DatasetH(handler=PrecomputedDataHandler(combined_df, segments), segments=segments)


def load_alpha158_raw(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    handler = Alpha158(
        instruments=args.market,
        start_time=args.start_time,
        end_time=args.test_end,
        fit_start_time=args.train_start,
        fit_end_time=args.train_end,
        infer_processors=[],
        learn_processors=[],
    )
    feature_df = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_R)
    label_df = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_R)
    feature_df = slice_dates(feature_df, args.start_time, args.test_end)
    label_df = slice_dates(label_df, args.start_time, args.test_end)
    return to_float32(feature_df), to_float32(label_df)


def load_market_label(args, price_field: str) -> pd.DataFrame:
    stock_list = D.instruments(args.market)
    label_df = D.features(
        stock_list,
        [build_label_expression(price_field)],
        args.start_time,
        args.test_end,
        freq="day",
    )
    label_df.columns = ["LABEL0"]
    label_df = slice_dates(label_df, args.start_time, args.test_end)
    return to_float32(label_df)


def load_177_cache(path: str, start_time: str, end_time: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.read_pickle(path)
    combined = normalize_index(combined)
    combined = slice_dates(combined, start_time, end_time)
    feature_df = combined["feature"].copy()
    label_df = combined["label"].copy()
    return to_float32(feature_df), to_float32(label_df)


def choose_shared_test_end(args) -> str:
    if args.test_end:
        return args.test_end

    qa_feature_dates = pd.read_pickle(args.qa_feature_cache).index.get_level_values("datetime")
    legacy_dates = pd.read_pickle(args.legacy_cache).index.get_level_values("datetime")
    shared_end = min(qa_feature_dates.max(), legacy_dates.max())
    return pd.Timestamp(shared_end).strftime("%Y-%m-%d")


def build_model(args):
    model_config = copy.deepcopy(DEFAULT_MODEL_CONFIG)
    model_config["kwargs"]["num_threads"] = args.num_threads
    return init_instance_by_config(model_config)


def filter_invalid_price_signals(
    pred: pd.Series,
    stock_list: List[str],
    start_time: str,
    end_time: str,
    deal_price: str,
) -> tuple[pd.Series, dict]:
    price_field = f"${deal_price}"
    price_data = D.features(
        stock_list,
        [price_field],
        start_time=start_time,
        end_time=end_time,
        freq="day",
    )
    price_data = normalize_index(price_data)
    pred_df = normalize_index(pred.to_frame("score"))
    invalid_mask = price_data[price_field].isna() | (price_data[price_field] == 0)
    invalid_index = price_data.index[invalid_mask]
    filtered_index = pred_df.index.intersection(invalid_index)
    pred_df.loc[filtered_index, "score"] = np.nan
    return pred_df["score"], {
        "invalid_price_field": price_field,
        "invalid_price_rows": int(invalid_mask.sum()),
        "filtered_signal_rows": int(len(filtered_index)),
    }


def get_next_trade_date(date: str) -> pd.Timestamp:
    date_ts = pd.Timestamp(date)
    calendar = D.calendar(start_time=date_ts, future=True, freq="day")
    for cal_date in calendar:
        if cal_date > date_ts:
            return pd.Timestamp(cal_date)
    return date_ts


def build_deal_to_deal_return_frame(instruments: List[str], deal_price: str, start_time: str, end_time: str) -> pd.DataFrame:
    fetch_end = get_next_trade_date(end_time)
    price_field = f"${deal_price}"
    price_df = D.features(
        instruments,
        [price_field],
        start_time=start_time,
        end_time=fetch_end,
        freq="day",
    )
    price_df = normalize_index(price_df)
    price_panel = price_df[price_field].unstack(level="instrument").sort_index().ffill()
    return price_panel.pct_change().shift(-1).loc[pd.Timestamp(start_time) : pd.Timestamp(end_time)]


def compute_deal_to_deal_metrics(
    positions: dict,
    report_df: pd.DataFrame,
    benchmark: str,
    deal_price: str,
    args,
) -> tuple[dict[str, float], dict[str, int]]:
    stock_weight_df = get_stock_weight_df(positions).sort_index()
    stock_weight_df.index = pd.to_datetime(stock_weight_df.index)
    stock_weight_df = stock_weight_df.loc[pd.Timestamp(args.test_start) : pd.Timestamp(args.test_end)]
    stock_weight_df = stock_weight_df.fillna(0)

    instruments = sorted(stock_weight_df.columns)
    if len(instruments) == 0:
        return {}, {"portfolio_return_rows": 0, "benchmark_return_rows": 0}

    stock_ret = build_deal_to_deal_return_frame(instruments, deal_price, args.test_start, args.test_end)
    stock_ret = stock_ret.reindex(index=stock_weight_df.index, columns=stock_weight_df.columns)
    stock_ret = stock_ret.fillna(0)

    portfolio_return = stock_weight_df.mul(stock_ret).sum(axis=1)

    bench_ret_df = build_deal_to_deal_return_frame([benchmark], deal_price, args.test_start, args.test_end)
    bench_ret = bench_ret_df.iloc[:, 0].reindex(stock_weight_df.index).fillna(0)

    cost = (
        report_df["cost"].copy()
        if isinstance(report_df, pd.DataFrame) and "cost" in report_df.columns
        else pd.Series(0, index=stock_weight_df.index, dtype="float64")
    )
    cost.index = pd.to_datetime(cost.index)
    cost = cost.reindex(stock_weight_df.index).fillna(0)

    excess_return_with_cost = (portfolio_return - bench_ret - cost).dropna()
    if len(excess_return_with_cost) == 0:
        return {}, {"portfolio_return_rows": 0, "benchmark_return_rows": 0}

    analysis = risk_analysis(excess_return_with_cost)
    analysis = analysis["risk"] if "risk" in analysis.columns else analysis.iloc[:, 0]
    metrics = {
        "1day.excess_return_with_cost.annualized_return": float(analysis.get("annualized_return", np.nan)),
        "1day.excess_return_with_cost.information_ratio": float(analysis.get("information_ratio", np.nan)),
        "1day.excess_return_with_cost.max_drawdown": float(analysis.get("max_drawdown", np.nan)),
    }
    diagnostics = {
        "portfolio_return_rows": int(len(portfolio_return)),
        "benchmark_return_rows": int(len(bench_ret)),
    }
    return metrics, diagnostics


def run_backtest(pred: pd.Series, deal_price: str, args) -> tuple[dict, dict]:
    market = D.instruments(args.market)
    stock_list = D.list_instruments(
        market,
        start_time=args.test_start,
        end_time=args.test_end,
        as_list=True,
    )
    filtered_pred, diagnostics = filter_invalid_price_signals(
        pred,
        stock_list,
        args.test_start,
        args.test_end,
        deal_price,
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
            "deal_price": deal_price,
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    )

    metrics: dict[str, float] = {}
    if portfolio_metric_dict and "1day" in portfolio_metric_dict:
        report_df, positions = portfolio_metric_dict["1day"]
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
                analysis = analysis["risk"] if "risk" in analysis.columns else analysis.iloc[:, 0]
                diagnostics["qlib_mark_to_close.annualized_return"] = float(analysis.get("annualized_return", np.nan))
                diagnostics["qlib_mark_to_close.information_ratio"] = float(analysis.get("information_ratio", np.nan))
                diagnostics["qlib_mark_to_close.max_drawdown"] = float(analysis.get("max_drawdown", np.nan))
        if positions:
            deal_to_deal_metrics, deal_to_deal_diag = compute_deal_to_deal_metrics(
                positions=positions,
                report_df=report_df,
                benchmark=args.benchmark,
                deal_price=deal_price,
                args=args,
            )
            metrics.update(deal_to_deal_metrics)
            diagnostics.update({f"deal_to_deal.{k}": v for k, v in deal_to_deal_diag.items()})
            diagnostics["return_calculation"] = f"{deal_price}_to_{deal_price}"
    return metrics, diagnostics


def compute_signal_metrics(pred_df: pd.DataFrame, label_df: pd.DataFrame, args) -> dict:
    label_test = slice_dates(label_df, args.test_start, args.test_end)
    pred_df = normalize_index(pred_df)
    label_test = normalize_index(label_test)
    common_index = pred_df.index.intersection(label_test.index)
    pred_s = pred_df.loc[common_index].iloc[:, 0]
    label_s = label_test.loc[common_index].iloc[:, 0]
    ic, ric = calc_ic(pred_s, label_s, dropna=True)
    ic_std = ic.std() if len(ic) > 1 else np.nan
    ric_std = ric.std() if len(ric) > 1 else np.nan
    metrics = {
        "IC": float(ic.mean()) if len(ic) > 0 else np.nan,
        "ICIR": float(ic.mean() / ic_std) if pd.notna(ic_std) and ic_std != 0 else np.nan,
        "Rank IC": float(ric.mean()) if len(ric) > 0 else np.nan,
        "Rank ICIR": float(ric.mean() / ric_std) if pd.notna(ric_std) and ric_std != 0 else np.nan,
    }
    return metrics


def run_single_spec(spec: RunSpec, feature_df: pd.DataFrame, label_raw: pd.DataFrame, args) -> dict:
    print(f"[run] {spec.run_name}", flush=True)
    label_df = preprocess_label(label_raw, use_qa_rank=spec.label_qa)
    feature_df, label_df = align_feature_label(feature_df, label_df)
    dataset = build_dataset(feature_df, label_df, args)
    model = build_model(args)
    model.fit(dataset)
    pred = model.predict(dataset)
    pred_df = pred.to_frame("score") if isinstance(pred, pd.Series) else pred.copy()
    signal_metrics = compute_signal_metrics(pred_df, label_df, args)
    backtest_metrics, backtest_diag = run_backtest(pred_df.iloc[:, 0], spec.deal_price, args)
    metrics = {**signal_metrics, **backtest_metrics}
    result = {
        **asdict(spec),
        "run_name": spec.run_name,
        "feature_rows": int(len(feature_df)),
        "feature_count": int(feature_df.shape[1]),
        "metrics": metrics,
        "backtest_diagnostics": backtest_diag,
    }
    del dataset, model, pred, pred_df, label_df
    gc.collect()
    return result


def metric_value(run: dict, metric_name: str) -> float:
    value = run["metrics"].get(metric_name)
    return float(value) if value is not None else np.nan


def compute_main_effects(runs: List[dict]) -> dict:
    metrics = [
        "IC",
        "Rank IC",
        "1day.excess_return_with_cost.annualized_return",
        "1day.excess_return_with_cost.information_ratio",
        "1day.excess_return_with_cost.max_drawdown",
    ]
    effect_map = {
        "feature_qa": {},
        "label_qa": {},
        "deal_price_open": {},
    }
    for metric_name in metrics:
        feature_on = [metric_value(r, metric_name) for r in runs if r["feature_qa"]]
        feature_off = [metric_value(r, metric_name) for r in runs if not r["feature_qa"]]
        label_on = [metric_value(r, metric_name) for r in runs if r["label_qa"]]
        label_off = [metric_value(r, metric_name) for r in runs if not r["label_qa"]]
        open_on = [metric_value(r, metric_name) for r in runs if r["deal_price"] == "open"]
        close_off = [metric_value(r, metric_name) for r in runs if r["deal_price"] == "close"]

        effect_map["feature_qa"][metric_name] = float(np.nanmean(feature_on) - np.nanmean(feature_off))
        effect_map["label_qa"][metric_name] = float(np.nanmean(label_on) - np.nanmean(label_off))
        effect_map["deal_price_open"][metric_name] = float(np.nanmean(open_on) - np.nanmean(close_off))
    return effect_map


def run_family_alpha158(args) -> List[dict]:
    raw_feature, raw_label_close = load_alpha158_raw(args)
    label_by_price = {DEFAULT_LABEL_PRICE: raw_label_close}
    if args.label_follows_deal_price:
        label_by_price["open"] = load_market_label(args, "open")
    else:
        label_by_price["open"] = raw_label_close
    prepared_features = {
        False: preprocess_alpha158_feature(raw_feature, use_qa=False),
        True: preprocess_alpha158_feature(raw_feature, use_qa=True),
    }
    results = []
    for feature_qa in [False, True]:
        for label_qa in [False, True]:
            for deal_price in ["close", "open"]:
                spec = RunSpec("alpha158", feature_qa, label_qa, deal_price)
                results.append(
                    run_single_spec(spec, prepared_features[feature_qa], label_by_price[deal_price], args)
                )
    return results


def run_family_177(args) -> List[dict]:
    legacy_feature, legacy_label_close = load_177_cache(args.legacy_cache, args.start_time, args.test_end)
    qa_feature, _qa_label_unused = load_177_cache(args.qa_feature_cache, args.start_time, args.test_end)

    legacy_label_close = legacy_label_close.loc[legacy_label_close.notna().all(axis=1)].copy()
    label_by_price = {DEFAULT_LABEL_PRICE: legacy_label_close}
    if args.label_follows_deal_price:
        label_by_price["open"] = load_market_label(args, "open")
    else:
        label_by_price["open"] = legacy_label_close

    prepared_features = {
        False: preprocess_177_feature(legacy_feature),
        True: preprocess_177_feature(qa_feature),
    }

    results = []
    for feature_qa in [False, True]:
        for label_qa in [False, True]:
            for deal_price in ["close", "open"]:
                spec = RunSpec("factors177", feature_qa, label_qa, deal_price)
                results.append(
                    run_single_spec(spec, prepared_features[feature_qa], label_by_price[deal_price], args)
                )
    return results


def build_summary(alpha_runs: List[dict], factor_runs: List[dict], args, provider_uri: str, provider_source: str) -> dict:
    datasets = {
        "alpha158": {
            "runs": alpha_runs,
            "main_effects": compute_main_effects(alpha_runs),
        },
        "factors177": {
            "runs": factor_runs,
            "main_effects": compute_main_effects(factor_runs),
        },
    }

    for dataset_name, payload in datasets.items():
        runs = payload["runs"]
        payload["best_by_ann_ret"] = max(
            runs,
            key=lambda item: metric_value(item, "1day.excess_return_with_cost.annualized_return"),
        )
        payload["best_by_rank_ic"] = max(runs, key=lambda item: metric_value(item, "Rank IC"))
        payload["baseline_close_close"] = next(
            item
            for item in runs
            if (not item["feature_qa"]) and (not item["label_qa"]) and item["deal_price"] == "close"
        )
        payload["full_qa_open"] = next(
            item
            for item in runs
            if item["feature_qa"] and item["label_qa"] and item["deal_price"] == "open"
        )

    return {
        "provider_uri": provider_uri,
        "provider_source": provider_source,
        "segments": {
            "train": [args.train_start, args.train_end],
            "valid": [args.valid_start, args.valid_end],
            "test": [args.test_start, args.test_end],
        },
        "settings": {
            "market": args.market,
            "benchmark": args.benchmark,
            "topk": args.topk,
            "n_drop": args.n_drop,
            "account": args.account,
            "num_threads": args.num_threads,
            "label_follows_deal_price": args.label_follows_deal_price,
            "label_expression_by_deal_price": {
                "close": build_label_expression("close"),
                "open": build_label_expression("open") if args.label_follows_deal_price else build_label_expression("close"),
            },
            "legacy_cache": str(Path(args.legacy_cache).resolve()),
            "qa_feature_cache": str(Path(args.qa_feature_cache).resolve()),
        },
        "datasets": datasets,
    }


def main():
    args = parse_args()
    args.test_end = choose_shared_test_end(args)
    provider_uri, provider_source = resolve_provider_uri(args)
    qlib.init(provider_uri=provider_uri, region=args.region)
    print(f"using provider_uri={provider_uri} (source={provider_source})", flush=True)
    print(
        f"segments: train={args.train_start}:{args.train_end}, "
        f"valid={args.valid_start}:{args.valid_end}, test={args.test_start}:{args.test_end}",
        flush=True,
    )
    if args.label_follows_deal_price:
        print("label mode: deal_price-aware (close=>close-to-close, open=>open-to-open)", flush=True)
    else:
        print("label mode: fixed close-to-close for both close/open runs", flush=True)

    alpha_runs = run_family_alpha158(args)
    factor_runs = run_family_177(args)
    summary = build_summary(alpha_runs, factor_runs, args, provider_uri, provider_source)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"saved ablation summary to {output_path}", flush=True)


if __name__ == "__main__":
    main()
