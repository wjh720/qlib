import argparse
import copy
import json
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd

import qlib
from qlib.config import C
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandler

import run_all_factors_library_workflow as workflow
import run_qa_trick_ablation as qa_ablation


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


class PrecomputedDataHandler(DataHandler):
    def __init__(self, data_df: pd.DataFrame, segments: dict):
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
    parser = argparse.ArgumentParser(description="Run a factor JSON with QA-style feature/label tricks.")
    parser.add_argument("--factor-json", default="all_factors_library_new.json")
    parser.add_argument("--provider-uri", default="")
    parser.add_argument("--region", default="cn")
    parser.add_argument("--market", default="csi300")
    parser.add_argument("--benchmark", default="SH000300")
    parser.add_argument("--start-time", default="2016-01-01")
    parser.add_argument("--end-time", default="2025-10-10")
    parser.add_argument("--train-start", default="2016-01-01")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--valid-start", default="2021-01-01")
    parser.add_argument("--valid-end", default="2021-12-31")
    parser.add_argument("--test-start", default="2022-01-01")
    parser.add_argument("--test-end", default="2025-10-10")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--n-drop", type=int, default=5)
    parser.add_argument("--account", type=float, default=100000000.0)
    parser.add_argument("--deal-price", choices=["open", "close"], default="open")
    parser.add_argument("--num-threads", type=int, default=20)
    parser.add_argument(
        "--feature-workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel factor generation.",
    )
    parser.add_argument("--uri-folder", default="mlruns_all_factors_library_new_qa_tricks")
    parser.add_argument("--exp-name", default="all_factors_library_new_lgbm_qa_tricks")
    parser.add_argument("--feature-cache", default="outputs/all_factors_library_new_qa_tricks/features.pkl")
    parser.add_argument("--output", default="outputs/all_factors_library_new_qa_tricks/metrics.json")
    parser.add_argument(
        "--feature-pipeline",
        choices=["quantaalpha", "legacy"],
        default="quantaalpha",
        help="Feature generation path. `quantaalpha` uses the vendored QA-compatible path inside qlib.",
    )
    parser.add_argument(
        "--quantaalpha-root",
        default=str(workflow._default_quantaalpha_root()),
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
    return parser.parse_args()


def resolve_provider_uri(args) -> tuple[str, str]:
    if args.feature_pipeline == "quantaalpha":
        args.provider_uri = args.provider_uri or DEFAULT_PROVIDER_URI
        return workflow._resolve_provider_uri(args)

    if args.provider_uri:
        return os.path.expanduser(args.provider_uri), "cli"

    candidate_jsons = [
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


def load_factor_library(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "factors" in payload:
        factors = []
        for factor_id, factor_info in payload["factors"].items():
            factor_payload = dict(factor_info)
            factor_payload.setdefault("factor_id", factor_id)
            factor_payload.setdefault("factor_name", factor_payload.get("factorName", factor_id))
            factor_payload.setdefault("factor_expression", factor_payload.get("factorExpression"))
            factors.append(factor_payload)
        return factors

    if isinstance(payload, list):
        factors = []
        for item in payload:
            factor_payload = dict(item)
            factor_payload["factor_id"] = item["factorId"]
            factor_payload["factor_name"] = item["factorName"]
            factor_payload["factor_expression"] = item["factorExpression"]
            factors.append(factor_payload)
        return factors

    raise ValueError(f"Unsupported factor JSON shape: {type(payload)}")


def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        raise TypeError(f"expected MultiIndex, got {type(df.index)}")
    names = list(df.index.names)
    if names != ["datetime", "instrument"]:
        df = df.copy()
        df.index = df.index.set_names(["datetime", "instrument"])
    return df.sort_index()


def centered_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level="datetime", group_keys=False).rank(pct=True) - 0.5


def apply_qa_feature_preprocess(feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = normalize_index(feature_df.copy())
    feature_df = feature_df.replace([np.inf, -np.inf], 0)
    feature_df = feature_df.fillna(0)
    feature_df = centered_rank(feature_df)
    return feature_df.astype(np.float32)


def apply_qa_label_preprocess(label_df: pd.DataFrame) -> pd.DataFrame:
    label_df = normalize_index(label_df.copy())
    label_df = label_df.loc[label_df.notna().all(axis=1)]
    label_df = centered_rank(label_df)
    return label_df.astype(np.float32)


def build_label_from_field_panel(field_panels: dict[str, pd.DataFrame], price_field: str) -> pd.DataFrame:
    label_panel = field_panels[price_field].shift(-2) / field_panels[price_field].shift(-1) - 1.0
    return workflow._stack_panel(label_panel, "LABEL0").to_frame().sort_index()


def _compute_feature_chunk(
    chunk_id: int,
    factors: list[dict],
    provider_uri: str,
    region: str,
    market: str,
    start_time: str,
    end_time: str,
    output_path: str,
) -> tuple[int, str]:
    qlib.init(provider_uri=provider_uri, region=region)
    field_panels = workflow._load_base_fields(market, start_time, end_time)
    feature_df, _ = workflow._build_legacy_feature_label_frames(factors, field_panels)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_pickle(output_path)
    return chunk_id, output_path


def build_feature_df(
    factors: list[dict],
    provider_uri: str,
    args,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    if args.feature_pipeline == "quantaalpha":
        feature_df, _unused_label_df, feature_metadata = workflow._build_quantaalpha_feature_label_frames(
            factors, args
        )
        field_panels = workflow._load_base_fields(args.market, args.start_time, args.end_time)
        return feature_df, field_panels, feature_metadata

    if args.feature_workers <= 1 or len(factors) <= 20:
        field_panels = workflow._load_base_fields(args.market, args.start_time, args.end_time)
        raw_feature_df, _ = workflow._build_legacy_feature_label_frames(factors, field_panels)
        return raw_feature_df, field_panels, {
            "feature_pipeline": "legacy",
            "factor_count_requested": len(factors),
            "factor_count_loaded": len(raw_feature_df.columns),
            "feature_rows_before_qa_preprocess": len(raw_feature_df),
        }

    field_panels = workflow._load_base_fields(args.market, args.start_time, args.end_time)
    chunk_size = math.ceil(len(factors) / args.feature_workers)
    chunks = [factors[idx : idx + chunk_size] for idx in range(0, len(factors), chunk_size)]
    tmp_dir = Path(args.feature_cache).resolve().parent / "feature_parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"building {len(factors)} factors with {len(chunks)} parallel chunks (workers={args.feature_workers})",
        flush=True,
    )

    from concurrent.futures import ProcessPoolExecutor

    part_paths: list[tuple[int, str]] = []
    with ProcessPoolExecutor(max_workers=min(args.feature_workers, len(chunks))) as executor:
        futures = []
        for chunk_id, chunk in enumerate(chunks):
            part_path = tmp_dir / f"part_{chunk_id:02d}.pkl"
            futures.append(
                executor.submit(
                    _compute_feature_chunk,
                    chunk_id,
                    chunk,
                    provider_uri,
                    args.region,
                    args.market,
                    args.start_time,
                    args.end_time,
                    str(part_path),
                )
            )

        for future in futures:
            chunk_id, path = future.result()
            print(f"finished feature chunk {chunk_id + 1}/{len(chunks)} -> {path}", flush=True)
            part_paths.append((chunk_id, path))

    raw_feature_df = pd.concat(
        [pd.read_pickle(path) for _, path in sorted(part_paths, key=lambda item: item[0])],
        axis=1,
    ).sort_index()
    return raw_feature_df, field_panels, {
        "feature_pipeline": "legacy",
        "factor_count_requested": len(factors),
        "factor_count_loaded": len(raw_feature_df.columns),
        "feature_rows_before_qa_preprocess": len(raw_feature_df),
        "feature_workers": args.feature_workers,
        "feature_chunks": len(chunks),
    }


def build_dataset(feature_df: pd.DataFrame, label_df: pd.DataFrame, args) -> DatasetH:
    common_index = feature_df.index.intersection(label_df.index)
    feature_df = feature_df.loc[common_index].sort_index()
    label_df = label_df.loc[common_index].sort_index()
    combined_df = pd.concat({"feature": feature_df, "label": label_df}, axis=1).sort_index()
    segments = {
        "train": (args.train_start, args.train_end),
        "valid": (args.valid_start, args.valid_end),
        "test": (args.test_start, args.test_end),
    }
    return DatasetH(handler=PrecomputedDataHandler(combined_df, segments), segments=segments)


def run_backtest(pred: pd.Series, args) -> tuple[dict, dict]:
    return qa_ablation.run_backtest(pred, args.deal_price, args)


def main():
    args = parse_args()
    output_path = Path(args.output).resolve()
    factor_path = Path(args.factor_json).resolve()
    provider_uri, provider_source = resolve_provider_uri(args)

    exp_manager = copy.deepcopy(C["exp_manager"])
    exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / args.uri_folder)
    qlib.init(provider_uri=provider_uri, region=args.region, exp_manager=exp_manager)
    print(f"using provider_uri={provider_uri} (source={provider_source})", flush=True)

    factors = load_factor_library(factor_path)
    print(f"loaded {len(factors)} factors from {factor_path}", flush=True)

    feature_df, field_panels, feature_metadata = build_feature_df(factors, provider_uri, args)
    raw_label_df = build_label_from_field_panel(field_panels, args.deal_price)
    if args.feature_pipeline == "legacy":
        feature_df = apply_qa_feature_preprocess(feature_df)
    label_df = apply_qa_label_preprocess(raw_label_df)

    common_index = feature_df.index.intersection(label_df.index)
    feature_df = feature_df.loc[common_index].sort_index()
    label_df = label_df.loc[common_index].sort_index()

    cache_path = Path(args.feature_cache).resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat({"feature": feature_df, "label": label_df}, axis=1).to_pickle(cache_path)
    print(f"saved feature cache to {cache_path}", flush=True)

    dataset = build_dataset(feature_df, label_df, args)
    model_config = copy.deepcopy(DEFAULT_MODEL_CONFIG)
    model_config["kwargs"]["num_threads"] = args.num_threads
    model = LGBModel(**model_config["kwargs"])
    model.fit(dataset)

    pred = model.predict(dataset)
    pred_df = pred.to_frame("score") if isinstance(pred, pd.Series) else pred.copy()
    signal_metrics = qa_ablation.compute_signal_metrics(pred_df, label_df, args)
    backtest_metrics, backtest_diagnostics = run_backtest(pred_df.iloc[:, 0], args)

    summary = {
        "factor_json": str(factor_path),
        "factor_count": len(factors),
        "provider_uri": provider_uri,
        "provider_source": provider_source,
        "feature_pipeline": args.feature_pipeline,
        "feature_metadata": {
            **feature_metadata,
            "feature_rows_after_label_align": int(len(feature_df)),
            "feature_count_after_label_align": int(feature_df.shape[1]),
        },
        "feature_preprocess": (
            "QuantaAlpha factor pipeline + QA cross-sectional rank(pct)-0.5"
            if args.feature_pipeline == "quantaalpha"
            else "Legacy evaluator + QA style: fillna(0), inf->0, cross-sectional rank(pct)-0.5"
        ),
        "label_preprocess": f"QA style on {args.deal_price} label: dropna then cross-sectional rank(pct)-0.5",
        "deal_price": args.deal_price,
        "segments": {
            "train": [args.train_start, args.train_end],
            "valid": [args.valid_start, args.valid_end],
            "test": [args.test_start, args.test_end],
        },
        "feature_count": int(feature_df.shape[1]),
        "feature_rows": int(len(feature_df)),
        "model_config": model_config["kwargs"],
        "backtest_diagnostics": backtest_diagnostics,
        "metrics": {
            **signal_metrics,
            **backtest_metrics,
        },
        "all_metrics": {
            **signal_metrics,
            **backtest_metrics,
        },
        "params": {
            "market": args.market,
            "benchmark": args.benchmark,
            "deal_price": args.deal_price,
            "feature_pipeline": args.feature_pipeline,
            "train_segment": f"{args.train_start}:{args.train_end}",
            "valid_segment": f"{args.valid_start}:{args.valid_end}",
            "test_segment": f"{args.test_start}:{args.test_end}",
            "feature_preprocess": (
                "quantaalpha_factor_pipeline"
                if args.feature_pipeline == "quantaalpha"
                else "fillna0+inf0+csranks(pct)-0.5"
            ),
            "label_preprocess": f"{args.deal_price}_label+dropna+csranks(pct)-0.5",
            "factor_count": str(len(factors)),
            "feature_count": str(feature_df.shape[1]),
            "feature_rows": str(len(feature_df)),
            "factor_json": str(factor_path),
            "cmd-sys.argv": " ".join(os.sys.argv),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"saved metrics to {output_path}")
    print(f"IC={summary['metrics']['IC']}")
    print(f"Rank IC={summary['metrics']['Rank IC']}")


if __name__ == "__main__":
    main()
