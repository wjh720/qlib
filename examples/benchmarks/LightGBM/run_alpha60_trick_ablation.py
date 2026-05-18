import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import qlib

import run_all_factors_library_workflow as workflow
import run_qa_trick_ablation as qa_ablation


def parse_args():
    parser = argparse.ArgumentParser(description="Run 2^3 trick ablation on a factor JSON feature set.")
    parser.add_argument("--factor-json", default="all_factors_library_new.json")
    parser.add_argument(
        "--dataset-name",
        default="",
        help="Dataset label used in run names and summary output; defaults to factor JSON stem.",
    )
    parser.add_argument("--provider-uri", default="")
    parser.add_argument("--provider-metrics-json", default="outputs/all_factors_library_quantaalpha_feature_metrics.json")
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
    parser.add_argument("--num-threads", type=int, default=20)
    parser.add_argument(
        "--label-follows-deal-price",
        action="store_true",
        default=True,
        help="Use close-to-close label for deal=close and open-to-open label for deal=open.",
    )
    parser.add_argument(
        "--output",
        default="outputs/alpha60_trick_ablation/summary.json",
        help="Path to save the ablation summary JSON.",
    )
    return parser.parse_args()


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


def build_label_from_field_panel(field_panels: dict[str, pd.DataFrame], price_field: str) -> pd.DataFrame:
    label_panel = field_panels[price_field].shift(-2) / field_panels[price_field].shift(-1) - 1.0
    return workflow._stack_panel(label_panel, "LABEL0").to_frame().sort_index()


def preprocess_raw_feature(feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = qa_ablation.normalize_index(feature_df.copy())
    return qa_ablation.to_float32(feature_df)


def preprocess_qa_feature(feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = qa_ablation.normalize_index(feature_df.copy())
    feature_df = feature_df.replace([np.inf, -np.inf], 0)
    feature_df = feature_df.fillna(0)
    feature_df = qa_ablation.centered_rank(feature_df)
    return qa_ablation.to_float32(feature_df)


def build_summary(
    runs: list[dict],
    args,
    provider_uri: str,
    provider_source: str,
    factor_json: str,
    factor_count: int,
    dataset_name: str,
) -> dict:
    payload = {
        "runs": runs,
        "main_effects": qa_ablation.compute_main_effects(runs),
    }
    payload["best_by_ann_ret"] = max(
        runs,
        key=lambda item: qa_ablation.metric_value(item, "1day.excess_return_with_cost.annualized_return"),
    )
    payload["best_by_rank_ic"] = max(runs, key=lambda item: qa_ablation.metric_value(item, "Rank IC"))
    payload["baseline_close_close"] = next(
        item for item in runs if (not item["feature_qa"]) and (not item["label_qa"]) and item["deal_price"] == "close"
    )
    payload["full_qa_open"] = next(
        item for item in runs if item["feature_qa"] and item["label_qa"] and item["deal_price"] == "open"
    )

    return {
        "provider_uri": provider_uri,
        "provider_source": provider_source,
        "factor_json": factor_json,
        "factor_count": factor_count,
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
                "close": qa_ablation.build_label_expression("close"),
                "open": qa_ablation.build_label_expression("open")
                if args.label_follows_deal_price
                else qa_ablation.build_label_expression("close"),
            },
        },
        "datasets": {
            dataset_name: payload,
        },
    }


def main():
    args = parse_args()
    provider_uri, provider_source = qa_ablation.resolve_provider_uri(args)
    factor_path = Path(args.factor_json).resolve()
    dataset_name = args.dataset_name or factor_path.stem

    qlib.init(provider_uri=provider_uri, region=args.region)
    print(f"using provider_uri={provider_uri} (source={provider_source})", flush=True)
    print(
        f"segments: train={args.train_start}:{args.train_end}, "
        f"valid={args.valid_start}:{args.valid_end}, test={args.test_start}:{args.test_end}",
        flush=True,
    )
    print("label mode: deal_price-aware (close=>close-to-close, open=>open-to-open)", flush=True)

    factors = load_factor_library(factor_path)
    print(f"loaded {len(factors)} factors from {factor_path}", flush=True)

    field_panels = workflow._load_base_fields(args.market, args.start_time, args.end_time)
    raw_feature_df, _ = workflow._build_legacy_feature_label_frames(factors, field_panels)

    label_by_price = {
        "close": build_label_from_field_panel(field_panels, "close"),
        "open": build_label_from_field_panel(field_panels, "open"),
    }
    if not args.label_follows_deal_price:
        label_by_price["open"] = label_by_price["close"]

    prepared_features = {
        False: preprocess_raw_feature(raw_feature_df),
        True: preprocess_qa_feature(raw_feature_df),
    }

    runs = []
    for feature_qa in [False, True]:
        for label_qa in [False, True]:
            for deal_price in ["close", "open"]:
                spec = qa_ablation.RunSpec(dataset_name, feature_qa, label_qa, deal_price)
                runs.append(
                    qa_ablation.run_single_spec(spec, prepared_features[feature_qa], label_by_price[deal_price], args)
                )

    summary = build_summary(
        runs=runs,
        args=args,
        provider_uri=provider_uri,
        provider_source=provider_source,
        factor_json=str(factor_path),
        factor_count=len(factors),
        dataset_name=dataset_name,
    )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"saved ablation summary to {output_path}", flush=True)


if __name__ == "__main__":
    main()
