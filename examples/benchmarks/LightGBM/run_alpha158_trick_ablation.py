import argparse
import json
from pathlib import Path

import qlib

import run_qa_trick_ablation as qa_ablation


def parse_args():
    parser = argparse.ArgumentParser(description="Run 2^3 trick ablation on Alpha158 only.")
    parser.add_argument("--provider-uri", default="", help="Override provider URI. Leave empty to auto-resolve.")
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
        default="outputs/alpha158_trick_ablation/summary.json",
        help="Path to save the ablation summary JSON.",
    )
    return parser.parse_args()


def build_summary(runs: list[dict], args, provider_uri: str, provider_source: str) -> dict:
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
            "alpha158": payload,
        },
    }


def main():
    args = parse_args()
    provider_uri, provider_source = qa_ablation.resolve_provider_uri(args)
    qlib.init(provider_uri=provider_uri, region=args.region)
    print(f"using provider_uri={provider_uri} (source={provider_source})", flush=True)
    print(
        f"segments: train={args.train_start}:{args.train_end}, "
        f"valid={args.valid_start}:{args.valid_end}, test={args.test_start}:{args.test_end}",
        flush=True,
    )
    print("label mode: deal_price-aware (close=>close-to-close, open=>open-to-open)", flush=True)

    raw_feature, raw_label_close = qa_ablation.load_alpha158_raw(args)
    label_by_price = {qa_ablation.DEFAULT_LABEL_PRICE: raw_label_close}
    if args.label_follows_deal_price:
        label_by_price["open"] = qa_ablation.load_market_label(args, "open")
    else:
        label_by_price["open"] = raw_label_close

    prepared_features = {
        False: qa_ablation.preprocess_alpha158_feature(raw_feature, use_qa=False),
        True: qa_ablation.preprocess_alpha158_feature(raw_feature, use_qa=True),
    }

    runs = []
    for feature_qa in [False, True]:
        for label_qa in [False, True]:
            for deal_price in ["close", "open"]:
                spec = qa_ablation.RunSpec("alpha158", feature_qa, label_qa, deal_price)
                runs.append(
                    qa_ablation.run_single_spec(spec, prepared_features[feature_qa], label_by_price[deal_price], args)
                )

    summary = build_summary(runs, args, provider_uri, provider_source)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"saved ablation summary to {output_path}", flush=True)


if __name__ == "__main__":
    main()
