import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


FACTOR_JSONS = [
    "all_factors_library.json",
    "all_factors_library_new.json",
    "all_factors_library_new3.json",
    "all_factors_library_new4.json",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run QA-compatible rerun + ablation suite for factor JSONs and Alpha158."
    )
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
    parser.add_argument("--num-threads", type=int, default=20)
    parser.add_argument(
        "--python-bin",
        default="",
        help="Interpreter used for child experiment scripts. Leave empty to auto-detect a local env with numpy/lightgbm.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/quantaalpha_compat_suite_20260518",
        help="Directory that stores per-dataset ablation results plus the combined suite summary.",
    )
    return parser.parse_args()


def common_args(args) -> list[str]:
    values = [
        "--region",
        args.region,
        "--market",
        args.market,
        "--benchmark",
        args.benchmark,
        "--start-time",
        args.start_time,
        "--end-time",
        args.end_time,
        "--train-start",
        args.train_start,
        "--train-end",
        args.train_end,
        "--valid-start",
        args.valid_start,
        "--valid-end",
        args.valid_end,
        "--test-start",
        args.test_start,
        "--test-end",
        args.test_end,
        "--topk",
        str(args.topk),
        "--n-drop",
        str(args.n_drop),
        "--account",
        str(args.account),
        "--num-threads",
        str(args.num_threads),
    ]
    if args.provider_uri:
        values.extend(["--provider-uri", args.provider_uri])
    return values


def run_command(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as fp:
        process = subprocess.run(cmd, stdout=fp, stderr=subprocess.STDOUT, text=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(cmd)}")


def _candidate_python_bins(user_override: str) -> list[str]:
    candidates: list[str] = []
    if user_override:
        candidates.append(user_override)

    env_override = os.environ.get("QALIB_PYTHON")
    if env_override:
        candidates.append(env_override)

    candidates.extend(
        [
            sys.executable,
            "/home/wjh/miniconda3/envs/qlib310/bin/python",
            "/home/wjh/miniconda3/envs/quantaalpha/bin/python",
            "python3",
        ]
    )

    seen: set[str] = set()
    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)
    return unique_candidates


def resolve_python_bin(user_override: str) -> str:
    probe = "import numpy, lightgbm, pandas; print('ok')"
    for candidate in _candidate_python_bins(user_override):
        if "/" in candidate and not Path(candidate).exists():
            continue
        result = subprocess.run(
            [candidate, "-c", probe],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return candidate
    raise RuntimeError(
        "Failed to find a runnable Python interpreter with numpy/lightgbm/pandas. "
        "Pass --python-bin explicitly."
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_dataset_payload(summary: dict, dataset_name: str) -> dict:
    datasets = summary.get("datasets", {})
    if dataset_name not in datasets:
        raise KeyError(f"Dataset {dataset_name} not found in summary")
    return datasets[dataset_name]


def build_compat_metrics(dataset_name: str, summary: dict, dataset_payload: dict, ablation_summary_path: Path) -> dict:
    compat_run = dict(dataset_payload["full_qa_open"])
    compat_run["family"] = dataset_name
    return {
        "dataset": dataset_name,
        "source_summary": str(ablation_summary_path.resolve()),
        "mode": "full_qa_open",
        "provider_uri": summary.get("provider_uri"),
        "provider_source": summary.get("provider_source"),
        "segments": summary.get("segments", {}),
        "settings": summary.get("settings", {}),
        "metrics": compat_run.get("metrics", {}),
        "run": compat_run,
    }


def materialize_dataset_outputs(
    dataset_name: str,
    summary: dict,
    dataset_payload: dict,
    ablation_summary_path: Path,
    output_root: Path,
) -> dict:
    dataset_dir = output_root / dataset_name
    compat_metrics = build_compat_metrics(dataset_name, summary, dataset_payload, ablation_summary_path)
    compat_path = dataset_dir / "compat_run" / "metrics.json"
    compat_path.parent.mkdir(parents=True, exist_ok=True)
    compat_path.write_text(json.dumps(compat_metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return {
        "compat_run_metrics_path": str(compat_path.resolve()),
        "ablation_summary_path": str(ablation_summary_path.resolve()),
        "compat_run": compat_metrics,
        "baseline_close_close": dataset_payload["baseline_close_close"],
        "best_by_ann_ret": dataset_payload["best_by_ann_ret"],
        "best_by_rank_ic": dataset_payload["best_by_rank_ic"],
        "main_effects": dataset_payload["main_effects"],
    }


def run_factor_dataset(factor_json: str, args, output_root: Path, python_bin: str) -> dict:
    dataset_name = Path(factor_json).stem
    dataset_dir = output_root / dataset_name
    ablation_summary = dataset_dir / "ablation" / "summary.json"
    ablation_log = dataset_dir / "ablation" / "run.log"

    cmd = [
        python_bin,
        "examples/benchmarks/LightGBM/run_factor_json_trick_ablation.py",
        "--factor-json",
        factor_json,
        "--dataset-name",
        dataset_name,
        "--raw-feature-cache",
        str((dataset_dir / "ablation" / "raw_feature.pkl").resolve()),
        "--qa-feature-cache",
        str((dataset_dir / "ablation" / "qa_feature.pkl").resolve()),
        "--output",
        str(ablation_summary.resolve()),
        *common_args(args),
    ]
    run_command(cmd, ablation_log)

    summary = load_json(ablation_summary)
    dataset_payload = extract_dataset_payload(summary, dataset_name)
    materialized = materialize_dataset_outputs(dataset_name, summary, dataset_payload, ablation_summary, output_root)
    materialized["factor_json"] = str(Path(factor_json).resolve())
    return materialized


def run_alpha158_dataset(args, output_root: Path, python_bin: str) -> dict:
    dataset_name = "alpha158"
    dataset_dir = output_root / dataset_name
    ablation_summary = dataset_dir / "ablation" / "summary.json"
    ablation_log = dataset_dir / "ablation" / "run.log"

    cmd = [
        python_bin,
        "examples/benchmarks/LightGBM/run_alpha158_trick_ablation.py",
        "--output",
        str(ablation_summary.resolve()),
        *common_args(args),
    ]
    run_command(cmd, ablation_log)

    summary = load_json(ablation_summary)
    dataset_payload = extract_dataset_payload(summary, dataset_name)
    return materialize_dataset_outputs(dataset_name, summary, dataset_payload, ablation_summary, output_root)


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    python_bin = resolve_python_bin(args.python_bin)
    print(f"using child python={python_bin}", flush=True)

    suite_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "segments": {
            "train": [args.train_start, args.train_end],
            "valid": [args.valid_start, args.valid_end],
            "test": [args.test_start, args.test_end],
        },
        "settings": {
            "provider_uri_override": args.provider_uri,
            "region": args.region,
            "market": args.market,
            "benchmark": args.benchmark,
            "topk": args.topk,
            "n_drop": args.n_drop,
            "account": args.account,
            "num_threads": args.num_threads,
            "python_bin": python_bin,
        },
        "datasets": {},
    }

    for factor_json in FACTOR_JSONS:
        dataset_name = Path(factor_json).stem
        suite_payload["datasets"][dataset_name] = run_factor_dataset(factor_json, args, output_root, python_bin)

    suite_payload["datasets"]["alpha158"] = run_alpha158_dataset(args, output_root, python_bin)

    suite_summary_path = output_root / "suite_summary.json"
    suite_summary_path.write_text(
        json.dumps(suite_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"saved suite summary to {suite_summary_path}", flush=True)


if __name__ == "__main__":
    main()
