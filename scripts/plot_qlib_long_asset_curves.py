#!/usr/bin/env python3
"""
Plot cumulative return curves for Qlib backtest runs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import mlflow
import pandas as pd

from qlib.contrib.report.analysis_position.parse_position import get_position_data

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _cumsum_return(series: pd.Series) -> pd.Series:
    return series.fillna(0).cumsum()


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _resolve_run_id(run_id: str | None, metrics_json: str | None) -> str:
    if run_id:
        return run_id
    if not metrics_json:
        raise ValueError("Either --run-id or --metrics-json must be provided")
    payload = json.loads(Path(metrics_json).read_text())
    resolved = payload.get("run_id")
    if not resolved:
        raise ValueError(f"run_id not found in {metrics_json}")
    return resolved


def _configure_tracking_uri(mlruns_dir: str | None) -> None:
    if mlruns_dir:
        mlflow.set_tracking_uri("file:" + str(Path(mlruns_dir).resolve()))


def _download_artifact(run_id: str, artifact_path: str) -> Path:
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    return Path(local_path)


def _load_run_objects(run_id: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    report_path = _download_artifact(run_id, "portfolio_analysis/report_normal_1day.pkl")
    positions_path = _download_artifact(run_id, "portfolio_analysis/positions_normal_1day.pkl")
    label_path = _download_artifact(run_id, "label.pkl")

    report_df = pd.read_pickle(report_path)
    positions = pd.read_pickle(positions_path)
    label_df = pd.read_pickle(label_path)
    return report_df, positions, label_df


def build_asset_trade_detail(position: dict, report_df: pd.DataFrame, label_data: pd.DataFrame) -> pd.DataFrame:
    position_df = get_position_data(
        position=position,
        report_normal=report_df,
        label_data=label_data,
    ).reset_index()
    if position_df.empty:
        return pd.DataFrame()

    traded_assets = sorted(position_df["instrument"].dropna().unique())
    asset_panel = label_data.loc[label_data.index.get_level_values("instrument").isin(traded_assets)].reset_index()
    asset_panel = asset_panel.rename(columns={"label": "asset_ret"})
    bench_df = report_df[["bench"]].reset_index().rename(columns={"index": "datetime", "bench": "bench_ret"})

    position_df = position_df.drop(columns=["label"], errors="ignore")
    detail_df = asset_panel.merge(position_df, on=["instrument", "datetime"], how="left")
    detail_df = detail_df.merge(bench_df, on="datetime", how="left")
    detail_df["weight"] = detail_df["weight"].fillna(0)
    detail_df["status"] = detail_df["status"].fillna(0)
    detail_df["model_trade_ret"] = detail_df["asset_ret"].fillna(0) * detail_df["weight"]
    detail_df["is_held"] = detail_df["weight"] > 0
    detail_df = detail_df.sort_values(["instrument", "datetime"]).reset_index(drop=True)
    return detail_df


def build_daily_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()

    def _agg_day(day_df: pd.DataFrame) -> pd.Series:
        held_mask = day_df["is_held"]
        held_ret = day_df.loc[held_mask, "asset_ret"]
        traded_asset_ret = held_ret.mean() if len(held_ret) > 0 else 0.0
        bench_ret = day_df["bench_ret"].dropna().iloc[0] if day_df["bench_ret"].notna().any() else 0.0
        return pd.Series(
            {
                "model_ret": day_df["model_trade_ret"].sum(),
                "traded_asset_ret": traded_asset_ret,
                "bench_ret": bench_ret,
                "held_asset_count": int(held_mask.sum()),
                "held_weight_sum": day_df.loc[held_mask, "weight"].sum(),
            }
        )

    daily_df = detail_df.groupby("datetime", sort=True, group_keys=False).apply(_agg_day).reset_index()
    daily_df["cum_model_ret"] = _cumsum_return(daily_df["model_ret"])
    daily_df["cum_traded_asset_ret"] = _cumsum_return(daily_df["traded_asset_ret"])
    daily_df["cum_bench_ret"] = _cumsum_return(daily_df["bench_ret"])
    return daily_df


def build_asset_info(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()

    def _agg_asset(asset_df: pd.DataFrame) -> pd.Series:
        held_df = asset_df[asset_df["is_held"]]
        first_date = held_df["datetime"].min() if len(held_df) > 0 else pd.NaT
        last_date = held_df["datetime"].max() if len(held_df) > 0 else pd.NaT
        return pd.Series(
            {
                "start_date": asset_df["datetime"].min(),
                "end_date": asset_df["datetime"].max(),
                "first_held_date": first_date,
                "last_held_date": last_date,
                "held_days": int(asset_df["is_held"].sum()),
                "avg_weight": held_df["weight"].mean() if len(held_df) > 0 else 0.0,
                "max_weight": held_df["weight"].max() if len(held_df) > 0 else 0.0,
                "total_asset_ret": asset_df["asset_ret"].fillna(0).sum(),
                "total_model_trade_ret": asset_df["model_trade_ret"].fillna(0).sum(),
                "plot_file": f"{_sanitize_filename(asset_df['instrument'].iloc[0])}.png",
            }
        )

    return detail_df.groupby("instrument", sort=True, group_keys=False).apply(_agg_asset).reset_index()


def plot_overall_curves(daily_df: pd.DataFrame, output_path: Path, benchmark_label: str) -> None:
    if daily_df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(daily_df["datetime"], daily_df["cum_model_ret"], label="Model cumulative ret", linewidth=2.2)
    ax.plot(
        daily_df["datetime"],
        daily_df["cum_traded_asset_ret"],
        label="Held assets cumulative ret",
        linewidth=1.8,
        alpha=0.9,
    )
    ax.plot(
        daily_df["datetime"],
        daily_df["cum_bench_ret"],
        label=f"{benchmark_label} cumulative ret",
        linewidth=1.8,
        alpha=0.9,
    )
    ax.set_title(f"Model vs Held Assets vs {benchmark_label}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_single_asset(args: tuple[str, pd.DataFrame, str]) -> str:
    asset, asset_df, output_dir = args
    asset_df = asset_df.sort_values("datetime").copy()
    asset_df["cum_asset_ret"] = _cumsum_return(asset_df["asset_ret"])
    asset_df["cum_model_trade_ret"] = _cumsum_return(asset_df["model_trade_ret"])

    fig, (ax_ret, ax_weight) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_ret_right = ax_ret.twinx()

    ret_line_left = ax_ret.plot(
        asset_df["datetime"],
        asset_df["cum_asset_ret"],
        label=f"{asset} cumulative ret",
        linewidth=2.0,
        color="tab:blue",
    )[0]
    ret_line_right = ax_ret_right.plot(
        asset_df["datetime"],
        asset_df["cum_model_trade_ret"],
        label=f"{asset} model traded cumulative ret",
        linewidth=1.8,
        alpha=0.9,
        color="tab:orange",
    )[0]

    ax_ret.set_title(f"{asset} Return vs Model Traded Return")
    ax_ret.set_ylabel(f"{asset} cumulative ret", color="tab:blue")
    ax_ret_right.set_ylabel("Model traded cumulative ret", color="tab:orange")
    ax_ret.tick_params(axis="y", labelcolor="tab:blue")
    ax_ret_right.tick_params(axis="y", labelcolor="tab:orange")
    ax_ret.grid(True, alpha=0.25)
    ax_ret.legend(
        [ret_line_left, ret_line_right],
        [ret_line_left.get_label(), ret_line_right.get_label()],
        loc="upper left",
    )

    ax_weight.plot(
        asset_df["datetime"],
        asset_df["weight"].fillna(0),
        label="Position weight",
        linewidth=1.6,
        color="tab:green",
    )
    ax_weight.fill_between(
        asset_df["datetime"],
        0,
        asset_df["weight"].fillna(0),
        color="tab:green",
        alpha=0.2,
    )
    ax_weight.set_xlabel("Date")
    ax_weight.set_ylabel("Weight")
    ax_weight.grid(True, alpha=0.25)
    ax_weight.legend(loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = Path(output_dir) / f"{_sanitize_filename(asset)}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def plot_asset_curves_parallel(detail_df: pd.DataFrame, output_dir: Path, n_jobs: int = 8) -> list[Path]:
    if detail_df.empty:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    grouped_frames = [(asset, asset_df.copy(), str(output_dir)) for asset, asset_df in detail_df.groupby("instrument")]
    if not grouped_frames:
        return []

    max_workers = max(1, min(int(n_jobs), len(grouped_frames)))
    if max_workers == 1:
        return [Path(_plot_single_asset(item)) for item in grouped_frames]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return [
            Path(path)
            for path in executor.map(
                _plot_single_asset,
                grouped_frames,
                chunksize=max(1, math.ceil(len(grouped_frames) / max_workers)),
            )
        ]


def _artifact_output_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "detail_parquet": output_dir / f"{prefix}_asset_trade_detail.parquet",
        "daily_parquet": output_dir / f"{prefix}_asset_trade_daily.parquet",
        "asset_info_parquet": output_dir / f"{prefix}_asset_info.parquet",
        "overall_plot": output_dir / f"{prefix}_model_vs_assets.png",
        "asset_plot_dir": output_dir / f"{prefix}_asset_curves",
        "summary_json": output_dir / f"{prefix}_plot_summary.json",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Plot saved Qlib backtest asset trade curves")
    parser.add_argument("--run-id", default="", help="MLflow run id")
    parser.add_argument("--metrics-json", default="", help="Path to saved metrics json containing run_id")
    parser.add_argument("--mlruns-dir", default="mlruns", help="Local mlruns directory")
    parser.add_argument("--output-dir", default="outputs/qlib_plots", help="Directory to save plot outputs")
    parser.add_argument("--prefix", default="", help="Output file prefix; defaults to run_id")
    parser.add_argument("--benchmark-label", default="CSI300")
    parser.add_argument("--n-jobs", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    run_id = _resolve_run_id(args.run_id or None, args.metrics_json or None)
    _configure_tracking_uri(args.mlruns_dir)

    report_df, positions, label_df = _load_run_objects(run_id)
    detail_df = build_asset_trade_detail(position=positions, report_df=report_df, label_data=label_df)
    if detail_df.empty:
        raise ValueError("No traded asset detail available for plotting")

    daily_df = build_daily_summary(detail_df)
    asset_info_df = build_asset_info(detail_df)

    prefix = args.prefix or run_id
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _artifact_output_paths(output_dir, prefix)

    detail_df.to_parquet(paths["detail_parquet"], index=False)
    daily_df.to_parquet(paths["daily_parquet"], index=False)
    asset_info_df.to_parquet(paths["asset_info_parquet"], index=False)

    plot_overall_curves(daily_df, paths["overall_plot"], benchmark_label=args.benchmark_label)
    plot_asset_curves_parallel(detail_df, paths["asset_plot_dir"], n_jobs=args.n_jobs)

    summary = {
        "run_id": run_id,
        "detail_parquet": str(paths["detail_parquet"]),
        "daily_parquet": str(paths["daily_parquet"]),
        "asset_info_parquet": str(paths["asset_info_parquet"]),
        "overall_plot": str(paths["overall_plot"]),
        "asset_plot_dir": str(paths["asset_plot_dir"]),
    }
    paths["summary_json"].write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
