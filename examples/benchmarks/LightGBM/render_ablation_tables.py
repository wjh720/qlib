import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TableSpec:
    feature_set: str
    nominal_feature_count: int
    summary_path: str
    dataset_key: str


DEFAULT_TABLE_SPECS = [
    TableSpec(
        feature_set="all_factors_library_new.json",
        nominal_feature_count=60,
        summary_path="outputs/alpha60_trick_ablation/summary.json",
        dataset_key="alpha60",
    ),
    TableSpec(
        feature_set="all_factors_library_new4.json",
        nominal_feature_count=110,
        summary_path="outputs/all_factors_library_new4_trick_ablation/summary.json",
        dataset_key="all_factors_library_new4",
    ),
    TableSpec(
        feature_set="alpha158",
        nominal_feature_count=158,
        summary_path="outputs/qa_trick_ablation_label_by_deal_price_deal_to_deal/summary.json",
        dataset_key="alpha158",
    ),
    TableSpec(
        feature_set="all_factors_library.json",
        nominal_feature_count=177,
        summary_path="outputs/qa_trick_ablation_label_by_deal_price_deal_to_deal/summary.json",
        dataset_key="factors177",
    ),
    TableSpec(
        feature_set="all_factors_library_new3.json",
        nominal_feature_count=271,
        summary_path="outputs/all_factors_library_new3_trick_ablation/summary.json",
        dataset_key="all_factors_library_new3",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render one markdown ablation table per feature set using the correct return-calculation summary."
    )
    parser.add_argument(
        "--output",
        default="outputs/correct_ablation_tables.md",
        help="Markdown output path.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sort_key(run: dict) -> tuple[int, int, int]:
    return (
        1 if run["feature_qa"] else 0,
        1 if run["label_qa"] else 0,
        1 if run["deal_price"] == "open" else 0,
    )


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_float(value: float) -> str:
    return f"{value:.4f}"


def render_table(spec: TableSpec, summary: dict) -> str:
    dataset = summary["datasets"][spec.dataset_key]
    runs = sorted(dataset["runs"], key=sort_key)

    lines = [
        f"## {spec.nominal_feature_count} features - `{spec.feature_set}`",
        f"Source: `{Path(spec.summary_path)}`",
        "",
        "| feature | label | deal | effective_feature_count | IC | Rank IC | annualized_return | IR | max_drawdown |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for run in runs:
        metrics = run["metrics"]
        feature_tag = "qa" if run["feature_qa"] else "raw"
        label_tag = "qa" if run["label_qa"] else "default"
        lines.append(
            "| "
            + " | ".join(
                [
                    feature_tag,
                    label_tag,
                    run["deal_price"],
                    str(run["feature_count"]),
                    format_float(metrics["IC"]),
                    format_float(metrics["Rank IC"]),
                    format_pct(metrics["1day.excess_return_with_cost.annualized_return"]),
                    format_float(metrics["1day.excess_return_with_cost.information_ratio"]),
                    format_pct(metrics["1day.excess_return_with_cost.max_drawdown"]),
                ]
            )
            + " |"
        )

    best_ic = dataset["best_by_rank_ic"]
    best_ann = dataset["best_by_ann_ret"]
    lines.extend(
        [
            "",
            f"- Best IC row: `{best_ic['run_name']}` | IC={format_float(best_ic['metrics']['IC'])} | "
            f"Rank IC={format_float(best_ic['metrics']['Rank IC'])} | "
            f"AnnRet={format_pct(best_ic['metrics']['1day.excess_return_with_cost.annualized_return'])}",
            f"- Best AnnRet row: `{best_ann['run_name']}` | IC={format_float(best_ann['metrics']['IC'])} | "
            f"Rank IC={format_float(best_ann['metrics']['Rank IC'])} | "
            f"AnnRet={format_pct(best_ann['metrics']['1day.excess_return_with_cost.annualized_return'])}",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    output_path = Path(args.output).resolve()

    rendered_sections = [
        "# Ablation Tables",
        "",
        "These tables use the deal-to-deal return summaries for `alpha158` and `all_factors_library.json`,",
        "so `deal=open` rows are evaluated with `open_to_open` returns rather than qlib's default mark-to-close.",
        "",
    ]

    for spec in DEFAULT_TABLE_SPECS:
        summary_path = Path(spec.summary_path).resolve()
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary file: {summary_path}")
        summary = load_json(summary_path)
        rendered_sections.append(render_table(spec, summary))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rendered_sections).rstrip() + "\n", encoding="utf-8")
    print(f"saved markdown tables to {output_path}")


if __name__ == "__main__":
    main()
