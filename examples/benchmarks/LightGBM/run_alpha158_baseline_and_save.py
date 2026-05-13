import argparse
import json
from pathlib import Path

from ruamel.yaml import YAML

import qlib
from qlib.cli.run import render_template, sys_config
from qlib.config import C
from qlib.model.trainer import task_train
from qlib.utils.data import update_config


def load_config(config_path: Path) -> dict:
    rendered_yaml = render_template(str(config_path))
    yaml = YAML(typ="safe", pure=True)
    config = yaml.load(rendered_yaml)

    base_config_path = config.get("BASE_CONFIG_PATH")
    if base_config_path:
        base_path = Path(base_config_path)
        if not base_path.exists():
            base_path = config_path.parent / base_path
        with open(base_path) as fp:
            base_config = yaml.load(fp)
        config = update_config(base_config, config)

    sys_config(config, str(config_path))
    return config


def init_qlib(config: dict, uri_folder: str) -> None:
    if "exp_manager" in config.get("qlib_init", {}):
        qlib.init(**config["qlib_init"])
        return

    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(Path.cwd() / uri_folder)
    qlib.init(**config["qlib_init"], exp_manager=exp_manager)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Alpha158 baseline and save metrics to file.")
    parser.add_argument(
        "--config",
        default="examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml",
        help="Path to qrun yaml config.",
    )
    parser.add_argument("--experiment-name", default="alpha158_baseline")
    parser.add_argument("--uri-folder", default="mlruns")
    parser.add_argument("--data-start", default="2016-01-01")
    parser.add_argument("--data-end", default="2025-12-26")
    parser.add_argument("--train-start", default="2016-01-01")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--valid-start", default="2021-01-01")
    parser.add_argument("--valid-end", default="2021-12-31")
    parser.add_argument("--test-start", default="2022-01-01")
    parser.add_argument("--test-end", default="2025-12-26")
    parser.add_argument(
        "--output",
        default="alpha158_baseline_metrics.json",
        help="Path to save summary metrics json.",
    )
    return parser.parse_args()


def override_date_config(config: dict, args) -> None:
    handler_kwargs = config["task"]["dataset"]["kwargs"]["handler"]["kwargs"]
    handler_kwargs["start_time"] = args.data_start
    handler_kwargs["end_time"] = args.data_end
    if "fit_start_time" in handler_kwargs:
        handler_kwargs["fit_start_time"] = args.train_start
    if "fit_end_time" in handler_kwargs:
        handler_kwargs["fit_end_time"] = args.train_end

    segments = config["task"]["dataset"]["kwargs"]["segments"]
    segments["train"] = [args.train_start, args.train_end]
    segments["valid"] = [args.valid_start, args.valid_end]
    segments["test"] = [args.test_start, args.test_end]

    port_cfg = None
    for record in config["task"].get("record", []):
        if record.get("class") == "PortAnaRecord":
            port_cfg = record["kwargs"]["config"]
            break
    if port_cfg is not None:
        port_cfg["backtest"]["start_time"] = args.test_start
        port_cfg["backtest"]["end_time"] = args.test_end


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    output_path = Path(args.output).resolve()

    config = load_config(config_path)
    override_date_config(config, args)
    experiment_name = config.get("experiment_name", args.experiment_name)

    init_qlib(config, args.uri_folder)
    recorder = task_train(config["task"], experiment_name=experiment_name)
    recorder.save_objects(config=config)

    metrics = recorder.list_metrics()
    params = recorder.list_params()

    summary = {
        "run_id": recorder.id,
        "experiment_id": recorder.experiment_id,
        "config_path": str(config_path),
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
