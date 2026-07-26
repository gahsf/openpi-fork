import argparse
import collections
import glob
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq

ACTION_DIM = 7


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LIBERO rollout actions with training actions.")
    parser.add_argument("--train-glob", required=True, help="Glob matching LIBERO training parquet files.")
    parser.add_argument("--eval-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def _stats(actions: list[np.ndarray], train_q01: np.ndarray | None = None, train_q99: np.ndarray | None = None) -> dict:
    if not actions:
        return {"count": 0}
    values = np.concatenate(actions, axis=0).astype(np.float64, copy=False)
    if values.ndim != 2 or values.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected actions shaped [N, {ACTION_DIM}], got {values.shape}")
    result = {
        "count": int(values.shape[0]),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "gripper_positive_fraction": float(np.mean(values[:, -1] > 0)),
        "gripper_negative_fraction": float(np.mean(values[:, -1] < 0)),
    }
    if train_q01 is not None and train_q99 is not None:
        outside = (values < train_q01[None, :]) | (values > train_q99[None, :])
        result["outside_train_q01_q99_fraction"] = outside.mean(axis=0).tolist()
        result["outside_train_q01_q99_any_dim_fraction"] = float(outside.any(axis=1).mean())
    return result


def _load_training_actions(pattern: str) -> list[np.ndarray]:
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No parquet files matched: {pattern}")
    batches = []
    for path in paths:
        table = pq.read_table(path, columns=["actions"])
        actions = np.asarray(table.column("actions").combine_chunks().values).reshape(-1, ACTION_DIM)
        batches.append(actions)
    return batches


def _load_trace(path: pathlib.Path) -> tuple[dict[tuple[int, int], list[list[float]]], dict[tuple[int, int], bool]]:
    episode_actions: dict[tuple[int, int], list[list[float]]] = collections.defaultdict(list)
    outcomes = {}
    with path.open(encoding="utf-8") as trace_file:
        for line in trace_file:
            record = json.loads(line)
            key = (int(record["task_id"]), int(record["episode_idx"]))
            if record["record_type"] == "action":
                episode_actions[key].append(record["action"])
            elif record["record_type"] == "episode_end":
                outcomes[key] = bool(record["success"])
    if set(episode_actions) != set(outcomes):
        raise ValueError(f"Incomplete trace file: {path}")
    return episode_actions, outcomes


def _switch_rate(episodes: list[np.ndarray]) -> float:
    switches = 0
    transitions = 0
    for actions in episodes:
        signs = actions[:, -1] > 0
        switches += int(np.count_nonzero(signs[1:] != signs[:-1]))
        transitions += max(0, len(signs) - 1)
    return switches / transitions if transitions else 0.0


def main() -> None:
    args = _parse_args()
    training_batches = _load_training_actions(args.train_glob)
    training = _stats(training_batches)
    train_q01 = np.asarray(training["q01"])
    train_q99 = np.asarray(training["q99"])

    report = {"training": training, "models": {}}
    for model_dir in sorted(path for path in args.eval_root.iterdir() if path.is_dir()):
        groups: dict[str, list[np.ndarray]] = collections.defaultdict(list)
        model_report = {"suites": {}}
        for trace_path in sorted(model_dir.glob("*/traces/action_trace.jsonl")):
            suite = trace_path.parents[1].name
            episode_actions, outcomes = _load_trace(trace_path)
            suite_groups: dict[str, list[np.ndarray]] = collections.defaultdict(list)
            for key, action_rows in episode_actions.items():
                actions = np.asarray(action_rows, dtype=np.float64)
                outcome = "success" if outcomes[key] else "failure"
                suite_groups["all"].append(actions)
                suite_groups[outcome].append(actions)
                groups["all"].append(actions)
                groups[outcome].append(actions)
            model_report["suites"][suite] = {
                name: {
                    **_stats(values, train_q01, train_q99),
                    "gripper_switch_rate": _switch_rate(values),
                    "episodes": len(values),
                }
                for name, values in suite_groups.items()
            }
        model_report["overall"] = {
            name: {
                **_stats(values, train_q01, train_q99),
                "gripper_switch_rate": _switch_rate(values),
                "episodes": len(values),
            }
            for name, values in groups.items()
        }
        report["models"][model_dir.name] = model_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
