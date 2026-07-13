import argparse
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing import event_accumulator


DEFAULT_TAGS = [
    "train/loss_condition",
    "train/loss_condition_decouple_raw",
    "train/loss_condition_common_raw",
    "train/loss_condition_unique_raw",
    "train/loss_weather_aux",
    "train/weather_aux_acc",
    "train/loss_contrastive",
]


def find_event_file(exp_dir: Path, subdir: str) -> Path:
    event_files = sorted((exp_dir / subdir).glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No event files found under {(exp_dir / subdir)!s}")
    return event_files[0]


def load_scalars(event_file: Path):
    ea = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={"scalars": 0},
    )
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    scalar_map = {}
    for tag in tags:
        scalar_map[tag] = ea.Scalars(tag)
    return scalar_map


def summarize_tag(events, iters_per_epoch: int, recent_points: int, recent_epochs: int):
    steps = np.array([e.step for e in events], dtype=np.int64)
    values = np.array([e.value for e in events], dtype=np.float64)

    summary = {
        "count": int(values.size),
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "first_value": float(values[0]),
        "last_value": float(values[-1]),
        "min": float(values.min()),
        "max": float(values.max()),
        "recent_step_mean": float(values[-min(recent_points, values.size):].mean()),
    }

    epoch_stats = []
    if iters_per_epoch > 0:
        epoch_ids = steps // int(iters_per_epoch)
        unique_epochs = np.unique(epoch_ids)
        for epoch_id in unique_epochs[-recent_epochs:]:
            mask = epoch_ids == epoch_id
            epoch_vals = values[mask]
            epoch_stats.append(
                {
                    "epoch": int(epoch_id),
                    "count": int(epoch_vals.size),
                    "mean": float(epoch_vals.mean()),
                    "min": float(epoch_vals.min()),
                    "max": float(epoch_vals.max()),
                    "last": float(epoch_vals[-1]),
                }
            )
    summary["recent_epochs"] = epoch_stats
    return summary


def print_summary(tag: str, summary):
    print(tag)
    print(
        "  "
        f"count={summary['count']} "
        f"step={summary['first_step']}->{summary['last_step']} "
        f"first={summary['first_value']:.6f} "
        f"last={summary['last_value']:.6f} "
        f"recent_mean={summary['recent_step_mean']:.6f} "
        f"min={summary['min']:.6f} max={summary['max']:.6f}"
    )
    if summary["recent_epochs"]:
        print("  recent epoch means:")
        for item in summary["recent_epochs"]:
            print(
                "    "
                f"epoch {item['epoch']}: "
                f"mean={item['mean']:.6f} "
                f"last={item['last']:.6f} "
                f"min={item['min']:.6f} max={item['max']:.6f} "
                f"count={item['count']}"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Experiment directory")
    parser.add_argument("--iters-per-epoch", type=int, default=4570)
    parser.add_argument("--recent-points", type=int, default=200)
    parser.add_argument("--recent-epochs", type=int, default=5)
    parser.add_argument("--tags", nargs="*", default=DEFAULT_TAGS)
    return parser.parse_args()


def main():
    args = parse_args()
    exp_dir = Path(args.exp).resolve()
    event_file = find_event_file(exp_dir, "train_iter")
    scalar_map = load_scalars(event_file)

    print(f"Experiment: {exp_dir}")
    print(f"Event file: {event_file.name}")
    print(f"Iters/epoch: {args.iters_per_epoch}")

    for tag in args.tags:
        events = scalar_map.get(tag, [])
        if not events:
            print(f"{tag}\n  MISSING")
            continue
        summary = summarize_tag(
            events,
            iters_per_epoch=args.iters_per_epoch,
            recent_points=args.recent_points,
            recent_epochs=args.recent_epochs,
        )
        print_summary(tag, summary)


if __name__ == "__main__":
    main()
