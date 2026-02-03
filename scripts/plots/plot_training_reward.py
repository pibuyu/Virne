import argparse

from _plot_common import (
    add_common_args,
    choose_x_column,
    ensure_out_path,
    load_csv,
    plot_line,
    resolve_csv_path,
    warn_missing_column,
)

PRIMARY_COL = "value/reward"
FALLBACK_COL = "value/return"
METRIC_SLUG = "training_reward"


def main():
    parser = argparse.ArgumentParser(description="Plot training reward per update")
    add_common_args(parser, kind="training")
    args = parser.parse_args()

    csv_path = resolve_csv_path(args.csv_path, args.run_dir, kind="training")
    df = load_csv(csv_path)
    if PRIMARY_COL in df.columns:
        y_col = PRIMARY_COL
    elif FALLBACK_COL in df.columns:
        y_col = FALLBACK_COL
        print(f"Warning: {PRIMARY_COL} not found, using {FALLBACK_COL}")
    else:
        warn_missing_column(df, PRIMARY_COL)

    x_col = choose_x_column(df, args.x, ["update_time", "step", "global_step"])
    if x_col is None:
        df["_idx"] = range(len(df))
        x_col = "_idx"

    out_path = ensure_out_path(args.out, csv_path, METRIC_SLUG)
    title = args.title or "Training Reward"
    plot_line(df, x_col, y_col, out_path, title, x_col, y_col)


if __name__ == "__main__":
    main()
