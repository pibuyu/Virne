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

METRIC_COL = "info/approx_kl"
METRIC_SLUG = "training_approx_kl"


def main():
    parser = argparse.ArgumentParser(description="Plot training approximate KL per update")
    add_common_args(parser, kind="training")
    args = parser.parse_args()

    csv_path = resolve_csv_path(args.csv_path, args.run_dir, kind="training")
    df = load_csv(csv_path)
    if METRIC_COL not in df.columns:
        warn_missing_column(df, METRIC_COL)

    x_col = choose_x_column(df, args.x, ["update_time", "step", "global_step"])
    if x_col is None:
        df["_idx"] = range(len(df))
        x_col = "_idx"

    out_path = ensure_out_path(args.out, csv_path, METRIC_SLUG)
    title = args.title or "Training Approx KL"
    plot_line(df, x_col, METRIC_COL, out_path, title, x_col, METRIC_COL)


if __name__ == "__main__":
    main()
