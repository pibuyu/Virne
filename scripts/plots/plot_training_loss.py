import argparse

from _plot_common import (
    add_common_args,
    add_plot_style_args,
    choose_x_column,
    ensure_out_path,
    load_csv,
    plot_line,
    resolve_csv_path,
    warn_missing_column,
)

METRIC_COL = "loss/loss"
METRIC_SLUG = "training_loss"


def main():
    parser = argparse.ArgumentParser(description="Plot training total loss per update")
    add_common_args(parser, kind="training")
    add_plot_style_args(parser)
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
    title = args.title or "Training Loss"
    yscale = "log" if args.logy else None
    plot_line(
        df,
        x_col,
        METRIC_COL,
        out_path,
        title,
        x_col,
        METRIC_COL,
        yscale=yscale,
        ymin=args.ymin,
        ymax=args.ymax,
        clip_quantile=args.clip_quantile,
    )


if __name__ == "__main__":
    main()
