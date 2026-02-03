import argparse

from _plot_common import (
    add_common_args,
    choose_x_column,
    ensure_out_path,
    filter_enter_events,
    load_csv,
    plot_line,
    resolve_csv_path,
    warn_missing_column,
)

METRIC_COL = "v_net_r2c_ratio"
METRIC_SLUG = "v_net_r2c_ratio"


def main():
    parser = argparse.ArgumentParser(description="Plot per-request r2c ratio over time")
    add_common_args(parser, kind="records")
    parser.add_argument("--window", type=int, default=1, help="Rolling mean window (default: 1)")
    args = parser.parse_args()

    csv_path = resolve_csv_path(args.csv_path, args.run_dir, kind="records")
    df = load_csv(csv_path)
    if METRIC_COL not in df.columns:
        warn_missing_column(df, METRIC_COL)

    df = filter_enter_events(df)
    if args.window and args.window > 1:
        df[METRIC_COL] = df[METRIC_COL].rolling(args.window, min_periods=1).mean()

    x_col = choose_x_column(df, args.x, ["event_id", "event_time", "v_net_arrival_time"])
    if x_col is None:
        df["_idx"] = range(len(df))
        x_col = "_idx"

    out_path = ensure_out_path(args.out, csv_path, METRIC_SLUG)
    title = args.title or "V-Net R2C Ratio"
    ylabel = METRIC_COL + (f" (rolling={args.window})" if args.window > 1 else "")
    plot_line(df, x_col, METRIC_COL, out_path, title, x_col, ylabel)


if __name__ == "__main__":
    main()
