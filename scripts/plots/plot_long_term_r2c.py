import argparse
import numpy as np

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

METRIC_COL = "long_term_r2c_ratio"
METRIC_SLUG = "long_term_r2c"


def main():
    parser = argparse.ArgumentParser(description="Plot long-term r2c ratio over time")
    add_common_args(parser, kind="records")
    args = parser.parse_args()

    csv_path = resolve_csv_path(args.csv_path, args.run_dir, kind="records")
    df = load_csv(csv_path)

    if METRIC_COL not in df.columns:
        if "total_revenue" in df.columns and "total_cost" in df.columns:
            df[METRIC_COL] = df["total_revenue"] / df["total_cost"].replace(0, np.nan)
        else:
            warn_missing_column(df, METRIC_COL)

    df = filter_enter_events(df)

    x_col = choose_x_column(df, args.x, ["event_id", "event_time", "v_net_arrival_time"])
    if x_col is None:
        df["_idx"] = range(len(df))
        x_col = "_idx"

    out_path = ensure_out_path(args.out, csv_path, METRIC_SLUG)
    title = args.title or "Long-term R2C Ratio"
    plot_line(df, x_col, METRIC_COL, out_path, title, x_col, METRIC_COL)


if __name__ == "__main__":
    main()
