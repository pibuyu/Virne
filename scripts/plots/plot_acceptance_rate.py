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

METRIC_COL = "acceptance_rate"
METRIC_SLUG = "acceptance_rate"


def main():
    parser = argparse.ArgumentParser(description="Plot acceptance rate over time")
    add_common_args(parser, kind="records")
    args = parser.parse_args()

    csv_path = resolve_csv_path(args.csv_path, args.run_dir, kind="records")
    df = load_csv(csv_path)

    if "success_count" not in df.columns or "v_net_count" not in df.columns:
        missing = [c for c in ["success_count", "v_net_count"] if c not in df.columns]
        warn_missing_column(df, missing[0])

    df = filter_enter_events(df)
    df[METRIC_COL] = df["success_count"] / df["v_net_count"].replace(0, np.nan)

    x_col = choose_x_column(df, args.x, ["event_id", "event_time", "v_net_arrival_time"])
    if x_col is None:
        df["_idx"] = range(len(df))
        x_col = "_idx"

    out_path = ensure_out_path(args.out, csv_path, METRIC_SLUG)
    title = args.title or "Acceptance Rate"
    plot_line(df, x_col, METRIC_COL, out_path, title, x_col, METRIC_COL)


if __name__ == "__main__":
    main()
