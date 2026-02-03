import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def add_common_args(parser, *, kind):
    parser.add_argument("--run-dir", help="Run directory that contains logs/ or records/")
    if kind == "training":
        parser.add_argument("--training-csv", dest="csv_path", help="Path to training_info.csv")
    elif kind == "records":
        parser.add_argument("--records-csv", dest="csv_path", help="Path to records csv")
        parser.add_argument("--only-enter", action="store_true", help="Filter to event_type==1 if available")
    else:
        raise ValueError(f"Unknown kind: {kind}")
    parser.add_argument("--x", default=None, help="X-axis column name")
    parser.add_argument("--out", default=None, help="Output image path (png)")
    parser.add_argument("--title", default=None, help="Plot title override")
    return parser


def add_plot_style_args(parser):
    parser.add_argument("--logy", action="store_true", help="Use log scale on Y axis")
    parser.add_argument("--ymin", type=float, default=None, help="Y-axis minimum")
    parser.add_argument("--ymax", type=float, default=None, help="Y-axis maximum")
    parser.add_argument(
        "--clip-quantile",
        type=float,
        default=None,
        help="Clip Y values to this upper quantile (e.g. 0.99)",
    )
    return parser


def resolve_csv_path(csv_path, run_dir, kind):
    if csv_path:
        path = csv_path
    else:
        if not run_dir:
            raise SystemExit("Provide --run-dir or a direct CSV path.")
        if kind == "training":
            path = os.path.join(run_dir, "logs", "training_info.csv")
        elif kind == "records":
            records_dir = os.path.join(run_dir, "records")
            if not os.path.isdir(records_dir):
                raise SystemExit(f"Records dir not found: {records_dir}")
            candidates = [
                p for p in glob.glob(os.path.join(records_dir, "*.csv"))
                if not os.path.basename(p).startswith("temp-")
            ]
            if not candidates:
                raise SystemExit(f"No record CSV found in {records_dir}")
            path = max(candidates, key=os.path.getmtime)
        else:
            raise ValueError(f"Unknown kind: {kind}")
    if not os.path.exists(path):
        raise SystemExit(f"CSV not found: {path}")
    return path


def load_csv(path):
    return pd.read_csv(path)


def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def choose_x_column(df, preferred, fallback):
    if preferred:
        if preferred not in df.columns:
            raise SystemExit(f"X column not found: {preferred}")
        return preferred
    for col in fallback:
        if col in df.columns:
            return col
    return None


def ensure_out_path(out_path, csv_path, metric_slug):
    if out_path:
        path = out_path
    else:
        base_dir = os.path.dirname(csv_path)
        path = os.path.join(base_dir, f"{metric_slug}.png")
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return path


def filter_enter_events(df):
    if "event_type" in df.columns:
        try:
            return df[df["event_type"] == 1].copy()
        except Exception:
            return df[df["event_type"].astype(str) == "1"].copy()
    return df


def plot_line(
    df,
    x_col,
    y_col,
    out_path,
    title,
    xlabel,
    ylabel,
    yscale=None,
    ymin=None,
    ymax=None,
    clip_quantile=None,
):
    clean = df[[x_col, y_col]].copy()
    clean[x_col] = to_numeric(clean[x_col])
    clean[y_col] = to_numeric(clean[y_col])
    clean = clean.dropna()
    clean = clean.sort_values(by=x_col)
    if clip_quantile is not None:
        if not (0 < clip_quantile <= 1):
            raise SystemExit("--clip-quantile must be in (0, 1].")
        upper = clean[y_col].quantile(clip_quantile)
        clean[y_col] = clean[y_col].clip(upper=upper)
    if yscale == "log":
        clean = clean[clean[y_col] > 0]
    if clean.empty:
        raise SystemExit("No valid data to plot after cleaning.")

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(clean[x_col].to_numpy(), clean[y_col].to_numpy(), linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if yscale:
        ax.set_yscale(yscale)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved: {out_path}")


def warn_missing_column(df, col):
    cols = ", ".join(df.columns)
    raise SystemExit(f"Column not found: {col}\nAvailable columns: {cols}")


def add_script_dir_to_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
