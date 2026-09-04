import os
DATA_DIR = os.environ.get("PHYSICAL_DATA_DIR", "./results/physical")
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# =========================================================
# CONFIG
# =========================================================
CSV_FILES = {
    "RRT": os.path.join(DATA_DIR, "rrt.csv"),
    "RRT_CONNECT": os.path.join(DATA_DIR, "rrt_connect.csv"),
    "RRT_STAR": os.path.join(DATA_DIR, "rrtstar.csv"),
    "RRTX": os.path.join(DATA_DIR, "rrtx.csv"),
}

OUTPUT_DIR = Path(os.environ.get("OUT_DIR", "./figures"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# If True, use only solved rows for path/time/path-length/clearance style metrics.
USE_SOLVED_ROWS_FOR_PATH_METRICS = True

# =========================================================
# HELPERS
# =========================================================
def safe_mean(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(s.mean())

def safe_last(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(s.iloc[-1])

def add_value_labels(ax, fmt="{:.1f}"):
    ymax = 0
    for p in ax.patches:
        h = p.get_height()
        if pd.isna(h):
            continue
        ymax = max(ymax, h)
    offset = 0.02 * ymax if ymax > 0 else 0.1
    for p in ax.patches:
        h = p.get_height()
        if pd.isna(h):
            continue
        ax.text(
            p.get_x() + p.get_width() / 2.0,
            h + offset,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=10
        )

def save_bar_plot(summary_df, metric_col, title, ylabel, filename, value_fmt="{:.1f}"):
    plot_df = summary_df.copy()

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.bar(plot_df["planner_name"], plot_df[metric_col])
    ax.set_title(title)
    ax.set_xlabel("Planner")
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    add_value_labels(ax, value_fmt)
    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved: {out_path}")

# =========================================================
# LOAD + SUMMARIZE
# =========================================================
rows = []

for planner_name, csv_path in CSV_FILES.items():
    df = pd.read_csv(csv_path)

    df["timestamp_iso"] = pd.to_datetime(df["timestamp_iso"], errors="coerce")

    solved_df = df[df["solved"] == 1].copy()
    metric_df = solved_df if USE_SOLVED_ROWS_FOR_PATH_METRICS else df

    # Basic session-level summary
    row = {
        "planner_name": planner_name,
        "num_rows": len(df),
        "num_solved_rows": int((df["solved"] == 1).sum()),
        "solved_fraction_percent": 100.0 * float((df["solved"] == 1).mean()),
        "avg_time_to_goal_ms": safe_mean(metric_df["time_to_goal_ms"]),
        "avg_total_planning_time_ms": safe_mean(metric_df["total_cumulative_planning_time_ms"]),
        "avg_path_length_px": safe_mean(metric_df["total_executed_path_length"]),
        "avg_min_clearance_px": safe_mean(metric_df["minimum_obstacle_clearance_px"]),
        "avg_recovery_time_ms": safe_mean(metric_df["recovery_time_ms"]),
        "last_replanning_events": safe_last(df["replanning_events"]),
        "avg_replanning_events": safe_mean(df["replanning_events"]),
        "last_node_count": safe_last(df["node_count"]),
        "avg_node_count": safe_mean(df["node_count"]),
        "first_timestamp": df["timestamp_iso"].min(),
        "last_timestamp": df["timestamp_iso"].max(),
    }

    # Approximate session duration from available rows
    if pd.notna(row["first_timestamp"]) and pd.notna(row["last_timestamp"]):
        row["session_duration_s"] = (row["last_timestamp"] - row["first_timestamp"]).total_seconds()
    else:
        row["session_duration_s"] = np.nan

    rows.append(row)

summary = pd.DataFrame(rows)

# Keep planner order fixed
planner_order = ["RRT", "RRT_CONNECT", "RRT_STAR", "RRTX"]
summary["planner_name"] = pd.Categorical(summary["planner_name"], categories=planner_order, ordered=True)
summary = summary.sort_values("planner_name").reset_index(drop=True)

# Save summary CSV
summary_csv = OUTPUT_DIR / "physical_validation_summary.csv"
summary.to_csv(summary_csv, index=False)
print(f"Saved: {summary_csv}")

# =========================================================
# PLOTS
# =========================================================

# 1) Solved-row fraction across logged events
save_bar_plot(
    summary,
    "solved_fraction_percent",
    "Solved Fraction Across Logged Events (Physical Validation Session)",
    "Solved Fraction (%)",
    "physical_solved_fraction.png",
    value_fmt="{:.1f}"
)

# 2) Average time to goal
save_bar_plot(
    summary,
    "avg_time_to_goal_ms",
    "Average Time to Goal (Physical Validation Session)",
    "Average Time to Goal (ms)",
    "physical_time_to_goal.png",
    value_fmt="{:.1f}"
)

# 3) Average cumulative planning time
save_bar_plot(
    summary,
    "avg_total_planning_time_ms",
    "Average Cumulative Planning Time (Physical Validation Session)",
    "Average Cumulative Planning Time (ms)",
    "physical_total_planning_time.png",
    value_fmt="{:.1f}"
)

# 4) Average executed path length
save_bar_plot(
    summary,
    "avg_path_length_px",
    "Average Executed Path Length (Physical Validation Session)",
    "Average Executed Path Length (px)",
    "physical_path_length.png",
    value_fmt="{:.1f}"
)

# 5) Average minimum obstacle clearance
save_bar_plot(
    summary,
    "avg_min_clearance_px",
    "Average Minimum Obstacle Clearance (Physical Validation Session)",
    "Average Minimum Obstacle Clearance (px)",
    "physical_min_clearance.png",
    value_fmt="{:.1f}"
)

# 6) Average replanning events
save_bar_plot(
    summary,
    "avg_replanning_events",
    "Average Replanning Events (Physical Validation Session)",
    "Average Replanning Events",
    "physical_avg_replanning_events.png",
    value_fmt="{:.1f}"
)

# 7) Final replanning-event count
save_bar_plot(
    summary,
    "last_replanning_events",
    "Final Replanning Event Count (Physical Validation Session)",
    "Final Replanning Event Count",
    "physical_final_replanning_events.png",
    value_fmt="{:.0f}"
)

# 8) Average node count
save_bar_plot(
    summary,
    "avg_node_count",
    "Average Node Count (Physical Validation Session)",
    "Average Node Count",
    "physical_avg_node_count.png",
    value_fmt="{:.0f}"
)

print("\nDone.")
print("Important:")
print("- These plots summarize a single physical validation session.")
print("- They should be used for validation / qualitative support, not as a batch statistical comparison.")