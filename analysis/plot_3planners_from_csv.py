import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================
CSV_PATH = os.path.join(os.environ.get("RESULTS_DIR", "./results/simulation"), "benchmark_detailed.csv")
OUTPUT_DIR = "report_plots_3planners"

PLANNERS_TO_KEEP = ["RRT", "RRT_CONNECT", "RRT_STAR"]

# Recommended:
# Keep this False for main report plots.
# Turn it True only for a secondary sensitivity check.
REMOVE_OUTLIERS = False
IQR_MULTIPLIER = 1.5

# =========================
# HELPERS
# =========================
def remove_outliers_iqr(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Remove outliers per planner using IQR rule."""
    kept = []
    for _, group in df.groupby("planner_name"):
        s = group[metric].dropna()

        # Too few samples -> do not filter
        if len(s) < 4:
            kept.append(group)
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            kept.append(group)
            continue

        low = q1 - IQR_MULTIPLIER * iqr
        high = q3 + IQR_MULTIPLIER * iqr

        mask = group[metric].isna() | ((group[metric] >= low) & (group[metric] <= high))
        kept.append(group.loc[mask])

    return pd.concat(kept, ignore_index=True)


def make_bar_plot(summary: pd.DataFrame, metric: str, title: str, ylabel: str, filename: str):
    plt.figure(figsize=(8, 5))
    plt.bar(summary["planner_name"], summary[metric])
    plt.title(title)
    plt.xlabel("Planner")
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_path = Path(OUTPUT_DIR) / filename
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved: {out_path}")


# =========================
# LOAD DATA
# =========================
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Keep only the 3 main planners
df = df[df["planner_name"].isin(PLANNERS_TO_KEEP)].copy()

# Optional: enforce order in plots
df["planner_name"] = pd.Categorical(
    df["planner_name"],
    categories=PLANNERS_TO_KEEP,
    ordered=True
)

# =========================
# MAIN REPORT PLOTS
# =========================

# 1) Success rate
# This should stay over ALL runs.
success_summary = (
    df.groupby("planner_name", as_index=False)["success_rate"]
    .mean(numeric_only=True)
    .sort_values("planner_name")
)
make_bar_plot(
    success_summary,
    metric="success_rate",
    title="Average Success Rate (%)",
    ylabel="Average Success Rate (%)",
    filename="success_rate_3planners.png"
)

# Metrics below should be based on successful / valid final runs only
valid_df = df[df["found_valid_path"] == 1].copy()

plot_specs = [
    ("first_solution_ms", "Average Time to First Solution (ms)", "Average Time to First Solution (ms)", "first_solution_ms_3planners.png"),
    ("total_planning_ms", "Average Total Planning Time (ms)", "Average Total Planning Time (ms)", "total_planning_ms_3planners.png"),
    ("replans", "Average Number of Replans", "Average Number of Replans", "replans_3planners.png"),
    ("path_length", "Average Planned Path Length (px)", "Average Planned Path Length (px)", "path_length_3planners.png"),
]

# Optional supporting metric:
optional_specs = [
    ("min_clearance_px", "Average Minimum Clearance (px)", "Average Minimum Clearance (px)", "min_clearance_3planners.png"),
]

for metric, title, ylabel, filename in plot_specs + optional_specs:
    plot_df = valid_df.copy()

    if REMOVE_OUTLIERS:
        plot_df = remove_outliers_iqr(plot_df, metric)

    summary = (
        plot_df.groupby("planner_name", as_index=False)[metric]
        .mean(numeric_only=True)
        .sort_values("planner_name")
    )

    make_bar_plot(
        summary,
        metric=metric,
        title=title,
        ylabel=ylabel,
        filename=filename
    )

# =========================
# SAVE SUMMARY TABLE TOO
# =========================
summary_cols = [
    "success_rate",
    "found_valid_path",
    "first_solution_ms",
    "total_planning_ms",
    "replans",
    "path_length",
    "min_clearance_px",
]

summary_table = (
    df.groupby("planner_name", as_index=False)[summary_cols]
    .mean(numeric_only=True)
    .sort_values("planner_name")
)

summary_csv_path = Path(OUTPUT_DIR) / "summary_3planners.csv"
summary_table.to_csv(summary_csv_path, index=False)
print(f"Saved: {summary_csv_path}")

print("\nDone.")
print(f"Outlier removal used: {REMOVE_OUTLIERS}")