import os
import csv
import importlib.util
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

# Put this script in the SAME folder as your original project file.
ORIGINAL_MODULE_FILENAME = "projection_RRT_rrt_rrtconnect_rrtstar_rrtx_overlay_v6_csv_fix2.py"

# Main toggles
VISUALIZE_COMPARISON_2X2 = True
RUN_BATCH_EXPERIMENTS = True
GENERATE_PLOTS = True
SAVE_FRAME_SNAPSHOTS = True
SAVE_EVERY_N_FRAMES = 20

# Plot / statistics toggles
# Keep this OFF by default. Turn it on only for a secondary sensitivity check.
PLOT_REMOVE_OUTLIERS = True
OUTLIER_IQR_MULTIPLIER = 1.5

# Runtime / experiment settings
MAP_W = 1000
MAP_H = 700
STEP_BUDGET_PER_FRAME = 120
MAX_FRAMES = 260
FRAME_DELAY_MS = 35
BATCH_RUNS = 20
BASE_SEED = 45

SNAPSHOT_DIR = "comparison_snapshots"
RESULTS_DIR = "comparison_results"

# Planner colors in BGR (OpenCV)
COLOR_RRT = (255, 0, 0)            # blue
COLOR_RRT_CONNECT = (255, 0, 255)  # magenta / pink
COLOR_RRT_STAR = (0, 255, 255)     # yellow
COLOR_RRTX = (255, 255, 0)         # turquoise / cyan

Point = Tuple[int, int]
ProjRect = Tuple[int, int, int, int, float]


# =============================================================================
# LOAD ORIGINAL MODULE (keeps the actual planner algorithms unchanged)
# =============================================================================

def load_original_module():
    here = Path(__file__).resolve().parent
    module_path = here / ORIGINAL_MODULE_FILENAME
    if not module_path.exists():
        raise FileNotFoundError(
            f"Could not find original file: {module_path}\n"
            f"Put this script in the same folder as the original file, "
            f"or change ORIGINAL_MODULE_FILENAME at the top."
        )

    spec = importlib.util.spec_from_file_location("orig_project_module", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {module_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


orig = load_original_module()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RunResult:
    planner_name: str
    seed: int
    scenario_name: str
    success_rate: float
    found_valid_path: int
    first_solution_ms: Optional[float]
    total_planning_ms: Optional[float]
    replans: Optional[float]
    repairs: Optional[float]
    path_length: Optional[float]
    min_clearance_px: Optional[float]
    recovery_ms: Optional[float]
    total_nodes: int
    status: str


# =============================================================================
# SCENARIO
# =============================================================================

def rect_from_center(cx: float, cy: float, w: int, h: int) -> ProjRect:
    x = int(round(cx - w / 2.0))
    y = int(round(cy - h / 2.0))
    return (x, y, w, h, 1.0)


def rects_overlap(r1: ProjRect, r2: ProjRect, tol: int = 12) -> bool:
    x1, y1, w1, h1, _ = r1
    x2, y2, w2, h2, _ = r2

    left1, right1 = x1 - tol, x1 + w1 + tol
    top1, bottom1 = y1 - tol, y1 + h1 + tol

    left2, right2 = x2 - tol, x2 + w2 + tol
    top2, bottom2 = y2 - tol, y2 + h2 + tol

    return not (
        right1 < left2 or
        right2 < left1 or
        bottom1 < top2 or
        bottom2 < top1
    )


def scenario_total_three_obstacles(frame_idx: int) -> List[ProjRect]:
    ax = 500
    ay = 430 - 0.70 * frame_idx
    a = rect_from_center(ax, ay, 90, 300)

    bx = 820 - 0.55 * frame_idx
    by = 235 + 0.38 * frame_idx
    b = rect_from_center(bx, by, 110, 80)

    cx = 170 + 0.70 * frame_idx
    cy = 545
    c = rect_from_center(cx, cy, 120, 70)

    assert not rects_overlap(a, b), f"A and B overlap at frame {frame_idx}"
    assert not rects_overlap(a, c), f"A and C overlap at frame {frame_idx}"
    assert not rects_overlap(b, c), f"B and C overlap at frame {frame_idx}"

    return [a, b, c]


def build_obstacles(frame_idx: int) -> List[ProjRect]:
    return scenario_total_three_obstacles(frame_idx)


def scenario_name() -> str:
    return "three_total_smooth_obstacles_2x2"


def default_start_goal() -> Tuple[Point, Point]:
    return (110, 590), (875, 120)


# =============================================================================
# HELPERS
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def planner_factory(name: str):
    if name == "RRT":
        return orig.RRTPlanner(color=COLOR_RRT)
    if name == "RRT_CONNECT":
        return orig.RRTConnectPlanner(color=COLOR_RRT_CONNECT)
    if name == "RRT_STAR":
        return orig.RRTStarPlanner(color=COLOR_RRT_STAR)
    if name == "RRTX":
        return orig.RRTXLitePlanner(color=COLOR_RRTX)
    raise ValueError(f"Unknown planner name: {name}")


def planner_names() -> List[str]:
    return ["RRT", "RRT_CONNECT", "RRT_STAR", "RRTX"]


def current_total_nodes(planner) -> int:
    if hasattr(planner, "total_nodes"):
        return int(planner.total_nodes())
    nodes = getattr(planner, "nodes", [])
    return len(nodes)


def obstacle_mask(frame_idx: int) -> np.ndarray:
    rects = build_obstacles(frame_idx)

    inflation = 8
    inflated = []
    for x, y, w, h, c in rects:
        inflated.append((x - inflation, y - inflation, w + 2 * inflation, h + 2 * inflation, c))

    return orig.obstacle_rects_to_mask(MAP_W, MAP_H, inflated)


def update_planner_for_obstacle_change(planner, mask: np.ndarray, start: Point, goal: Point) -> None:
    # Keep RRTX behavior as before: repair on every obstacle change.
    planner.update_obstacles(mask)
    is_rrtx = planner.__class__.__name__ == "RRTXLitePlanner"

    if is_rrtx:
        planner.start_recovery()
        planner.repair_after_obstacle_change()
    else:
        if getattr(planner, "path", None) is None or not planner.path_is_valid():
            planner.start_recovery()
            planner.reset(start, goal, MAP_W, MAP_H, mask)


def maybe_finish_recovery(planner) -> None:
    if getattr(planner, "recovering", False):
        path = getattr(planner, "path", None)
        if path is not None and planner.path_is_valid():
            planner.finish_recovery()


def blank_canvas() -> np.ndarray:
    return np.full((MAP_H, MAP_W, 3), 245, dtype=np.uint8)


def format_optional(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float) and np.isnan(x):
        return "NA"
    return f"{x:.1f}"


def ensure_dirs():
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)


def draw_single_panel(frame_idx: int, start: Point, goal: Point, planner_name: str, planner) -> np.ndarray:
    canvas = blank_canvas()

    for x, y, w, h, _ in build_obstacles(frame_idx):
        cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), (50, 50, 50), -1)
        cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), (100, 100, 100), 2)

    planner.draw(canvas)

    cv2.circle(canvas, start, 14, (0, 200, 0), -1)
    cv2.circle(canvas, goal, 14, (0, 0, 220), -1)
    cv2.putText(canvas, "S", (start[0] - 7, start[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "G", (goal[0] - 7, goal[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    lines = [
        f"{planner_name}",
        f"Frame: {frame_idx}/{MAX_FRAMES}",
        f"Status: {getattr(planner, 'status', 'NA')}",
        f"Nodes: {current_total_nodes(planner)}",
        f"Replans: {int(getattr(planner, 'replans', 0))}",
        f"Repairs: {int(getattr(planner, 'repairs', 0))}",
        f"First ms: {format_optional(getattr(planner, 'first_solution_time_ms', None))}",
        f"Plan ms: {getattr(planner, 'solve_time_ms', 0.0):.1f}",
        f"Recovery: {format_optional(getattr(planner, 'last_recovery_ms', None))}",
    ]
    y = 25
    for line in lines:
        cv2.putText(canvas, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
        y += 28

    return canvas


def make_2x2_view(frame_idx: int, start: Point, goal: Point, planners: Dict[str, object]) -> np.ndarray:
    p1 = draw_single_panel(frame_idx, start, goal, "RRT", planners["RRT"])
    p2 = draw_single_panel(frame_idx, start, goal, "RRT_CONNECT", planners["RRT_CONNECT"])
    p3 = draw_single_panel(frame_idx, start, goal, "RRT_STAR", planners["RRT_STAR"])
    p4 = draw_single_panel(frame_idx, start, goal, "RRTX", planners["RRTX"])

    top = np.hstack([p1, p2])
    bottom = np.hstack([p3, p4])
    big = np.vstack([top, bottom])

    cv2.putText(
        big,
        f"                 Seed: {BASE_SEED}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return big


def save_snapshot(img: np.ndarray, label: str, seed: int) -> None:
    ensure_dirs()
    path = Path(SNAPSHOT_DIR) / f"{label}_seed{seed}.png"
    cv2.imwrite(str(path), img)


def result_from_planner(planner, planner_name: str, seed: int) -> RunResult:
    path = getattr(planner, "path", None)
    valid_path = (path is not None) and planner.path_is_valid()

    if valid_path:
        path_length = float(planner.path_length_of(path))
        final_mask = obstacle_mask(MAX_FRAMES)
        clearance = orig.planner_min_clearance(path, final_mask)
        min_clearance_px = np.nan if clearance is None else float(clearance)

        first_solution_ms = getattr(planner, "first_solution_time_ms", None)
        total_planning_ms = float(getattr(planner, "solve_time_ms", 0.0))
        replans = float(getattr(planner, "replans", 0))
        repairs = float(getattr(planner, "repairs", 0))
        recovery_ms = getattr(planner, "last_recovery_ms", None)
    else:
        path_length = np.nan
        min_clearance_px = np.nan
        first_solution_ms = np.nan
        total_planning_ms = np.nan
        replans = np.nan
        repairs = np.nan
        recovery_ms = np.nan

    return RunResult(
        planner_name=planner_name,
        seed=seed,
        scenario_name=scenario_name(),
        success_rate=orig.planner_task_completion_rate(planner),
        found_valid_path=1 if valid_path else 0,
        first_solution_ms=first_solution_ms,
        total_planning_ms=total_planning_ms,
        replans=replans,
        repairs=repairs,
        path_length=path_length,
        min_clearance_px=min_clearance_px,
        recovery_ms=recovery_ms,
        total_nodes=current_total_nodes(planner),
        status=str(getattr(planner, "status", "UNKNOWN")),
    )


# =============================================================================
# 2X2 COMPARISON RUNNER
# =============================================================================

def run_comparison_2x2(seed: int, visualize: bool = True, save_snapshots_flag: bool = True) -> Dict[str, RunResult]:
    set_seed(seed)
    start, goal = default_start_goal()

    planners = {name: planner_factory(name) for name in planner_names()}
    initial_mask = obstacle_mask(0)

    for planner in planners.values():
        planner.reset(start, goal, MAP_W, MAP_H, initial_mask)

    prev_mask = initial_mask.copy()
    ensure_dirs()

    cv2.namedWindow("2x2 Planner Comparison", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("2x2 Planner Comparison", 1200, 800)
    cv2.moveWindow("2x2 Planner Comparison", 200, 100)

    for frame_idx in range(MAX_FRAMES + 1):
        current_mask = obstacle_mask(frame_idx)

        if orig.masks_changed(prev_mask, current_mask):
            for planner in planners.values():
                update_planner_for_obstacle_change(planner, current_mask, start, goal)
            prev_mask = current_mask.copy()

        elapsed_ms = (frame_idx + 1) * FRAME_DELAY_MS
        for planner in planners.values():
            planner.grow(STEP_BUDGET_PER_FRAME, elapsed_ms)
            maybe_finish_recovery(planner)

        big = make_2x2_view(frame_idx, start, goal, planners)

        if save_snapshots_flag and (frame_idx % SAVE_EVERY_N_FRAMES == 0):
            save_snapshot(big, f"frame_{frame_idx:03d}", seed)

        if visualize:
            screen_w = 1400
            h, w = big.shape[:2]
            scale = screen_w / w

            display_img = cv2.resize(
                big,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LINEAR
            )

            cv2.imshow("2x2 Planner Comparison", display_img)
            key = cv2.waitKey(FRAME_DELAY_MS) & 0xFF
            if key == 27 or key == ord("q"):
                break

    if visualize:
        cv2.destroyAllWindows()

    return {name: result_from_planner(planners[name], name, seed) for name in planner_names()}


# =============================================================================
# BATCH / CSV / PLOTS
# =============================================================================

def run_batch(seeds: List[int]) -> pd.DataFrame:
    rows: List[Dict] = []
    for planner_name in planner_names():
        print(f"\n=== {planner_name} ===")
        for seed in seeds:
            set_seed(seed)
            start, goal = default_start_goal()
            planner = planner_factory(planner_name)

            initial_mask = obstacle_mask(0)
            planner.reset(start, goal, MAP_W, MAP_H, initial_mask)
            prev_mask = initial_mask.copy()

            for frame_idx in range(MAX_FRAMES + 1):
                current_mask = obstacle_mask(frame_idx)
                if orig.masks_changed(prev_mask, current_mask):
                    update_planner_for_obstacle_change(planner, current_mask, start, goal)
                    prev_mask = current_mask.copy()

                elapsed_ms = (frame_idx + 1) * FRAME_DELAY_MS
                planner.grow(STEP_BUDGET_PER_FRAME, elapsed_ms)
                maybe_finish_recovery(planner)

            result = result_from_planner(planner, planner_name, seed)
            rows.append(asdict(result))
            print(
                f"seed={seed} | success={result.success_rate:.0f}% | "
                f"first_ms={format_optional(result.first_solution_ms)} | "
                f"plan_ms={format_optional(result.total_planning_ms)} | "
                f"replans={format_optional(result.replans)} | "
                f"repairs={format_optional(result.repairs)} | "
                f"path_len={format_optional(result.path_length)} | "
                f"clearance={format_optional(result.min_clearance_px)} | "
                f"recovery={format_optional(result.recovery_ms)}"
            )
    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame) -> Tuple[Path, Path]:
    ensure_dirs()
    detailed_path = Path(RESULTS_DIR) / "benchmark_detailed.csv"
    summary_path = Path(RESULTS_DIR) / "benchmark_summary.csv"

    df.to_csv(detailed_path, index=False)

    numeric_cols = [
        "success_rate", "found_valid_path", "first_solution_ms", "total_planning_ms",
        "replans", "repairs", "path_length", "min_clearance_px", "recovery_ms", "total_nodes"
    ]
    summary = df.groupby("planner_name")[numeric_cols].mean(numeric_only=True).reset_index()
    summary.to_csv(summary_path, index=False)

    return detailed_path, summary_path


def remove_outliers_for_plotting(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if not PLOT_REMOVE_OUTLIERS:
        return df

    filtered_parts = []
    for _, group in df.groupby("planner_name"):
        s = group[metric].dropna()
        if len(s) < 4:
            filtered_parts.append(group)
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            filtered_parts.append(group)
            continue

        low = q1 - OUTLIER_IQR_MULTIPLIER * iqr
        high = q3 + OUTLIER_IQR_MULTIPLIER * iqr
        keep_mask = group[metric].isna() | ((group[metric] >= low) & (group[metric] <= high))
        filtered_parts.append(group.loc[keep_mask])

    return pd.concat(filtered_parts, ignore_index=True)


def generate_plots(df: pd.DataFrame) -> List[Path]:
    ensure_dirs()
    plot_paths: List[Path] = []

    plot_specs = [
        ("success_rate", "Average Success Rate (%)", "success_rate.png", False),
        ("first_solution_ms", "Average Time to First Solution (ms)", "first_solution_ms.png", True),
        ("total_planning_ms", "Average Total Planning Time (ms)", "total_planning_ms.png", True),
        ("replans", "Average Number of Replans", "replans.png", True),
        ("repairs", "Average Number of Repairs", "repairs.png", True),
        ("path_length", "Average Planned Path Length (px)", "path_length.png", True),
        ("recovery_ms", "Average Recovery Time (ms)", "recovery_ms.png", True),
        ("min_clearance_px", "Average Minimum Clearance (px)", "min_clearance.png", True),
    ]

    for col, title, fname, use_success_only in plot_specs:
        plot_df = df.copy()

        if use_success_only:
            plot_df = plot_df[plot_df["found_valid_path"] == 1]

        plot_df = remove_outliers_for_plotting(plot_df, col)
        summary = plot_df.groupby("planner_name", as_index=False)[col].mean(numeric_only=True)

        plt.figure(figsize=(8, 5))
        plt.bar(summary["planner_name"], summary[col])
        plt.title(title)
        plt.xlabel("Planner")
        plt.ylabel(title)
        plt.tight_layout()
        out = Path(RESULTS_DIR) / fname
        plt.savefig(out, dpi=180)
        plt.close()
        plot_paths.append(out)

    return plot_paths


# =============================================================================
# MAIN
# =============================================================================

def main():
    seeds = [BASE_SEED + i for i in range(BATCH_RUNS)]

    print(f"Scenario: {scenario_name()}")
    print(f"Map size: {MAP_W} x {MAP_H}")
    print(f"Planners: {planner_names()}")
    print(f"Seeds: {seeds}")
    print(f"2x2 visualization: {VISUALIZE_COMPARISON_2X2}")
    print(f"Run batch experiments: {RUN_BATCH_EXPERIMENTS}")
    print(f"Generate plots: {GENERATE_PLOTS}")
    print(f"Save every N frames: {SAVE_EVERY_N_FRAMES}")
    print(f"Plot outlier removal: {PLOT_REMOVE_OUTLIERS}")

    if VISUALIZE_COMPARISON_2X2:
        print("\nShowing 2x2 live comparison. Press q or Esc to close early.")
        results = run_comparison_2x2(
            seed=BASE_SEED,
            visualize=True,
            save_snapshots_flag=SAVE_FRAME_SNAPSHOTS,
        )
        print("\nSingle-run final results:")
        for name, result in results.items():
            print(name, asdict(result))

    if RUN_BATCH_EXPERIMENTS:
        print("\nRunning batch experiments...")
        df = run_batch(seeds)
        detailed_path, summary_path = save_results(df)
        print(f"\nSaved detailed results to: {detailed_path}")
        print(f"Saved summary results to: {summary_path}")

        if GENERATE_PLOTS:
            plot_paths = generate_plots(df)
            print("\nSaved plots:")
            for p in plot_paths:
                print(p)

        print("\nBatch summary (means):")
        print(df.groupby("planner_name")[[
            "success_rate", "first_solution_ms", "total_planning_ms",
            "replans", "repairs", "path_length", "min_clearance_px", "recovery_ms"
        ]].mean(numeric_only=True).round(2))


if __name__ == "__main__":
    main()
