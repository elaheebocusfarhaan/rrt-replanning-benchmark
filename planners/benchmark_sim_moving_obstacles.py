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
VISUALIZE_SINGLE_RUN = True      # show one live run in OpenCV
RUN_BATCH_EXPERIMENTS = False    # keep off for now
GENERATE_PLOTS = False           # keep off for now
SAVE_FRAME_SNAPSHOTS = True      # save useful before/after screenshots

# Runtime / experiment settings
MAP_W = 1000
MAP_H = 700
STEP_BUDGET_PER_FRAME = 120
MAX_FRAMES = 260
FRAME_DELAY_MS = 35              # used only when VISUALIZE_SINGLE_RUN=True
BATCH_RUNS = 10
BASE_SEED = 42

SNAPSHOT_DIR = "benchmark_snapshots"
RESULTS_DIR = "benchmark_results"

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
            f"Put this benchmark script in the same folder as the original file, "
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
    total_planning_ms: float
    replans: int
    repairs: int
    path_length: float
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
    """
    Exactly 3 obstacles total.
    All 3 move smoothly.
    Designed so they stay separated with margin and do not collide.
    """

    # Obstacle A: tall vertical wall in the middle.
    # It goes from low to high, but never seals the full top or bottom route.
    # Only a slight vertical drift so the scene stays readable.
    ax = 500
    ay = 430 - 0.70 * frame_idx
    a = rect_from_center(ax, ay, 90, 300)

    # Obstacle B: diagonal motion from upper-right toward mid-right.
    # Chosen to stay to the RIGHT of obstacle A with margin.
    bx = 820 - 0.55 * frame_idx
    by = 235 + 0.38 * frame_idx
    b = rect_from_center(bx, by, 110, 80)

    # Obstacle C: left-to-right motion in the lower-left / lower-middle region.
    # Chosen to stay BELOW obstacle A and BELOW obstacle B.
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
    return "three_total_smooth_obstacles_v2"


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

    inflation = 8  # pixels (tune between 5–15)

    inflated = []
    for x, y, w, h, c in rects:
        inflated.append((
            x - inflation,
            y - inflation,
            w + 2 * inflation,
            h + 2 * inflation,
            c
        ))

    return orig.obstacle_rects_to_mask(MAP_W, MAP_H, inflated)


def update_planner_for_obstacle_change(planner, mask: np.ndarray, start: Point, goal: Point) -> None:
    """
    Keep algorithm code unchanged.
    - RRTX repairs its existing tree.
    - Others update obstacle mask and replan from scratch for the same query.
    """
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


def draw_scene(img: np.ndarray, frame_idx: int, start: Point, goal: Point, planner_name: str, planner) -> np.ndarray:
    canvas = img.copy()

    # Draw obstacles
    for x, y, w, h, _ in build_obstacles(frame_idx):
        cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), (50, 50, 50), -1)
        cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), (100, 100, 100), 2)

    # Draw planner tree/path using original planner draw methods
    planner.draw(canvas)

    # Start / goal
    cv2.circle(canvas, start, 14, (0, 200, 0), -1)
    cv2.circle(canvas, goal, 14, (0, 0, 220), -1)
    cv2.putText(canvas, "S", (start[0] - 7, start[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "G", (goal[0] - 7, goal[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Text panel
    lines = [
        f"Planner: {planner_name}",
        f"Frame: {frame_idx}/{MAX_FRAMES}",
        f"Status: {getattr(planner, 'status', 'NA')}",
        f"Nodes: {current_total_nodes(planner)}",
        f"Replans: {int(getattr(planner, 'replans', 0))}",
        f"Repairs: {int(getattr(planner, 'repairs', 0))}",
        f"First solution ms: {format_optional(getattr(planner, 'first_solution_time_ms', None))}",
        f"Planning ms: {getattr(planner, 'solve_time_ms', 0.0):.1f}",
        f"Recovery ms: {format_optional(getattr(planner, 'last_recovery_ms', None))}",
    ]
    y = 25
    for line in lines:
        cv2.putText(canvas, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
        y += 28

    return canvas


def blank_canvas() -> np.ndarray:
    return np.full((MAP_H, MAP_W, 3), 245, dtype=np.uint8)


def format_optional(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.1f}"


def ensure_dirs():
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)


def save_snapshot(img: np.ndarray, planner_name: str, label: str, seed: int) -> None:
    ensure_dirs()
    path = Path(SNAPSHOT_DIR) / f"{planner_name.lower()}_{label}_seed{seed}.png"
    cv2.imwrite(str(path), img)


def result_from_planner(planner, planner_name: str, seed: int) -> RunResult:
    path = getattr(planner, "path", None)
    path_length = planner.path_length_of(path) if path is not None else 0.0
    final_mask = obstacle_mask(MAX_FRAMES)
    clearance = orig.planner_min_clearance(path, final_mask)

    return RunResult(
        planner_name=planner_name,
        seed=seed,
        scenario_name=scenario_name(),
        success_rate=orig.planner_task_completion_rate(planner),
        found_valid_path=1 if path is not None and planner.path_is_valid() else 0,
        first_solution_ms=getattr(planner, "first_solution_time_ms", None),
        total_planning_ms=float(getattr(planner, "solve_time_ms", 0.0)),
        replans=int(getattr(planner, "replans", 0)),
        repairs=int(getattr(planner, "repairs", 0)),
        path_length=float(path_length),
        min_clearance_px=None if clearance is None else float(clearance),
        recovery_ms=getattr(planner, "last_recovery_ms", None),
        total_nodes=current_total_nodes(planner),
        status=str(getattr(planner, "status", "UNKNOWN")),
    )


# =============================================================================
# CORE RUNNERS
# =============================================================================

def run_single(planner_name: str, seed: int, visualize: bool = True, save_snapshots: bool = True) -> RunResult:
    set_seed(seed)
    start, goal = default_start_goal()
    planner = planner_factory(planner_name)

    initial_mask = obstacle_mask(0)
    planner.reset(start, goal, MAP_W, MAP_H, initial_mask)

    prev_mask = initial_mask.copy()
    ensure_dirs()

    snap_1_saved = False
    snap_2_saved = False
    snap_3_saved = False

    for frame_idx in range(MAX_FRAMES + 1):
        current_mask = obstacle_mask(frame_idx)

        if orig.masks_changed(prev_mask, current_mask):
            update_planner_for_obstacle_change(planner, current_mask, start, goal)
            prev_mask = current_mask.copy()

        elapsed_ms = frame_idx * FRAME_DELAY_MS
        planner.grow(STEP_BUDGET_PER_FRAME, elapsed_ms)
        maybe_finish_recovery(planner)

        frame = draw_scene(blank_canvas(), frame_idx, start, goal, planner_name, planner)

        if save_snapshots:
            if frame_idx == 40 and not snap_1_saved:
                save_snapshot(frame, planner_name, "early_motion", seed)
                snap_1_saved = True
            if frame_idx == 120 and not snap_2_saved:
                save_snapshot(frame, planner_name, "mid_motion", seed)
                snap_2_saved = True
            if frame_idx == 210 and not snap_3_saved:
                save_snapshot(frame, planner_name, "late_motion", seed)
                snap_3_saved = True

        if visualize:
            cv2.imshow("Benchmark Simulation", frame)
            key = cv2.waitKey(FRAME_DELAY_MS) & 0xFF
            if key == 27 or key == ord("q"):
                break

    if visualize:
        cv2.destroyAllWindows()

    return result_from_planner(planner, planner_name, seed)


def run_batch(planner_list: List[str], seeds: List[int]) -> pd.DataFrame:
    rows: List[Dict] = []
    for planner_name in planner_list:
        print(f"\n=== {planner_name} ===")
        for seed in seeds:
            result = run_single(
                planner_name=planner_name,
                seed=seed,
                visualize=False,
                save_snapshots=False,
            )
            rows.append(asdict(result))
            print(
                f"seed={seed} | success={result.success_rate:.0f}% | "
                f"first_ms={format_optional(result.first_solution_ms)} | "
                f"plan_ms={result.total_planning_ms:.1f} | replans={result.replans} | "
                f"repairs={result.repairs} | path_len={result.path_length:.1f} | "
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


def generate_plots(df: pd.DataFrame) -> List[Path]:
    ensure_dirs()
    plot_paths: List[Path] = []

    summary = df.groupby("planner_name", as_index=False).agg({
        "success_rate": "mean",
        "first_solution_ms": "mean",
        "total_planning_ms": "mean",
        "replans": "mean",
        "repairs": "mean",
        "path_length": "mean",
        "min_clearance_px": "mean",
        "recovery_ms": "mean",
    })

    plot_specs = [
        ("success_rate", "Average Success Rate (%)", "success_rate.png"),
        ("first_solution_ms", "Average Time to First Solution (ms)", "first_solution_ms.png"),
        ("total_planning_ms", "Average Total Planning Time (ms)", "total_planning_ms.png"),
        ("replans", "Average Number of Replans", "replans.png"),
        ("path_length", "Average Planned Path Length (px)", "path_length.png"),
        ("recovery_ms", "Average Recovery Time (ms)", "recovery_ms.png"),
        ("min_clearance_px", "Average Minimum Clearance (px)", "min_clearance.png"),
    ]

    for col, title, fname in plot_specs:
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
    names = planner_names()

    print(f"Scenario: {scenario_name()}")
    print(f"Map size: {MAP_W} x {MAP_H}")
    print(f"Planners: {names}")
    print(f"Seeds: {seeds}")
    print(f"Visualize single run: {VISUALIZE_SINGLE_RUN}")
    print(f"Run batch experiments: {RUN_BATCH_EXPERIMENTS}")
    print(f"Generate plots: {GENERATE_PLOTS}")

    if VISUALIZE_SINGLE_RUN:
        print("\nShowing one live run per planner. Press q or Esc to close a window early.")
        for name in names:
            print(f"\nLive visualization: {name}")
            result = run_single(
                planner_name=name,
                seed=BASE_SEED,
                visualize=True,
                save_snapshots=SAVE_FRAME_SNAPSHOTS,
            )
            print(asdict(result))

    if RUN_BATCH_EXPERIMENTS:
        print("\nRunning batch experiments...")
        df = run_batch(names, seeds)
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
