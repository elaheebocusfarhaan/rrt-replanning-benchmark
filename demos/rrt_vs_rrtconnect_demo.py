import os
# demo_compare_rrt_rrtconnect_60x40.py
# Side-by-side Basic RRT vs RRT-Connect in a 60x40 environment
#
# Features:
# - Similar scale/style to the earlier 60x40 demo
# - Feasible environment (not too narrow, not too cluttered)
# - Shared outer iteration counter
# - Basic RRT on left, RRT-Connect on right
# - Reliable MP4 saving
#
# Run:
#   python demo_compare_rrt_rrtconnect_60x40.py
#
# Requirements:
#   pip install matplotlib numpy imageio-ffmpeg

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import imageio_ffmpeg

Point = Tuple[float, float]
Rect = Tuple[float, float, float, float]  # xmin, ymin, xmax, ymax


# =========================
# USER SETTINGS
# =========================
WIDTH = 60
HEIGHT = 40

STEP_SIZE = 3
MAX_ITERS = 350
GOAL_SAMPLE_RATE = 0.10          # occasional goal sampling for both planners
GOAL_RADIUS = 3.0                # Basic RRT succeeds when within this radius of goal
CONNECT_MAX_STEPS = 2000
COLLISION_CHECK_RESOLUTION = 0.45
RANDOM_SEED = 3

PAUSE_EXTEND = 0.1
PAUSE_CONNECT_STEP = 0.05
PAUSE_FINAL = 1.8

SAVE_VIDEO = True
VIDEO_FILENAME = "rrt_vs_rrtconnect_60x40.mp4"
VIDEO_FPS = 12
VIDEO_DPI = 180

START_TREE_COLOR = "#1f77b4"   # blue
GOAL_TREE_COLOR = "#2ca02c"    # green
OBSTACLE_COLOR = "#666666"
PATH_COLOR = "#d62728"

TREE_NODE_SIZE = 14
ROOT_SIZE = 70
PATH_WIDTH = 3.0
# =========================


@dataclass
class Node:
    p: Point
    parent: Optional[int]


class Tree:
    def __init__(self, root: Point):
        self.nodes: List[Node] = [Node(p=root, parent=None)]

    def nearest_index(self, q: Point) -> int:
        qx, qy = q
        best_i = 0
        best_d2 = float("inf")
        for i, n in enumerate(self.nodes):
            nx, ny = n.p
            d2 = (nx - qx) ** 2 + (ny - qy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    def add_node(self, p: Point, parent: int) -> int:
        self.nodes.append(Node(p=p, parent=parent))
        return len(self.nodes) - 1


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def steer(from_p: Point, to_p: Point, step: float) -> Point:
    fx, fy = from_p
    tx, ty = to_p
    dx, dy = tx - fx, ty - fy
    d = math.hypot(dx, dy)
    if d <= step:
        return (tx, ty)
    s = step / d
    return (fx + s * dx, fy + s * dy)


def in_bounds(p: Point) -> bool:
    x, y = p
    return 0.0 <= x <= WIDTH and 0.0 <= y <= HEIGHT


def point_in_rect(p: Point, r: Rect) -> bool:
    x, y = p
    xmin, ymin, xmax, ymax = r
    return xmin <= x <= xmax and ymin <= y <= ymax


def point_collision_free(p: Point, obstacles: List[Rect]) -> bool:
    if not in_bounds(p):
        return False
    for r in obstacles:
        if point_in_rect(p, r):
            return False
    return True


def segment_collision_free(a: Point, b: Point, obstacles: List[Rect]) -> bool:
    ax, ay = a
    bx, by = b
    d = math.hypot(bx - ax, by - ay)
    if d == 0:
        return point_collision_free(a, obstacles)

    steps = max(2, int(d / COLLISION_CHECK_RESOLUTION) + 1)
    for i in range(steps):
        t = i / (steps - 1)
        x = ax + t * (bx - ax)
        y = ay + t * (by - ay)
        if not point_collision_free((x, y), obstacles):
            return False
    return True


REACHED = "REACHED"
ADVANCED = "ADVANCED"
TRAPPED = "TRAPPED"


def sample_point(goal: Point, obstacles: List[Rect]) -> Point:
    if random.random() < GOAL_SAMPLE_RATE:
        return goal

    for _ in range(12000):
        p = (random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
        if point_collision_free(p, obstacles):
            return p
    return goal


def extend(T: Tree, q_target: Point, obstacles: List[Rect], step_size: float) -> Tuple[str, Optional[int]]:
    i_near = T.nearest_index(q_target)
    q_near = T.nodes[i_near].p
    q_new = steer(q_near, q_target, step_size)

    if not point_collision_free(q_new, obstacles):
        return (TRAPPED, None)
    if not segment_collision_free(q_near, q_new, obstacles):
        return (TRAPPED, None)

    i_new = T.add_node(q_new, i_near)

    # tolerant reached criterion, helps finish a bit faster
    if dist(q_new, q_target) < 1e-9 or dist(q_new, q_target) < step_size * 0.5:
        return (REACHED, i_new)
    return (ADVANCED, i_new)


def trace_path(T: Tree, idx: int) -> List[Point]:
    path = []
    cur = idx
    while cur is not None:
        path.append(T.nodes[cur].p)
        cur = T.nodes[cur].parent
    path.reverse()
    return path


def reconstruct_bidirectional(Ta: Tree, idx_a: int, Tb: Tree, idx_b: int) -> List[Point]:
    pa = trace_path(Ta, idx_a)
    pb = trace_path(Tb, idx_b)
    pb.reverse()
    if pa and pb and pa[-1] == pb[0]:
        pb = pb[1:]
    return pa + pb


class BasicRRTState:
    def __init__(self, start: Point, goal: Point):
        self.tree = Tree(start)
        self.start = start
        self.goal = goal
        self.solved = False
        self.solve_iter: Optional[int] = None
        self.path: Optional[List[Point]] = None

    def step_once(self, q_rand: Point, obstacles: List[Rect], outer_iter: int):
        if self.solved:
            return

        _, i_new = extend(self.tree, q_rand, obstacles, STEP_SIZE)

        if i_new is not None:
            q_new = self.tree.nodes[i_new].p
            if dist(q_new, self.goal) <= GOAL_RADIUS and segment_collision_free(q_new, self.goal, obstacles):
                i_goal = self.tree.add_node(self.goal, i_new)
                self.path = trace_path(self.tree, i_goal)
                self.solved = True
                self.solve_iter = outer_iter


class RRTConnectState:
    def __init__(self, start: Point, goal: Point):
        self.T_start = Tree(start)
        self.T_goal = Tree(goal)

        self.Ta = self.T_start
        self.Tb = self.T_goal

        self.solved = False
        self.solve_iter: Optional[int] = None
        self.path: Optional[List[Point]] = None

        self.pending_connect = False
        self.current_target: Optional[Point] = None
        self.current_ext_new_idx: Optional[int] = None
        self.current_outer_iter: Optional[int] = None

    def extend_phase(self, q_rand: Point, obstacles: List[Rect], outer_iter: int):
        if self.solved:
            return

        _, ext_new_idx = extend(self.Ta, q_rand, obstacles, STEP_SIZE)
        self.current_outer_iter = outer_iter

        if ext_new_idx is not None:
            self.pending_connect = True
            self.current_ext_new_idx = ext_new_idx
            self.current_target = self.Ta.nodes[ext_new_idx].p
        else:
            self.pending_connect = False
            self.current_ext_new_idx = None
            self.current_target = None

    def connect_once(self, obstacles: List[Rect]):
        if self.solved or not self.pending_connect or self.current_target is None:
            return

        status, i_new = extend(self.Tb, self.current_target, obstacles, STEP_SIZE)

        if status == REACHED and i_new is not None and self.current_ext_new_idx is not None:
            if self.Ta is self.T_start:
                idx_start_side = self.current_ext_new_idx
                idx_goal_side = i_new
            else:
                idx_start_side = i_new
                idx_goal_side = self.current_ext_new_idx

            self.path = reconstruct_bidirectional(self.T_start, idx_start_side, self.T_goal, idx_goal_side)
            self.solved = True
            self.solve_iter = self.current_outer_iter
            self.pending_connect = False
            self.current_target = None
            self.current_ext_new_idx = None
            return

        if status != ADVANCED:
            self.pending_connect = False
            self.current_target = None
            self.current_ext_new_idx = None

    def finish_outer_iteration(self):
        if self.solved:
            return
        self.Ta, self.Tb = self.Tb, self.Ta


def draw_obstacles(ax, obstacles: List[Rect]):
    for (xmin, ymin, xmax, ymax) in obstacles:
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                facecolor=OBSTACLE_COLOR,
                edgecolor=OBSTACLE_COLOR,
                alpha=0.35,
            )
        )


def draw_tree(ax, T: Tree, color: str):
    for n in T.nodes:
        if n.parent is None:
            continue
        p = T.nodes[n.parent].p
        c = n.p
        ax.plot([p[0], c[0]], [p[1], c[1]], "-", linewidth=1.0, color=color)
    ax.scatter([n.p[0] for n in T.nodes], [n.p[1] for n in T.nodes], s=TREE_NODE_SIZE, color=color, zorder=3)


def draw_path(ax, path: Optional[List[Point]]):
    if not path or len(path) < 2:
        return
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    ax.plot(xs, ys, "-", linewidth=PATH_WIDTH, color=PATH_COLOR, zorder=6)


def setup_axes(ax, title: str):
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title(title, fontsize=13, fontweight="bold")


def draw_panel_basic(ax, state: BasicRRTState, obstacles: List[Rect]):
    ax.clear()
    setup_axes(ax, "Basic RRT")
    draw_obstacles(ax, obstacles)
    draw_tree(ax, state.tree, START_TREE_COLOR)
    draw_path(ax, state.path)

    ax.plot(state.start[0], state.start[1], "o", markersize=8, color=START_TREE_COLOR)
    ax.plot(state.goal[0], state.goal[1], "s", markersize=8, color=GOAL_TREE_COLOR)

    status_text = "Solved" if state.solved else "Running"
    if state.solved and state.solve_iter is not None:
        status_text += f" @ iter {state.solve_iter}"

    ax.text(
        0.02, 0.98, status_text,
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"),
    )


def draw_panel_connect(ax, state: RRTConnectState, start: Point, goal: Point, obstacles: List[Rect]):
    ax.clear()
    setup_axes(ax, "RRT-Connect")
    draw_obstacles(ax, obstacles)
    draw_tree(ax, state.T_start, START_TREE_COLOR)
    draw_tree(ax, state.T_goal, GOAL_TREE_COLOR)
    draw_path(ax, state.path)

    ax.plot(start[0], start[1], "o", markersize=8, color=START_TREE_COLOR)
    ax.plot(goal[0], goal[1], "s", markersize=8, color=GOAL_TREE_COLOR)

    status_text = "Solved" if state.solved else "Running"
    if state.solved and state.solve_iter is not None:
        status_text += f" @ iter {state.solve_iter}"

    ax.text(
        0.02, 0.98, status_text,
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"),
    )


def grab_with_hold(writer, fig, seconds: float):
    if writer is None:
        return
    n = max(1, int(round(seconds * VIDEO_FPS)))
    for _ in range(n):
        writer.grab_frame()


def configure_ffmpeg():
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_exe
    return ffmpeg_exe


def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    script_dir = Path(__file__).resolve().parent
    video_path = script_dir / VIDEO_FILENAME

    # Similar feel to your previous 60x40 environment, but easier / cleaner.
    # There is a clear winding route with no very sharp or tiny corridor.
    obstacles: List[Rect] = [
        (13, 0, 17, 18),
        (13, 28, 17, 40),

        (27, 14, 31, 40),

        (41, 0, 45, 22),

        (51, 18, 55, 40),
    ]

    start = (3, 10)
    goal = (58, 25)

    basic = BasicRRTState(start, goal)
    connect = RRTConnectState(start, goal)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    writer = None
    ffmpeg_exe = None
    if SAVE_VIDEO:
        ffmpeg_exe = configure_ffmpeg()
        writer = FFMpegWriter(
            fps=VIDEO_FPS,
            metadata={"title": "Basic RRT vs RRT-Connect (60x40)"},
        )

    def render_frame(outer_iter: int, hold: float):
        draw_panel_basic(ax1, basic, obstacles)
        draw_panel_connect(ax2, connect, start, goal, obstacles)
        fig.suptitle(f"Outer iteration = {outer_iter}", fontsize=15, fontweight="bold")
        fig.canvas.draw()
        fig.canvas.flush_events()
        grab_with_hold(writer, fig, hold)
        plt.pause(hold)

    def run_loop():
        outer_iter = 0

        for outer_iter in range(1, MAX_ITERS + 1):
            if not plt.fignum_exists(fig.number):
                print("Window closed — exiting.")
                return

            q_rand = sample_point(goal, obstacles)

            if not basic.solved:
                basic.step_once(q_rand, obstacles, outer_iter)

            if not connect.solved:
                connect.extend_phase(q_rand, obstacles, outer_iter)

            render_frame(outer_iter, PAUSE_EXTEND)

            connect_steps = 0
            while (not connect.solved) and connect.pending_connect and connect_steps < CONNECT_MAX_STEPS:
                connect_steps += 1
                if not plt.fignum_exists(fig.number):
                    print("Window closed — exiting.")
                    return
                connect.connect_once(obstacles)
                render_frame(outer_iter, PAUSE_CONNECT_STEP)

            if not connect.solved:
                connect.finish_outer_iteration()

            if basic.solved and connect.solved:
                render_frame(outer_iter, PAUSE_FINAL)
                break

        render_frame(outer_iter, PAUSE_FINAL)

    try:
        if writer is not None:
            with writer.saving(fig, str(video_path), dpi=VIDEO_DPI):
                run_loop()
        else:
            run_loop()
    finally:
        plt.ioff()

    if SAVE_VIDEO:
        print(f"Using ffmpeg from: {ffmpeg_exe}")
        print(f"Saved video: {video_path}")

    plt.show()


if __name__ == "__main__":
    main()