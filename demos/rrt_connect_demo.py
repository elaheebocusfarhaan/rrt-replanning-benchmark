import os
# compare_rrt_vs_rrtconnect_async.py
# Side-by-side comparison:
#   left  = Basic RRT
#   right = RRT-Connect
#
# Behavior:
# - Shared OUTER iteration counter
# - Basic RRT shows exactly one EXTEND per outer iteration
# - RRT-Connect shows one EXTEND, then all CONNECT substeps in the same outer iteration
# - While RRT-Connect is showing CONNECT substeps, Basic RRT stays frozen
# - If RRT-Connect solves first, it stays frozen while Basic RRT continues
# - MP4 saving in same folder as this script
#
# Run:
#   python compare_rrt_vs_rrtconnect_async.py
#
# Requirements:
#   pip install matplotlib imageio-ffmpeg

import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.animation import FFMpegWriter
import imageio_ffmpeg


# =========================
# USER SETTINGS
# =========================
MAX_ITERATIONS = 300
STEP_SIZE = 0.05
RANDOM_SEED = 3
CONNECT_MAX_STEPS = 600
GOAL_RADIUS = 0.04

X_LIM = (0.0, 0.45)
Y_LIM = (0.0, 0.45)

# Animation speed (seconds shown on screen)
PAUSE_BASIC_EXTEND = 2.0
PAUSE_CONNECT_EXTEND = 2.0
PAUSE_CONNECT_STEP = 1.4
PAUSE_FINAL = 2.5

# Video saving
SAVE_VIDEO = False
VIDEO_FILENAME = "compare_rrt_vs_rrtconnect_async.mp4"
VIDEO_FPS = 2
VIDEO_DPI = 140

SHOW_FLOATING_TEXT = True
SHOW_ROLE_BOX = True

SHOW_QRAND = True
SHOW_QNEAR = True
SHOW_QNEW = True
SHOW_QTARGET = True

SHOW_LABEL_QRAND = True
SHOW_LABEL_QNEAR = True
SHOW_LABEL_QNEW = True
SHOW_LABEL_QTARGET = True
# =========================


# =========================
# VISUAL STYLE
# =========================
START_TREE_COLOR = "#1f77b4"   # blue
GOAL_TREE_COLOR = "#2ca02c"    # green
OBSTACLE_COLOR = "#7f7f7f"     # gray

QRAND_COLOR = "black"
QNEAR_COLOR = "#ff7f0e"
QNEW_COLOR = "#d62728"
QTARGET_COLOR = "#9467bd"

TREE_NODE_SIZE = 26
ROOT_SIZE = 75
SPECIAL_SIZE = 95
PATH_COLOR = "#d62728"
# =========================


COLLISION_CHECKS = {
    "rrt": 0,
    "rrt_connect": 0,
}


@dataclass
class RectObs:
    x: float
    y: float
    w: float
    h: float

    def contains(self, px: float, py: float) -> bool:
        return (self.x <= px <= self.x + self.w) and (self.y <= py <= self.y + self.h)


def collision_free_segment(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    obstacles: List[RectObs],
    counter_key: str,
    resolution: float = 0.003,
) -> bool:
    COLLISION_CHECKS[counter_key] += 1

    x1, y1 = p1
    x2, y2 = p2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist == 0:
        return True

    n = max(2, int(dist / resolution))
    for i in range(n + 1):
        t = i / n
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        for obs in obstacles:
            if obs.contains(x, y):
                return False
    return True


@dataclass
class Node:
    x: float
    y: float
    parent: Optional[int] = None


class Tree:
    def __init__(self, root: Tuple[float, float]):
        self.nodes: List[Node] = [Node(root[0], root[1], None)]

    def nearest_index(self, q: Tuple[float, float]) -> int:
        qx, qy = q
        best_i = 0
        best_d = float("inf")
        for i, n in enumerate(self.nodes):
            d = (n.x - qx) ** 2 + (n.y - qy) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def add_node(self, pt: Tuple[float, float], parent_idx: int) -> int:
        idx = len(self.nodes)
        self.nodes.append(Node(pt[0], pt[1], parent_idx))
        return idx


REACHED = "Reached"
ADVANCED = "Advanced"
TRAPPED = "Trapped"


def steer(from_pt: Tuple[float, float], to_pt: Tuple[float, float], eps: float) -> Tuple[float, float]:
    x1, y1 = from_pt
    x2, y2 = to_pt
    dx, dy = x2 - x1, y2 - y1
    d = math.hypot(dx, dy)
    if d <= eps:
        return (x2, y2)
    s = eps / d
    return (x1 + s * dx, y1 + s * dy)


def EXTEND(
    T: Tree,
    q_target: Tuple[float, float],
    obstacles: List[RectObs],
    eps: float,
    counter_key: str,
):
    i_near = T.nearest_index(q_target)
    q_near = (T.nodes[i_near].x, T.nodes[i_near].y)
    q_new = steer(q_near, q_target, eps)

    if not collision_free_segment(q_near, q_new, obstacles, counter_key=counter_key):
        return TRAPPED, i_near, None, q_near, q_new

    i_new = T.add_node(q_new, i_near)
    status = REACHED if q_new == q_target else ADVANCED
    return status, i_near, i_new, q_near, q_new


def extract_path(T: Tree, idx: int) -> List[Tuple[float, float]]:
    path = []
    cur = idx
    while cur is not None:
        n = T.nodes[cur]
        path.append((n.x, n.y))
        cur = n.parent
    path.reverse()
    return path


def extract_bidirectional_path(
    T_start: Tree,
    T_goal: Tree,
    idx_start_side: int,
    idx_goal_side: int,
) -> List[Tuple[float, float]]:
    path_start = extract_path(T_start, idx_start_side)
    path_goal = extract_path(T_goal, idx_goal_side)
    path_goal.reverse()
    return path_start + path_goal[1:]


def sample_free(obstacles: List[RectObs]) -> Tuple[float, float]:
    for _ in range(8000):
        x = random.uniform(X_LIM[0], X_LIM[1])
        y = random.uniform(Y_LIM[0], Y_LIM[1])
        if not any(obs.contains(x, y) for obs in obstacles):
            return (x, y)
    return ((X_LIM[0] + X_LIM[1]) / 2, (Y_LIM[0] + Y_LIM[1]) / 2)


def setup_axes(ax, title=""):
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.set_title(title, fontsize=13, fontweight="bold")


def draw_obstacles(ax, obstacles: List[RectObs]):
    for obs in obstacles:
        ax.add_patch(
            plt.Rectangle(
                (obs.x, obs.y), obs.w, obs.h,
                facecolor=OBSTACLE_COLOR, edgecolor=OBSTACLE_COLOR, alpha=0.18
            )
        )


def draw_tree(ax, T: Tree, color: str):
    for n in T.nodes:
        if n.parent is None:
            continue
        p = T.nodes[n.parent]
        ax.plot([p.x, n.x], [p.y, n.y], linewidth=1.4, color=color)
    ax.scatter([n.x for n in T.nodes], [n.y for n in T.nodes], s=TREE_NODE_SIZE, color=color, zorder=3)


def draw_path(ax, path: Optional[List[Tuple[float, float]]]):
    if not path or len(path) < 2:
        return
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    ax.plot(xs, ys, linewidth=3.0, color=PATH_COLOR, zorder=5)


def annotate_point(ax, pt, text, dx=0.008, dy=0.008, fontsize=9):
    if pt is None:
        return
    ax.text(
        pt[0] + dx,
        pt[1] + dy,
        text,
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.72, edgecolor="none"),
        zorder=8,
    )


def draw_special(ax, q_start, q_goal, q_rand=None, q_near=None, q_new=None, q_target=None):
    ax.scatter([q_start[0]], [q_start[1]], s=ROOT_SIZE, marker="o", color=START_TREE_COLOR, zorder=6)
    ax.scatter([q_goal[0]], [q_goal[1]], s=ROOT_SIZE, marker="s", color=GOAL_TREE_COLOR, zorder=6)

    if SHOW_QRAND and q_rand is not None:
        ax.scatter([q_rand[0]], [q_rand[1]], s=SPECIAL_SIZE, marker="*", color=QRAND_COLOR, zorder=7)
    if SHOW_QNEAR and q_near is not None:
        ax.scatter([q_near[0]], [q_near[1]], s=SPECIAL_SIZE, marker="o", color=QNEAR_COLOR, zorder=7)
    if SHOW_QNEW and q_new is not None:
        ax.scatter([q_new[0]], [q_new[1]], s=SPECIAL_SIZE, marker="x", color=QNEW_COLOR, zorder=7)
    if SHOW_QTARGET and q_target is not None:
        ax.scatter([q_target[0]], [q_target[1]], s=SPECIAL_SIZE, marker="x", color=QTARGET_COLOR, zorder=7)

    if SHOW_LABEL_QRAND and q_rand is not None:
        annotate_point(ax, q_rand, "q_rand", dx=0.008, dy=0.010)
    if SHOW_LABEL_QNEAR and q_near is not None:
        annotate_point(ax, q_near, "q_near", dx=0.008, dy=0.010)
    if SHOW_LABEL_QNEW and q_new is not None:
        annotate_point(ax, q_new, "q_new", dx=0.008, dy=-0.014)
    if SHOW_LABEL_QTARGET and q_target is not None:
        annotate_point(ax, q_target, "q_target", dx=0.008, dy=-0.014)


def add_legend(ax):
    handles = [
        Line2D([0], [0], color=START_TREE_COLOR, lw=2, label="Start tree"),
        Line2D([0], [0], color=GOAL_TREE_COLOR, lw=2, label="Goal tree"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor=OBSTACLE_COLOR, alpha=0.18, label="Obstacle"),
        Line2D([0], [0], color=PATH_COLOR, lw=3, label="Path"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=QRAND_COLOR, markersize=12, label="q_rand"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=QNEAR_COLOR, markersize=10, label="q_near"),
        Line2D([0], [0], marker="x", color=QNEW_COLOR, markersize=10, label="q_new"),
        Line2D([0], [0], marker="x", color=QTARGET_COLOR, markersize=10, label="q_target"),
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.85, fontsize=8)


def draw_info_box(ax, text: str, x=0.02, y=0.98):
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.75, edgecolor="none"),
        zorder=8,
    )


class BasicRRTState:
    def __init__(self, q_start, q_goal):
        self.tree = Tree(q_start)
        self.q_start = q_start
        self.q_goal = q_goal

        self.solved = False
        self.path = None
        self.solve_iteration = None
        self.solve_time = None

        self.last_qrand = None
        self.last_qnear = None
        self.last_qnew = None
        self.last_status = "initial"
        self.last_mode = "waiting"

    def step_once(self, obstacles: List[RectObs], q_rand, outer_iter: int, t0: float):
        if self.solved:
            self.last_mode = "solved"
            return

        status, _, i_new, q_near, q_new = EXTEND(
            self.tree, q_rand, obstacles, STEP_SIZE, counter_key="rrt"
        )

        self.last_qrand = q_rand
        self.last_qnear = q_near
        self.last_qnew = q_new if i_new is not None else None
        self.last_status = status
        self.last_mode = "EXTEND"

        if i_new is not None:
            dist_to_goal = math.hypot(q_new[0] - self.q_goal[0], q_new[1] - self.q_goal[1])
            if dist_to_goal <= GOAL_RADIUS:
                i_goal = self.tree.add_node(self.q_goal, i_new)
                self.path = extract_path(self.tree, i_goal)
                self.solved = True
                self.solve_iteration = outer_iter
                self.solve_time = time.perf_counter() - t0
                self.last_status = "Solved"
                self.last_mode = "solved"


class RRTConnectState:
    def __init__(self, q_start, q_goal):
        self.T_start = Tree(q_start)
        self.T_goal = Tree(q_goal)
        self.q_start = q_start
        self.q_goal = q_goal

        self.Ta = self.T_start
        self.Tb = self.T_goal
        self.Ta_name = "T_start"
        self.Tb_name = "T_goal"

        self.solved = False
        self.path = None
        self.solve_iteration = None
        self.solve_time = None

        self.last_qrand = None
        self.last_qnear = None
        self.last_qnew = None
        self.last_qtarget = None
        self.last_status = "initial"
        self.last_mode = "waiting"

        self.pending_connect = False
        self.current_target = None
        self.current_ext_new_idx = None
        self.current_outer_iter = None

    def step_extend_phase(self, obstacles: List[RectObs], q_rand, outer_iter: int):
        if self.solved:
            self.last_mode = "solved"
            return

        ext_status, _, ext_new_idx, q_near, q_new_prop = EXTEND(
            self.Ta, q_rand, obstacles, STEP_SIZE, counter_key="rrt_connect"
        )
        q_new = q_new_prop if ext_new_idx is not None else None

        self.last_qrand = q_rand
        self.last_qnear = q_near
        self.last_qnew = q_new
        self.last_qtarget = None
        self.last_status = ext_status
        self.last_mode = "EXTEND"

        self.current_outer_iter = outer_iter

        if ext_new_idx is not None:
            self.pending_connect = True
            self.current_target = q_new
            self.current_ext_new_idx = ext_new_idx
        else:
            self.pending_connect = False
            self.current_target = None
            self.current_ext_new_idx = None

    def step_connect_once(self, obstacles: List[RectObs], t0: float):
        if self.solved or not self.pending_connect:
            return

        q_target = self.current_target
        conn_status, _, i_new, c_near, c_new_prop = EXTEND(
            self.Tb, q_target, obstacles, STEP_SIZE, counter_key="rrt_connect"
        )
        c_new = c_new_prop if i_new is not None else None

        self.last_qrand = None
        self.last_qnear = c_near
        self.last_qnew = c_new
        self.last_qtarget = q_target
        self.last_status = conn_status
        self.last_mode = "CONNECT"

        if conn_status == REACHED:
            if self.Ta is self.T_start:
                idx_start_side = self.current_ext_new_idx
                idx_goal_side = i_new
            else:
                idx_start_side = i_new
                idx_goal_side = self.current_ext_new_idx

            self.path = extract_bidirectional_path(
                self.T_start, self.T_goal, idx_start_side, idx_goal_side
            )
            self.solved = True
            self.solve_iteration = self.current_outer_iter
            self.solve_time = time.perf_counter() - t0
            self.last_status = "Solved"
            self.last_mode = "solved"
            self.pending_connect = False
            self.current_target = None
            self.current_ext_new_idx = None
            return

        if conn_status != ADVANCED:
            self.pending_connect = False
            self.current_target = None
            self.current_ext_new_idx = None

    def finish_outer_iteration(self):
        if self.solved:
            return
        self.Ta, self.Tb = self.Tb, self.Ta
        self.Ta_name, self.Tb_name = self.Tb_name, self.Ta_name


def render_basic_rrt(ax, obstacles, state: BasicRRTState):
    setup_axes(ax, "Basic RRT")
    draw_obstacles(ax, obstacles)
    draw_tree(ax, state.tree, START_TREE_COLOR)
    draw_path(ax, state.path)

    show_markers = not state.solved
    draw_special(
        ax,
        state.q_start,
        state.q_goal,
        q_rand=state.last_qrand if show_markers else None,
        q_near=state.last_qnear if show_markers else None,
        q_new=state.last_qnew if show_markers else None,
        q_target=None,
    )

    add_legend(ax)

    info = f"mode: {state.last_mode}\nstatus: {state.last_status}"
    if state.solved:
        info += f"\nsolved at iter {state.solve_iteration}"
    if SHOW_FLOATING_TEXT:
        draw_info_box(ax, info, x=0.02, y=0.98)


def render_rrt_connect(ax, obstacles, state: RRTConnectState):
    setup_axes(ax, "RRT-Connect")
    draw_obstacles(ax, obstacles)
    draw_tree(ax, state.T_start, START_TREE_COLOR)
    draw_tree(ax, state.T_goal, GOAL_TREE_COLOR)
    draw_path(ax, state.path)

    show_markers = not state.solved
    draw_special(
        ax,
        state.q_start,
        state.q_goal,
        q_rand=state.last_qrand if (state.last_mode == "EXTEND" and show_markers) else None,
        q_near=state.last_qnear if show_markers else None,
        q_new=state.last_qnew if show_markers else None,
        q_target=state.last_qtarget if (state.last_mode == "CONNECT" and show_markers) else None,
    )

    add_legend(ax)

    if SHOW_FLOATING_TEXT:
        text = f"mode: {state.last_mode}\nstatus: {state.last_status}"
        if state.solved:
            text += f"\nsolved at iter {state.solve_iteration}"
        draw_info_box(ax, text, x=0.02, y=0.98)

    if SHOW_ROLE_BOX:
        role_text = f"Ta = {state.Ta_name}\nTb = {state.Tb_name}"
        draw_info_box(ax, role_text, x=0.02, y=0.77)


def draw_both(fig, ax1, ax2, obstacles, basic, connect, outer_iter, subtitle=""):
    ax1.clear()
    ax2.clear()

    render_basic_rrt(ax1, obstacles, basic)
    render_rrt_connect(ax2, obstacles, connect)

    main_title = f"Outer iteration = {outer_iter}"
    if subtitle:
        main_title += f"   |   {subtitle}"
    fig.suptitle(main_title, fontsize=16, fontweight="bold")

    fig.canvas.draw()
    fig.canvas.flush_events()


def maybe_capture(writer, fig):
    if writer is not None:
        writer.grab_frame()


def do_pause(seconds: float):
    plt.pause(seconds)


def configure_ffmpeg():
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_exe
    return ffmpeg_exe


def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    script_dir = Path(__file__).resolve().parent
    video_path = script_dir / VIDEO_FILENAME

    obstacles = [
        RectObs(0.20, 0.00, 0.03, 0.20),
        RectObs(0.20, 0.20, 0.18, 0.03),
    ]

    q_start = (0.05, 0.05)
    q_goal = (0.40, 0.40)

    basic = BasicRRTState(q_start, q_goal)
    connect = RRTConnectState(q_start, q_goal)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    writer = None
    ffmpeg_exe = None

    if SAVE_VIDEO:
        ffmpeg_exe = configure_ffmpeg()
        metadata = dict(title="Basic RRT vs RRT-Connect", artist="OpenAI")
        writer = FFMpegWriter(fps=VIDEO_FPS, metadata=metadata)

    t0 = time.perf_counter()

    def render_and_optionally_save(outer_iter, subtitle, pause_seconds):
        draw_both(fig, ax1, ax2, obstacles, basic, connect, outer_iter, subtitle=subtitle)
        maybe_capture(writer, fig)
        if pause_seconds > 0:
            do_pause(pause_seconds)

    try:
        if writer is not None:
            with writer.saving(fig, str(video_path), dpi=VIDEO_DPI):
                run_simulation(render_and_optionally_save, basic, connect, obstacles, t0, fig)
        else:
            run_simulation(render_and_optionally_save, basic, connect, obstacles, t0, fig)
    finally:
        plt.ioff()

    print("\n===== SUMMARY =====")
    if basic.solved:
        print(f"Basic RRT solved at outer iteration: {basic.solve_iteration}")
        print(f"Basic RRT time to first path: {basic.solve_time:.4f} s")
    else:
        print("Basic RRT did not solve within max iterations.")
    print(f"Basic RRT collision checks: {COLLISION_CHECKS['rrt']}")

    print()

    if connect.solved:
        print(f"RRT-Connect solved at outer iteration: {connect.solve_iteration}")
        print(f"RRT-Connect time to first path: {connect.solve_time:.4f} s")
    else:
        print("RRT-Connect did not solve within max iterations.")
    print(f"RRT-Connect collision checks: {COLLISION_CHECKS['rrt_connect']}")

    print("\nImportant note:")
    print("Outer iterations are not a perfectly fair metric by themselves,")
    print("because RRT-Connect can add multiple nodes during one CONNECT call.")

    if SAVE_VIDEO:
        print(f"\nUsing ffmpeg from: {ffmpeg_exe}")
        print(f"Saved video: {video_path}")

    plt.show()


def run_simulation(render_and_optionally_save, basic, connect, obstacles, t0, fig):
    for outer_iter in range(1, MAX_ITERATIONS + 1):
        if not plt.fignum_exists(fig.number):
            print("Window closed — exiting.")
            return

        q_rand = sample_free(obstacles)

        # -------------------------
        # Basic RRT: one EXTEND only
        # -------------------------
        if not basic.solved:
            basic.step_once(obstacles, q_rand, outer_iter, t0)

        # -------------------------
        # RRT-Connect: first EXTEND
        # -------------------------
        if not connect.solved:
            connect.step_extend_phase(obstacles, q_rand, outer_iter)

        render_and_optionally_save(
            outer_iter,
            "Basic: one EXTEND  |  Connect: EXTEND phase",
            max(PAUSE_BASIC_EXTEND, PAUSE_CONNECT_EXTEND),
        )

        # -------------------------
        # RRT-Connect: repeated CONNECT substeps
        # Basic stays frozen here
        # -------------------------
        while (not connect.solved) and connect.pending_connect:
            if not plt.fignum_exists(fig.number):
                print("Window closed — exiting.")
                return

            connect.step_connect_once(obstacles, t0)

            render_and_optionally_save(
                outer_iter,
                "Connect: repeated CONNECT substep",
                PAUSE_CONNECT_STEP,
            )

            if connect.solved:
                break

            if not connect.pending_connect:
                break

        # Finish outer iteration for RRT-Connect only after all connect substeps
        if not connect.solved:
            connect.finish_outer_iteration()

        # Stop early only if both have solved
        if basic.solved and connect.solved:
            render_and_optionally_save(outer_iter, "Both solved", PAUSE_FINAL)
            break

    # Final freeze frame if one solved and the other did not
    if plt.fignum_exists(fig.number):
        render_and_optionally_save(
            outer_iter if 'outer_iter' in locals() else 0,
            "Finished run",
            PAUSE_FINAL,
        )


if __name__ == "__main__":
    main()