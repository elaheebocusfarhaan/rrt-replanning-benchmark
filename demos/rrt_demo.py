import os
# demo1_hq_video.py — RRT-Connect step-by-step visualization
# - consistent colors + legend
# - optional floating status text
# - optional point labels
# - optional compact Ta/Tb role box
# - high-quality MP4 export using imageio-ffmpeg
#
# Run:
#   python demo1_hq_video.py
#
# Requirements:
#   pip install matplotlib imageio-ffmpeg

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.animation import FFMpegWriter
import matplotlib as mpl
import imageio_ffmpeg


# =========================
# USER SETTINGS (EDIT HERE)
# =========================
MAX_ITERATIONS = 80
STEP_SIZE = 0.05
RANDOM_SEED = 3
CONNECT_MAX_STEPS = 600

# Smaller environment
X_LIM = (0.0, 0.45)
Y_LIM = (0.0, 0.45)

# On-screen pause durations (seconds)
PAUSE_AFTER_EXTEND = 5
PAUSE_DURING_CONNECT = 4
PAUSE_INITIAL = 4
PAUSE_FINAL = 2.5

# Show special markers during steps
SHOW_QRAND = True
SHOW_QNEAR = True
SHOW_QNEW = True
SHOW_QTARGET = True

# Floating text box toggle
SHOW_FLOATING_TEXT = True

# Point label toggles
SHOW_LABEL_QRAND = True
SHOW_LABEL_QNEAR = True
SHOW_LABEL_QNEW = True
SHOW_LABEL_QTARGET = True

# Small role box toggle
SHOW_ROLE_BOX = True

# ---------- Video export ----------
SAVE_VIDEO = True
VIDEO_FILENAME = "demo1_rrt_connect.mp4"
VIDEO_FPS = 8
VIDEO_DPI = 220
# =========================


# =========================
# VISUAL STYLE
# =========================
START_TREE_COLOR = "#1f77b4"   # blue
GOAL_TREE_COLOR  = "#2ca02c"   # green
OBSTACLE_COLOR   = "#7f7f7f"   # gray

QRAND_COLOR   = "black"
QNEAR_COLOR   = "#ff7f0e"
QNEW_COLOR    = "#d62728"
QTARGET_COLOR = "#9467bd"

TREE_NODE_SIZE = 28
START_ROOT_SIZE = 70
GOAL_ROOT_SIZE  = 70
SPECIAL_SIZE = 95
# =========================


# -----------------------------
# Obstacles + collision checking
# -----------------------------
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
    resolution: float = 0.003,
) -> bool:
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


# -----------------------------
# Tree / Node structures
# -----------------------------
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


# -----------------------------
# EXTEND / CONNECT
# -----------------------------
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
) -> Tuple[str, int, Optional[int], Tuple[float, float], Tuple[float, float]]:
    i_near = T.nearest_index(q_target)
    q_near = (T.nodes[i_near].x, T.nodes[i_near].y)
    q_new = steer(q_near, q_target, eps)

    if not collision_free_segment(q_near, q_new, obstacles):
        return TRAPPED, i_near, None, q_near, q_new

    i_new = T.add_node(q_new, i_near)
    status = REACHED if q_new == q_target else ADVANCED
    return status, i_near, i_new, q_near, q_new


# -----------------------------
# Sampling
# -----------------------------
def sample_free(obstacles: List[RectObs]) -> Tuple[float, float]:
    for _ in range(8000):
        x = random.uniform(X_LIM[0], X_LIM[1])
        y = random.uniform(Y_LIM[0], Y_LIM[1])
        if not any(obs.contains(x, y) for obs in obstacles):
            return (x, y)
    return ((X_LIM[0] + X_LIM[1]) / 2, (Y_LIM[0] + Y_LIM[1]) / 2)


# -----------------------------
# Plotting
# -----------------------------
def setup_axes(ax):
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)


def draw_obstacles(ax, obstacles: List[RectObs]):
    for obs in obstacles:
        ax.add_patch(
            plt.Rectangle(
                (obs.x, obs.y),
                obs.w,
                obs.h,
                facecolor=OBSTACLE_COLOR,
                edgecolor=OBSTACLE_COLOR,
                alpha=0.18,
            )
        )


def draw_tree(ax, T: Tree, color: str):
    for n in T.nodes:
        if n.parent is None:
            continue
        p = T.nodes[n.parent]
        ax.plot([p.x, n.x], [p.y, n.y], linewidth=1.3, color=color)
    ax.scatter([n.x for n in T.nodes], [n.y for n in T.nodes], s=TREE_NODE_SIZE, color=color, zorder=3)


def draw_special(ax, q_start, q_goal, q_rand=None, q_near=None, q_new=None, q_target=None):
    ax.scatter([q_start[0]], [q_start[1]], s=START_ROOT_SIZE, marker="o", color=START_TREE_COLOR, zorder=6)
    ax.scatter([q_goal[0]], [q_goal[1]], s=GOAL_ROOT_SIZE, marker="s", color=GOAL_TREE_COLOR, zorder=6)

    if SHOW_QRAND and q_rand is not None:
        ax.scatter([q_rand[0]], [q_rand[1]], s=SPECIAL_SIZE, marker="*", color=QRAND_COLOR, zorder=7)
    if SHOW_QNEAR and q_near is not None:
        ax.scatter([q_near[0]], [q_near[1]], s=SPECIAL_SIZE, marker="o", color=QNEAR_COLOR, zorder=7)
    if SHOW_QNEW and q_new is not None:
        ax.scatter([q_new[0]], [q_new[1]], s=SPECIAL_SIZE, marker="x", color=QNEW_COLOR, zorder=7)
    if SHOW_QTARGET and q_target is not None:
        ax.scatter([q_target[0]], [q_target[1]], s=SPECIAL_SIZE, marker="x", color=QTARGET_COLOR, zorder=7)


def add_legend(ax):
    handles = [
        Line2D([0], [0], color=START_TREE_COLOR, lw=2, label="Start tree"),
        Line2D([0], [0], color=GOAL_TREE_COLOR, lw=2, label="Goal tree"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor=OBSTACLE_COLOR, alpha=0.18, label="Obstacle"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=QRAND_COLOR, markersize=12, label="q_rand"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=QNEAR_COLOR, markersize=10, label="q_near"),
        Line2D([0], [0], marker="x", color=QNEW_COLOR, markersize=10, label="q_new"),
        Line2D([0], [0], marker="x", color=QTARGET_COLOR, markersize=10, label="q_target"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=START_TREE_COLOR, markersize=10, label="q_start"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=GOAL_TREE_COLOR, markersize=10, label="q_goal"),
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.85, fontsize=9)


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


def draw_role_box(ax, Ta_name: str, Tb_name: str):
    if not SHOW_ROLE_BOX:
        return
    role_text = f"Ta = {Ta_name}  (active / expanding)\nTb = {Tb_name}  (connecting)"
    ax.text(
        0.02,
        0.80,
        role_text,
        transform=ax.transAxes,
        va="top",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.78, edgecolor="none"),
        zorder=8,
    )


def render(
    fig,
    ax,
    obstacles: List[RectObs],
    T_start: Tree,
    T_goal: Tree,
    q_start: Tuple[float, float],
    q_goal: Tuple[float, float],
    title: str,
    floating_text: Optional[str],
    Ta_name: Optional[str] = None,
    Tb_name: Optional[str] = None,
    q_rand=None,
    q_near=None,
    q_new=None,
    q_target=None,
):
    ax.clear()
    setup_axes(ax)
    draw_obstacles(ax, obstacles)
    draw_tree(ax, T_start, START_TREE_COLOR)
    draw_tree(ax, T_goal, GOAL_TREE_COLOR)
    draw_special(ax, q_start, q_goal, q_rand=q_rand, q_near=q_near, q_new=q_new, q_target=q_target)

    ax.set_title(title, fontsize=12, fontweight="bold")
    add_legend(ax)

    if Ta_name is not None and Tb_name is not None:
        draw_role_box(ax, Ta_name, Tb_name)

    if SHOW_LABEL_QRAND and q_rand is not None:
        annotate_point(ax, q_rand, "q_rand", dx=0.008, dy=0.010)
    if SHOW_LABEL_QNEAR and q_near is not None:
        annotate_point(ax, q_near, "q_near", dx=0.008, dy=0.010)
    if SHOW_LABEL_QNEW and q_new is not None:
        annotate_point(ax, q_new, "q_new", dx=0.008, dy=-0.014)
    if SHOW_LABEL_QTARGET and q_target is not None:
        annotate_point(ax, q_target, "q_target", dx=0.008, dy=-0.014)

    if SHOW_FLOATING_TEXT and floating_text:
        ax.text(
            0.02,
            0.98,
            floating_text,
            transform=ax.transAxes,
            va="top",
            fontsize=11,
            bbox=dict(alpha=0.08),
        )

    fig.canvas.draw()
    fig.canvas.flush_events()


# -----------------------------
# Video helper
# -----------------------------
def grab_with_hold(writer, fig, hold_seconds: float):
    if writer is None:
        return
    n = max(1, int(round(hold_seconds * VIDEO_FPS)))
    for _ in range(n):
        writer.grab_frame()


# -----------------------------
# Main loop
# -----------------------------
def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    obstacles = [
        RectObs(0.20, 0.00, 0.03, 0.20),
        RectObs(0.20, 0.20, 0.18, 0.03),
    ]

    q_start = (0.05, 0.05)
    q_goal = (0.40, 0.40)

    T_start = Tree(q_start)
    T_goal = Tree(q_goal)

    Ta, Tb = T_start, T_goal
    Ta_name, Tb_name = "T_start", "T_goal"

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    writer = None

    if SAVE_VIDEO:
        mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = FFMpegWriter(
            fps=VIDEO_FPS,
            metadata={"title": "RRT-Connect Demo", "artist": "OpenAI"},
        )

    def draw_and_record(hold_seconds: float, **render_kwargs):
        render(fig, ax, obstacles, T_start, T_goal, q_start, q_goal, **render_kwargs)
        grab_with_hold(writer, fig, hold_seconds)
        plt.pause(hold_seconds)

    def run():
        nonlocal Ta, Tb, Ta_name, Tb_name

        draw_and_record(
            PAUSE_INITIAL,
            title="i=0 (initial)",
            floating_text="Two roots: q_start and q_goal",
            Ta_name=Ta_name,
            Tb_name=Tb_name,
            q_rand=None,
            q_near=None,
            q_new=None,
            q_target=None,
        )

        for it in range(1, MAX_ITERATIONS + 1):
            if not plt.fignum_exists(fig.number):
                print("Window closed — exiting.")
                break

            q_rand = sample_free(obstacles)
            ext_status, _, ext_new_idx, q_near, q_new_prop = EXTEND(Ta, q_rand, obstacles, STEP_SIZE)
            q_new = q_new_prop if ext_new_idx is not None else None

            draw_and_record(
                PAUSE_AFTER_EXTEND,
                title=f"i={it} | EXTEND on {Ta_name} → {ext_status}",
                floating_text=(
                    f"EXTEND({Ta_name}, q_rand)\n"
                    f"q_near is nearest node in {Ta_name}\n"
                    f"q_new is one step toward q_rand"
                ),
                Ta_name=Ta_name,
                Tb_name=Tb_name,
                q_rand=q_rand,
                q_near=q_near,
                q_new=q_new,
                q_target=None,
            )

            conn_status = None
            q_target = None

            if ext_new_idx is not None:
                q_target = q_new
                steps = 0
                while steps < CONNECT_MAX_STEPS:
                    steps += 1
                    status, _, i_new, c_near, c_new_prop = EXTEND(Tb, q_target, obstacles, STEP_SIZE)
                    conn_status = status
                    c_new = c_new_prop if i_new is not None else None

                    draw_and_record(
                        PAUSE_DURING_CONNECT,
                        title=f"i={it} | CONNECT on {Tb_name} (step {steps}) → {conn_status}",
                        floating_text=(
                            f"CONNECT({Tb_name}, target node)\n"
                            f"Repeated EXTEND toward same target\n"
                            f"until Reached / Trapped"
                        ),
                        Ta_name=Ta_name,
                        Tb_name=Tb_name,
                        q_rand=None,
                        q_near=c_near,
                        q_new=c_new,
                        q_target=q_target,
                    )

                    if status != ADVANCED:
                        break
            else:
                draw_and_record(
                    PAUSE_AFTER_EXTEND,
                    title=f"i={it} | EXTEND on {Ta_name} → TRAPPED (CONNECT skipped)",
                    floating_text="EXTEND returned TRAPPED\nno new node added → skip CONNECT",
                    Ta_name=Ta_name,
                    Tb_name=Tb_name,
                    q_rand=q_rand,
                    q_near=q_near,
                    q_new=None,
                    q_target=None,
                )

            if conn_status == REACHED:
                draw_and_record(
                    PAUSE_FINAL,
                    title=f"CONNECTED at i={it} ✅",
                    floating_text="CONNECT returned REACHED → trees met",
                    Ta_name=Ta_name,
                    Tb_name=Tb_name,
                    q_rand=None,
                    q_near=None,
                    q_new=None,
                    q_target=q_target,
                )
                break

            Ta, Tb = Tb, Ta
            Ta_name, Tb_name = Tb_name, Ta_name

    if writer is not None:
        with writer.saving(fig, VIDEO_FILENAME, dpi=VIDEO_DPI):
            run()
    else:
        run()

    plt.ioff()
    plt.close("all")
    print("Done.")
    if SAVE_VIDEO:
        print(f"Saved video: {VIDEO_FILENAME}")


if __name__ == "__main__":
    main()