import os
import csv
import json
import os
import time
from datetime import datetime
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path

import cv2
import numpy as np


Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]
ProjRect = Tuple[int, int, int, int, float]


@dataclass
class AppConfig:
    camera_index: int = 6
    frame_w: int = 1280
    frame_h: int = 720
    projector_x: int = 100
    projector_y: int = 100
    projector_init_w: int = 1000
    projector_init_h: int = 700
    min_dot_area: int = 80
    min_obstacle_area: int = 600
    obstacle_bbox_padding_px: int = 22
    obstacle_deadzone_px: int = 28
    sample_half_h: int = 10
    sample_half_s: int = 60
    sample_half_v: int = 60
    save_file: str = "detector_config.json"
    auto_calib_frames_per_corner: int = 3
    auto_calib_max_corner_distance_px: int = 420
    auto_calib_grid_cols: int = 8
    auto_calib_grid_rows: int = 6
    auto_calib_margin_px: int = 50
    auto_calib_target_radius_px: int = 18
    auto_calib_warmup_frames: int = 12
    checker_cols: int = 8   # inner corners
    checker_rows: int = 5   # inner corners
    checker_frames_to_average: int = 10
    intrinsic_samples_target: int = 20


@dataclass
class AppState:
    current_mode: str = "start"  # start | goal | obs | corners | proj_cal
    latest_hsv_frame: Optional[np.ndarray] = None

    camera_corners: List[Point] = field(default_factory=list)
    camera_to_projector_h: Optional[np.ndarray] = None
    projector_to_camera_h: Optional[np.ndarray] = None

    projector_observed_corners_cam: List[Point] = field(default_factory=list)
    logical_to_projector_prewarp_h: Optional[np.ndarray] = None
    projector_calibration_index: int = 0
    auto_calibration_samples: List[Point] = field(default_factory=list)
    auto_projector_points: List[Point] = field(default_factory=list)
    auto_camera_points: List[Point] = field(default_factory=list)
    auto_background_gray: Optional[np.ndarray] = None
    auto_warmup_counter: int = 0
    tune_tx: float = 0.0
    tune_ty: float = 0.0
    tune_sx: float = 1.0
    tune_sy: float = 1.0
    checker_projector_points: List[Point] = field(default_factory=list)
    checker_camera_samples: List[np.ndarray] = field(default_factory=list)
    checker_miss_counter: int = 0
    intrinsic_obj_points: List[np.ndarray] = field(default_factory=list)
    intrinsic_img_points: List[np.ndarray] = field(default_factory=list)
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None

    planner_enabled: bool = True
    active_planner: str = "rrt"
    planner_view_mode: str = "overlay"  # overlay | sequential
    planner_step_budget: int = 120
    obstacle_mask_proj: Optional[np.ndarray] = None
    last_planner_start: Optional[Point] = None
    last_planner_goal: Optional[Point] = None
    last_planner_obstacle_mask: Optional[np.ndarray] = None
    planner_episode_start_ms: float = field(default_factory=lambda: time.time() * 1000.0)

    stable_start_proj: Optional[Point] = None
    stable_goal_proj: Optional[Point] = None
    pending_start_proj: Optional[Point] = None
    pending_goal_proj: Optional[Point] = None
    pending_start_count: int = 0
    pending_goal_count: int = 0
    start_miss_count: int = 0
    goal_miss_count: int = 0
    freeze_start_goal: bool = False
    frozen_start_proj: Optional[Point] = None
    frozen_goal_proj: Optional[Point] = None

    stable_obstacle_proj_rects: List[ProjRect] = field(default_factory=list)

    recording_enabled: bool = False
    recording_session_id: Optional[str] = None
    recording_dir: Optional[str] = None
    recording_event_index: int = 0
    recording_files: Dict[str, str] = field(default_factory=dict)
    last_logged_path_signature: Dict[str, str] = field(default_factory=dict)
    last_logged_obstacle_signature: Optional[str] = None

    sampled_ranges: Dict[str, object] = field(
        default_factory=lambda: {
            "start": {
                "lower": np.array([40, 80, 80]),
                "upper": np.array([90, 255, 255]),
            },
            "goal": {
                "lower1": np.array([0, 100, 80]),
                "upper1": np.array([10, 255, 255]),
                "lower2": np.array([170, 100, 80]),
                "upper2": np.array([180, 255, 255]),
            },
            "obs_list": [{"lower": np.array([8, 100, 100]), "upper": np.array([25, 255, 255])}],
            "obs_dynamic_list": [],
        }
    )


DRAW_GREEN = (0, 255, 0)
DRAW_RED = (0, 0, 255)
DRAW_ORANGE = (0, 165, 255)
DRAW_WHITE = (255, 255, 255)
DRAW_CYAN = (255, 255, 0)
DRAW_YELLOW = (0, 255, 255)
DRAW_MAGENTA = (255, 0, 255)
DRAW_BLUE = (255, 0, 0)

CAMERA_WINDOW = "Camera Debug"
PROJECTOR_WINDOW = "Projector Output"
METRICS_WINDOW = "Detection Metrics"
WARPED_WINDOW = "Warped ROI"


@dataclass
class Node:
    x: int
    y: int
    parent: int

    @property
    def point(self) -> Point:
        return (self.x, self.y)


@dataclass
class CostNode:
    x: int
    y: int
    parent: int
    cost: float

    @property
    def point(self) -> Point:
        return (self.x, self.y)


@dataclass
class RepairNode:
    x: int
    y: int
    parent: int
    cost: float
    active: bool = True
    children: set = field(default_factory=set)

    @property
    def point(self) -> Point:
        return (self.x, self.y)


class RRTPlanner:
    MAX_NODES: int = 2000
    MAX_ITERATIONS: int = 8000

    def __init__(self, color: Tuple[int, int, int] = (255, 120, 0)) -> None:
        self.color = color
        self.start: Optional[Point] = None
        self.goal: Optional[Point] = None
        self.width: int = 0
        self.height: int = 0
        self.obstacle_mask: Optional[np.ndarray] = None
        self.step_size: int = 22
        self.goal_radius: int = 24
        self.collision_step: int = 5
        self.goal_bias_every: int = 8

        self.solve_time_ms: float = 0.0
        self.first_solution_time_ms: Optional[float] = None
        self.best_path_length: float = 0.0
        self.status: str = "WAITING"
        self.replans: int = 0
        self.recovering: bool = False
        self.recovery_start_ms: Optional[float] = None
        self.last_recovery_ms: Optional[float] = None

        self._nodes: List[Node] = []
        self._path: Optional[List[Point]] = None
        self._goal_reached: bool = False
        self._failed: bool = False
        self._iter_count: int = 0

    @property
    def path(self) -> Optional[List[Point]]:
        return self._path

    @property
    def nodes(self) -> List[Node]:
        return self._nodes

    def clear(self) -> None:
        self.start = None
        self.goal = None
        self.obstacle_mask = None
        self._nodes = []
        self._path = None
        self._goal_reached = False
        self._failed = False
        self._iter_count = 0
        self.solve_time_ms = 0.0
        self.first_solution_time_ms = None
        self.best_path_length = 0.0
        self.status = "WAITING"
        self.replans = 0
        self.recovering = False
        self.recovery_start_ms = None
        self.last_recovery_ms = None

    def reset(self, start: Point, goal: Point, width: int, height: int, obstacle_mask: np.ndarray) -> None:
        self.start = start
        self.goal = goal
        self.width = width
        self.height = height
        self.obstacle_mask = obstacle_mask.copy()
        self._nodes = [Node(start[0], start[1], -1)]
        self._path = None
        self._goal_reached = False
        self._failed = False
        self._iter_count = 0
        self.solve_time_ms = 0.0
        self.first_solution_time_ms = None
        self.best_path_length = 0.0
        self.replans += 1
        self.recovering = False
        self.recovery_start_ms = None
        self.last_recovery_ms = None
        self.status = "WAITING"
        if self._is_occupied(start) or self._is_occupied(goal):
            self._failed = True
            self.status = "FAILED"

    def ready(self) -> bool:
        return (
            self.start is not None
            and self.goal is not None
            and len(self._nodes) > 0
            and not self._failed
            and not self._goal_reached
        )

    def update_obstacles(self, obstacle_mask: np.ndarray) -> None:
        self.obstacle_mask = obstacle_mask.copy()

    def total_nodes(self) -> int:
        return len(self._nodes)

    def path_is_valid(self) -> bool:
        return self._path is not None and self._check_path_valid(self._path)

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _in_bounds(self, p: Point) -> bool:
        return 0 <= p[0] < self.width and 0 <= p[1] < self.height

    def _is_occupied(self, p: Point) -> bool:
        return self.obstacle_mask is not None and bool(self.obstacle_mask[p[1], p[0]] > 0)

    def _steer(self, a: Point, b: Point) -> Point:
        dx, dy = b[0] - a[0], b[1] - a[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return a
        scale = min(self.step_size / dist, 1.0)
        return (int(round(a[0] + dx * scale)), int(round(a[1] + dy * scale)))

    def _collision_free(self, a: Point, b: Point) -> bool:
        dist = max(1.0, self._distance(a, b))
        steps = max(2, int(dist / self.collision_step))
        for i in range(steps + 1):
            t = i / steps
            x = int(round(a[0] * (1 - t) + b[0] * t))
            y = int(round(a[1] * (1 - t) + b[1] * t))
            if not self._in_bounds((x, y)) or self._is_occupied((x, y)):
                return False
        return True

    def _check_path_valid(self, path: List[Point]) -> bool:
        return all(self._collision_free(path[i], path[i + 1]) for i in range(len(path) - 1))

    def path_length_of(self, path: Optional[List[Point]]) -> float:
        if path is None or len(path) < 2:
            return 0.0
        return sum(self._distance(path[i], path[i + 1]) for i in range(len(path) - 1))

    def grow(self, n: int, elapsed_ms: float) -> None:
        if not self.ready() or self.goal is None:
            return
        t0 = time.time()
        added = 0
        attempts = 0
        max_attempts = max(50, n * 30)

        while added < n and attempts < max_attempts:
            attempts += 1
            self._iter_count += 1
            if self.total_nodes() >= self.MAX_NODES or self._iter_count >= self.MAX_ITERATIONS:
                self._failed = True
                self.status = "FAILED"
                break

            sample: Point = (
                self.goal
                if self._iter_count % self.goal_bias_every == 0
                else (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            )
            nearest_idx = min(range(len(self._nodes)), key=lambda i: self._distance(self._nodes[i].point, sample))
            nearest_pt = self._nodes[nearest_idx].point
            new_pt = self._steer(nearest_pt, sample)

            if not self._in_bounds(new_pt) or self._is_occupied(new_pt) or not self._collision_free(nearest_pt, new_pt):
                continue

            self._nodes.append(Node(new_pt[0], new_pt[1], nearest_idx))
            added += 1

            if self._distance(new_pt, self.goal) <= self.goal_radius and self._collision_free(new_pt, self.goal):
                self._nodes.append(Node(self.goal[0], self.goal[1], len(self._nodes) - 1))
                self._path = self._backtrack(len(self._nodes) - 1)
                self._goal_reached = True
                self.best_path_length = self.path_length_of(self._path)
                if self.first_solution_time_ms is None:
                    self.first_solution_time_ms = elapsed_ms
                    self.solve_time_ms = elapsed_ms
                self.status = "FOUND"
                break

        if self.first_solution_time_ms is None:
            self.solve_time_ms += (time.time() - t0) * 1000.0
        if self._goal_reached:
            self.status = "FOUND"
        elif self._failed:
            self.status = "FAILED"
        elif len(self._nodes) > 1:
            self.status = "SEARCHING"
        else:
            self.status = "WAITING"

    def _backtrack(self, idx: int) -> List[Point]:
        path: List[Point] = []
        while idx != -1:
            path.append(self._nodes[idx].point)
            idx = self._nodes[idx].parent
        return list(reversed(path))

    def start_recovery(self) -> None:
        if not self.recovering:
            self.recovering = True
            self.recovery_start_ms = time.time() * 1000.0
            self.last_recovery_ms = None

    def finish_recovery(self) -> None:
        if self.recovering and self.recovery_start_ms is not None:
            self.last_recovery_ms = time.time() * 1000.0 - self.recovery_start_ms
        self.recovering = False
        self.recovery_start_ms = None

    def start_recovery(self) -> None:
        if not self.recovering:
            self.recovering = True
            self.recovery_start_ms = time.time() * 1000.0
            self.last_recovery_ms = None

    def finish_recovery(self) -> None:
        if self.recovering and self.recovery_start_ms is not None:
            self.last_recovery_ms = time.time() * 1000.0 - self.recovery_start_ms
        self.recovering = False
        self.recovery_start_ms = None

    def start_recovery(self) -> None:
        if not self.recovering:
            self.recovering = True
            self.recovery_start_ms = time.time() * 1000.0
            self.last_recovery_ms = None

    def finish_recovery(self) -> None:
        if self.recovering and self.recovery_start_ms is not None:
            self.last_recovery_ms = time.time() * 1000.0 - self.recovery_start_ms
        self.recovering = False
        self.recovery_start_ms = None

    def draw(self, img: np.ndarray) -> None:
        for node in self._nodes:
            if node.parent != -1:
                cv2.line(img, node.point, self._nodes[node.parent].point, self.color, 1, cv2.LINE_AA)
        if self._path is not None:
            for i in range(len(self._path) - 1):
                cv2.line(img, self._path[i], self._path[i + 1], self.color, 4, cv2.LINE_AA)


class RRTConnectPlanner:
    MAX_NODES_TOTAL: int = 3000
    MAX_ITERATIONS: int = 8000

    def __init__(self, color: Tuple[int, int, int] = (255, 0, 255)) -> None:
        self.color = color
        self.start: Optional[Point] = None
        self.goal: Optional[Point] = None
        self.width: int = 0
        self.height: int = 0
        self.obstacle_mask: Optional[np.ndarray] = None
        self.step_size: int = 22
        self.goal_radius: int = 24
        self.collision_step: int = 5

        self.solve_time_ms: float = 0.0
        self.first_solution_time_ms: Optional[float] = None
        self.best_path_length: float = 0.0
        self.status: str = "WAITING"
        self.replans: int = 0
        self.recovering: bool = False
        self.recovery_start_ms: Optional[float] = None
        self.last_recovery_ms: Optional[float] = None

        self._start_tree: List[Node] = []
        self._goal_tree: List[Node] = []
        self._path: Optional[List[Point]] = None
        self._connected: bool = False
        self._failed: bool = False
        self._iter_count: int = 0
        self._grow_from_start: bool = True

    @property
    def path(self) -> Optional[List[Point]]:
        return self._path

    @property
    def nodes(self) -> List[Node]:
        return self._start_tree + self._goal_tree

    def clear(self) -> None:
        self.start = None
        self.goal = None
        self.obstacle_mask = None
        self._start_tree = []
        self._goal_tree = []
        self._path = None
        self._connected = False
        self._failed = False
        self._iter_count = 0
        self._grow_from_start = True
        self.solve_time_ms = 0.0
        self.first_solution_time_ms = None
        self.best_path_length = 0.0
        self.status = "WAITING"
        self.replans = 0
        self.recovering = False
        self.recovery_start_ms = None
        self.last_recovery_ms = None

    def reset(self, start: Point, goal: Point, width: int, height: int, obstacle_mask: np.ndarray) -> None:
        self.start = start
        self.goal = goal
        self.width = width
        self.height = height
        self.obstacle_mask = obstacle_mask.copy()
        self._start_tree = [Node(start[0], start[1], -1)]
        self._goal_tree = [Node(goal[0], goal[1], -1)]
        self._path = None
        self._connected = False
        self._failed = False
        self._iter_count = 0
        self._grow_from_start = True
        self.solve_time_ms = 0.0
        self.first_solution_time_ms = None
        self.best_path_length = 0.0
        self.replans += 1
        self.recovering = False
        self.recovery_start_ms = None
        self.last_recovery_ms = None
        self.status = "WAITING"
        if self._is_occupied(start) or self._is_occupied(goal):
            self._failed = True
            self.status = "FAILED"

    def ready(self) -> bool:
        return (
            self.start is not None
            and self.goal is not None
            and len(self._start_tree) > 0
            and len(self._goal_tree) > 0
            and not self._failed
            and not self._connected
        )

    def update_obstacles(self, obstacle_mask: np.ndarray) -> None:
        self.obstacle_mask = obstacle_mask.copy()

    def total_nodes(self) -> int:
        return len(self._start_tree) + len(self._goal_tree)

    def path_is_valid(self) -> bool:
        return self._path is not None and self._check_path_valid(self._path)

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _in_bounds(self, p: Point) -> bool:
        return 0 <= p[0] < self.width and 0 <= p[1] < self.height

    def _is_occupied(self, p: Point) -> bool:
        return self.obstacle_mask is not None and bool(self.obstacle_mask[p[1], p[0]] > 0)

    def _steer(self, a: Point, b: Point) -> Point:
        dx, dy = b[0] - a[0], b[1] - a[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return a
        scale = min(self.step_size / dist, 1.0)
        return (int(round(a[0] + dx * scale)), int(round(a[1] + dy * scale)))

    def _collision_free(self, a: Point, b: Point) -> bool:
        dist = max(1.0, self._distance(a, b))
        steps = max(2, int(dist / self.collision_step))
        for i in range(steps + 1):
            t = i / steps
            x = int(round(a[0] * (1 - t) + b[0] * t))
            y = int(round(a[1] * (1 - t) + b[1] * t))
            if not self._in_bounds((x, y)) or self._is_occupied((x, y)):
                return False
        return True

    def _check_path_valid(self, path: List[Point]) -> bool:
        return all(self._collision_free(path[i], path[i + 1]) for i in range(len(path) - 1))

    def path_length_of(self, path: Optional[List[Point]]) -> float:
        if path is None or len(path) < 2:
            return 0.0
        return sum(self._distance(path[i], path[i + 1]) for i in range(len(path) - 1))

    def grow(self, n: int, elapsed_ms: float) -> None:
        if not self.ready():
            return
        t0 = time.time()
        for _ in range(n):
            self._iter_count += 1
            if self.total_nodes() >= self.MAX_NODES_TOTAL or self._iter_count >= self.MAX_ITERATIONS:
                self._failed = True
                self.status = "FAILED"
                break

            tree_a, tree_b = (self._start_tree, self._goal_tree) if self._grow_from_start else (self._goal_tree, self._start_tree)
            a_is_start = self._grow_from_start
            sample: Point = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            new_idx = self._extend(tree_a, sample)

            if new_idx is not None:
                connect_idx = self._connect(tree_b, tree_a[new_idx].point)
                if connect_idx is not None:
                    self._connected = True
                    if a_is_start:
                        self._path = self._build_full_path(self._start_tree, new_idx, self._goal_tree, connect_idx)
                    else:
                        self._path = self._build_full_path(self._start_tree, connect_idx, self._goal_tree, new_idx)
                    self.best_path_length = self.path_length_of(self._path)
                    if self.first_solution_time_ms is None:
                        self.first_solution_time_ms = elapsed_ms
                        self.solve_time_ms = elapsed_ms
                    self.status = "FOUND"
                    break

            self._grow_from_start = not self._grow_from_start

        if self.first_solution_time_ms is None:
            self.solve_time_ms += (time.time() - t0) * 1000.0
        if self._connected:
            self.status = "FOUND"
        elif self._failed:
            self.status = "FAILED"
        elif self.total_nodes() > 2:
            self.status = "SEARCHING"
        else:
            self.status = "WAITING"

    def _extend(self, tree: List[Node], target: Point) -> Optional[int]:
        nearest_idx = min(range(len(tree)), key=lambda i: self._distance(tree[i].point, target))
        nearest_pt = tree[nearest_idx].point
        new_pt = self._steer(nearest_pt, target)
        if not self._in_bounds(new_pt) or self._is_occupied(new_pt) or not self._collision_free(nearest_pt, new_pt):
            return None
        tree.append(Node(new_pt[0], new_pt[1], nearest_idx))
        return len(tree) - 1

    def _connect(self, tree: List[Node], target: Point) -> Optional[int]:
        while True:
            if self.total_nodes() >= self.MAX_NODES_TOTAL:
                self._failed = True
                return None
            nearest_idx = min(range(len(tree)), key=lambda i: self._distance(tree[i].point, target))
            nearest_pt = tree[nearest_idx].point
            if self._distance(nearest_pt, target) <= self.goal_radius:
                if self._collision_free(nearest_pt, target):
                    tree.append(Node(target[0], target[1], nearest_idx))
                    return len(tree) - 1
                return None
            new_pt = self._steer(nearest_pt, target)
            if not self._in_bounds(new_pt) or self._is_occupied(new_pt) or not self._collision_free(nearest_pt, new_pt):
                return None
            tree.append(Node(new_pt[0], new_pt[1], nearest_idx))

    def _backtrack_tree(self, tree: List[Node], idx: int) -> List[Point]:
        path: List[Point] = []
        while idx != -1:
            path.append(tree[idx].point)
            idx = tree[idx].parent
        return list(reversed(path))

    def _build_full_path(self, st: List[Node], si: int, gt: List[Node], gi: int) -> List[Point]:
        a = self._backtrack_tree(st, si)
        b = self._backtrack_tree(gt, gi)
        b.reverse()
        if a and b and a[-1] == b[0]:
            b = b[1:]
        return a + b

    def draw(self, img: np.ndarray) -> None:
        for node in self._start_tree:
            if node.parent != -1:
                cv2.line(img, node.point, self._start_tree[node.parent].point, (0, 180, 255), 1, cv2.LINE_AA)
        for node in self._goal_tree:
            if node.parent != -1:
                cv2.line(img, node.point, self._goal_tree[node.parent].point, (180, 120, 255), 1, cv2.LINE_AA)
        if self._path is not None:
            for i in range(len(self._path) - 1):
                cv2.line(img, self._path[i], self._path[i + 1], self.color, 4, cv2.LINE_AA)



def obstacle_rects_to_mask(width: int, height: int, obstacle_rects: Sequence[ProjRect]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for x, y, w, h, _ in obstacle_rects:
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(width - 1, int(x + w))
        y1 = min(height - 1, int(y + h))
        if x1 >= x0 and y1 >= y0:
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    return mask




def planner_task_completion_rate(planner: object) -> float:
    return 100.0 if getattr(planner, 'path', None) is not None and planner.path_is_valid() else 0.0


def planner_replan_events(planner: object) -> int:
    return max(0, int(getattr(planner, 'replans', 0)))


def planner_min_clearance(path: Optional[List[Point]], obstacle_mask: Optional[np.ndarray]) -> Optional[float]:
    if path is None or obstacle_mask is None or obstacle_mask.size == 0:
        return None
    if np.count_nonzero(obstacle_mask) == 0:
        return None
    free_mask = np.where(obstacle_mask > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(free_mask, cv2.DIST_L2, 3)
    min_clear = float('inf')
    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]
        seg_len = max(1.0, math.hypot(b[0] - a[0], b[1] - a[1]))
        steps = max(2, int(seg_len / 3.0))
        for j in range(steps + 1):
            t = j / steps
            x = int(round(a[0] * (1 - t) + b[0] * t))
            y = int(round(a[1] * (1 - t) + b[1] * t))
            x = max(0, min(obstacle_mask.shape[1] - 1, x))
            y = max(0, min(obstacle_mask.shape[0] - 1, y))
            d = float(dist[y, x])
            if d < min_clear:
                min_clear = d
    if min_clear == float('inf'):
        return None
    return min_clear


class RRTStarPlanner:
    MAX_NODES: int = 5000
    MAX_ITERATIONS: int = 10000
    REWIRE_RADIUS: float = 60.0

    def __init__(self, color: Tuple[int, int, int] = (255, 200, 0)) -> None:
        self.color = color
        self.start: Optional[Point] = None
        self.goal: Optional[Point] = None
        self.width: int = 0
        self.height: int = 0
        self.obstacle_mask: Optional[np.ndarray] = None
        self.step_size: int = 22
        self.goal_radius: int = 24
        self.collision_step: int = 5
        self.goal_bias_every: int = 8

        self.solve_time_ms: float = 0.0
        self.first_solution_time_ms: Optional[float] = None
        self.best_path_length: float = 0.0
        self.status: str = "WAITING"
        self.replans: int = 0
        self.recovering: bool = False
        self.recovery_start_ms: Optional[float] = None
        self.last_recovery_ms: Optional[float] = None

        self._nodes: List[CostNode] = []
        self._path: Optional[List[Point]] = None
        self._goal_reached: bool = False
        self._failed: bool = False
        self._iter_count: int = 0
        self._best_goal_idx: Optional[int] = None

    @property
    def path(self) -> Optional[List[Point]]:
        return self._path

    @property
    def nodes(self) -> List[CostNode]:
        return self._nodes

    def clear(self) -> None:
        self.start = None
        self.goal = None
        self.obstacle_mask = None
        self._nodes = []
        self._path = None
        self._goal_reached = False
        self._failed = False
        self._iter_count = 0
        self._best_goal_idx = None
        self.solve_time_ms = 0.0
        self.first_solution_time_ms = None
        self.best_path_length = 0.0
        self.status = "WAITING"
        self.replans = 0
        self.recovering = False
        self.recovery_start_ms = None
        self.last_recovery_ms = None

    def reset(self, start: Point, goal: Point, width: int, height: int, obstacle_mask: np.ndarray) -> None:
        self.start = start
        self.goal = goal
        self.width = width
        self.height = height
        self.obstacle_mask = obstacle_mask.copy()
        self._nodes = [CostNode(start[0], start[1], -1, 0.0)]
        self._path = None
        self._goal_reached = False
        self._failed = False
        self._iter_count = 0
        self._best_goal_idx = None
        self.solve_time_ms = 0.0
        self.first_solution_time_ms = None
        self.best_path_length = 0.0
        self.replans += 1
        self.recovering = False
        self.recovery_start_ms = None
        self.last_recovery_ms = None
        self.status = "WAITING"
        if self._is_occupied(start) or self._is_occupied(goal):
            self._failed = True
            self.status = "FAILED"

    def ready(self) -> bool:
        return (
            self.start is not None
            and self.goal is not None
            and len(self._nodes) > 0
            and not self._failed
        )

    def update_obstacles(self, obstacle_mask: np.ndarray) -> None:
        self.obstacle_mask = obstacle_mask.copy()

    def total_nodes(self) -> int:
        return len(self._nodes)

    def path_is_valid(self) -> bool:
        return self._path is not None and self._check_path_valid(self._path)

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _in_bounds(self, p: Point) -> bool:
        return 0 <= p[0] < self.width and 0 <= p[1] < self.height

    def _is_occupied(self, p: Point) -> bool:
        return self.obstacle_mask is not None and bool(self.obstacle_mask[p[1], p[0]] > 0)

    def _steer(self, a: Point, b: Point) -> Point:
        dx, dy = b[0] - a[0], b[1] - a[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return a
        scale = min(self.step_size / dist, 1.0)
        return (int(round(a[0] + dx * scale)), int(round(a[1] + dy * scale)))

    def _collision_free(self, a: Point, b: Point) -> bool:
        dist = max(1.0, self._distance(a, b))
        steps = max(2, int(dist / self.collision_step))
        for i in range(steps + 1):
            t = i / steps
            x = int(round(a[0] * (1 - t) + b[0] * t))
            y = int(round(a[1] * (1 - t) + b[1] * t))
            if not self._in_bounds((x, y)) or self._is_occupied((x, y)):
                return False
        return True

    def _check_path_valid(self, path: List[Point]) -> bool:
        return all(self._collision_free(path[i], path[i + 1]) for i in range(len(path) - 1))

    def path_length_of(self, path: Optional[List[Point]]) -> float:
        if path is None or len(path) < 2:
            return 0.0
        return sum(self._distance(path[i], path[i + 1]) for i in range(len(path) - 1))

    def grow(self, n: int, elapsed_ms: float) -> None:
        if not self.ready() or self.goal is None:
            return
        t0 = time.time()
        added = 0
        attempts = 0
        max_attempts = max(50, n * 30)

        while added < n and attempts < max_attempts:
            attempts += 1
            self._iter_count += 1

            if len(self._nodes) >= self.MAX_NODES or self._iter_count >= self.MAX_ITERATIONS:
                if not self._goal_reached:
                    self._failed = True
                    self.status = "FAILED"
                break

            sample: Point = (
                self.goal if self._iter_count % self.goal_bias_every == 0
                else (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            )

            nearest_idx = min(range(len(self._nodes)), key=lambda i: self._distance(self._nodes[i].point, sample))
            nearest_pt = self._nodes[nearest_idx].point
            new_pt = self._steer(nearest_pt, sample)

            if not self._in_bounds(new_pt) or self._is_occupied(new_pt) or not self._collision_free(nearest_pt, new_pt):
                continue

            near = [i for i, nd in enumerate(self._nodes) if self._distance(nd.point, new_pt) <= self.REWIRE_RADIUS]
            parent_idx, new_cost = self._choose_parent(nearest_idx, near, new_pt)

            new_idx = len(self._nodes)
            self._nodes.append(CostNode(new_pt[0], new_pt[1], parent_idx, new_cost))
            added += 1

            self._rewire(new_idx, near)
            self._try_connect_goal(new_idx, elapsed_ms)

        self.solve_time_ms += (time.time() - t0) * 1000.0
        if self._goal_reached:
            self.status = "FOUND"
        elif self._failed:
            self.status = "FAILED"
        elif len(self._nodes) > 1:
            self.status = "SEARCHING"
        else:
            self.status = "WAITING"

    def _choose_parent(self, nearest_idx: int, near: List[int], new_pt: Point) -> Tuple[int, float]:
        best_parent = nearest_idx
        best_cost = self._nodes[nearest_idx].cost + self._distance(self._nodes[nearest_idx].point, new_pt)
        for idx in near:
            nd = self._nodes[idx]
            if not self._collision_free(nd.point, new_pt):
                continue
            c = nd.cost + self._distance(nd.point, new_pt)
            if c < best_cost:
                best_parent, best_cost = idx, c
        return best_parent, best_cost

    def _rewire(self, new_idx: int, near: List[int]) -> None:
        new_node = self._nodes[new_idx]
        for idx in near:
            nd = self._nodes[idx]
            nc = new_node.cost + self._distance(new_node.point, nd.point)
            if nc + 1e-9 < nd.cost and self._collision_free(new_node.point, nd.point):
                self._nodes[idx].parent = new_idx
                self._nodes[idx].cost = nc
                self._propagate_costs(idx)
        if self._best_goal_idx is not None and self._best_goal_idx < len(self._nodes):
            self._path = self._backtrack(self._best_goal_idx)
            self.best_path_length = self.path_length_of(self._path)

    def _propagate_costs(self, start_idx: int) -> None:
        queue: List[int] = [start_idx]
        while queue:
            parent_idx = queue.pop(0)
            parent_cost = self._nodes[parent_idx].cost
            for idx, nd in enumerate(self._nodes):
                if nd.parent == parent_idx:
                    self._nodes[idx].cost = parent_cost + self._distance(self._nodes[parent_idx].point, nd.point)
                    queue.append(idx)

    def _try_connect_goal(self, new_idx: int, elapsed_ms: float) -> None:
        if self.goal is None:
            return
        nd = self._nodes[new_idx]
        d = self._distance(nd.point, self.goal)
        if d > self.goal_radius or not self._collision_free(nd.point, self.goal):
            return
        goal_cost = nd.cost + d
        if self._best_goal_idx is None:
            self._best_goal_idx = len(self._nodes)
            self._nodes.append(CostNode(self.goal[0], self.goal[1], new_idx, goal_cost))
            self._goal_reached = True
            self._path = self._backtrack(self._best_goal_idx)
            self.best_path_length = self.path_length_of(self._path)
            if self.first_solution_time_ms is None:
                self.first_solution_time_ms = elapsed_ms
                self.solve_time_ms = elapsed_ms
        elif goal_cost + 1e-9 < self._nodes[self._best_goal_idx].cost:
            self._nodes[self._best_goal_idx].parent = new_idx
            self._nodes[self._best_goal_idx].cost = goal_cost
            self._goal_reached = True
            self._path = self._backtrack(self._best_goal_idx)
            self.best_path_length = self.path_length_of(self._path)

    def _backtrack(self, idx: int) -> List[Point]:
        path: List[Point] = []
        while idx != -1:
            path.append(self._nodes[idx].point)
            idx = self._nodes[idx].parent
        return list(reversed(path))

    def draw(self, img: np.ndarray) -> None:
        for node in self._nodes:
            if node.parent != -1:
                cv2.line(img, node.point, self._nodes[node.parent].point, self.color, 1, cv2.LINE_AA)
        if self._path is not None:
            for i in range(len(self._path) - 1):
                cv2.line(img, self._path[i], self._path[i + 1], self.color, 3, cv2.LINE_AA)



def points_changed(a: Optional[Point], b: Optional[Point], thresh: float = 18.0) -> bool:
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    return math.hypot(a[0] - b[0], a[1] - b[1]) > thresh


def masks_changed(m1: Optional[np.ndarray], m2: np.ndarray, ratio_thresh: float = 0.005) -> bool:
    if m1 is None:
        return True
    if m1.shape != m2.shape:
        return True
    diff = cv2.absdiff(m1, m2)
    return (np.count_nonzero(diff) / diff.size) > ratio_thresh


# ----------------------------- math + transforms -----------------------------

class RRTXLitePlanner:
    MAX_NODES: int = 5000
    MAX_ITERATIONS: int = 12000
    REWIRE_RADIUS: float = 70.0

    def __init__(self, color: Tuple[int, int, int] = (0, 180, 90)) -> None:
        self.color = color
        self.start: Optional[Point] = None
        self.goal: Optional[Point] = None
        self.width: int = 0
        self.height: int = 0
        self.obstacle_mask: Optional[np.ndarray] = None
        self.step_size: int = 22
        self.goal_radius: int = 24
        self.collision_step: int = 5
        self.goal_bias_every: int = 8
        self.solve_time_ms: float = 0.0
        self.first_solution_time_ms: Optional[float] = None
        self.best_path_length: float = 0.0
        self.status: str = "WAITING"
        self.replans: int = 0
        self.repairs: int = 0
        self.last_pruned_count: int = 0
        self.recovering: bool = False
        self.recovery_start_ms: Optional[float] = None
        self.last_recovery_ms: Optional[float] = None
        self._nodes: List[RepairNode] = []
        self._path: Optional[List[Point]] = None
        self._goal_reached: bool = False
        self._failed: bool = False
        self._iter_count: int = 0
        self._best_goal_idx: Optional[int] = None

    @property
    def path(self) -> Optional[List[Point]]:
        return self._path

    def clear(self) -> None:
        self.start = None
        self.goal = None
        self.obstacle_mask = None
        self._nodes = []
        self._path = None
        self._goal_reached = False
        self._failed = False
        self._iter_count = 0
        self._best_goal_idx = None
        self.repairs = 0
        self.last_pruned_count = 0
        self.recovering = False
        self.recovery_start_ms = None
        self.last_recovery_ms = None
        self.solve_time_ms = 0.0
        self.first_solution_time_ms = None
        self.best_path_length = 0.0
        self.status = "WAITING"
        self.replans = 0

    def reset(self, start: Point, goal: Point, width: int, height: int, obstacle_mask: np.ndarray) -> None:
        self.start = start
        self.goal = goal
        self.width = width
        self.height = height
        self.obstacle_mask = obstacle_mask.copy()
        self._nodes = [RepairNode(start[0], start[1], -1, 0.0)]
        self._path = None
        self._goal_reached = False
        self._failed = False
        self._iter_count = 0
        self._best_goal_idx = None
        self.solve_time_ms = 0.0
        self.first_solution_time_ms = None
        self.best_path_length = 0.0
        self.replans += 1
        self.recovering = False
        self.recovery_start_ms = None
        self.last_recovery_ms = None
        self.status = "WAITING"
        if self._is_occupied(start) or self._is_occupied(goal):
            self._failed = True
            self.status = "FAILED"

    def ready(self) -> bool:
        return self.start is not None and self.goal is not None and len(self._active_indices()) > 0 and not self._failed

    def update_obstacles(self, obstacle_mask: np.ndarray) -> None:
        self.obstacle_mask = obstacle_mask.copy()

    def total_nodes(self) -> int:
        return len(self._active_indices())

    def path_is_valid(self) -> bool:
        return self._path is not None and self._check_path_valid(self._path)

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _in_bounds(self, p: Point) -> bool:
        return 0 <= p[0] < self.width and 0 <= p[1] < self.height

    def _is_occupied(self, p: Point) -> bool:
        return self.obstacle_mask is not None and bool(self.obstacle_mask[p[1], p[0]] > 0)

    def _steer(self, a: Point, b: Point) -> Point:
        dx, dy = b[0] - a[0], b[1] - a[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return a
        scale = min(self.step_size / dist, 1.0)
        return (int(round(a[0] + dx * scale)), int(round(a[1] + dy * scale)))

    def _collision_free(self, a: Point, b: Point) -> bool:
        dist = max(1.0, self._distance(a, b))
        steps = max(2, int(dist / self.collision_step))
        for i in range(steps + 1):
            t = i / steps
            x = int(round(a[0] * (1 - t) + b[0] * t))
            y = int(round(a[1] * (1 - t) + b[1] * t))
            if not self._in_bounds((x, y)) or self._is_occupied((x, y)):
                return False
        return True

    def _check_path_valid(self, path: List[Point]) -> bool:
        return all(self._collision_free(path[i], path[i + 1]) for i in range(len(path) - 1))

    def path_length_of(self, path: Optional[List[Point]]) -> float:
        if path is None or len(path) < 2:
            return 0.0
        return sum(self._distance(path[i], path[i + 1]) for i in range(len(path) - 1))

    def grow(self, n: int, elapsed_ms: float) -> None:
        if not self.ready() or self.goal is None:
            return
        t0 = time.time()
        added = 0
        attempts = 0
        max_attempts = max(50, n * 30)
        while added < n and attempts < max_attempts:
            attempts += 1
            self._iter_count += 1
            if len(self._nodes) >= self.MAX_NODES or self._iter_count >= self.MAX_ITERATIONS:
                if not self._goal_reached:
                    self._failed = True
                    self.status = "FAILED"
                break
            sample = self.goal if self._iter_count % self.goal_bias_every == 0 else (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            nearest_idx = self._nearest_index(sample)
            nearest_pt = self._nodes[nearest_idx].point
            new_pt = self._steer(nearest_pt, sample)
            if not self._in_bounds(new_pt) or self._is_occupied(new_pt) or not self._collision_free(nearest_pt, new_pt):
                continue
            near = self._near_indices(new_pt)
            parent_idx, new_cost = self._choose_parent(nearest_idx, near, new_pt)
            new_idx = len(self._nodes)
            self._nodes.append(RepairNode(new_pt[0], new_pt[1], parent_idx, new_cost))
            self._nodes[parent_idx].children.add(new_idx)
            added += 1
            self._rewire(new_idx, near)
            self._try_connect_goal(new_idx, elapsed_ms)
        if self.first_solution_time_ms is None:
            self.solve_time_ms += (time.time() - t0) * 1000.0
        if self.recovering and self._path is not None and self.path_is_valid():
            self.finish_recovery()
        if self._goal_reached:
            self.status = "FOUND"
        elif self._failed:
            self.status = "FAILED"
        elif len(self._active_indices()) > 1:
            self.status = "REPAIRING"
        else:
            self.status = "WAITING"

    def repair_after_obstacle_change(self) -> None:
        if not self._nodes:
            return
        self.repairs += 1
        invalid = set()
        for idx, nd in enumerate(self._nodes):
            if not nd.active:
                continue
            if idx == 0:
                if self._is_occupied(nd.point):
                    invalid.add(idx)
                continue
            parent = self._nodes[nd.parent]
            if not parent.active or self._is_occupied(nd.point) or not self._collision_free(parent.point, nd.point):
                invalid.add(idx)
        to_prune = set()
        for idx in invalid:
            self._collect_descendants(idx, to_prune)
        self.last_pruned_count = len(to_prune)
        for idx in to_prune:
            nd = self._nodes[idx]
            nd.active = False
            if idx != 0 and 0 <= nd.parent < len(self._nodes):
                self._nodes[nd.parent].children.discard(idx)
            nd.children.clear()
        self._refresh_reachability_and_costs()
        if self._best_goal_idx is not None:
            goal_node = self._nodes[self._best_goal_idx] if self._best_goal_idx < len(self._nodes) else None
            if goal_node is None or not goal_node.active:
                self._best_goal_idx = None
                self._path = None
                self._goal_reached = False
            else:
                self._path = self._backtrack(self._best_goal_idx)
                self.best_path_length = self.path_length_of(self._path)
                self._goal_reached = self.path_is_valid()
                if not self._goal_reached:
                    self._path = None

    def start_recovery(self) -> None:
        if not self.recovering:
            self.recovering = True
            self.recovery_start_ms = time.time() * 1000.0
            self.last_recovery_ms = None

    def finish_recovery(self) -> None:
        if self.recovering and self.recovery_start_ms is not None:
            self.last_recovery_ms = time.time() * 1000.0 - self.recovery_start_ms
        self.recovering = False
        self.recovery_start_ms = None

    def _active_indices(self):
        return [i for i, nd in enumerate(self._nodes) if nd.active]

    def _collect_descendants(self, idx, out):
        stack = [idx]
        while stack:
            cur = stack.pop()
            if cur in out or cur < 0 or cur >= len(self._nodes):
                continue
            out.add(cur)
            stack.extend(self._nodes[cur].children)

    def _refresh_reachability_and_costs(self):
        if not self._nodes or not self._nodes[0].active:
            return
        from collections import deque
        reachable = {0}
        q = deque([0])
        self._nodes[0].cost = 0.0
        while q:
            idx = q.popleft()
            nd = self._nodes[idx]
            for c in list(nd.children):
                if c < 0 or c >= len(self._nodes):
                    continue
                child = self._nodes[c]
                if child.active:
                    child.cost = nd.cost + self._distance(nd.point, child.point)
                    reachable.add(c)
                    q.append(c)
        for idx, nd in enumerate(self._nodes):
            if nd.active and idx not in reachable:
                nd.active = False
        if self._best_goal_idx is not None and self._best_goal_idx < len(self._nodes) and self._nodes[self._best_goal_idx].active:
            self._path = self._backtrack(self._best_goal_idx)
            self.best_path_length = self.path_length_of(self._path)

    def _nearest_index(self, target):
        active = self._active_indices()
        return min(active, key=lambda i: self._distance(self._nodes[i].point, target))

    def _near_indices(self, point):
        return [i for i, nd in enumerate(self._nodes) if nd.active and self._distance(nd.point, point) <= self.REWIRE_RADIUS]

    def _choose_parent(self, nearest_idx, near, new_pt):
        best_parent = nearest_idx
        best_cost = self._nodes[nearest_idx].cost + self._distance(self._nodes[nearest_idx].point, new_pt)
        for idx in near:
            nd = self._nodes[idx]
            if not self._collision_free(nd.point, new_pt):
                continue
            c = nd.cost + self._distance(nd.point, new_pt)
            if c < best_cost:
                best_parent, best_cost = idx, c
        return best_parent, best_cost

    def _rewire(self, new_idx, near):
        new_node = self._nodes[new_idx]
        for idx in near:
            if idx == new_node.parent or not self._nodes[idx].active:
                continue
            nd = self._nodes[idx]
            nc = new_node.cost + self._distance(new_node.point, nd.point)
            if nc + 1e-9 < nd.cost and self._collision_free(new_node.point, nd.point):
                if nd.parent >= 0:
                    self._nodes[nd.parent].children.discard(idx)
                self._nodes[idx].parent = new_idx
                self._nodes[idx].cost = nc
                self._nodes[new_idx].children.add(idx)
                self._propagate_costs(idx)
        if self._best_goal_idx is not None and self._best_goal_idx < len(self._nodes) and self._nodes[self._best_goal_idx].active:
            self._path = self._backtrack(self._best_goal_idx)
            self.best_path_length = self.path_length_of(self._path)

    def _propagate_costs(self, start_idx):
        from collections import deque
        q = deque([start_idx])
        while q:
            idx = q.popleft()
            nd = self._nodes[idx]
            for c in list(nd.children):
                if 0 <= c < len(self._nodes) and self._nodes[c].active:
                    self._nodes[c].cost = nd.cost + self._distance(nd.point, self._nodes[c].point)
                    q.append(c)

    def _try_connect_goal(self, new_idx, elapsed_ms):
        if self.goal is None:
            return
        nd = self._nodes[new_idx]
        d = self._distance(nd.point, self.goal)
        if d > self.goal_radius or not self._collision_free(nd.point, self.goal):
            return
        goal_cost = nd.cost + d
        if self._best_goal_idx is None:
            self._best_goal_idx = len(self._nodes)
            self._nodes.append(RepairNode(self.goal[0], self.goal[1], new_idx, goal_cost))
            self._nodes[new_idx].children.add(self._best_goal_idx)
            self._goal_reached = True
            self._path = self._backtrack(self._best_goal_idx)
            self.best_path_length = self.path_length_of(self._path)
            if self.first_solution_time_ms is None:
                self.first_solution_time_ms = elapsed_ms
                self.solve_time_ms = elapsed_ms
        elif self._nodes[self._best_goal_idx].active and goal_cost + 1e-9 < self._nodes[self._best_goal_idx].cost:
            old_parent = self._nodes[self._best_goal_idx].parent
            if old_parent >= 0:
                self._nodes[old_parent].children.discard(self._best_goal_idx)
            self._nodes[self._best_goal_idx].parent = new_idx
            self._nodes[self._best_goal_idx].cost = goal_cost
            self._nodes[self._best_goal_idx].active = True
            self._nodes[new_idx].children.add(self._best_goal_idx)
            self._path = self._backtrack(self._best_goal_idx)
            self._goal_reached = True
            self.best_path_length = self.path_length_of(self._path)

    def _backtrack(self, idx):
        path = []
        while idx != -1 and 0 <= idx < len(self._nodes):
            path.append(self._nodes[idx].point)
            idx = self._nodes[idx].parent
        return list(reversed(path))

    def draw(self, img: np.ndarray) -> None:
        for idx, nd in enumerate(self._nodes):
            if not nd.active or nd.parent == -1:
                continue
            parent = self._nodes[nd.parent]
            if parent.active:
                cv2.line(img, nd.point, parent.point, self.color, 1, cv2.LINE_AA)
        if self._path is not None:
            for i in range(len(self._path) - 1):
                cv2.line(img, self._path[i], self._path[i + 1], self.color, 4, cv2.LINE_AA)


def clamp_hsv(h: int, s: int, v: int) -> Tuple[int, int, int]:
    return max(0, min(180, int(h))), max(0, min(255, int(s))), max(0, min(255, int(v)))


def normalize_homography(h: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if h is None:
        return None
    denom = float(h[2, 2])
    if abs(denom) <= 1e-9:
        return h
    return h / denom


def safe_inverse(h: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if h is None:
        return None
    try:
        return np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return None


def order_points(pts: Sequence[Point]) -> np.ndarray:
    """
    Return points ordered as top-left, top-right, bottom-right, bottom-left.
    Uses the classic sum/diff heuristic, which is stable for 4-corner quads.
    """
    arr = np.array(pts, dtype=np.float32)
    if arr.shape != (4, 2):
        raise ValueError(f"order_points expects 4 points, got shape={arr.shape}")

    s = arr.sum(axis=1)
    d = np.diff(arr, axis=1).reshape(-1)

    top_left = arr[np.argmin(s)]
    bottom_right = arr[np.argmax(s)]
    top_right = arr[np.argmin(d)]
    bottom_left = arr[np.argmax(d)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def get_projector_rect_points(width: int, height: int) -> np.ndarray:
    return np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)


def warp_with_src_to_dst(src: np.ndarray, h_src_to_dst: Optional[np.ndarray], out_w: int, out_h: int) -> np.ndarray:
    h_inv = safe_inverse(h_src_to_dst)
    if h_inv is None:
        return src.copy()
    return cv2.warpPerspective(src, h_inv, (out_w, out_h))


def normalize_prewarp_to_canvas(h_src_to_dst: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """
    Prevent auto-calibration from shrinking output coverage by normalizing
    mapped projector corners back to the full projector canvas.
    """
    corners = get_projector_rect_points(out_w, out_h).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, h_src_to_dst).reshape(-1, 2)
    min_x, min_y = float(np.min(warped[:, 0])), float(np.min(warped[:, 1]))
    max_x, max_y = float(np.max(warped[:, 0])), float(np.max(warped[:, 1]))
    bw = max(1e-6, max_x - min_x)
    bh = max(1e-6, max_y - min_y)

    sx = float((out_w - 1) / bw)
    sy = float((out_h - 1) / bh)
    n = np.array([[sx, 0.0, -min_x * sx], [0.0, sy, -min_y * sy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return normalize_homography(n @ h_src_to_dst)


def apply_user_tune_homography(base_h: np.ndarray, state: AppState) -> np.ndarray:
    tune = np.array(
        [
            [state.tune_sx, 0.0, state.tune_tx],
            [0.0, state.tune_sy, state.tune_ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return normalize_homography(tune @ base_h)


def transform_point(h: Optional[np.ndarray], pt: Optional[Point]) -> Optional[Point]:
    if h is None or pt is None:
        return None
    pts = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(pts, h)
    x, y = warped[0, 0]
    return int(round(x)), int(round(y))


def projector_rect_to_camera_rect(h_inv: Optional[np.ndarray], rect: ProjRect) -> Optional[Rect]:
    if h_inv is None:
        return None
    x, y, w, h, _ = rect
    corners = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.float32)
    cam_pts = cv2.perspectiveTransform(corners, h_inv).reshape(-1, 2)

    x0, y0 = int(round(float(np.min(cam_pts[:, 0])))), int(round(float(np.min(cam_pts[:, 1]))))
    x1, y1 = int(round(float(np.max(cam_pts[:, 0])))), int(round(float(np.max(cam_pts[:, 1]))))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


# ----------------------------- detection stabilization -----------------------------
STABLE_POINT_ACCEPT_RADIUS_PX = 18
STABLE_POINT_SWITCH_RADIUS_PX = 80
STABLE_POINT_CONFIRM_FRAMES = 8
STABLE_POINT_MISS_FRAMES = 15

def _pt_dist(a: Optional[Point], b: Optional[Point]) -> float:
    if a is None or b is None:
        return float("inf")
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))

def stabilize_detected_point(
    detected_pt: Optional[Point],
    stable_pt: Optional[Point],
    pending_pt: Optional[Point],
    pending_count: int,
    miss_count: int,
) -> Tuple[Optional[Point], Optional[Point], int, int]:
    """Apply hysteresis to noisy marker centroids so small jitter does not trigger replans.

    Rules:
    - If the new detection stays close to the stable point, keep the stable point locked.
    - If the detection moves meaningfully, require several consecutive frames before switching.
    - If detections disappear briefly, hold the last stable point for a few frames.
    """
    if detected_pt is None:
        miss_count += 1
        if miss_count >= STABLE_POINT_MISS_FRAMES:
            return None, None, 0, miss_count
        return stable_pt, pending_pt, pending_count, miss_count

    miss_count = 0

    if stable_pt is None:
        if pending_pt is None or _pt_dist(detected_pt, pending_pt) > STABLE_POINT_ACCEPT_RADIUS_PX:
            return None, detected_pt, 1, miss_count
        pending_count += 1
        if pending_count >= STABLE_POINT_CONFIRM_FRAMES:
            return detected_pt, None, 0, miss_count
        return None, pending_pt, pending_count, miss_count

    if _pt_dist(detected_pt, stable_pt) <= STABLE_POINT_ACCEPT_RADIUS_PX:
        return stable_pt, None, 0, miss_count

    if pending_pt is None or _pt_dist(detected_pt, pending_pt) > STABLE_POINT_SWITCH_RADIUS_PX:
        return stable_pt, detected_pt, 1, miss_count

    pending_count += 1
    if pending_count >= STABLE_POINT_CONFIRM_FRAMES:
        return detected_pt, None, 0, miss_count

    return stable_pt, pending_pt, pending_count, miss_count


OBSTACLE_RECT_STABLE_CENTER_PX = 14
OBSTACLE_RECT_STABLE_SIZE_PX = 14

def stabilize_obstacle_rects(
    detected_rects: Sequence[ProjRect],
    stable_rects: Sequence[ProjRect],
    center_tol_px: int = OBSTACLE_RECT_STABLE_CENTER_PX,
    size_tol_px: int = OBSTACLE_RECT_STABLE_SIZE_PX,
) -> List[ProjRect]:
    """Keep an existing rectangle when the new detection only jitters slightly.

    This makes stationary obstacle boxes look visually locked instead of wavering
    by a few pixels each frame.
    """
    if not detected_rects:
        return []
    if not stable_rects:
        return list(detected_rects)

    used_old: set[int] = set()
    stabilized: List[ProjRect] = []

    for rect in detected_rects:
        x, y, w, h, area = rect
        cx = x + 0.5 * w
        cy = y + 0.5 * h

        best_idx = None
        best_score = float("inf")
        for i, old in enumerate(stable_rects):
            if i in used_old:
                continue
            ox, oy, ow, oh, _ = old
            ocx = ox + 0.5 * ow
            ocy = oy + 0.5 * oh
            dc = math.hypot(cx - ocx, cy - ocy)
            ds = abs(w - ow) + abs(h - oh)
            if dc <= center_tol_px and ds <= 2 * size_tol_px:
                score = dc + 0.25 * ds
                if score < best_score:
                    best_score = score
                    best_idx = i

        if best_idx is not None:
            used_old.add(best_idx)
            stabilized.append(stable_rects[best_idx])
        else:
            stabilized.append(rect)

    return stabilized


# ----------------------------- calibration logic -----------------------------
def update_camera_homography(state: AppState, projector_w: int, projector_h: int) -> None:
    if len(state.camera_corners) != 4:
        state.camera_to_projector_h = None
        state.projector_to_camera_h = None
        return

    src = order_points(state.camera_corners)
    dst = get_projector_rect_points(projector_w, projector_h)
    state.camera_to_projector_h = cv2.getPerspectiveTransform(src, dst)
    state.projector_to_camera_h = cv2.getPerspectiveTransform(dst, src)


def update_output_prewarp(state: AppState, projector_w: int, projector_h: int) -> None:
    if len(state.camera_corners) != 4:
        state.logical_to_projector_prewarp_h = None
        return

    raw_projector_to_camera_h: Optional[np.ndarray] = None
    if len(state.auto_projector_points) >= 4 and len(state.auto_projector_points) == len(state.auto_camera_points):
        src = np.array(state.auto_projector_points, dtype=np.float32)
        dst = np.array(state.auto_camera_points, dtype=np.float32)
        raw_projector_to_camera_h, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    elif len(state.projector_observed_corners_cam) == 4:
        logical_corners = get_projector_rect_points(projector_w, projector_h)
        observed_cam = np.array(state.projector_observed_corners_cam, dtype=np.float32)
        raw_projector_to_camera_h = cv2.getPerspectiveTransform(logical_corners, observed_cam)

    if raw_projector_to_camera_h is None:
        state.logical_to_projector_prewarp_h = None
        return

    logical_corners = get_projector_rect_points(projector_w, projector_h)
    desired_cam = order_points(state.camera_corners)
    desired_logical_to_camera_h = cv2.getPerspectiveTransform(logical_corners, desired_cam)

    inv_raw = safe_inverse(raw_projector_to_camera_h)
    if inv_raw is None:
        state.logical_to_projector_prewarp_h = None
        return

    prewarp = normalize_homography(inv_raw @ desired_logical_to_camera_h)
    state.logical_to_projector_prewarp_h = normalize_prewarp_to_canvas(prewarp, projector_w, projector_h)


# ----------------------------- detection logic -----------------------------
def get_color_mask(hsv: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def update_range_from_hsv_pixel(cfg: AppConfig, state: AppState, target_name: str, hsv_pixel: np.ndarray, append_obstacle: bool = False) -> None:
    h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])

    if target_name == "goal":
        low_h, high_h = h - cfg.sample_half_h, h + cfg.sample_half_h
        s_low, s_high = max(0, s - cfg.sample_half_s), min(255, s + cfg.sample_half_s)
        v_low, v_high = max(0, v - cfg.sample_half_v), min(255, v + cfg.sample_half_v)

        if low_h < 0:
            state.sampled_ranges["goal"]["lower1"] = np.array([0, s_low, v_low])
            state.sampled_ranges["goal"]["upper1"] = np.array([high_h, s_high, v_high])
            state.sampled_ranges["goal"]["lower2"] = np.array([180 + low_h, s_low, v_low])
            state.sampled_ranges["goal"]["upper2"] = np.array([180, s_high, v_high])
        elif high_h > 180:
            state.sampled_ranges["goal"]["lower1"] = np.array([0, s_low, v_low])
            state.sampled_ranges["goal"]["upper1"] = np.array([high_h - 180, s_high, v_high])
            state.sampled_ranges["goal"]["lower2"] = np.array([low_h, s_low, v_low])
            state.sampled_ranges["goal"]["upper2"] = np.array([180, s_high, v_high])
        else:
            state.sampled_ranges["goal"]["lower1"] = np.array([low_h, s_low, v_low])
            state.sampled_ranges["goal"]["upper1"] = np.array([high_h, s_high, v_high])
            state.sampled_ranges["goal"]["lower2"] = np.array([0, 0, 0])
            state.sampled_ranges["goal"]["upper2"] = np.array([0, 0, 0])
        return

    low = clamp_hsv(h - cfg.sample_half_h, s - cfg.sample_half_s, v - cfg.sample_half_v)
    high = clamp_hsv(h + cfg.sample_half_h, s + cfg.sample_half_s, v + cfg.sample_half_v)

    if target_name in ("obs", "obs_dynamic"):
        new_range = {"lower": np.array(low), "upper": np.array(high)}
        list_key = "obs_list" if target_name == "obs" else "obs_dynamic_list"
        if append_obstacle:
            state.sampled_ranges[list_key].append(new_range)
        else:
            state.sampled_ranges[list_key] = [new_range]
        return

    state.sampled_ranges[target_name]["lower"] = np.array(low)
    state.sampled_ranges[target_name]["upper"] = np.array(high)


def build_masks(state: AppState, hsv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start_mask = get_color_mask(hsv, state.sampled_ranges["start"]["lower"], state.sampled_ranges["start"]["upper"])

    goal_mask_1 = get_color_mask(hsv, state.sampled_ranges["goal"]["lower1"], state.sampled_ranges["goal"]["upper1"])
    goal_mask_2 = get_color_mask(hsv, state.sampled_ranges["goal"]["lower2"], state.sampled_ranges["goal"]["upper2"])
    goal_mask = cv2.bitwise_or(goal_mask_1, goal_mask_2)

    obs_static_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for obs_range in state.sampled_ranges["obs_list"]:
        obs_static_mask = cv2.bitwise_or(obs_static_mask, get_color_mask(hsv, obs_range["lower"], obs_range["upper"]))

    obs_dynamic_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for obs_range in state.sampled_ranges["obs_dynamic_list"]:
        obs_dynamic_mask = cv2.bitwise_or(obs_dynamic_mask, get_color_mask(hsv, obs_range["lower"], obs_range["upper"]))

    return start_mask, goal_mask, obs_static_mask, obs_dynamic_mask


def largest_centroid(mask: np.ndarray, min_area: int) -> Tuple[Optional[Point], float]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_center: Optional[Point] = None
    best_area = 0.0
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
        if area > best_area:
            best_area = area
            best_center = (cx, cy)
    return best_center, best_area


def extract_obstacle_rectangles(
    mask: np.ndarray, min_area: int, padding_px: int = 0, deadzone_px: int = 0
) -> List[ProjRect]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[ProjRect] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        candidates.append((x, y, w, h, area))

    if not candidates:
        return []

    # Sort large to small, and suppress detections whose centers are within deadzone
    # of a stronger obstacle. This keeps distinct obstacles while merging close neighbors.
    candidates.sort(key=lambda r: r[4], reverse=True)
    kept: List[ProjRect] = []
    dz2 = float(max(0, deadzone_px) ** 2)
    for cand in candidates:
        x, y, w, h, area = cand
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        skip = False
        for kx, ky, kw, kh, _ in kept:
            kcx = kx + 0.5 * kw
            kcy = ky + 0.5 * kh
            if ((cx - kcx) ** 2 + (cy - kcy) ** 2) <= dz2:
                skip = True
                break
        if not skip:
            kept.append(cand)

    if padding_px <= 0:
        return kept

    h_img, w_img = mask.shape[:2]
    padded: List[ProjRect] = []
    for x, y, w, h, area in kept:
        x0 = max(0, x - padding_px)
        y0 = max(0, y - padding_px)
        x1 = min(w_img - 1, x + w + padding_px)
        y1 = min(h_img - 1, y + h + padding_px)
        padded.append((x0, y0, max(1, x1 - x0), max(1, y1 - y0), area))
    return padded


def build_auto_calibration_projector_points(cfg: AppConfig, projector_w: int, projector_h: int) -> List[Point]:
    cols = max(2, cfg.auto_calib_grid_cols)
    rows = max(2, cfg.auto_calib_grid_rows)
    margin = max(0, cfg.auto_calib_margin_px)
    x0, x1 = margin, max(margin + 1, projector_w - 1 - margin)
    y0, y1 = margin, max(margin + 1, projector_h - 1 - margin)
    xs = np.linspace(x0, x1, cols)
    ys = np.linspace(y0, y1, rows)
    points: List[Point] = []
    for yy in ys:
        for xx in xs:
            points.append((int(round(float(xx))), int(round(float(yy)))))
    return points


def build_checkerboard_pattern(
    cfg: AppConfig, projector_w: int, projector_h: int
) -> Tuple[np.ndarray, List[Point]]:
    """
    Build a standard checkerboard projection pattern and return:
    - rendered projector image
    - logical projector coordinates of checkerboard inner corners
    """
    cols = max(3, cfg.checker_cols)
    rows = max(3, cfg.checker_rows)
    squares_x = cols + 1
    squares_y = rows + 1

    cell = int(min((projector_w - 40) / squares_x, (projector_h - 40) / squares_y))
    cell = max(20, cell)
    board_w = squares_x * cell
    board_h = squares_y * cell
    x0 = max(0, (projector_w - board_w) // 2)
    y0 = max(0, (projector_h - board_h) // 2)

    canvas = np.zeros((projector_h, projector_w, 3), dtype=np.uint8)
    for r in range(squares_y):
        for c in range(squares_x):
            color = 255 if ((r + c) % 2 == 0) else 0
            cv2.rectangle(
                canvas,
                (x0 + c * cell, y0 + r * cell),
                (x0 + (c + 1) * cell, y0 + (r + 1) * cell),
                (color, color, color),
                -1,
            )

    inner_points: List[Point] = []
    for r in range(rows):
        for c in range(cols):
            inner_points.append((x0 + (c + 1) * cell, y0 + (r + 1) * cell))

    cv2.putText(canvas, "STANDARD CHECKERBOARD CALIBRATION", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 255), 2)
    return canvas, inner_points


def find_checker_corners(gray: np.ndarray, cols: int, rows: int) -> Optional[np.ndarray]:
    pattern_size = (cols, rows)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    corners = None

    # Prefer the newer SB detector when available.
    if hasattr(cv2, "findChessboardCornersSB"):
        ok_sb, corners_sb = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
        if ok_sb:
            corners = corners_sb.reshape(-1, 2)

    if corners is None:
        ok, corners_std = cv2.findChessboardCorners(gray, pattern_size)
        if not ok:
            return None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.01)
        corners_std = cv2.cornerSubPix(gray, corners_std, (9, 9), (-1, -1), criteria)
        corners = corners_std.reshape(-1, 2)

    if corners.shape[0] != cols * rows:
        return None
    return corners


def step_checkerboard_calibration(cfg: AppConfig, state: AppState, frame_bgr: np.ndarray) -> None:
    if not state.checker_projector_points:
        return
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners = find_checker_corners(gray, cfg.checker_cols, cfg.checker_rows)
    if corners is None:
        state.checker_miss_counter += 1
        if state.checker_miss_counter % 60 == 0:
            print("Checkerboard not detected yet. Ensure pattern fully visible and in focus.")
        return
    state.checker_miss_counter = 0

    rows, cols = cfg.checker_rows, cfg.checker_cols
    corner_grid = corners.reshape(rows, cols, 2)

    # findChessboardCorners can return equivalent flipped orderings.
    # Pick the orientation that best matches expected camera locations.
    candidates = [
        corner_grid,
        corner_grid[::-1, :, :],
        corner_grid[:, ::-1, :],
        corner_grid[::-1, ::-1, :],
        np.transpose(corner_grid, (1, 0, 2)),
        np.transpose(corner_grid, (1, 0, 2))[::-1, :, :],
        np.transpose(corner_grid, (1, 0, 2))[:, ::-1, :],
        np.transpose(corner_grid, (1, 0, 2))[::-1, ::-1, :],
    ]

    expected_cam: Optional[np.ndarray] = None
    if state.projector_to_camera_h is not None and state.checker_projector_points:
        exp = np.array(state.checker_projector_points, dtype=np.float32).reshape(-1, 1, 2)
        expected_cam = cv2.perspectiveTransform(exp, state.projector_to_camera_h).reshape(-1, 2)

    best = candidates[0]
    if expected_cam is not None:
        best_err = float("inf")
        for cand in candidates:
            flat = cand.reshape(-1, 2)
            if flat.shape[0] != expected_cam.shape[0]:
                continue
            err = float(np.mean(np.sum((flat - expected_cam) ** 2, axis=1)))
            if err < best_err:
                best_err = err
                best = cand

    corners_oriented = best.reshape(-1, 2)
    state.checker_camera_samples.append(corners_oriented.copy())
    state.projector_calibration_index = len(state.checker_camera_samples)
    if len(state.checker_camera_samples) % 3 == 0:
        print(
            f"Checkerboard samples: {len(state.checker_camera_samples)}/"
            f"{cfg.checker_frames_to_average}"
        )

    if len(state.checker_camera_samples) < cfg.checker_frames_to_average:
        return

    stack = np.stack(state.checker_camera_samples, axis=0)  # N x K x 2
    med = np.median(stack, axis=0)
    cam_points = [(int(round(float(p[0]))), int(round(float(p[1])))) for p in med]
    state.auto_projector_points = list(state.checker_projector_points)
    state.auto_camera_points = cam_points
    state.current_mode = "start"
    state.checker_camera_samples = []
    print(f"Checkerboard calibration complete ({len(cam_points)} points).")


def step_intrinsic_calibration(cfg: AppConfig, state: AppState, frame_bgr: np.ndarray) -> None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners = find_checker_corners(gray, cfg.checker_cols, cfg.checker_rows)
    if corners is None:
        return
    corners = corners.reshape(-1, 1, 2).astype(np.float32)

    # avoid near-duplicate samples
    if state.intrinsic_img_points:
        prev = state.intrinsic_img_points[-1].reshape(-1, 2)
        cur = corners.reshape(-1, 2)
        motion = float(np.mean(np.linalg.norm(cur - prev, axis=1)))
        if motion < 2.0:
            return

    objp = np.zeros((cfg.checker_rows * cfg.checker_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0 : cfg.checker_cols, 0 : cfg.checker_rows].T.reshape(-1, 2)

    state.intrinsic_obj_points.append(objp)
    state.intrinsic_img_points.append(corners)
    state.projector_calibration_index = len(state.intrinsic_img_points)
    print(f"Intrinsic samples: {len(state.intrinsic_img_points)}/{cfg.intrinsic_samples_target}")

    if len(state.intrinsic_img_points) < cfg.intrinsic_samples_target:
        return

    h, w = gray.shape[:2]
    ok_calib, cam_mtx, dist, _, _ = cv2.calibrateCamera(
        state.intrinsic_obj_points, state.intrinsic_img_points, (w, h), None, None
    )
    if not ok_calib:
        print("Intrinsic calibration failed.")
        return
    state.camera_matrix = cam_mtx
    state.dist_coeffs = dist
    state.current_mode = "start"
    print("Intrinsic calibration complete. Undistortion enabled.")


def detect_projected_target(
    frame_bgr: np.ndarray,
    background_gray: Optional[np.ndarray] = None,
    expected_pt: Optional[Point] = None,
    roi_polygon: Optional[np.ndarray] = None,
    max_dist_px: Optional[int] = None,
) -> Optional[Point]:
    """
    Detect a bright projected calibration target in camera space.
    We intentionally render a black screen + white target during auto-calibration
    so this brightness detector is robust to projector color shifts.
    """
    gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if background_gray is not None and background_gray.shape == gray_full.shape:
        gray_full = cv2.subtract(gray_full, background_gray)
    h, w = gray_full.shape[:2]

    x0, y0, x1, y1 = 0, 0, w, h
    if expected_pt is not None and max_dist_px is not None:
        r = max(40, int(max_dist_px))
        x0 = max(0, int(expected_pt[0] - r))
        y0 = max(0, int(expected_pt[1] - r))
        x1 = min(w, int(expected_pt[0] + r))
        y1 = min(h, int(expected_pt[1] + r))

    gray = gray_full[y0:y1, x0:x1]
    if gray.size == 0:
        return None

    # Adaptive threshold works better than a hard value across different projectors.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_thr, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Enforce a minimum brightness gate so general bright areas are less likely.
    min_gate = int(max(185, otsu_thr))
    mask = cv2.inRange(blur, min_gate, 255)

    if roi_polygon is not None:
        poly_mask = np.zeros(gray_full.shape, dtype=np.uint8)
        cv2.fillPoly(poly_mask, [roi_polygon.astype(np.int32)], 255)
        poly_crop = poly_mask[y0:y1, x0:x1]
        mask = cv2.bitwise_and(mask, poly_crop)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Tuple[Point, float]] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 8:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = int(m["m10"] / m["m00"]) + x0
        cy = int(m["m01"] / m["m00"]) + y0
        candidates.append(((cx, cy), area))

    # Fallback: if no contour survives, use brightest pixel in ROI.
    if not candidates:
        _, max_val, _, max_loc = cv2.minMaxLoc(blur)
        if max_val < 35:
            return None
        candidates.append(((int(max_loc[0] + x0), int(max_loc[1] + y0)), 1.0))

    if expected_pt is None:
        return max(candidates, key=lambda item: item[1])[0]

    def dist2(p: Point, q: Point) -> float:
        dx, dy = float(p[0] - q[0]), float(p[1] - q[1])
        return dx * dx + dy * dy

    best = min(candidates, key=lambda item: dist2(item[0], expected_pt))
    if max_dist_px is not None:
        if dist2(best[0], expected_pt) > float(max_dist_px * max_dist_px):
            return None
    return best[0]


def step_auto_projector_calibration(cfg: AppConfig, state: AppState, frame_bgr: np.ndarray) -> None:
    """
    Accumulate robust centroid samples for each projected corner target.
    This removes manual clicking and reduces human alignment error.
    """
    if len(state.camera_corners) != 4 or not state.auto_projector_points:
        return
    ordered_cam = order_points(state.camera_corners)
    idx = min(state.projector_calibration_index, len(state.auto_projector_points) - 1)
    expected_pt = transform_point(state.projector_to_camera_h, state.auto_projector_points[idx])
    detected = detect_projected_target(
        frame_bgr,
        background_gray=state.auto_background_gray,
        expected_pt=expected_pt,
        roi_polygon=ordered_cam,
        max_dist_px=cfg.auto_calib_max_corner_distance_px,
    )
    if detected is None:
        detected = detect_projected_target(
            frame_bgr,
            background_gray=state.auto_background_gray,
            expected_pt=expected_pt,
            roi_polygon=None,
            max_dist_px=None,
        )
    if detected is None:
        return

    state.auto_calibration_samples.append(detected)
    if len(state.auto_calibration_samples) < cfg.auto_calib_frames_per_corner:
        return

    pts = np.array(state.auto_calibration_samples, dtype=np.float32)
    med = np.median(pts, axis=0)
    finalized = (int(round(float(med[0]))), int(round(float(med[1]))))
    state.auto_camera_points.append(finalized)
    state.auto_calibration_samples = []
    state.projector_calibration_index = len(state.auto_camera_points)
    print(
        f"Auto-calibration locked point {state.projector_calibration_index}/"
        f"{len(state.auto_projector_points)}: {finalized}"
    )

    if len(state.auto_camera_points) >= len(state.auto_projector_points):
        state.current_mode = "start"
        state.projector_calibration_index = 0
        print("Automatic projector calibration complete (multi-point).")


# ----------------------------- persistence -----------------------------
def _serialize_ranges(state: AppState) -> Dict[str, object]:
    return {
        "start": {
            "lower": state.sampled_ranges["start"]["lower"].tolist(),
            "upper": state.sampled_ranges["start"]["upper"].tolist(),
        },
        "goal": {
            "lower1": state.sampled_ranges["goal"]["lower1"].tolist(),
            "upper1": state.sampled_ranges["goal"]["upper1"].tolist(),
            "lower2": state.sampled_ranges["goal"]["lower2"].tolist(),
            "upper2": state.sampled_ranges["goal"]["upper2"].tolist(),
        },
        "obs_list": [
            {"lower": item["lower"].tolist(), "upper": item["upper"].tolist()}
            for item in state.sampled_ranges["obs_list"]
        ],
        "obs_dynamic_list": [
            {"lower": item["lower"].tolist(), "upper": item["upper"].tolist()}
            for item in state.sampled_ranges["obs_dynamic_list"]
        ],
    }


def save_settings(state: AppState, filepath: str) -> None:
    data = {
        "camera_corners": [[int(x), int(y)] for x, y in state.camera_corners],
        "projector_observed_corners_cam": [[int(x), int(y)] for x, y in state.projector_observed_corners_cam],
        "auto_projector_points": [[int(x), int(y)] for x, y in state.auto_projector_points],
        "auto_camera_points": [[int(x), int(y)] for x, y in state.auto_camera_points],
        "camera_matrix": state.camera_matrix.tolist() if state.camera_matrix is not None else None,
        "dist_coeffs": state.dist_coeffs.tolist() if state.dist_coeffs is not None else None,
        **_serialize_ranges(state),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved settings to {os.path.abspath(filepath)}")


def _read_required_triplet(d: Dict[str, object], key: str) -> np.ndarray:
    raw = d.get(key)
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"invalid HSV vector for {key}: {raw}")
    return np.array([int(raw[0]), int(raw[1]), int(raw[2])])


def load_settings(state: AppState, filepath: str) -> None:
    if not os.path.exists(filepath):
        print(f"No saved config found at {os.path.abspath(filepath)}")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        camera_corners = [tuple(map(int, pt)) for pt in data.get("camera_corners", [])]
        projector_points = [tuple(map(int, pt)) for pt in data.get("projector_observed_corners_cam", [])]
        auto_projector_points = [tuple(map(int, pt)) for pt in data.get("auto_projector_points", [])]
        auto_camera_points = [tuple(map(int, pt)) for pt in data.get("auto_camera_points", [])]
        camera_matrix = data.get("camera_matrix")
        dist_coeffs = data.get("dist_coeffs")

        start = data.get("start", {})
        goal = data.get("goal", {})
        obs_list = data.get("obs_list", [])
        obs_dynamic_list = data.get("obs_dynamic_list", [])

        state.sampled_ranges["start"]["lower"] = _read_required_triplet(start, "lower")
        state.sampled_ranges["start"]["upper"] = _read_required_triplet(start, "upper")

        state.sampled_ranges["goal"]["lower1"] = _read_required_triplet(goal, "lower1")
        state.sampled_ranges["goal"]["upper1"] = _read_required_triplet(goal, "upper1")
        state.sampled_ranges["goal"]["lower2"] = _read_required_triplet(goal, "lower2")
        state.sampled_ranges["goal"]["upper2"] = _read_required_triplet(goal, "upper2")

        new_obs_list = []
        for item in obs_list:
            new_obs_list.append({"lower": _read_required_triplet(item, "lower"), "upper": _read_required_triplet(item, "upper")})

        new_obs_dynamic_list = []
        for item in obs_dynamic_list:
            new_obs_dynamic_list.append({"lower": _read_required_triplet(item, "lower"), "upper": _read_required_triplet(item, "upper")})

        state.camera_corners = camera_corners[:4]
        state.projector_observed_corners_cam = projector_points[:4]
        state.auto_projector_points = auto_projector_points
        state.auto_camera_points = auto_camera_points
        state.camera_matrix = np.array(camera_matrix) if camera_matrix is not None else None
        state.dist_coeffs = np.array(dist_coeffs) if dist_coeffs is not None else None
        state.sampled_ranges["obs_list"] = new_obs_list
        state.sampled_ranges["obs_dynamic_list"] = new_obs_dynamic_list
    except Exception as exc:  # config validation boundary
        print(f"WARNING: Failed to load config ({exc}). Keeping existing settings.")
        return

    print(
        f"Loaded settings from {os.path.abspath(filepath)} "
        f"({len(state.sampled_ranges['obs_list'])} obstacle colors, "
        f""
        f"{len(state.camera_corners)} ROI corners, "
        f"{len(state.projector_observed_corners_cam)} projector-correction points)"
    )


# ----------------------------- rendering -----------------------------
def draw_camera_overlay(
    frame: np.ndarray,
    cfg: AppConfig,
    state: AppState,
    start_cam_pt: Optional[Point],
    goal_cam_pt: Optional[Point],
    static_obstacle_cam_rects: Sequence[Rect],
    dynamic_obstacle_cam_rects: Sequence[Rect],
) -> np.ndarray:
    out = frame.copy()

    for i, pt in enumerate(state.camera_corners):
        cv2.circle(out, pt, 7, DRAW_YELLOW, -1)
        cv2.putText(out, f"C{i+1}", (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, DRAW_YELLOW, 2)

    if len(state.camera_corners) == 4:
        cv2.polylines(out, [order_points(state.camera_corners).astype(int)], True, DRAW_CYAN, 2)

    for i, pt in enumerate(state.projector_observed_corners_cam):
        cv2.circle(out, pt, 8, DRAW_MAGENTA, 2)
        cv2.putText(out, f"P{i+1}", (pt[0] + 8, pt[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, DRAW_MAGENTA, 2)

    for x, y, w, h in static_obstacle_cam_rects:
        cv2.rectangle(out, (x, y), (x + w, y + h), DRAW_ORANGE, 2)
        cv2.putText(out, "STATIC", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, DRAW_ORANGE, 2)

    for x, y, w, h in dynamic_obstacle_cam_rects:
        cv2.rectangle(out, (x, y), (x + w, y + h), DRAW_BLUE, 2)
        cv2.putText(out, "MOVING", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, DRAW_BLUE, 2)

    if start_cam_pt is not None:
        cv2.circle(out, start_cam_pt, 10, DRAW_GREEN, -1)
        cv2.putText(out, "START", (start_cam_pt[0] + 12, start_cam_pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, DRAW_GREEN, 2)

    if goal_cam_pt is not None:
        cv2.circle(out, goal_cam_pt, 10, DRAW_RED, -1)
        cv2.putText(out, "GOAL", (goal_cam_pt[0] + 12, goal_cam_pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, DRAW_RED, 2)

    cv2.putText(out, "Modes: c=ROI corners, 1=start, 2=goal, 3=obstacle, f=freeze start/goal", (20, cfg.frame_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.54, DRAW_CYAN, 2)
    cv2.putText(out, "Warped ROI is shown in a dedicated window; click here for sampling/calibration.", (20, cfg.frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.54, DRAW_WHITE, 2)
    return out



def draw_projector_output(
    width: int,
    height: int,
    start_pt: Optional[Point],
    goal_pt: Optional[Point],
    static_obstacle_rects: Sequence[ProjRect],
    dynamic_obstacle_rects: Sequence[ProjRect],
    planners: Optional[Dict[str, object]] = None,
    planner_view_mode: str = "overlay",
    active_planner: str = "rrt",
    freeze_start_goal: bool = False,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for x, y, w, h, _ in static_obstacle_rects:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (225, 243, 255), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), DRAW_ORANGE, 3)

    for x, y, w, h, _ in dynamic_obstacle_rects:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (230, 238, 255), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), DRAW_BLUE, 3)

    if planners:
        if planner_view_mode == "overlay":
            for key in ("rrt", "rrt_connect", "rrt_star", "rrtx"):
                if key in planners:
                    planners[key].draw(canvas)
        else:
            planners[active_planner].draw(canvas)

        legend_items = [
            ("RRT", (255, 140, 0)),
            ("RRT-Connect", (180, 70, 255)),
            ("RRT*", (0, 210, 255)),
            ("RRTX", (0, 180, 90)),
            ("Obstacle", DRAW_ORANGE),
        ]
        lx, ly = 18, 18
        for name, color in legend_items:
            cv2.rectangle(canvas, (lx, ly), (lx + 18, ly + 18), color, -1)
            cv2.putText(canvas, name, (lx + 28, ly + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
            ly += 24
        mode_text = "OVERLAY" if planner_view_mode == "overlay" else f"SEQUENTIAL: {active_planner.upper()}"
        freeze_text = "START/GOAL FROZEN" if freeze_start_goal else "START/GOAL LIVE"
        cv2.putText(canvas, mode_text, (18, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(canvas, freeze_text, (18, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.56, DRAW_GREEN if freeze_start_goal else (40, 40, 40), 2, cv2.LINE_AA)

    if start_pt is not None:
        cv2.circle(canvas, start_pt, 12, DRAW_GREEN, -1)
        cv2.circle(canvas, start_pt, 18, DRAW_GREEN, 2)
    if goal_pt is not None:
        cv2.circle(canvas, goal_pt, 12, DRAW_RED, -1)
        cv2.circle(canvas, goal_pt, 18, DRAW_RED, 2)
    return canvas


def draw_corner_target(canvas: np.ndarray, pt: np.ndarray, label: str, color: Tuple[int, int, int] = DRAW_RED) -> None:
    x, y = int(pt[0]), int(pt[1])
    arm, thickness, radius = 70, 8, 12
    h, w = canvas.shape[:2]

    x2 = min(w - 1, x + arm) if x == 0 else max(0, x - arm)
    y2 = min(h - 1, y + arm) if y == 0 else max(0, y - arm)
    cv2.line(canvas, (x, y), (x2, y), color, thickness)
    cv2.line(canvas, (x, y), (x, y2), color, thickness)
    cv2.circle(canvas, (x, y), radius, color, -1)

    tx = min(w - 120, x + 20) if x < w // 2 else max(10, x - 110)
    ty = min(h - 20, y + 40) if y < h // 2 else max(30, y - 20)
    cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, DRAW_BLUE, 2)


def draw_projector_calibration_frame(width: int, height: int, active_index: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    corners = get_projector_rect_points(width, height).astype(int)
    labels = ["TL", "TR", "BR", "BL"]

    for i, pt in enumerate(corners):
        draw_corner_target(canvas, pt, labels[i], color=(180, 180, 255) if i != active_index else DRAW_RED)

    cv2.putText(canvas, "PROJECTOR CALIBRATION", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, DRAW_BLUE, 2)
    cv2.putText(canvas, f"Click in CAMERA view where {labels[active_index]} target lands", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.72, DRAW_BLUE, 2)
    cv2.putText(canvas, "Order: TL -> TR -> BR -> BL", (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.72, DRAW_BLUE, 2)
    return canvas


def draw_auto_projector_calibration_frame(
    width: int,
    height: int,
    projector_points: Sequence[Point],
    active_index: Optional[int],
    radius: int,
) -> np.ndarray:
    """
    Auto calibration frame: black background with one bright white target.
    This pattern is intentionally high-contrast for robust camera detection.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    for i, pt in enumerate(projector_points):
        color = (80, 80, 80)
        r = max(4, radius // 2)
        if active_index is not None and i == active_index:
            color = (255, 255, 255)
            r = radius
        cv2.circle(canvas, pt, r, color, -1)

    if projector_points:
        i = 0 if active_index is None else min(active_index, len(projector_points) - 1)
        cv2.putText(canvas, "AUTO PROJECTOR CALIBRATION", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
        if active_index is None:
            cv2.putText(canvas, "Warmup: capturing dark baseline...", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 220, 220), 2)
        else:
            cv2.putText(canvas, f"Point {i + 1}/{len(projector_points)}", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 220, 220), 2)
        cv2.putText(canvas, "Do not click. Hold scene steady.", (30, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (220, 220, 220), 2)
    return canvas



def build_metrics_panel(
    state: AppState,
    fps: float,
    proj_w: int,
    proj_h: int,
    start_pt: Optional[Point],
    goal_pt: Optional[Point],
    obstacle_rects: Tuple[Sequence[ProjRect], Sequence[ProjRect]],
    obstacle_mask: Optional[np.ndarray],
    warped_ready: bool,
    output_cal_ready: bool,
    planners: Dict[str, object],
    obstacle_changed: bool,
) -> np.ndarray:
    panel_h, panel_w = 1020, 1160
    panel = np.full((panel_h, panel_w, 3), (18, 23, 30), dtype=np.uint8)

    def rounded_card(x1: int, y1: int, x2: int, y2: int, fill: Tuple[int, int, int], border: Tuple[int, int, int]) -> None:
        cv2.rectangle(panel, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(panel, (x1, y1), (x2, y2), border, 1)

    def put(x: int, y: int, text: str, color: Tuple[int, int, int] = (235, 240, 245), scale: float = 0.55, thick: int = 1) -> None:
        cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def chip(x: int, y: int, text: str, fg: Tuple[int, int, int], bg: Tuple[int, int, int]) -> int:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        w = tw + 22
        h = 28
        cv2.rectangle(panel, (x, y), (x + w, y + h), bg, -1)
        cv2.rectangle(panel, (x, y), (x + w, y + h), fg, 1)
        put(x + 11, y + 19, text, fg, 0.48, 1)
        return w

    planner_names = {"rrt": "RRT", "rrt_connect": "RRT-Connect", "rrt_star": "RRT*", "rrtx": "RRTX"}
    planner_colors = {"rrt": (255, 140, 0), "rrt_connect": (255, 0, 255), "rrt_star": (0, 210, 255), "rrtx": (0, 180, 90)}

    put(24, 34, "Projection-Mapped Detection + Multi-Planner Overlay", (245, 250, 255), 0.86, 2)
    put(24, 58, "Locked start/goal detection, stabilized obstacle boxes, simultaneous or sequential planner view", (150, 175, 195), 0.50, 1)

    x = 24
    x += chip(x, 72, f"FPS {fps:.1f}", (120, 220, 255), (28, 44, 58)) + 10
    x += chip(x, 72, f"Mode {state.current_mode.upper()}", (255, 210, 120), (55, 44, 24)) + 10
    x += chip(x, 72, "INPUT WARP ON" if warped_ready else "INPUT WARP OFF", (110, 240, 155) if warped_ready else (255, 120, 120), (24, 45, 34) if warped_ready else (58, 28, 28)) + 10
    x += chip(x, 72, "OUTPUT CORRECTION ON" if output_cal_ready else "OUTPUT CORRECTION OFF", (110, 240, 155) if output_cal_ready else (255, 120, 120), (24, 45, 34) if output_cal_ready else (58, 28, 28)) + 10
    x += chip(x, 72, "OVERLAY VIEW" if state.planner_view_mode == "overlay" else "SEQUENTIAL VIEW", (210, 200, 255), (42, 34, 58)) + 10
    chip(x, 72, f"CSV {'REC ON' if state.recording_enabled else 'REC OFF'}", (110, 240, 155) if state.recording_enabled else (205, 215, 225), (24, 45, 34) if state.recording_enabled else (40, 44, 52))

    rounded_card(24, 112, 340, 270, (24, 30, 38), (55, 73, 95))
    rounded_card(364, 112, 680, 270, (24, 30, 38), (55, 73, 95))
    rounded_card(704, 112, 1056, 270, (24, 30, 38), (55, 73, 95))

    put(42, 138, "Detection", (180, 215, 255), 0.68, 2)
    put(42, 170, f"Start: {'YES' if start_pt is not None else 'NO'}", (110, 240, 155) if start_pt else (255, 120, 120), 0.58, 1)
    if start_pt is not None:
        put(42, 196, f"Start @ ({start_pt[0]}, {start_pt[1]})", (225, 235, 245), 0.52, 1)
    put(42, 224, f"Goal: {'YES' if goal_pt is not None else 'NO'}", (110, 240, 155) if goal_pt else (255, 120, 120), 0.58, 1)
    if goal_pt is not None:
        put(42, 250, f"Goal @ ({goal_pt[0]}, {goal_pt[1]})", (225, 235, 245), 0.52, 1)

    put(382, 138, "Scene Geometry", (180, 215, 255), 0.68, 2)
    put(382, 170, f"Projector canvas: {proj_w} x {proj_h}", (225, 235, 245), 0.54, 1)
    put(382, 198, f"ROI corners: {len(state.camera_corners)}/4", (255, 210, 120), 0.54, 1)
    put(382, 226, f"Obstacle colors: {len(state.sampled_ranges['obs_list'])}", (255, 175, 95), 0.54, 1)
    put(382, 254, f"Obstacle boxes: {len(obstacle_rects[0])} | changed: {'YES' if obstacle_changed else 'NO'}", (255, 175, 95), 0.50, 1)

    put(722, 138, "Warp Tuning", (180, 215, 255), 0.68, 2)
    put(722, 170, f"tx={state.tune_tx:.1f}   ty={state.tune_ty:.1f}", (225, 235, 245), 0.54, 1)
    put(722, 198, f"sx={state.tune_sx:.3f}   sy={state.tune_sy:.3f}", (225, 235, 245), 0.54, 1)
    put(722, 226, f"Start/Goal freeze: {'ON' if state.freeze_start_goal else 'OFF'}", DRAW_GREEN if state.freeze_start_goal else (225, 235, 245), 0.52, 1)
    put(722, 254, f"Output pts manual/auto: {len(state.projector_observed_corners_cam)}/4  |  {len(state.auto_camera_points)}/{max(0, len(state.auto_projector_points))}", (235, 170, 255), 0.50, 1)

    planner_keys = ["rrt", "rrt_connect", "rrt_star", "rrtx"]
    card_w = 500
    card_h = 214
    x_positions = [24, 556]
    y_positions = [300, 530]
    for idx, key in enumerate(planner_keys):
        row = idx // 2
        col = idx % 2
        x0 = x_positions[col]
        y0 = y_positions[row]
        planner = planners[key]
        focus = key == state.active_planner
        fill = (26, 32, 42) if focus else (22, 27, 35)
        border = planner_colors[key] if focus else (60, 72, 88)
        rounded_card(x0, y0, x0 + card_w, y0 + card_h, fill, border)
        put(x0 + 18, y0 + 28, planner_names[key], planner_colors[key], 0.70, 2)
        status_color = (110, 240, 155) if planner.status == "FOUND" else (255, 190, 95) if planner.status in ("SEARCHING", "REPAIRING") else (255, 120, 120) if planner.status == "FAILED" else (205, 215, 225)
        put(x0 + 18, y0 + 58, f"Status: {planner.status}", status_color, 0.54, 1)
        completion = planner_task_completion_rate(planner)
        solve_txt = f"{planner.solve_time_ms:.1f} ms"
        first_txt = "-" if planner.first_solution_time_ms is None else f"{planner.first_solution_time_ms:.1f} ms"
        replans_txt = str(planner_replan_events(planner))
        clearance = planner_min_clearance(planner.path, obstacle_mask)
        clearance_txt = "-" if clearance is None else f"{clearance:.1f} px"
        rec_txt = "-" if getattr(planner, 'last_recovery_ms', None) is None else f"{planner.last_recovery_ms:.1f} ms"
        state_txt = "YES" if getattr(planner, 'recovering', False) else "NO"
        put(x0 + 18, y0 + 84, f"Completion: {completion:.0f}%", (225, 235, 245), 0.52, 1)
        put(x0 + 18, y0 + 110, f"Time-to-goal: {first_txt}", (225, 235, 245), 0.50, 1)
        put(x0 + 18, y0 + 136, f"Planning time: {solve_txt}", (225, 235, 245), 0.50, 1)
        put(x0 + 18, y0 + 162, f"Replans: {replans_txt}", (225, 235, 245), 0.50, 1)
        put(x0 + 250, y0 + 84, f"Nodes: {planner.total_nodes()}", (225, 235, 245), 0.52, 1)
        put(x0 + 250, y0 + 110, f"Path length: {planner.best_path_length:.1f}", (225, 235, 245), 0.50, 1)
        put(x0 + 250, y0 + 136, f"Min clearance: {clearance_txt}", (225, 235, 245), 0.50, 1)
        put(x0 + 250, y0 + 162, f"Recovery: {rec_txt}", (255, 190, 95) if getattr(planner, 'recovering', False) else (225, 235, 245), 0.50, 1)
        if key == "rrtx":
            put(x0 + 18, y0 + 186, f"Repairs: {planner.repairs}   Pruned last: {planner.last_pruned_count}   Recovering: {state_txt}", (110, 240, 155) if not planner.recovering else (255, 190, 95), 0.46, 1)

    rounded_card(24, 800, 1136, 990, (24, 30, 38), (55, 73, 95))
    put(42, 828, "Planner View", (180, 215, 255), 0.68, 2)
    put(42, 860, f"Enabled: {'YES' if state.planner_enabled else 'NO'}", (110, 240, 155) if state.planner_enabled else (255, 120, 120), 0.56, 1)
    put(42, 888, f"Display mode: {'Overlay (all on top)' if state.planner_view_mode == 'overlay' else 'Sequential (focus one)'}", (225, 235, 245), 0.54, 1)
    put(42, 916, f"Focus planner: {planner_names[state.active_planner]}", planner_colors[state.active_planner], 0.56, 1)

    put(520, 828, "Controls", (180, 215, 255), 0.68, 2)
    controls = [
        "h planner on/off", "y cycle focus planner", "i toggle overlay/sequential",
        "j reset planners", "c ROI corners", "1/2/3 sample start/goal/obstacle",
        "p manual output cal", "a auto output cal", "b checkerboard cal", "g intrinsic cal",
        "8/4/5/6 nudge output", "z/v scale X", "n/m scale Y", "t reset tuning", "e csv rec on/off", "k save  l load  q quit",
    ]
    cy = 860
    for i, txt in enumerate(controls):
        put(430, cy + i * 22, txt, (225, 235, 245), 0.48, 1)

    return panel




# ----------------------------- app loop -----------------------------
def make_mouse_callback(cfg: AppConfig, state: AppState):
    def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN or state.latest_hsv_frame is None:
            return
        if not (0 <= y < state.latest_hsv_frame.shape[0] and 0 <= x < state.latest_hsv_frame.shape[1]):
            return

        if state.current_mode == "corners":
            if len(state.camera_corners) < 4:
                state.camera_corners.append((x, y))
                print(f"Added ROI corner {len(state.camera_corners)} at ({x}, {y})")
            else:
                print("Already have 4 ROI corners. Press 'r' to reset or 'u' to undo.")
            return

        if state.current_mode == "proj_cal":
            if len(state.projector_observed_corners_cam) < 4:
                state.projector_observed_corners_cam.append((x, y))
                state.projector_calibration_index = len(state.projector_observed_corners_cam)
                print(f"Recorded projector calibration point {len(state.projector_observed_corners_cam)} at ({x}, {y})")
                if len(state.projector_observed_corners_cam) == 4:
                    state.current_mode = "start"
                    state.projector_calibration_index = 0
                    print("Projector calibration point collection complete.")
            else:
                print("Already have 4 projector calibration points.")
            return

        if state.current_mode in ("auto_proj_cal", "checker_cal"):
            return

        hsv_pixel = state.latest_hsv_frame[y, x]
        append_obstacle = state.current_mode in ("obs", "obs_dynamic")
        update_range_from_hsv_pixel(cfg, state, state.current_mode, hsv_pixel, append_obstacle=append_obstacle)

        if state.current_mode == "obs":
            print(f"Added obstacle color at ({x}, {y}) -> HSV {hsv_pixel.tolist()} (total colors: {len(state.sampled_ranges['obs_list'])})")
        elif state.current_mode == "obs_dynamic":
            print(f"Added obstacle color at ({x}, {y}) -> HSV {hsv_pixel.tolist()} (total colors: {len(state.sampled_ranges['obs_list'])})")
        else:
            print(f"Sampled {state.current_mode} at ({x}, {y}) -> HSV {hsv_pixel.tolist()}")

    return mouse_callback


def open_capture(camera_index: int) -> cv2.VideoCapture:
    if os.name == "nt":
        return cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(camera_index)


def run() -> None:
    cfg = AppConfig()
    state = AppState()
    planners = {"rrt": RRTPlanner(color=(255, 140, 0)), "rrt_connect": RRTConnectPlanner(color=DRAW_MAGENTA), "rrt_star": RRTStarPlanner(color=(0, 210, 255)), "rrtx": RRTXLitePlanner(color=(0, 180, 90))}

    cap = open_capture(cfg.camera_index)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {cfg.camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_h)

    cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(PROJECTOR_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(METRICS_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WARPED_WINDOW, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(CAMERA_WINDOW, 960, 540)
    cv2.resizeWindow(METRICS_WINDOW, 900, 760)
    cv2.resizeWindow(WARPED_WINDOW, 640, 360)
    cv2.resizeWindow(PROJECTOR_WINDOW, cfg.projector_init_w, cfg.projector_init_h)
    cv2.moveWindow(PROJECTOR_WINDOW, cfg.projector_x, cfg.projector_y)

    cv2.setMouseCallback(CAMERA_WINDOW, make_mouse_callback(cfg, state))
    load_settings(state, cfg.save_file)

    fps_counter = 0
    fps = 0.0
    fps_t0 = time.time()

    print("Running projection-mapped detector with RRT / RRT-Connect / RRT* / RRTX. h=planner, y=cycle focus, i=overlay/sequential, j=reset planners, f=freeze start/goal, e=csv record.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: Failed to read frame.")
            break

        frame = cv2.resize(frame, (cfg.frame_w, cfg.frame_h))
        if state.camera_matrix is not None and state.dist_coeffs is not None:
            frame = cv2.undistort(frame, state.camera_matrix, state.dist_coeffs)
        state.latest_hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        _, _, proj_w, proj_h = cv2.getWindowImageRect(PROJECTOR_WINDOW)
        proj_w, proj_h = max(100, proj_w), max(100, proj_h)

        update_camera_homography(state, proj_w, proj_h)
        update_output_prewarp(state, proj_w, proj_h)

        warped_ready = state.camera_to_projector_h is not None
        output_cal_ready = state.logical_to_projector_prewarp_h is not None

        start_proj_pt: Optional[Point] = None
        goal_proj_pt: Optional[Point] = None
        static_obstacle_proj_rects: List[ProjRect] = []
        dynamic_obstacle_proj_rects: List[ProjRect] = []
        start_cam_pt: Optional[Point] = None
        goal_cam_pt: Optional[Point] = None
        static_obstacle_cam_rects: List[Rect] = []
        dynamic_obstacle_cam_rects: List[Rect] = []

        obstacle_changed = False
        if warped_ready:
            warped_bgr = cv2.warpPerspective(frame, state.camera_to_projector_h, (proj_w, proj_h))
            warped_hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)

            start_mask, goal_mask, obs_static_mask, _ = build_masks(state, warped_hsv)
            raw_start_proj_pt, _ = largest_centroid(start_mask, cfg.min_dot_area)
            raw_goal_proj_pt, _ = largest_centroid(goal_mask, cfg.min_dot_area)

            state.stable_start_proj, state.pending_start_proj, state.pending_start_count, state.start_miss_count = stabilize_detected_point(
                raw_start_proj_pt, state.stable_start_proj, state.pending_start_proj, state.pending_start_count, state.start_miss_count
            )
            state.stable_goal_proj, state.pending_goal_proj, state.pending_goal_count, state.goal_miss_count = stabilize_detected_point(
                raw_goal_proj_pt, state.stable_goal_proj, state.pending_goal_proj, state.pending_goal_count, state.goal_miss_count
            )

            if state.freeze_start_goal:
                if state.frozen_start_proj is None and state.stable_start_proj is not None:
                    state.frozen_start_proj = state.stable_start_proj
                if state.frozen_goal_proj is None and state.stable_goal_proj is not None:
                    state.frozen_goal_proj = state.stable_goal_proj
                start_proj_pt = state.frozen_start_proj
                goal_proj_pt = state.frozen_goal_proj
            else:
                start_proj_pt = state.stable_start_proj
                goal_proj_pt = state.stable_goal_proj

            detected_obstacle_proj_rects = extract_obstacle_rectangles(
                obs_static_mask,
                cfg.min_obstacle_area,
                cfg.obstacle_bbox_padding_px,
                cfg.obstacle_deadzone_px,
            )
            static_obstacle_proj_rects = stabilize_obstacle_rects(
                detected_obstacle_proj_rects,
                state.stable_obstacle_proj_rects,
            )
            state.stable_obstacle_proj_rects = list(static_obstacle_proj_rects)
            dynamic_obstacle_proj_rects = []

            state.obstacle_mask_proj = obstacle_rects_to_mask(proj_w, proj_h, static_obstacle_proj_rects)
            obstacle_changed = masks_changed(state.last_planner_obstacle_mask, state.obstacle_mask_proj)

            start_cam_pt = transform_point(state.projector_to_camera_h, start_proj_pt)
            goal_cam_pt = transform_point(state.projector_to_camera_h, goal_proj_pt)
            for rect in static_obstacle_proj_rects:
                mapped = projector_rect_to_camera_rect(state.projector_to_camera_h, rect)
                if mapped:
                    static_obstacle_cam_rects.append(mapped)

            warped_preview = cv2.resize(warped_bgr, (320, 190))
            if state.obstacle_mask_proj is not None:
                preview_mask = cv2.cvtColor(cv2.resize(state.obstacle_mask_proj, (320, 190)), cv2.COLOR_GRAY2BGR)
                warped_preview = np.hstack((warped_preview, preview_mask))
        else:
            state.obstacle_mask_proj = None
            warped_preview = np.full((190, 320, 3), 60, dtype=np.uint8)
            cv2.putText(warped_preview, "Pick 4 ROI corners", (24, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.8, DRAW_WHITE, 2)

        if (
            state.planner_enabled
            and warped_ready
            and start_proj_pt is not None
            and goal_proj_pt is not None
            and state.obstacle_mask_proj is not None
            and state.current_mode not in ("proj_cal", "auto_proj_cal", "checker_cal", "intrinsic_cal", "corners")
        ):
            points_moved = points_changed(state.last_planner_start, start_proj_pt) or points_changed(state.last_planner_goal, goal_proj_pt)
            force_full_reset = points_moved

            for p in planners.values():
                if p.path is not None and not p.path_is_valid():
                    if isinstance(p, RRTXLitePlanner):
                        p.start_recovery()
                        p.update_obstacles(state.obstacle_mask_proj)
                        p.repair_after_obstacle_change()
                    else:
                        p.start_recovery()
                        force_full_reset = True
                        break
                if p.total_nodes() == 0 and p.start is not None:
                    force_full_reset = True
                    break

            if any(p.start is None or p.goal is None for p in planners.values()) or force_full_reset:
                recovering_keys = {name for name, p in planners.items() if getattr(p, 'recovering', False)}
                for name, p in planners.items():
                    p.reset(start_proj_pt, goal_proj_pt, proj_w, proj_h, state.obstacle_mask_proj)
                    if name in recovering_keys:
                        p.start_recovery()
                state.planner_episode_start_ms = time.time() * 1000.0
                state.last_planner_start = start_proj_pt
                state.last_planner_goal = goal_proj_pt
                state.last_planner_obstacle_mask = state.obstacle_mask_proj.copy()
            else:
                for p in planners.values():
                    p.update_obstacles(state.obstacle_mask_proj)
                    if obstacle_changed and isinstance(p, RRTXLitePlanner):
                        if p.path is not None and not p.path_is_valid():
                            p.start_recovery()
                        p.repair_after_obstacle_change()
                    elif obstacle_changed and not isinstance(p, RRTXLitePlanner):
                        p.start_recovery()
                        p.reset(start_proj_pt, goal_proj_pt, proj_w, proj_h, state.obstacle_mask_proj)
                        p.start_recovery()
                        state.planner_episode_start_ms = time.time() * 1000.0

                state.last_planner_start = start_proj_pt
                state.last_planner_goal = goal_proj_pt
                state.last_planner_obstacle_mask = state.obstacle_mask_proj.copy()

            planner_elapsed_ms = time.time() * 1000.0 - state.planner_episode_start_ms
            for p in planners.values():
                if p.ready():
                    p.grow(state.planner_step_budget, planner_elapsed_ms)
                if getattr(p, 'recovering', False) and p.path is not None and p.path_is_valid():
                    p.finish_recovery()

            maybe_record_planner_events(
                cfg,
                state,
                planners,
                start_proj_pt,
                goal_proj_pt,
                static_obstacle_proj_rects,
                state.obstacle_mask_proj,
                proj_w,
                proj_h,
                obstacle_changed,
            )
        else:
            for p in planners.values():
                p.clear()
            state.last_planner_start = None
            state.last_planner_goal = None
            state.last_planner_obstacle_mask = None
        camera_view = draw_camera_overlay(frame, cfg, state, start_cam_pt, goal_cam_pt, static_obstacle_cam_rects, dynamic_obstacle_cam_rects)

        logical_canvas = draw_projector_output(
            proj_w,
            proj_h,
            start_proj_pt,
            goal_proj_pt,
            static_obstacle_proj_rects,
            dynamic_obstacle_proj_rects,
            planners if state.planner_enabled else None,
            state.planner_view_mode,
            state.active_planner,
            state.freeze_start_goal,
        )

        if state.current_mode == "proj_cal":
            projector_img = draw_projector_calibration_frame(proj_w, proj_h, min(state.projector_calibration_index, 3))
        elif state.current_mode == "auto_proj_cal":
            if (
                not state.auto_projector_points
                or len(state.auto_projector_points) != cfg.auto_calib_grid_cols * cfg.auto_calib_grid_rows
            ):
                state.auto_projector_points = build_auto_calibration_projector_points(cfg, proj_w, proj_h)
                state.auto_camera_points = []
                state.projector_calibration_index = 0
                state.auto_calibration_samples = []
                state.auto_background_gray = None
                state.auto_warmup_counter = 0
            active: Optional[int]
            if state.auto_warmup_counter < cfg.auto_calib_warmup_frames:
                active = None
            else:
                active = min(state.projector_calibration_index, max(0, len(state.auto_projector_points) - 1))
            projector_img = draw_auto_projector_calibration_frame(
                proj_w,
                proj_h,
                state.auto_projector_points,
                active,
                cfg.auto_calib_target_radius_px,
            )
        elif state.current_mode == "checker_cal":
            projector_img, checker_points = build_checkerboard_pattern(cfg, proj_w, proj_h)
            if not state.checker_projector_points or len(state.checker_projector_points) != len(checker_points):
                state.checker_projector_points = checker_points
                state.checker_camera_samples = []
        elif state.current_mode == "intrinsic_cal":
            projector_img, _ = build_checkerboard_pattern(cfg, proj_w, proj_h)
        elif output_cal_ready:
            tuned_h = apply_user_tune_homography(state.logical_to_projector_prewarp_h, state)
            projector_img = warp_with_src_to_dst(logical_canvas, tuned_h, proj_w, proj_h)
        else:
            projector_img = logical_canvas

        if state.current_mode == "auto_proj_cal":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if state.auto_warmup_counter < cfg.auto_calib_warmup_frames:
                state.auto_background_gray = gray
                state.auto_warmup_counter += 1
            else:
                step_auto_projector_calibration(cfg, state, frame)
        elif state.current_mode == "checker_cal":
            step_checkerboard_calibration(cfg, state, frame)
        elif state.current_mode == "intrinsic_cal":
            step_intrinsic_calibration(cfg, state, frame)

        metrics_panel = build_metrics_panel(state, fps, proj_w, proj_h, start_proj_pt, goal_proj_pt, (static_obstacle_proj_rects, dynamic_obstacle_proj_rects), state.obstacle_mask_proj, warped_ready, output_cal_ready, planners, obstacle_changed)

        fps_counter += 1
        now = time.time()
        if (now - fps_t0) >= 1.0:
            fps = fps_counter / (now - fps_t0)
            fps_counter = 0
            fps_t0 = now

        cv2.imshow(CAMERA_WINDOW, camera_view)
        cv2.imshow(WARPED_WINDOW, warped_preview)
        cv2.imshow(PROJECTOR_WINDOW, projector_img)
        cv2.imshow(METRICS_WINDOW, metrics_panel)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            state.current_mode = "corners"
            print("ROI corner mode active. Click 4 ROI corners in the camera window.")
        elif key == ord("1"):
            state.current_mode = "start"
        elif key == ord("2"):
            state.current_mode = "goal"
        elif key == ord("3"):
            state.current_mode = "obs"
        elif key == ord("p"):
            if len(state.camera_corners) != 4:
                print("Set the 4 ROI corners first before projector calibration.")
            else:
                state.projector_observed_corners_cam.clear()
                state.projector_calibration_index = 0
                state.current_mode = "proj_cal"
                state.auto_calibration_samples = []
        elif key == ord("a"):
            if len(state.camera_corners) != 4:
                print("Set the 4 ROI corners first before auto projector calibration.")
            else:
                state.projector_observed_corners_cam.clear()
                state.projector_calibration_index = 0
                state.auto_calibration_samples = []
                state.auto_projector_points = build_auto_calibration_projector_points(cfg, proj_w, proj_h)
                state.auto_camera_points = []
                state.auto_background_gray = None
                state.auto_warmup_counter = 0
                state.current_mode = "auto_proj_cal"
                print(
                    "Automatic projector calibration started "
                    f"({len(state.auto_projector_points)} points)."
                )
        elif key == ord("b"):
            if len(state.camera_corners) != 4:
                print("Set the 4 ROI corners first before checkerboard calibration.")
            else:
                state.checker_projector_points = []
                state.checker_camera_samples = []
                state.checker_miss_counter = 0
                state.current_mode = "checker_cal"
                print("Checkerboard calibration started. Keep pattern fully visible to the camera.")
        elif key == ord("g"):
            state.intrinsic_obj_points = []
            state.intrinsic_img_points = []
            state.current_mode = "intrinsic_cal"
            print("Intrinsic calibration started. Slightly move camera/view and keep checkerboard visible.")
        elif key == ord("o"):
            state.projector_observed_corners_cam.clear()
            state.projector_calibration_index = 0
            state.auto_calibration_samples = []
            state.auto_projector_points = []
            state.auto_camera_points = []
            state.auto_background_gray = None
            state.auto_warmup_counter = 0
            state.checker_projector_points = []
            state.checker_camera_samples = []
            state.checker_miss_counter = 0
            state.intrinsic_obj_points = []
            state.intrinsic_img_points = []
            print("Cleared projector output calibration.")
        elif key == ord("r"):
            state.camera_corners.clear()
            for p in planners.values():
                p.clear()
            state.last_planner_start = None
            state.last_planner_goal = None
            state.last_planner_obstacle_mask = None
            state.stable_start_proj = None
            state.stable_goal_proj = None
            state.pending_start_proj = None
            state.pending_goal_proj = None
            state.pending_start_count = 0
            state.pending_goal_count = 0
            state.start_miss_count = 0
            state.goal_miss_count = 0
            state.freeze_start_goal = False
            state.frozen_start_proj = None
            state.frozen_goal_proj = None
            state.stable_obstacle_proj_rects = []
            print("Reset ROI corners.")
        elif key == ord("u"):
            if state.current_mode == "proj_cal" and state.projector_observed_corners_cam:
                removed = state.projector_observed_corners_cam.pop()
                state.projector_calibration_index = len(state.projector_observed_corners_cam)
                print(f"Removed last projector calibration point: {removed}")
            elif state.camera_corners:
                removed = state.camera_corners.pop()
                print(f"Removed last ROI corner: {removed}")
        elif key == ord("k"):
            save_settings(state, cfg.save_file)
        elif key == ord("l"):
            load_settings(state, cfg.save_file)
        elif key == ord("x"):
            state.sampled_ranges["obs_list"] = []
            state.sampled_ranges["obs_dynamic_list"] = []
            state.stable_obstacle_proj_rects = []
            print("Cleared all obstacle color samples.")
        elif key == ord("f"):
            state.freeze_start_goal = not state.freeze_start_goal
            if state.freeze_start_goal:
                state.frozen_start_proj = state.stable_start_proj
                state.frozen_goal_proj = state.stable_goal_proj
            else:
                state.frozen_start_proj = None
                state.frozen_goal_proj = None
            print(f"Freeze start/goal: {state.freeze_start_goal}")
        elif key == ord("8"):
            state.tune_ty -= 2.0
        elif key == ord("5"):
            state.tune_ty += 2.0
        elif key == ord("7"):
            state.tune_tx -= 2.0
        elif key == ord("9"):
            state.tune_tx += 2.0
        elif key == ord("m"):
            state.tune_sy = min(1.25, state.tune_sy + 0.003)
        elif key == ord("n"):
            state.tune_sy = max(0.75, state.tune_sy - 0.003)
        elif key == ord("v"):
            state.tune_sx = min(1.25, state.tune_sx + 0.003)
        elif key == ord("z"):
            state.tune_sx = max(0.75, state.tune_sx - 0.003)
        elif key == ord("t"):
            state.tune_tx = 0.0
            state.tune_ty = 0.0
            state.tune_sx = 1.0
            state.tune_sy = 1.0
            print("Reset output tuning offsets/scales.")
        elif key == ord("h"):
            state.planner_enabled = not state.planner_enabled
            for p in planners.values():
                p.clear()
            state.last_planner_start = None
            state.last_planner_goal = None
            state.last_planner_obstacle_mask = None
            state.planner_episode_start_ms = time.time() * 1000.0
            print(f"Planner enabled: {state.planner_enabled}")
        elif key == ord("y"):
            planner_order = ["rrt", "rrt_connect", "rrt_star", "rrtx"]
            idx = planner_order.index(state.active_planner)
            state.active_planner = planner_order[(idx + 1) % len(planner_order)]
            print(f"Focus planner: {state.active_planner}")
        elif key == ord("i"):
            state.planner_view_mode = "sequential" if state.planner_view_mode == "overlay" else "overlay"
            print(f"Planner view mode: {state.planner_view_mode}")
        elif key == ord("j"):
            for p in planners.values():
                p.clear()
            state.last_planner_start = None
            state.last_planner_goal = None
            state.last_planner_obstacle_mask = None
            state.planner_episode_start_ms = time.time() * 1000.0
            print("Planner reset.")
        elif key == ord("e"):
            if state.recording_enabled:
                stop_csv_recording_session(state)
            else:
                start_csv_recording_session(state)
                for p in planners.values():
                    p.clear()
                state.last_planner_start = None
                state.last_planner_goal = None
                state.last_planner_obstacle_mask = None
                state.planner_episode_start_ms = time.time() * 1000.0
                print("CSV recording started and planners reset.")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ----------------------------- CSV recording + recovery patch helpers -----------------------------

def _planner_label(planner_name: str) -> str:
    return planner_name.replace('*', 'star').replace('-', '_').replace(' ', '_').lower()

def _ensure_recovery_methods(cls) -> None:
    if not hasattr(cls, 'start_recovery'):
        def start_recovery(self) -> None:
            if not getattr(self, 'recovering', False):
                self.recovering = True
                self.recovery_start_ms = time.time() * 1000.0
                self.last_recovery_ms = None
        cls.start_recovery = start_recovery
    if not hasattr(cls, 'finish_recovery'):
        def finish_recovery(self) -> None:
            if getattr(self, 'recovering', False) and getattr(self, 'recovery_start_ms', None) is not None:
                self.last_recovery_ms = time.time() * 1000.0 - self.recovery_start_ms
            self.recovering = False
            self.recovery_start_ms = None
        cls.finish_recovery = finish_recovery

for _cls in (RRTPlanner, RRTConnectPlanner, RRTStarPlanner, RRTXLitePlanner):
    _ensure_recovery_methods(_cls)

def _csv_headers() -> list[str]:
    return [
        'session_id','event_index','timestamp_iso','event_type','planner_key','planner_name','status',
        'solved','task_completion_rate','time_to_goal_ms','total_cumulative_planning_time_ms',
        'replanning_events','total_executed_path_length','minimum_obstacle_clearance_px','recovery_time_ms',
        'recovering','node_count','active_node_count','path_point_count','start_x','start_y','goal_x','goal_y',
        'projector_w','projector_h','planner_step_size','planner_goal_radius','planner_collision_step','planner_goal_bias_every',
        'roi_corners','obstacle_rects','path_points','obstacle_mask_nonzero','obstacle_mask_checksum','save_file'
    ]

def start_csv_recording_session(state: AppState) -> None:
    session_id = time.strftime('planning_record_%Y%m%d_%H%M%S')
    out_dir = Path.cwd() / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    state.recording_enabled = True
    state.recording_session_id = session_id
    state.recording_dir = str(out_dir)
    state.recording_event_index = 0
    state.recording_files = {}
    state._record_last_path_sig = {}
    state._record_last_obstacle_sig = None
    headers = _csv_headers()
    for key,name in [('rrt','RRT'),('rrt_connect','RRT-Connect'),('rrt_star','RRT*'),('rrtx','RRTX')]:
        path = out_dir / f'{_planner_label(name)}.csv'
        with path.open('w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(headers)
        state.recording_files[key] = str(path)
    print(f'CSV recording started: {out_dir}')

def stop_csv_recording_session(state: AppState) -> None:
    if state.recording_enabled:
        print(f'CSV recording stopped: {state.recording_dir}')
    state.recording_enabled = False
    state.recording_session_id = None
    state.recording_dir = None
    state.recording_files = {}

def _planner_active_node_count(planner) -> int:
    if hasattr(planner, '_active_indices'):
        try:
            return len(planner._active_indices())
        except Exception:
            pass
    if hasattr(planner, 'active_node_count'):
        try:
            return int(planner.active_node_count())
        except Exception:
            pass
    if hasattr(planner, 'total_nodes'):
        try:
            return int(planner.total_nodes())
        except Exception:
            pass
    return 0

def _planner_min_clearance(planner, obstacle_mask: np.ndarray):
    path = getattr(planner, 'path', None)
    if not path or obstacle_mask is None or obstacle_mask.size == 0:
        return None
    inv = cv2.bitwise_not(obstacle_mask)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    vals = []
    h,w = obstacle_mask.shape[:2]
    for x,y in path:
        if 0 <= x < w and 0 <= y < h:
            vals.append(float(dist[y, x]))
    return min(vals) if vals else None

def _write_csv_row(csv_path: str, row: list) -> None:
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(row)

def maybe_record_planner_events(cfg: AppConfig, state: AppState, planners: Dict[str, object],
                                start_proj_pt: Optional[Point], goal_proj_pt: Optional[Point],
                                obstacle_rects, obstacle_mask_proj: np.ndarray, proj_w: int, proj_h: int,
                                obstacle_changed: bool) -> None:
    if not getattr(state, 'recording_enabled', False) or not getattr(state, 'recording_files', None):
        return
    if start_proj_pt is None or goal_proj_pt is None:
        return
    last_path = getattr(state, '_record_last_path_sig', {})
    obstacle_checksum = int(np.sum(obstacle_mask_proj.astype(np.uint64)) % 1000000007) if obstacle_mask_proj is not None else 0
    obstacle_sig = (int(np.count_nonzero(obstacle_mask_proj)) if obstacle_mask_proj is not None else 0, obstacle_checksum, tuple(obstacle_rects or []))
    ts = time.strftime('%Y-%m-%dT%H:%M:%S')
    def make_row(event_type, key, planner, path):
        return [
            state.recording_session_id, state.recording_event_index, ts, event_type, key, getattr(planner,'name',key), getattr(planner,'status',''),
            int(bool(path)), '', getattr(planner,'first_solution_time_ms',None), getattr(planner,'solve_time_ms',None), getattr(planner,'replans',None),
            getattr(planner,'best_path_length',None), _planner_min_clearance(planner, obstacle_mask_proj), getattr(planner,'last_recovery_ms',None),
            int(bool(getattr(planner,'recovering',False))), getattr(planner,'total_nodes',lambda:0)(), _planner_active_node_count(planner),
            len(path or []), start_proj_pt[0], start_proj_pt[1], goal_proj_pt[0], goal_proj_pt[1], proj_w, proj_h,
            getattr(planner,'step_size',None), getattr(planner,'goal_radius',None), getattr(planner,'collision_step',None), getattr(planner,'goal_bias_every',None),
            json.dumps(state.camera_corners), json.dumps(obstacle_rects or []), json.dumps(path),
            int(np.count_nonzero(obstacle_mask_proj)) if obstacle_mask_proj is not None else 0, obstacle_checksum, cfg.save_file
        ]
    if obstacle_changed and getattr(state, '_record_last_obstacle_sig', None) != obstacle_sig:
        state._record_last_obstacle_sig = obstacle_sig
        for key, planner in planners.items():
            csv_path = state.recording_files.get(key)
            if not csv_path:
                continue
            state.recording_event_index += 1
            _write_csv_row(csv_path, make_row('obstacle_change', key, planner, getattr(planner,'path',None)))
    for key, planner in planners.items():
        csv_path = state.recording_files.get(key)
        if not csv_path:
            continue
        path = getattr(planner, 'path', None)
        sig = tuple(path) if path else None
        if sig is None or last_path.get(key) == sig:
            continue
        last_path[key] = sig
        state.recording_event_index += 1
        _write_csv_row(csv_path, make_row('new_path', key, planner, path))
    state._record_last_path_sig = last_path


if __name__ == "__main__":
    run()
