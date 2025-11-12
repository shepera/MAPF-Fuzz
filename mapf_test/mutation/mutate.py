from dataclasses import replace
from pprint import pprint
import re
import numpy as np
import copy
from numba import njit
from mapf_test.maptype import *
from functools import partial
from typing import Callable, Sequence
import random
from mapf_test.mutation.mutate_agent_fn import *
from mapf_test.settings import GUIDANCE_RATE
from .mutate_agent_no_guidance_fn import mutate_agent_no_guidance
from .mutate_map_fn import *
from codetiming import Timer
import cv2


@njit
def window_sum(y: int, x: int, sz, integral) -> float:
    y2, x2 = y + sz - 1, x + sz - 1
    total = integral[y2, x2]
    if y > 0:
        total -= integral[y - 1, x2]
    if x > 0:
        total -= integral[y2, x - 1]
    if y > 0 and x > 0:
        total += integral[y - 1, x - 1]
    return total


def critical_area(
    task: AgentTask, sz: int, num: int
) -> list[tuple[(int, int), (int, int)]]:
    freq: np.ndarray = task.analysis.freq_map.astype(np.float32, copy=False)
    H, W = freq.shape
    if sz > H or sz > W:
        raise ValueError(f"window size {sz} exceeds map size {freq.shape}")
    I = np.pad(freq, ((1, 0), (1, 0)), mode="constant")
    S = I[sz:, sz:] - I[:-sz, sz:] - I[sz:, :-sz] + I[:-sz, :-sz]
    flat = S.ravel()
    if num <= 0:
        return []
    K = min(len(flat), num * 4)
    idx_top = np.argpartition(flat, -K)[-K:]
    idx_top = idx_top[np.argsort(flat[idx_top])[::-1]]
    ys, xs = np.unravel_index(idx_top, S.shape)
    candidates = list(zip(flat[idx_top], ys, xs))
    random.shuffle(candidates[: max(4, num)])
    occupied = np.zeros_like(freq, dtype=bool)
    selected: list[tuple[(int, int), (int, int)]] = []
    for heat, y, x in candidates:
        if occupied[y : y + sz, x : x + sz].any():
            continue
        occupied[y : y + sz, x : x + sz] = True
        selected.append(((y, x), (y + sz - 1, x + sz - 1)))
        if len(selected) >= num:
            break
    return selected


from typing import Tuple
import uuid


@Timer(name="mutate", logger=None)
def mutate_task(task: AgentTask, sz: int = 32, num: int = 3) -> AgentTask:
    id = str(uuid.uuid4())
    if np.random.rand() < settings.MUTATE_AGENT_RATE:
        if np.random.rand() < settings.GUIDANCE_RATE:
            task = replace(mutate_agent(task), task_id=id)
        else:
            task = replace(mutate_agent_no_guidance(task), task_id=id)
    else:
        map = task.map
        critical_areas = critical_area(task, sz, num)
        for cr in critical_areas:
            map = mutate_area(map, cr, task)
        task = replace(task, map=map, task_id=id)
    return task


def mutate_area(
    m: MapInfo, area: Tuple[Tuple[int, int], Tuple[int, int]], task: AgentTask
) -> MapInfo:
    res = copy.deepcopy(m)
    task = replace(task, map=res)
    map_data = res.map_data
    (y1, x1), (y2, x2) = area
    for _ in range(50):
        sub = map_data[y1 : y2 + 1, x1 : x2 + 1].copy()
        h, w = sub.shape
        new_inner = generate_patch_with_border(h, w, border=sub)
        sub[:, :] = cal_new_area(sub, new_inner)
        map_data[y1 : y2 + 1, x1 : x2 + 1] = sub
        if success := task.check_task():
            break
    return res if success else m


def cal_new_area(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    if getattr(settings, "REPLACE_ONLY", False):
        return a2.copy()
    ops: list[tuple[float, Callable[[np.ndarray, np.ndarray], np.ndarray]]] = [
        (4, lambda x, y: y),
        (1, lambda x, y: np.logical_or(x, y)),
        (0.3, lambda x, y: np.logical_and(x, y)),
        (0.3, lambda x, y: np.logical_and(np.logical_not(x), y)),
        (0.3, lambda x, y: np.logical_and(x, np.logical_not(y))),
        (1, lambda x, y: np.logical_or(np.logical_not(x), y)),
        (1, lambda x, y: np.logical_or(x, np.logical_not(y))),
    ]
    weights, funcs = zip(*ops)
    probs = np.array(weights, dtype=float)
    probs /= probs.sum()
    chosen_func = np.random.choice(funcs, p=probs)
    return chosen_func(a1, a2).astype(np.uint8)


def add_road_or_wall(border: np.ndarray) -> np.ndarray:
    patch = border.copy()
    h, w = patch.shape
    two_line_prob = 0.2
    draw_two = np.random.rand() < two_line_prob
    line_val = np.random.choice([0, 1])
    r0, c0 = np.random.randint(0, h), np.random.randint(0, w)
    r1, c1 = np.random.randint(0, h), np.random.randint(0, w)
    if draw_two:
        dr = r1 - r0
        dc = c1 - c0
        pr = -np.sign(dc)
        pc = np.sign(dr)
        offset = np.random.randint(1, 3)
        r0b = np.clip(r0 + pr * offset, 0, h - 1)
        c0b = np.clip(c0 + pc * offset, 0, w - 1)
        r1b = np.clip(r1 + pr * offset, 0, h - 1)
        c1b = np.clip(c1 + pc * offset, 0, w - 1)

    def bresenham_line(y0, x0, y1, x1):
        points = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            points.append((y, x))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return points

    break_prob = 0.1
    pts_main = bresenham_line(r0, c0, r1, c1)
    lines = [pts_main]
    if draw_two:
        pts_parallel = bresenham_line(r0b, c0b, r1b, c1b)
        lines.append(pts_parallel)
    for pts in lines:
        for r, c in pts:
            if np.random.rand() < break_prob:
                continue
            if 0 <= r < h and 0 <= c < w:
                patch[r, c] = line_val
    return patch


PatchFn = Callable[[], np.ndarray]


def generate_patch_with_border(
    height: int,
    width: int,
    border: np.ndarray,
    *,
    weights: Sequence[float] = (2, 2, 1),
) -> np.ndarray:
    rng = np.random.default_rng()
    strategies: list[PatchFn] = [
        partial(random_patch, height, width),
        partial(cellular_automata_patch, height, width, border=border),
        partial(DLA_patch, height, width, border=border),
    ]
    probs = np.asarray(weights, dtype=float)
    probs /= probs.sum()
    patch_fn: PatchFn = rng.choice(strategies, p=probs)
    return patch_fn()


def random_patch(height: int, width: int, obs_density: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng()
    if np.random.rand() < 3:
        obs_density = random.uniform(0.1, 0.3)
        patch = (rng.random((height, width)) >= obs_density).astype(np.uint8)
    else:
        rng = random.Random()
        scale = height / float(rng.randint(1, 10))
        octaves = 5
        persistence = 0.5
        lacunarity = 2.0
        x_offset = rng.uniform(0, 1000)
        y_offset = rng.uniform(0, 1000)
        grid_values = np.ones((height, width))
        for r in range(height):
            for c in range(width):
                grid_values[r, c] = noise.pnoise2(
                    (r + x_offset) / scale,
                    (c + y_offset) / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                )
        sorted_noise = np.sort(grid_values.flatten())
        threshold_val = sorted_noise[int(obs_density * len(sorted_noise))]
        patch = (grid_values >= threshold_val).astype(np.uint8)
    return patch


@njit
def DLA_patch(height: int, width: int, border: np.ndarray) -> np.ndarray:
    patch = np.ones((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if i == 0 or j == 0 or i == height - 1 or j == width - 1:
                patch[i, j] = border[i, j]
    obstacle_count = int((patch == 0).sum())
    cy, cx = np.random.randint(1, height - 1), np.random.randint(1, width - 1)
    if patch[cy, cx] == 1:
        patch[cy, cx] = 0
        obstacle_count += 1
    target = 0.3 * height * width
    max_walkers = target * 5
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while obstacle_count < target and max_walkers > 0:
        max_walkers -= 1
        wy, wx = np.random.randint(1, height - 1), np.random.randint(1, width - 1)
        if patch[wy, wx] == 0:
            continue
        while True:
            dy, dx = neighbors[np.random.randint(0, 4)]
            wy, wx = wy + dy, wx + dx
            if not (0 < wy < height - 1 and 0 < wx < width - 1):
                break
            adhesion = False
            for ddy, ddx in neighbors:
                ny, nx = wy + ddy, wx + ddx
                if 0 <= ny < height and 0 <= nx < width and patch[ny, nx] == 0:
                    adhesion = True
                    break
            if adhesion:
                patch[wy, wx] = 0
                obstacle_count += 1
                break
    return patch
