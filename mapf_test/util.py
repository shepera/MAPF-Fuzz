from cv2 import line
import numpy as np
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from mapf_test.maptype import *
from numba import njit
import mapf_test.settings as settings
from codetiming import Timer
import numpy as np
import os


def arr2mapfile(arr: np.ndarray, filename: str):
    dir_name = os.path.dirname(filename)
    if len(dir_name) > 0 and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    h, w = arr.shape
    char_array = np.where(arr == 1.0, ".", "@")
    with open(filename, "w") as f:
        f.write("type octile\n")
        f.write(f"height {h}\n")
        f.write(f"width {w}\n")
        f.write("map\n")
        lines_to_write = ["".join(char_array[i]) + "\n" for i in range(h)]
        f.writelines(lines_to_write)


def _resize_nn_numpy(arr: np.ndarray, target: int) -> np.ndarray:
    h, w = arr.shape
    if h == target and w == target:
        return arr
    sy = target // h
    sx = target // w
    if sy * h != target or sx * w != target:
        raise ValueError()
    arr = np.repeat(arr, sy, axis=0)
    arr = np.repeat(arr, sx, axis=1)
    return arr.astype(np.uint8)


from scipy.ndimage import zoom


def _resize_nn_scipy(arr: np.ndarray, target: int) -> np.ndarray:
    h, w = arr.shape
    zy, zx = target / h, target / w
    arr = zoom(arr, zoom=(zy, zx), order=0)
    return arr.astype(np.uint8)


import cv2


def _resize_nn_cv2(arr: np.ndarray, target: int) -> np.ndarray:
    arr = cv2.resize(arr, (target, target), interpolation=cv2.INTER_NEAREST)
    return arr.astype(np.uint8)


def mapfile2arr(filename: str, map_size: int = settings.MAP_SIZE) -> np.ndarray:
    with open(filename, "r") as f:
        lines = f.readlines()
    idx = 0
    while idx < len(lines) and lines[idx].strip().lower() != "map":
        idx += 1
    idx += 1
    grid = np.array(
        [[1 if c == "." else 0 for c in line.strip()] for line in lines[idx:]],
        dtype=np.uint8,
    )
    if map_size > 0:
        if _resize_nn_cv2:
            grid = _resize_nn_cv2(grid, map_size)
        elif _resize_nn_scipy:
            grid = _resize_nn_scipy(grid, map_size)
        else:
            grid = _resize_nn_numpy(grid, map_size)
    return grid


def parse_result_file(file_path):
    result = {}
    with open(file_path, "r") as f:
        lines = f.readlines()
    parsing_solution = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line == "solution=":
            result["solution"] = []
            parsing_solution = True
            continue
        if parsing_solution and ":" in line:
            _, coords_str = line.split(":", 1)
            coords_clean = coords_str.strip().strip(",")
            numbers_only_str = coords_clean.replace("(", "").replace(")", "")
            string_numbers = numbers_only_str.split(",")
            coords = [
                (int(string_numbers[i + 1]), int(string_numbers[i]))
                for i in range(0, len(string_numbers), 2)
            ]
            result["solution"].append(coords)
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in [
                "agents",
                "solved",
                "soc",
                "lb_soc",
                "makespan",
                "lb_makespan",
                "comp_time",
            ]:
                try:
                    result[key] = int(value)
                except ValueError:
                    result[key] = value
            else:
                result[key] = value
    return result


def file2result(filename: str) -> MapfResult:
    res = parse_result_file(filename)
    result_obj = MapfResult(
        task=None,
        solver=res["solver"],
        solved=res["solved"],
        soc=res["soc"],
        makespan=res["makespan"],
        comp_time=res["comp_time"],
        solution=res["solution"],
    )
    return result_obj


def task2scen(instance: AgentTask, scen_filename: str = None, map_filename: str = None):
    mapinfo = instance.map
    map_height = mapinfo.map_data.shape[0]
    map_width = mapinfo.map_data.shape[1]
    actual_map_filename = (
        os.path.basename(map_filename) if map_filename else "unknown.map"
    )
    line_prefix_for_agents = f"3\t{actual_map_filename}\t{map_width}\t{map_height}"
    dir_name = os.path.dirname(scen_filename or "")
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    lines_to_write = ["version 1\n"]
    for i, (start, end) in enumerate(instance.agent_pairs):
        start_row, start_col = start
        end_row, end_col = end
        manhattan_distance = abs(start_row - end_row) + abs(start_col - end_col)
        agent_line = f"{line_prefix_for_agents}\t{start_col}\t{start_row}\t{end_col}\t{end_row}\t{manhattan_distance}\n"
        lines_to_write.append(agent_line)
    with open(scen_filename, "w") as f:
        f.writelines(lines_to_write)


def task2txt(instance: AgentTask, txt_filename: str = None, map_filename: str = None):
    actual_map_filename = (
        os.path.basename(map_filename) if map_filename else "unknown.map"
    )
    dir_name = os.path.dirname(txt_filename or "")
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    lines_to_write = []
    lines_to_write.append(f"map_file={actual_map_filename}\n")
    lines_to_write.append(f"agents={len(instance.agent_pairs)}\n")
    lines_to_write.append(f"seed=0\n")
    lines_to_write.append(f"random_problem=0\n")
    lines_to_write.append(f"max_timestep={settings.MAX_TIMESTEP}\n")
    lines_to_write.append(f"max_comp_time=10000\n")
    lines_to_write.append(f"\n")
    for _, (start, end) in enumerate(instance.agent_pairs):
        start_row, start_col = start
        end_row, end_col = end
        line = f"{start_col},{start_row},{end_col},{end_row}\n"
        lines_to_write.append(line)
    with open(txt_filename, "w") as f:
        f.writelines(lines_to_write)


def task2file(
    instance: AgentTask, filename: str = None, map_filename: str = None, type=None
):
    if type == "scen":
        task2scen(instance, filename, map_filename)
    elif type == "txt":
        task2txt(instance, filename, map_filename)
    else:
        task2scen(instance, filename + ".scen", map_filename)
        task2txt(instance, filename + ".txt", map_filename)


@njit(inline="always")
def _manhattan(ax: int, ay: int, bx: int, by: int) -> int:
    return abs(ax - bx) + abs(ay - by)


@njit
def _astar_len(map_arr: np.ndarray, sx: int, sy: int, gx: int, gy: int) -> int:
    if sx == gx and sy == gy:
        return 0
    h, w = map_arr.shape
    ncell = h * w
    goal_idx = gx * w + gy
    start_idx = sx * w + sy
    INF = np.int32(2**31 - 1)
    visited = np.zeros(ncell, dtype=np.uint8)
    gscore = np.full(ncell, INF, dtype=np.int32)
    gscore[start_idx] = 0
    heap_f = np.empty(ncell + 1, dtype=np.int32)
    heap_idx = np.empty(ncell + 1, dtype=np.int32)
    size = 1
    heap_f[size] = _manhattan(sx, sy, gx, gy)
    heap_idx[size] = start_idx
    dirs = (-1, 0), (1, 0), (0, -1), (0, 1)
    while size:
        cur_f = heap_f[1]
        cur_idx = heap_idx[1]
        last_f, last_idx = heap_f[size], heap_idx[size]
        size -= 1
        i = 1
        while True:
            l = i << 1
            r = l + 1
            if r <= size:
                child = l if heap_f[l] < heap_f[r] else r
            elif l <= size:
                child = l
            else:
                break
            if last_f <= heap_f[child]:
                break
            heap_f[i], heap_idx[i] = heap_f[child], heap_idx[child]
            i = child
        heap_f[i], heap_idx[i] = last_f, last_idx
        if cur_idx == goal_idx:
            return gscore[cur_idx]
        if visited[cur_idx]:
            continue
        visited[cur_idx] = 1
        cx, cy = divmod(cur_idx, w)
        g_next = gscore[cur_idx] + 1
        for k in range(4):
            nx = cx + dirs[k][0]
            ny = cy + dirs[k][1]
            if 0 <= nx < h and 0 <= ny < w and map_arr[nx, ny] == 1:
                nid = nx * w + ny
                if visited[nid]:
                    continue
                if g_next < gscore[nid]:
                    gscore[nid] = g_next
                    nf = g_next + _manhattan(nx, ny, gx, gy)
                    size += 1
                    j = size
                    while j > 1:
                        p = j >> 1
                        if nf >= heap_f[p]:
                            break
                        heap_f[j], heap_idx[j] = heap_f[p], heap_idx[p]
                        j = p
                    heap_f[j], heap_idx[j] = nf, nid
    return -1


def astar_shortest_paths(task) -> List[int]:
    map_arr: np.ndarray = task.map.map_data
    pairs = task.agent_pairs
    results: List[int] = []
    for (sx, sy), (gx, gy) in pairs:
        d = _astar_len(map_arr, int(sx), int(sy), int(gx), int(gy))
        results.append(int(d))
    return results

