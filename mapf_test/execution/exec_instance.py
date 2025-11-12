import mapf_test.settings as settings
from datetime import datetime
from functools import reduce
import shutil
import numpy as np
import os
from typing import Any, Dict, List, Optional, Tuple
from mapf_test.analysis.analysis import *
from mapf_test.maptype import MapfResult, SeedPool, MapInfo, AgentTask
import mapf_test.util as util
from scipy.signal import convolve2d
from typing import List, Dict, Callable, Optional
from pathlib import Path
from codetiming import Timer


@Timer(name="Metamorphic", logger=None)
def metamorphic(
    instance: AgentTask, result: MapfResult, number: int = 30
) -> List[AgentTask]:
    enabled_relations = settings.METAMORPHIC_RELATIONS
    relations_registry: Dict[str, Callable[[], AgentTask]] = {
        "exchange_start_end": lambda: exchange_start_end(instance),
        "reverse_order": lambda: reverse_order(instance),
        "fliplr": lambda: fliplr(instance),
        "flipud": lambda: flipud(instance),
        "flip": lambda: flip(instance),
    }
    dependent_relations = {"add_probe", "add_obstacle"}
    needs_prerun = any(name in dependent_relations for name in enabled_relations)
    if needs_prerun and result.solved:
        hot_data = hot_map(instance, result)
        relations_registry["add_probe"] = lambda: add_probe(
            instance, result, number, hot_data
        )
        relations_registry["add_obstacle"] = lambda: add_obstacle(
            instance, result, number, hot_data
        )
    output_tasks = []
    for name in enabled_relations:
        if name in relations_registry:
            transformed_task = relations_registry[name]()
            output_tasks.append(transformed_task)
    return output_tasks


from dataclasses import replace


def exchange_start_end(instance: AgentTask) -> AgentTask:
    new_list = []
    for start, end in instance.agent_pairs:
        new_list.append((end, start))
    res = replace(
        instance, agent_pairs=new_list, task_id=f"{instance.task_id}-exchange-start-end"
    )
    return res


def reverse_order(instance: AgentTask) -> AgentTask:
    reversed_pairs = list(reversed(instance.agent_pairs))
    res = replace(
        instance,
        agent_pairs=reversed_pairs,
        task_id=f"{instance.task_id}-reverse-order",
    )
    return res


def fliplr(instance: AgentTask) -> AgentTask:
    mapinfo = instance.map
    new_map = replace(
        mapinfo, map_data=np.fliplr(mapinfo.map_data), origin=mapinfo.origin + "-fliplr"
    )
    y = len(mapinfo.map_data[0])
    x = len(mapinfo.map_data)
    new_pair = [
        ((x1, y - y1 - 1), (x2, y - 1 - y2))
        for ((x1, y1), (x2, y2)) in instance.agent_pairs
    ]
    res = replace(
        instance,
        map=new_map,
        agent_pairs=new_pair,
        task_id=f"{instance.task_id}-fliplr",
    )
    return res


def flipud(instance: AgentTask) -> AgentTask:
    mapinfo = instance.map
    new_map = replace(
        mapinfo,
        map_data=np.flipud(mapinfo.map_data),
        origin=mapinfo.origin + "-meta-flipud",
    )
    y = len(mapinfo.map_data[0])
    x = len(mapinfo.map_data)
    new_pair = [
        ((x - x1 - 1, y1), (x - x2 - 1, y2))
        for ((x1, y1), (x2, y2)) in instance.agent_pairs
    ]
    res = replace(
        instance,
        map=new_map,
        agent_pairs=new_pair,
        task_id=f"{instance.task_id}-flipud",
    )
    return res


def flip(instance: AgentTask) -> AgentTask:
    return fliplr(flipud(instance))


import numpy as np
from functools import lru_cache
from typing import Sequence
from scipy.signal import convolve2d
import numpy as np
from functools import lru_cache
from scipy.signal import convolve2d


@lru_cache(maxsize=None)
def _gaussian_kernel(radius: int = 5, sigma: float = 2.0) -> np.ndarray:
    size = 2 * radius + 1
    coords = np.arange(size, dtype=np.float32) - radius
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    D = np.sqrt(X**2 + Y**2, dtype=np.float32)
    w = np.exp(-(D**2) / (2 * sigma**2), dtype=np.float32)
    w[radius, radius] = 0.0
    w /= D + 1e-06
    return w


def hot_map(instance: "AgentTask", res: "MapfResult") -> np.ndarray:
    map_data = instance.map.map_data
    h, w = map_data.shape
    freq = np.zeros((h, w), dtype=np.float32)
    mask = map_data == 1
    goals = np.asarray([g for (_, g) in instance.agent_pairs], dtype=np.int16)
    g_row, g_col = goals[:, 0], goals[:, 1]
    for step in res.solution:
        pos = np.asarray(step, dtype=np.int16)
        r_all, c_all = pos[:, 0], pos[:, 1]
        mask[r_all, c_all] = False
        arrived = (r_all == g_row) & (c_all == g_col)
        r_vis, c_vis = r_all[~arrived], c_all[~arrived]
        np.add.at(freq, (r_vis, c_vis), 1.0)
    hot_full = convolve2d(
        freq,
        _gaussian_kernel(radius=5, sigma=2.0),
        mode="same",
        boundary="fill",
        fillvalue=0,
    ).astype(np.float32)
    hot = np.where(mask, hot_full, 0.0)
    if getattr(settings, "DEBUG", False):
        _debug_show(freq, hot)
    return hot


def hot_map0(instance: AgentTask, res: MapfResult) -> np.array:
    freq = np.zeros(instance.map.map_data.shape)
    obstacle = np.where(instance.map.map_data == 0, 0, 1)
    task = instance
    goal = [agent[1] for agent in task.agent_pairs]
    for step in res.solution:
        for i, (start, end) in enumerate(step):
            if not (start == goal[i][0] and end == goal[i][1]):
                freq[start][end] += 1
            obstacle[start][end] = 0
    radius = 5
    sigma = 2.0
    size = 2 * radius + 1
    coords = np.arange(size) - radius
    X, Y = np.meshgrid(coords, coords)
    D = np.sqrt(X**2 + Y**2)
    weights = np.exp(-(D**2) / (2 * sigma**2))
    weights[radius, radius] = 0
    weights = weights / (D + 1e-06)
    hot_full = convolve2d(freq, weights, mode="same", boundary="fill", fillvalue=0)
    hot = np.where(obstacle == 0, 0, hot_full)
    if settings.DEBUG:
        import cv2

        norm = freq.astype(float)
        min_v, max_v = norm.min(), norm.max()
        if max_v > min_v:
            norm = (norm - min_v) / (max_v - min_v) * 255
        disp_freq = norm.astype(np.uint8)
        cv2.imshow("Original Image", disp_freq)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        ax1 = axes[0]
        im_freq = ax1.imshow(freq, interpolation="nearest")
        fig.colorbar(im_freq, ax=ax1, label="Frequency / Special Value")
        ax1.set_title("Frequency Map ('freq')")
        ax2 = axes[1]
        if hot.size > 0:
            im_hot = ax2.imshow(hot, cmap="hot", interpolation="nearest")
            fig.colorbar(im_hot, ax=ax2, label="Heat Value")
            ax2.set_title("Heatmap of 'hot' array")
        else:
            ax2.set_title("'hot' array is empty or invalid")
            ax2.imshow(np.zeros_like(freq, dtype=float), cmap="gray")
        plt.tight_layout()
        plt.show()
    return hot


def sample_hot_points(
    hot_map: np.ndarray, num_points: int = 1, replace: bool = False
) -> List[Tuple[int, int]]:

    flat = hot_map.ravel()
    total = flat.sum()

    if total <= 0:
        return []
    probs = flat / total
    num_available_unique_sources = np.count_nonzero(flat)
    actual_num_to_sample = num_points
    if not replace:
        if num_points > num_available_unique_sources:
            actual_num_to_sample = num_available_unique_sources
    if actual_num_to_sample == 0:
        return []
    sampled_indices_flat = np.random.choice(
        len(flat), size=actual_num_to_sample, replace=replace, p=probs
    )
    rows, cols = np.unravel_index(sampled_indices_flat, hot_map.shape)
    return list(zip(rows, cols))


def add_probe(
    instance: AgentTask, result: MapfResult, num: int, hot_map_data: np.ndarray
) -> AgentTask:
    candidates = sample_hot_points(hot_map_data, num_points=2 * num, replace=False)
    new_pairs = list(instance.agent_pairs)
    probes_added = 0
    h, w = instance.map.map_data.shape
    for p in candidates:
        if probes_added >= num:
            break
        i, j = p
        if hot_map_data[i][j] <= 0:
            continue
        neighbors = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
        valid_dests = []
        for ni, nj in neighbors:
            if (
                0 <= ni < h
                and 0 <= nj < w
                and instance.map.map_data[ni, nj] == 1
                and hot_map_data[ni][nj] > 0
            ):
                valid_dests.append((ni, nj))
        if not valid_dests:
            continue
        dest = valid_dests[np.random.randint(len(valid_dests))]
        new_pairs.append((p, dest))
        hot_map_data[dest[0], dest[1]] = 0
        hot_map_data[i][j] = 0
        probes_added += 1
    res = replace(
        instance, agent_pairs=new_pairs, task_id=f"{instance.task_id}-add-probe"
    )
    return res


def add_obstacle(
    instance: AgentTask, result: MapfResult, num: int, hot_map_data: np.ndarray
) -> AgentTask:
    pt = sample_hot_points(hot_map_data, num_points=num, replace=False)
    new_map = instance.map.map_data.copy()
    for p in pt:
        new_map[p[0]][p[1]] = 0
    new_map_info = replace(
        instance.map, map_data=new_map, origin=instance.map.origin + "-add-obstacle"
    )
    new_task = replace(
        instance, map=new_map_info, task_id=f"{instance.task_id}-add-obstacle"
    )
    return new_task


@Timer(name="algo_exec", logger=None)
def exec_task(task: AgentTask, algo: str, keep_files: bool = False):
    import subprocess
    import tempfile

    algo = algo.upper()
    if algo not in ["PIBT", "HCA", "PIBT_PLUS", "LACAM3"]:
        raise ValueError(f"Unsupported algorithm: {algo}")
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    task_id = task.task_id.replace("/", "_")
    PIBT_MAP_DIR = PROJECT_ROOT / "third_party/pibt2/map"
    PIBT_INST_DIR = PROJECT_ROOT / "third_party/pibt2/instances/mapf"
    PIBT_BUILD_DIR = PROJECT_ROOT / "third_party/pibt2/build"
    LACAM3_MAP_DIR = PROJECT_ROOT / "third_party/lacam3/assets"
    LACAM3_INST_DIR = PROJECT_ROOT / "third_party/lacam3/assets"
    LACAM3_BUILD_DIR = PROJECT_ROOT / "third_party/lacam3"
    map_dir = PIBT_MAP_DIR if algo != "LACAM3" else LACAM3_MAP_DIR
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".map", dir=map_dir
    ) as tmp_map:
        map_path = tmp_map.name
        util.arr2mapfile(task.map.map_data, map_path)
    inst_dir = PIBT_INST_DIR if algo != "LACAM3" else LACAM3_INST_DIR
    suffix = ".txt" if algo in {"PIBT", "HCA", "PIBT_PLUS"} else ".scen"
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, dir=inst_dir
    ) as tmp_inst:
        inst_path = tmp_inst.name
        if algo in {"PIBT", "HCA", "PIBT_PLUS"}:
            util.task2txt(task, inst_path, map_path)
        else:
            util.task2scen(task, inst_path, map_path)
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f"_{algo}_{task_id}.txt"
    ) as tmp:
        result_file = tmp.name
    if algo == "PIBT" or algo == "HCA" or algo == "PIBT_PLUS":
        cmd = f"./mapf -i {inst_path} -s {algo} -o {result_file} -v"
    elif algo == "LACAM3":
        no_thread_flag = "--no-multi-thread" if not settings.LACAM3_MULTI_THREAD else ""
        cmd = f"build/main -t {settings.LACAM_TIME} {no_thread_flag} -i {inst_path} -m {map_path} -o {result_file} -N {len(task.agent_pairs)} -v 3"
    subprocess.run(
        cmd,
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=PIBT_BUILD_DIR if algo != "LACAM3" else LACAM3_BUILD_DIR,
        timeout=settings.TIMEOUT,
    )

    result = util.file2result(result_file)
    result.task = task
    result.solver = algo
    if not keep_files:
        for f in [map_path, inst_path, result_file]:
            try:
                os.remove(f)
            except Exception:
                pass
        return result
    else:
        return result, (map_path, inst_path, result_file)


def exec_task_list(task_list: List[AgentTask], algo: str, keep_files: bool = False):
    results = []
    files = []
    for task in task_list:
        if keep_files:
            result, file_tuple = exec_task(task, algo, keep_files=True)
            if file_tuple is not None:
                files.append(file_tuple)
        else:
            result = exec_task(task, algo)
        results.append(result)
    if keep_files:
        return results, files
    else:
        return results


def eval_result_list(task_list: List[MapfResult]) -> Optional[float]:
    if not reduce(
        lambda x, y: x and y, [res.solved and res.soc >= 0 for res in task_list], True
    ):
        return None
    l = [x.soc for x in task_list]
    big = max(l)
    small = min(l)
    return abs((big - small) / small)


def pass_check(diff: Optional[float]) -> bool:
    if diff is None:
        return True
    return diff > settings.RESULT_THRESHOLD


def change_mr_weight(
    task_list: List[float], diff: Optional[float], weight: np.ndarray, quota: np.ndarray
):
    if diff is None:
        return
    first = task_list[0]
    l = np.array(task_list[1:])
    l = abs((l - first) / first)
    max_index = np.argmax(l)
    if pass_check(diff):
        quota[max_index] -= 1
        if quota[max_index] < 0:
            weight[max_index] *= 0.95


def cal_fitness(
    task_list: List[float], diff, weight: np.ndarray, quota: np.ndarray
) -> Optional[float]:
    if diff is None:
        return None
    first = task_list[0]
    l = np.array(task_list[1:])
    l = abs((l - first) / first)
    return max(l * weight)


def decide_keep_or_delete(
    task_list: List[MapfResult], files: List[Tuple[str, str, str]]
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]], str]:
    keep = [files[0]]
    keep_id = set()
    keep_id.add(0)
    delete = []
    reason = "default"
    if task_list[0].solved == 0:
        return keep, files[1:], "first_unsolved"
    low = task_list[0].soc
    low_id = 0
    high = task_list[0].soc
    high_id = 0
    flag = False
    for i, (res, f) in enumerate(zip(task_list[1:], files[1:])):
        if res.solved == 1:
            if res.soc < low:
                low = res.soc
                low_id = i + 1
            if res.soc > high:
                high = res.soc
                high_id = i + 1
        else:
            keep.append(f)
            keep_id.add(i + 1)
            flag = True
            reason = "got_unsolved"
    diff = (high - low) / low
    if not flag and diff > settings.RESULT_THRESHOLD:
        if low_id != 0:
            keep.append(files[low_id])
            keep_id.add(low_id)
        if high_id != 0:
            keep.append(files[high_id])
            keep_id.add(high_id)
        reason = f"high_low_diff{diff}-{high_id}-{low_id}"
    for i, (res, f) in enumerate(zip(task_list, files)):
        if i not in keep_id:
            delete.append(f)
    return keep, delete, reason


from scipy.special import expit
import numpy as np


def calculate_sigmoid_penalty_scipy(L, P_max, k, L_0):
    return P_max * expit(k * (L_0 - L))


def remove_files(files: List[Tuple[str, str, str]]):
    for file_tuple in files:
        for f in file_tuple:
            os.remove(f)


def metamorphic_test(task: AgentTask, algo: str, exp_id: str) -> list[MapfResult]:
    result0, file0 = exec_task(task, algo, keep_files=True)
    if not result0.solved:
        task.status = "fail"
        remove_files([file0])
        return [result0]
    tasks = metamorphic(task, result=result0)
    results, files = exec_task_list(tasks, algo, keep_files=True)
    results[0].soc = result0.soc
    results = [result0] + results
    files = [file0] + files
    diff = eval_result_list(results)
    print(f"Metamorphic test for task {task.task_id}, diff: {diff}")
    task.diff = diff
    fit, analysis_result = analysis(results[0])
    task.analysis = analysis_result
    task.score_list = [r.soc for r in results] if not diff is None else None
    if diff is not None:
        task.status = "solved"
        task.fitness = 0.2 * fit
    else:
        task.status = "fail"
    if not pass_check(diff):
        remove_files(files)
    else:
        keep, delete, reason = decide_keep_or_delete(results, files)
        remove_files(delete)
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        timestamp = datetime.now().strftime("%d_%H%M%S")
        fail_dir = (
            PROJECT_ROOT
            / f"fail_task/{algo}-{exp_id}-{settings.REPEAT_ID}/{timestamp}-{task.task_id}-{reason}"
        )
        os.makedirs(fail_dir, exist_ok=True)
        for file_tuple in keep:
            for f in file_tuple:
                try:
                    shutil.move(f, fail_dir / os.path.basename(f))
                except Exception:
                    pass
    return results


def length_multiplier(L, k=settings.LENGTH_K, L_0=settings.MAP_SIZE * 0.8):
    return expit(k * (L - L_0)) * 2
