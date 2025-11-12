import numpy as np
import mapf_test.settings as settings
from mapf_test.util import astar_shortest_paths
from mapf_test.maptype import MapfResult, SeedPool, MapInfo, AgentTask, AnalysisResult
from typing import Any, Dict, List, Optional, Tuple
from numba import njit
from typing import List, Tuple, Optional
from sklearn.cluster import DBSCAN
from scipy.ndimage import distance_transform_cdt
from scipy.signal import convolve2d
from codetiming import Timer


@njit
def _calculate_core_stats_numba(
    solution_arr: np.ndarray, agent_pairs_arr: np.ndarray, map_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    makespan, num_agents, _ = solution_arr.shape
    waiting_map_n = np.zeros(map_shape, dtype=np.int32)
    used_time_n = np.zeros(map_shape, dtype=np.int32)
    agent_waiting_time_n = np.zeros(num_agents, dtype=np.int32)
    agent_total_time_n = np.zeros(num_agents, dtype=np.int32)
    for t_loop_idx in range(makespan - 1):
        for i in range(num_agents):
            current_x, current_y = (
                solution_arr[t_loop_idx + 1, i, 0],
                solution_arr[t_loop_idx + 1, i, 1],
            )
            last_x, last_y = (
                solution_arr[t_loop_idx, i, 0],
                solution_arr[t_loop_idx, i, 1],
            )
            goal_x, goal_y = agent_pairs_arr[i, 1, 0], agent_pairs_arr[i, 1, 1]
            if goal_x != current_x and goal_y != current_y:
                if current_x == last_x and current_y == last_y:
                    waiting_map_n[current_x, current_y] += 1
                    used_time_n[current_x, current_y] += 1
                    agent_waiting_time_n[i] += 1
                else:
                    used_time_n[current_x, current_y] += 1
                    agent_total_time_n[i] += 1
    return waiting_map_n, used_time_n, agent_waiting_time_n, agent_total_time_n


@Timer(name="analysis", logger=None)
def analysis(res: MapfResult) -> tuple[float, Optional[AnalysisResult]]:
    if not res.solved:
        return -1.0, None
    task: AgentTask = res.task
    py_solution: List[List[Tuple[int, int]]] = res.solution
    map_info: MapInfo = task.map
    py_agent_pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = task.agent_pairs
    makespan = len(py_solution)
    if makespan <= 1:
        default_analysis_result = AnalysisResult(
            freq_map=np.zeros(map_info.map_data.shape, dtype=np.int32),
            agent_waiting_long=[],
            agent_used_long=[],
            waiting_map_long=[],
            used_map_long=[],
        )
        return -1.0 if makespan == 0 else 0.0, default_analysis_result
    num_agents = len(py_agent_pairs)
    if num_agents == 0:
        default_analysis_result = AnalysisResult(
            freq_map=np.zeros(map_info.map_data.shape, dtype=np.int32),
            agent_waiting_long=[],
            agent_used_long=[],
            waiting_map_long=[],
            used_map_long=[],
        )
        return 0.0, default_analysis_result
    solution_np = np.array(py_solution, dtype=np.int32)
    agent_pairs_np = np.array(py_agent_pairs, dtype=np.int32)
    agent_shortest_path_lengths = np.array(astar_shortest_paths(task), dtype=np.int32)
    waiting_map, used_time, agent_waiting_time, agent_total_time = (
        _calculate_core_stats_numba(
            solution_np, agent_pairs_np, map_info.map_data.shape
        )
    )
    agent_waiting_long = []
    agent_long_path = []
    avg_waiting_time = (
        np.mean(agent_waiting_time) if agent_waiting_time.size > 0 else 0.0
    )
    if avg_waiting_time > 1e-06:
        agent_waiting_long_indices = np.where(
            agent_waiting_time > 2 * avg_waiting_time
        )[0]
        agent_waiting_long = agent_waiting_long_indices.tolist()
    valid_shortest_paths = agent_shortest_path_lengths > 0
    condition_long_detour = (
        agent_total_time[valid_shortest_paths]
        > 2 * agent_shortest_path_lengths[valid_shortest_paths]
    )
    indices_detour = np.where(valid_shortest_paths)[0][condition_long_detour]
    indices_moved_from_goal = np.where(
        (agent_shortest_path_lengths == 0) & (agent_total_time > 0)
    )[0]
    agent_long_path_indices = np.union1d(indices_detour, indices_moved_from_goal)
    agent_long_path = agent_long_path_indices.tolist()
    used_time_long = []
    waiting_map_long = []
    map_size_total = map_info.map_data.sum().sum()
    if map_size_total > 0:
        avg_used_time = np.mean(used_time) if used_time.size > 0 else 0.0
        if avg_used_time > 1e-06:
            used_time_long_coords = np.argwhere(used_time > 2 * avg_used_time)
            used_time_long = [tuple(coord) for coord in used_time_long_coords]
        avg_waiting_map = np.mean(waiting_map) if waiting_map.size > 0 else 0.0
        if avg_waiting_map > 1e-06:
            waiting_map_long_coords = np.argwhere(waiting_map > 2 * avg_waiting_map)
            waiting_map_long = [tuple(coord) for coord in waiting_map_long_coords]
    num_agents_norm = float(num_agents) if num_agents > 0 else 1.0
    map_size_norm = float(map_size_total) if map_size_total > 0 else 1.0
    agent_waiting_ratio = len(agent_waiting_long) / num_agents_norm
    agent_long_ratio = len(agent_long_path) / num_agents_norm
    map_used_ratio = len(used_time_long) / map_size_norm
    map_wait_ratio = len(waiting_map_long) / map_size_norm
    weight = np.array([0.4, 0.4, 0.2, 0.2]) / 1.2
    sub_score = np.array(
        [agent_waiting_ratio, agent_long_ratio, map_used_ratio, map_wait_ratio]
    )
    score = np.dot(weight, sub_score) / weight.sum()
    if settings.ENABLE_DIVERSITY:
        task.feat = calculate_feat(
            res,
            agent_total_time,
            agent_shortest_path_lengths=agent_shortest_path_lengths,
            agent_waiting_time=agent_waiting_time,
        )
    analysis_data = AnalysisResult(
        freq_map=waiting_map,
        agent_waiting_long=agent_waiting_long,
        agent_used_long=agent_long_path,
        waiting_map_long=waiting_map_long,
        used_map_long=used_time_long,
    )
    return score, analysis_data


def calculate_feat(
    res, agent_total_time, agent_shortest_path_lengths, agent_waiting_time
) -> Dict[str, Any]:
    map_feat = cal_map_feat(res.task.map)
    agent_feat = cal_agent_feat(res.task, agent_shortest_path_lengths)
    dynamic_feat = cal_dynamic_feat(
        agent_total_time, agent_shortest_path_lengths, agent_waiting_time
    )
    values_list = (
        list(map_feat.values())
        + list(agent_feat.values())
        + list(dynamic_feat.values())
    )
    return np.array(values_list, dtype=np.float32)


def calculate_avg_distance_to_obstacle(map_data):
    distance_map = distance_transform_cdt(map_data, metric="cityblock")
    free_space_distances = distance_map[map_data == 1]
    if free_space_distances.size == 0:
        return 0.0
    avg_distance = np.mean(free_space_distances)
    return avg_distance


def find_dead_ends(map_data: np.ndarray) -> int:
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    walkable_neighbor_count = convolve2d(
        map_data, kernel, mode="same", boundary="fill", fillvalue=0
    )
    is_walkable = map_data == 1
    is_one_walkable_neighbor = walkable_neighbor_count == 1
    dead_ends_mask = np.logical_and(is_walkable, is_one_walkable_neighbor)
    return np.sum(dead_ends_mask)


def cal_map_feat(map_info: MapInfo) -> Dict[str, Any]:
    map_data = map_info.map_data
    density = np.sum(map_data) / map_data.size
    avg_gap = calculate_avg_distance_to_obstacle(map_data)
    dead_ends = find_dead_ends(map_data)
    return {"density": density, "avg_gap": avg_gap, "dead_ends": dead_ends}


def cal_agent_feat(task: AgentTask, shortest_paths: np.ndarray) -> Dict[str, Any]:
    pairs = task.agent_pairs
    start_feat = cal_dbscan([p[0] for p in pairs])
    goal_feat = cal_dbscan([p[1] for p in pairs])
    return {
        "avg_distance": np.mean(shortest_paths),
        "std_distance": np.std(shortest_paths),
        "start_cluster_num": start_feat[0],
        "start_noise_ratio": start_feat[1],
        "start_avg_cluster_size": start_feat[2],
        "goal_cluster_num": goal_feat[0],
        "goal_noise_ratio": goal_feat[1],
        "goal_avg_cluster_size": goal_feat[2],
    }


import numpy as np
from typing import Dict, Any


def cal_dynamic_feat(
    agent_total_time: np.ndarray,
    agent_shortest_path_lengths: np.ndarray,
    agent_waiting_time: np.ndarray,
) -> Dict[str, Any]:
    sum_shortest_paths = np.sum(agent_shortest_path_lengths)
    if sum_shortest_paths == 0:
        soc_soic = 1.0
    else:
        soc_soic = np.sum(agent_total_time) / sum_shortest_paths
    non_zero_mask = agent_shortest_path_lengths != 0
    detours = np.ones_like(agent_total_time, dtype=float)
    np.divide(
        agent_total_time, agent_shortest_path_lengths, out=detours, where=non_zero_mask
    )
    valid_detours = detours[non_zero_mask]
    avg_detour = np.mean(valid_detours) if valid_detours.size > 0 else 1.0
    std_detour = np.std(valid_detours) if valid_detours.size > 0 else 0.0
    avg_waiting_time = (
        np.mean(agent_waiting_time) if agent_waiting_time.size > 0 else 0.0
    )
    std_waiting_time = (
        np.std(agent_waiting_time) if agent_waiting_time.size > 0 else 0.0
    )
    return {
        "soc/soic": soc_soic,
        "avg_detour": avg_detour,
        "std_detour": std_detour,
        "avg_waiting_time": avg_waiting_time,
        "std_waiting_time": std_waiting_time,
    }


def cal_dbscan(
    dot_list: np.ndarray, eps: int = 25, min_samples: int = 3
) -> tuple[float, float, float]:
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples, metric="manhattan")
    labels = dbscan_model.fit_predict(dot_list)
    unique_labels = set(labels)
    n_clusters_ = len(unique_labels) - (1 if -1 in labels else 0)
    noise_ratio = np.sum(labels == -1) / len(labels) if len(labels) > 0 else 0.0
    cluster_sizes = [np.sum(labels == label) for label in unique_labels if label != -1]
    avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0.0
    return float(n_clusters_), noise_ratio, avg_cluster_size
