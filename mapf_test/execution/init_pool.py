import numpy as np
import os
from typing import Any, Dict, List
from mapf_test.maptype import SeedPool, MapInfo, AgentTask, GraphNode
import mapf_test.util as util
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import mapf_test.settings as settings


def newPool(max_size=1000):
    return SeedPool(max_size)


def get_street_map_list() -> list[MapInfo]:
    dir = "./dataset/"
    map_files = glob.glob(os.path.join(dir, "*.map"))
    maps = []
    for map_file in map_files:
        arr = util.mapfile2arr(map_file)
        maps.append(MapInfo(seed_id="", map_data=arr, origin="city"))
    return maps


import random
from scipy.ndimage import label
import numpy as np
import random
from scipy.ndimage import label
from typing import List


def generate_tasks_from_all_regions(
    map_info: MapInfo,
    num_agents_per_task: int = 3,
    target_total_tasks: int = 3,
    seed=None,
) -> List[AgentTask]:
    if not seed is None:
        random.seed(seed)
    if num_agents_per_task <= 0:
        return []
    if target_total_tasks <= 0:
        return []
    labeled_map, num_features = label(map_info.map_data)
    if num_features == 0:
        return []
    sid = getattr(map_info, "seed_id", "default_sid")
    eligible_regions_data = []
    min_coords_needed = 2 * num_agents_per_task
    for region_idx in range(1, num_features + 1):
        region_mask = labeled_map == region_idx
        size = np.sum(region_mask)
        if size >= 20 and size >= min_coords_needed:
            coords = list(zip(*np.where(region_mask)))
            eligible_regions_data.append(
                {"id": region_idx, "size": size, "coords": coords}
            )
    if not eligible_regions_data:
        return []
    total_eligible_size = sum(r["size"] for r in eligible_regions_data)
    if total_eligible_size == 0:
        return []
    for region in eligible_regions_data:
        region["ideal_tasks"] = (
            region["size"] / total_eligible_size * target_total_tasks
        )
        region["assigned_tasks"] = 0
    for region in eligible_regions_data:
        region["assigned_tasks"] = int(region["ideal_tasks"])
    current_assigned_sum = sum(r["assigned_tasks"] for r in eligible_regions_data)
    remaining_tasks = target_total_tasks - current_assigned_sum
    if remaining_tasks > 0:
        eligible_regions_data.sort(
            key=lambda r: r["ideal_tasks"] - r["assigned_tasks"], reverse=True
        )
        for i in range(min(remaining_tasks, len(eligible_regions_data))):
            eligible_regions_data[i]["assigned_tasks"] += 1
    all_generated_tasks = []
    task_count = 0
    for region_data in eligible_regions_data:
        if task_count >= target_total_tasks:
            break
        region_id = region_data["id"]
        num_to_generate_here = region_data["assigned_tasks"]
        region_coords = region_data["coords"]
        for _ in range(num_to_generate_here):
            if task_count >= target_total_tasks:
                break
            if len(region_coords) < min_coords_needed:
                break
            selected_coords = random.sample(region_coords, min_coords_needed)
            starts = selected_coords[:num_agents_per_task]
            goals = selected_coords[num_agents_per_task:]
            import uuid

            task_id_num = uuid.uuid4().int % 1000000
            task_id_str = f"{sid}-task-{region_id}-{task_id_num}"
            agent_task_instance = AgentTask(
                map=map_info,
                agent_pairs=list(zip(starts, goals)),
                task_id=task_id_str,
                fitness=None,
            )
            all_generated_tasks.append(agent_task_instance)
            task_count += 1
    return all_generated_tasks


from mapf_test.execution.exec_instance import metamorphic_test
from mapf_test.analysis.analysis import analysis


def add_init_instance_single(
    pool: SeedPool,
    algo: str,
    num_agent=30,
    size=100,
    exp_id=0,
    exec_dict: Dict[str, Any] = None,
):
    ann_raw_vector = []
    i = 0
    arr = get_street_map_list()
    for map in arr:
        if i >= size:
            break
        tasks = generate_tasks_from_all_regions(
            map, num_agents_per_task=num_agent, seed=42
        )
        for task in tasks:
            res_list = metamorphic_test(task, algo=algo, exp_id=exp_id)
            if task.status == "fail":
                print(f"Task {task.task_id} is not solvable, skipping.")
                continue
            pool.add_task(task, 2 * task.fitness)
            if exec_dict is not None:
                exec_dict[task.task_id] = GraphNode.new_node(
                    task_id=task.task_id,
                    parent=None,
                    score=task.fitness,
                    score_list=task.score_list,
                    feat=task.feat,
                )
            i += 1
            if settings.ENABLE_DIVERSITY:
                feature_vector = task.feat
                if feature_vector is not None:
                    ann_raw_vector.append(feature_vector)
    if settings.ENABLE_DIVERSITY:
        return ann_raw_vector
    else:
        return None


def worker_process_initial_task(
    task: "AgentTask", algo: str, exp_id: int
) -> "AgentTask":
    metamorphic_test(task, algo=algo, exp_id=exp_id)
    return task


def add_init_instance_concurrent(
    pool: "SeedPool",
    algo: str,
    num_agent: int = 30,
    size: int = 100,
    exp_id: int = 0,
    exec_dict: Dict[str, Any] = None,
):
    ann_raw_vector = []
    start_time_init = time.time()
    all_tasks_to_process: List["AgentTask"] = []
    map_list = get_street_map_list()
    tasks_generated_count = 0
    for map_info in map_list:
        if tasks_generated_count >= size:
            break
        generated_tasks = generate_tasks_from_all_regions(
            map_info, num_agents_per_task=num_agent, seed=42
        )
        for task in generated_tasks:
            if tasks_generated_count >= size:
                break
            all_tasks_to_process.append(task)
            tasks_generated_count += 1
    if not all_tasks_to_process:
        return
    num_workers = settings.NUM_WORKERS
    processed_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {
            executor.submit(worker_process_initial_task, task, algo, exp_id): task
            for task in all_tasks_to_process
        }
        for future in as_completed(future_to_task):
            original_task_info = future_to_task[future]
            try:
                processed_task = future.result()
                processed_count += 1
                if processed_task.status == "fail":
                    continue
                pool.add_task(processed_task, 2 * processed_task.fitness)
                if exec_dict is not None:
                    exec_dict[processed_task.task_id] = GraphNode.new_node(
                        task_id=processed_task.task_id,
                        parent=None,
                        score=processed_task.fitness,
                        score_list=processed_task.score_list,
                        feat=processed_task.feat,
                    )
                if settings.ENABLE_DIVERSITY:
                    feature_vector = processed_task.feat
                    if feature_vector is not None:
                        ann_raw_vector.append(feature_vector)
            except Exception as exc:
                print("exception in task processing:", exc)
    return ann_raw_vector if settings.ENABLE_DIVERSITY else None
