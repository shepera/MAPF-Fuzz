from dataclasses import replace
from re import A
import numpy as np
from mapf_test.maptype import *
import random
from mapf_test.execution.init_pool import generate_tasks_from_all_regions
import logging


def mutate_agent_no_guidance(task: AgentTask) -> AgentTask:
    r = np.random.rand()
    if r < 0.1:
        task = _change_all_agent(task)
    elif r < 0.4:
        task = partial_change_start_end(task)
    elif r < 0.6:
        task = exchange_ends(task)
    elif r < 0.8:
        task = change_starts(task)
    else:
        task = random_move(task)
    return task


def partial_change_start_end(task: AgentTask) -> AgentTask:
    for _ in range(50):
        agent_pairs = [
            (x, y) if random.random() < 0.5 else (y, x) for (x, y) in task.agent_pairs
        ]
        new_t = replace(task, agent_pairs=agent_pairs)
        if new_t.check_task():
            return new_t
    return task


def exchange_ends(task: AgentTask) -> AgentTask:
    new_agent_pairs = []
    i = 0
    while i < 50:
        i += 1
        end = [end for (_, end) in task.agent_pairs]
        random.shuffle(end)
        for start, _ in task.agent_pairs:
            new_agent_pairs.append((start, end.pop()))
        new_task = replace(task, agent_pairs=new_agent_pairs)
        if new_task.check_task():
            return new_task
    return task


def change_starts(task: AgentTask) -> AgentTask:
    new_agent_pairs = []
    i = 0
    while i < 50:
        i += 1
        start = [start for (start, _) in task.agent_pairs]
        random.shuffle(start)
        for _, end in task.agent_pairs:
            new_agent_pairs.append((start.pop(), end))
        new_task = replace(task, agent_pairs=new_agent_pairs)
        if new_task.check_task():
            return new_task
    return task


def _change_all_agent(task: AgentTask) -> AgentTask:
    agent_num = len(task.agent_pairs)
    while True:
        t = generate_tasks_from_all_regions(
            task.map, num_agents_per_task=agent_num, target_total_tasks=1
        )[0]
        if t is not None and t.check_task():
            new_task = replace(task, agent_pairs=t.agent_pairs)
            break
    return new_task


def random_move(task: AgentTask, p_end: float = 0.2, p_start: float = 0.2) -> AgentTask:
    h, w = task.map.map_data.shape

    def _get_random_valid_pos(max_tries: int = 100) -> Optional[tuple[int, int]]:
        for _ in range(max_tries):
            pos = random.randint(0, h - 1), random.randint(0, w - 1)
            if task.map.map_data[pos[0]][pos[1]] == 1:
                return pos
        return None

    for _ in range(50):
        new_agent_pairs = []
        all_agents_mutated_successfully = True
        for start, end in task.agent_pairs:
            new_start, new_end = start, end
            if random.random() < p_end:
                found = False
                for _ in range(30):
                    candidate_end = _get_random_valid_pos()
                    if candidate_end and candidate_end != new_start:
                        new_end = candidate_end
                        found = True
                        break
                if not found:
                    all_agents_mutated_successfully = False
                    break
            if random.random() < p_start:
                found = False
                for _ in range(30):
                    candidate_start = _get_random_valid_pos()
                    if candidate_start and candidate_start != new_end:
                        new_start = candidate_start
                        found = True
                        break
                if not found:
                    all_agents_mutated_successfully = False
                    break
            if not all_agents_mutated_successfully:
                continue
            new_agent_pairs.append((new_start, new_end))
        new_task = replace(task, agent_pairs=new_agent_pairs)
        if new_task.check_task():
            return new_task
    return task
