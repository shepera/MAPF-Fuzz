from dataclasses import replace
from pprint import pprint
import numpy as np
import copy
from numba import njit
from mapf_test.maptype import *
import random
from mapf_test.execution.init_pool import generate_tasks_from_all_regions
import logging


def cal_available_area(task: AgentTask) -> np.ndarray:
    available_area = task.map.map_data.copy()
    for (x1, y1), (x2, y2) in task.agent_pairs:
        available_area[x1, y1] = 0
        available_area[x2, y2] = 0
    return available_area


def mutate_agent(task: AgentTask) -> AgentTask:
    if np.random.rand() < 0.05:
        t = change_all_agent(task)
    ori_agent_num = len(task.agent_pairs)
    agent_change_num = 10
    area_change_num = 5
    waiting_long = task.analysis.agent_waiting_long
    run_long = task.analysis.agent_used_long
    num_agents = len(task.agent_pairs)
    prob = np.ones(num_agents, dtype=float)
    for i in waiting_long + run_long:
        prob[i] += 2
    prob = prob / prob.sum()
    area_wait = task.analysis.waiting_map_long
    area_run = task.analysis.used_map_long
    lst = area_run + area_wait
    area_list = np.random.choice(len(lst), size=area_change_num)
    agent_list = np.random.choice(len(task.agent_pairs), size=agent_change_num, p=prob)
    res = task
    for i in agent_list:
        x = np.random.rand()
        if x < 0.2:
            r = pertube_start_end(res, i, None)
        elif x < 0.4:
            r = add_appointment(res, i, None)
        elif x < 0.7:
            r = add_companion(res, i, None)
        else:
            r = add_opposite(res, i, None)
        if r.check_task():
            res = r
    for i in area_list:
        x = np.random.rand()
        if x < 0.5:
            r = add_traverser(res, lst[i], None)
        else:
            r = add_naughty(res, lst[i], None)
        if r.check_task():
            res = r
    x = len(res.agent_pairs) - ori_agent_num
    prob = np.ones(len(task.agent_pairs), dtype=float)
    for i in waiting_long + run_long:
        prob[i] -= 0.04
    prob = prob / prob.sum()
    lst = np.random.choice(ori_agent_num, size=x, p=prob)
    res = clean_agent(res, list(lst))
    return res


def pertube_start_end(task: AgentTask, i: int, available_area: np.ndarray) -> AgentTask:
    new_agent_pairs = list(task.agent_pairs)
    available_area = cal_available_area(task)
    h, w = available_area.shape
    start, end = task.agent_pairs[i]
    res = replace(task, agent_pairs=new_agent_pairs)
    for _ in range(100):
        s = start[0] + np.random.randint(-5, 5), start[1] + np.random.randint(-5, 5)
        e = end[0] + np.random.randint(-5, 5), end[1] + np.random.randint(-5, 5)
        if (
            0 <= s[0] < h
            and 0 <= s[1] < w
            and 0 <= e[0] < h
            and 0 <= e[1] < w
            and s != e
            and available_area[s[0]][s[1]] == 1
            and available_area[e[0]][e[1]] == 1
        ):
            new_agent_pairs[i] = s, e
            if res.check_task():
                return res
            else:
                new_agent_pairs[i] = start, end
    return task


def add_companion(task: AgentTask, i: int, available_area: np.ndarray) -> AgentTask:
    new_agent_pairs = list(task.agent_pairs)
    res = replace(task, agent_pairs=new_agent_pairs)
    available_area = cal_available_area(task)
    h, w = available_area.shape
    start, end = task.agent_pairs[i]
    for _ in range(100):
        s = start[0] + np.random.randint(-5, 5), start[1] + np.random.randint(-5, 5)
        e = end[0] + np.random.randint(-5, 5), end[1] + np.random.randint(-5, 5)
        if (
            0 <= s[0] < h
            and 0 <= s[1] < w
            and 0 <= e[0] < h
            and 0 <= e[1] < w
            and s != e
            and available_area[s[0]][s[1]] == 1
            and available_area[e[0]][e[1]] == 1
        ):
            new_agent_pairs.append((s, e))
            if res.check_task():
                return res
            else:
                new_agent_pairs.pop()
    return task


def add_opposite(task: AgentTask, i: int, available_area: np.ndarray) -> AgentTask:
    new_agent_pairs = list(task.agent_pairs)
    res = replace(task, agent_pairs=new_agent_pairs)
    available_area = cal_available_area(task)
    h, w = available_area.shape
    start, end = task.agent_pairs[i]
    for _ in range(100):
        x = end[0] + np.random.randint(-5, 5), end[1] + np.random.randint(-5, 5)
        y = start[0] + np.random.randint(-5, 5), start[1] + np.random.randint(-5, 5)
        if (
            0 <= x[0] < h
            and 0 <= x[1] < w
            and 0 <= y[0] < h
            and 0 <= y[1] < w
            and available_area[x[0]][x[1]] == 1
            and available_area[y[0]][y[1]] == 1
        ):
            new_agent_pairs.append((x, y))
            if res.check_task():
                return res
            else:
                new_agent_pairs.pop()
    return task


def add_appointment(task: AgentTask, i: int, available_area: np.ndarray) -> AgentTask:
    new_agent_pairs = list(task.agent_pairs)
    res = replace(task, agent_pairs=new_agent_pairs)
    available_area = cal_available_area(task)
    h, w = available_area.shape
    start, end = task.agent_pairs[i]
    for _ in range(100):
        if random.random() < 0.5:
            x = start[0] + np.random.randint(-5, 5), start[1] + np.random.randint(-5, 5)
            y = np.random.randint(0, h), np.random.randint(0, w)
        else:
            x = np.random.randint(0, h), np.random.randint(0, w)
            y = end[0] + np.random.randint(-5, 5), end[1] + np.random.randint(-5, 5)
        if (
            0 <= x[0] < h
            and 0 <= x[1] < w
            and 0 <= y[0] < h
            and 0 <= y[1] < w
            and x != y
            and available_area[x[0]][x[1]] == 1
            and available_area[y[0]][y[1]] == 1
        ):
            new_agent_pairs.append((x, y))
            if res.check_task():
                return res
            else:
                new_agent_pairs.pop()
    return task


def add_naughty(
    task: AgentTask, pos: tuple[int, int], available_area: np.ndarray
) -> AgentTask:
    available_area = cal_available_area(task)
    new_agent_pairs = list(task.agent_pairs)
    res = replace(task, agent_pairs=new_agent_pairs)
    h, w = available_area.shape
    for _ in range(100):
        start = pos[0] + np.random.randint(-5, 5), pos[1] + np.random.randint(-5, 5)
        end = pos[0] + np.random.randint(-5, 5), pos[1] + np.random.randint(-5, 5)
        if (
            0 <= start[0] < h
            and 0 <= start[1] < w
            and 0 <= end[0] < h
            and 0 <= end[1] < w
            and start != end
            and available_area[start[0]][start[1]] == 1
            and available_area[end[0]][end[1]] == 1
        ):
            new_agent_pairs.append((start, end))
            if res.check_task():
                return res
            else:
                new_agent_pairs.pop()
    return task


def add_traverser(
    task: AgentTask, pos: tuple[int, int], available_area: np.ndarray | None = None
) -> AgentTask:
    if available_area is None:
        available_area = cal_available_area(task)
    h, w = available_area.shape
    new_agent_pairs = list(task.agent_pairs)
    res = replace(task, agent_pairs=new_agent_pairs)
    for _ in range(100):
        horizontal = random.random() < 0.5
        if horizontal:
            row = pos[0] + np.random.randint(-5, 5)
            if not 0 <= row < h:
                continue
            left = [(row, c) for c in range(0, pos[1]) if available_area[row][c] == 1]
            right = [
                (row, c) for c in range(pos[1] + 1, w) if available_area[row][c] == 1
            ]
            if not left or not right:
                continue
            start = random.choice(left)
            end = random.choice(right)
        else:
            col = pos[1] + np.random.randint(-5, 5)
            if not 0 <= col < w:
                continue
            up = [(r, col) for r in range(0, pos[0]) if available_area[r][col] == 1]
            down = [
                (r, col) for r in range(pos[0] + 1, h) if available_area[r][col] == 1
            ]
            if not up or not down:
                continue
            start = random.choice(up)
            end = random.choice(down)
        if start == end:
            continue
        new_agent_pairs.append((start, end))
        if res.check_task():
            return res
        new_agent_pairs.pop()
    return task


def clean_agent(task: AgentTask, lst: list[int]) -> AgentTask:
    lst.sort(reverse=True)
    new_agent_pairs = list(task.agent_pairs)
    res = replace(task, agent_pairs=new_agent_pairs)
    for i in lst:
        res.agent_pairs.pop(i)
    return res


def change_all_agent(task: AgentTask) -> AgentTask:
    agent_num = len(task.agent_pairs)
    for _ in range(100):
        t = generate_tasks_from_all_regions(
            task.map, num_agents_per_task=agent_num, target_total_tasks=1
        )[0]
        if t is not None and t.check_task():
            return replace(task, agent_pairs=t.agent_pairs)
    return task
