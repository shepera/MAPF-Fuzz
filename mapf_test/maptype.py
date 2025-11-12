from __future__ import annotations
import copy
from dataclasses import replace
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
from PIL import Image
import numpy as np
import mapf_test.settings as settings
from scipy.ndimage import label
import time
from codetiming import Timer


@dataclass
class MapInfo:
    seed_id: str
    map_data: np.ndarray
    origin: str = "random"

    def show_map_pil(self, new_width: int, new_height: int, scale_factor: float):
        array = self.map_data
        img_array = (array * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode="L")
        original_width, original_height = img.size
        if new_width is not None and new_height is not None:
            if new_width > 0 and new_height > 0:
                target_size = new_width, new_height
            else:
                target_size = original_width, original_height
        elif scale_factor is not None:
            if scale_factor > 0:
                target_width = int(original_width * scale_factor)
                target_height = int(original_height * scale_factor)
                if target_width > 0 and target_height > 0:
                    target_size = target_width, target_height
                else:
                    target_size = original_width, original_height
            else:
                target_size = original_width, original_height
        else:
            target_size = original_width, original_height
        if target_size != (original_width, original_height):
            try:
                resample_filter = Image.Resampling.NEAREST
            except AttributeError:
                resample_filter = Image.NEAREST
            img_resized = img.resize(target_size, resample_filter)
            img_resized.show()
        else:
            img.show()


@dataclass
class AnalysisResult:
    freq_map: np.ndarray
    agent_waiting_long: List[int]
    agent_used_long: List[int]
    waiting_map_long: List[tuple[int, int]]
    used_map_long: List[tuple[int, int]]


@dataclass
class AgentTask:
    map: MapInfo
    agent_pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    task_id: str
    fitness: Optional[float] = None
    score_list: Optional[List[float]] = None
    status: str = "untested"
    diff: Optional[float] = None
    analysis: Optional[AnalysisResult] = None
    feat: Optional[np.ndarray] = None

    def check_task(self, debug: bool = False) -> bool:
        map_data = self.map.map_data
        H, W = map_data.shape
        free = map_data == 1
        labels, _ = label(free)
        starts_used: Set[Tuple[int, int]] = set()
        goals_used: Set[Tuple[int, int]] = set()
        for start, goal in self.agent_pairs:
            sy, sx = start
            gy, gx = goal
            if not (0 <= sy < H and 0 <= sx < W and 0 <= gy < H and 0 <= gx < W):
                return False
            if not (free[sy, sx] and free[gy, gx]):
                return False
            if labels[sy, sx] != labels[gy, gx]:
                return False
            if start in starts_used or goal in goals_used:
                return False
            starts_used.add(start)
            goals_used.add(goal)
        return True


@dataclass
class MapfResult:
    task: AgentTask
    solver: str
    solved: bool
    soc: int
    makespan: int
    comp_time: int
    solution: list[list[Tuple[int, int]]]


import heapq
import uuid


class SeedPool:
    def __init__(self, max_size):
        self.max_size = max_size
        (self.task_heap): List[Tuple[float, str, AgentTask]] = []

    def add_task(self, t: AgentTask, fitness: Optional[float] = None):
        if fitness is not None:
            heapq.heappush(self.task_heap, (-fitness, t.task_id, t))

    def select_best_task(self) -> Optional[AgentTask]:
        if self.task_heap:
            fitness, task_id, t = heapq.heappop(self.task_heap)
            fitness = abs(fitness) * settings.FITNESS_DECAY_FACTOR
            heapq.heappush(self.task_heap, (-fitness, task_id, t))
            return t
        return None

    def prune_heap(self, keep_k: int):
        heap_list = list(self.task_heap)
        heap_list.sort()
        pruned = heap_list[:keep_k]
        heapq.heapify(pruned)
        self.task_heap = pruned


import itertools

_ID_COUNTER = itertools.count()


@dataclass
class GraphNode:
    task_id: str
    generation: int
    score: float
    time: float = 0.0
    score_list: Optional[List[float]] = None
    parent: Optional["GraphNode"] = None
    novelty_score: Optional[float] = None
    feat: Optional[np.ndarray] = None
    id: int = field(init=False, default_factory=lambda: next(_ID_COUNTER))

    @staticmethod
    def new_node(
        task_id: str,
        score: float,
        score_list: Optional[List[float]] = None,
        parent: Optional["GraphNode"] = None,
        novelty_score: Optional[float] = None,
        feat: Optional[np.ndarray] = None,
        start_time: float = 0.0,
    ) -> "GraphNode":
        gen = parent.generation + 1 if parent else 0
        return GraphNode(
            task_id=task_id,
            generation=gen,
            score=score,
            score_list=score_list,
            parent=parent,
            novelty_score=novelty_score,
            feat=feat,
            time=time.time() - start_time if parent else 0.0,
        )


import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from annoy import AnnoyIndex


class AnnoyFuzzIndex:
    @Timer(name="ann", logger=None)
    def __init__(
        self,
        feature_dimensions: int,
        metric: str = "euclidean",
        rebuild_ratio: float = 0.1,
        min_rebuild_threshold: int = 200,
        max_rebuild_threshold: int = 1000,
        base_path: str = "fuzz_index",
    ):
        self.dims = feature_dimensions
        self.metric = metric
        self.rebuild_ratio = rebuild_ratio
        self.min_rebuild_threshold = min_rebuild_threshold
        self.max_rebuild_threshold = max_rebuild_threshold
        self.n_trees = 50
        self.low_novelty_threshold = 0.5
        self.high_novelty_threshold = 2.0
        self.index_path = f"{base_path}.ann"
        self.scaler_path = f"{base_path}_scaler.pkl"
        self.vectors_path = f"{base_path}_vectors.pkl"
        self.scaler = StandardScaler()
        self.annoy_index = AnnoyIndex(self.dims, self.metric)
        self.raw_vectors = []
        self.new_vectors_buffer = []
        self.is_ready = False

    @Timer(name="ann", logger=None)
    def _update_novelty_threshold(self):
        if len(self) < 20:
            return
        distances = []
        for i in range(len(self)):
            dists = self.annoy_index.get_nns_by_item(i, 2, include_distances=True)[1]
            if len(dists) > 1:
                distances.append(dists[1])
        if not distances:
            return
        self.low_novelty_threshold = np.percentile(distances, 25)
        self.high_novelty_threshold = np.percentile(distances, 75)

    @Timer(name="ann", logger=None)
    def _rebuild(self):
        if len(self.raw_vectors) == 0:
            return
        vectors_np = np.array(self.raw_vectors)
        self.scaler.fit(vectors_np)
        normalized_vectors = self.scaler.transform(vectors_np)
        self.annoy_index = AnnoyIndex(self.dims, self.metric)
        for i, vec in enumerate(normalized_vectors):
            self.annoy_index.add_item(i, vec)
        self.annoy_index.build(self.n_trees)
        self._update_novelty_threshold()
        self.new_vectors_buffer = []
        self.is_ready = True

    @Timer(name="ann", logger=None)
    def add_item(self, raw_vector: np.ndarray) -> bool:
        self.raw_vectors.append(raw_vector)
        self.new_vectors_buffer.append(raw_vector)
        current_threshold = max(
            self.min_rebuild_threshold, int(len(self.raw_vectors) * self.rebuild_ratio)
        )
        current_threshold = min(current_threshold, self.max_rebuild_threshold)
        if len(self.new_vectors_buffer) >= current_threshold:
            self._rebuild()
            return True
        return False

    @Timer(name="ann", logger=None)
    def build_from_scratch(self, initial_raw_vectors: list, n_trees: int = 50):
        self.raw_vectors = initial_raw_vectors
        self.n_trees = n_trees
        self._rebuild()

    def save(self):
        if not self.is_ready:
            raise RuntimeError("")
        self.annoy_index.save(self.index_path)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(self.vectors_path, "wb") as f:
            pickle.dump(self.raw_vectors, f)

    @Timer(name="ann", logger=None)
    def get_novelty_score(self, raw_vector: np.ndarray) -> float:
        if not self.is_ready or len(self) == 0:
            return float("inf")
        normalized_vector = self.scaler.transform(raw_vector.reshape(1, -1))[0]
        results = self.annoy_index.get_nns_by_vector(
            normalized_vector, 1, include_distances=True
        )
        return results[1][0] if results and results[1] else float("inf")

    def __len__(self):
        return len(self.raw_vectors)
