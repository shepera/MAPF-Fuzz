import numpy as np
import random
import noise
from numba import jit

_MOORE_OFFSETS = np.array(
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
    dtype=np.int8,
)
rules = {
    "Conway": {"birth": [3], "survival": [2, 3]},
    "HighLife": {"birth": [3, 6], "survival": [2, 3]},
    "Day & Night": {"birth": [3, 6, 7, 8], "survival": [3, 4, 6, 7, 8]},
    "Maze": {"birth": [3], "survival": [1, 2, 3, 4, 5]},
    "Mazectric": {"birth": [3], "survival": [1, 2, 3, 4]},
    "WalledCities": {"birth": [4, 5, 6, 7, 8], "survival": [2, 3, 4, 5]},
    "Seed": {"birth": [2], "survival": [-1]},
    "Diamoeba": {"birth": [3, 5, 6, 7, 8], "survival": [5, 6, 7, 8]},
    "Cave": {"birth": [5, 6, 7, 8, 9], "survival": [5, 6, 7, 8, 9]},
}


@jit(nopython=True, cache=True)
def _ca_core_v2(
    grid_initial: np.ndarray, iterations: int, birth_rules: list, survival_rules: list
) -> np.ndarray:
    height, width = grid_initial.shape
    grid = grid_initial.copy()
    for _ in range(iterations):
        new_grid = grid.copy()
        for r in range(0, height):
            for c in range(0, width):
                free_neighbors = 0
                for dr, dc in _MOORE_OFFSETS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        free_neighbors += grid[nr, nc]
                cell_state = grid[r, c]
                if cell_state == 1:
                    if free_neighbors in birth_rules:
                        new_grid[r, c] = 0
                elif not free_neighbors in survival_rules:
                    new_grid[r, c] = 1
        grid = new_grid
    return grid


def _cellular_automata_patch_inner(
    height: int,
    width: int,
    obs_density: float = 0.3,
    iterations: int = 4,
    noise_type="norm",
    rules: dict = None,
    border: np.ndarray = None,
) -> np.ndarray:
    if noise_type == "Perlin":
        rng = random.Random()
        scale = height / float(rng.randint(1, 10))
        octaves = 4
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
        initial_grid = (grid_values >= threshold_val).astype(np.uint8)
    elif noise_type == "same":
        initial_grid = (
            border.copy()
            if border is not None
            else np.ones((height, width), dtype=np.uint8)
        )
    else:
        initial_grid = (np.random.rand(height, width) > obs_density).astype(np.uint8)
    if border is not None:
        initial_grid[0, :] = border[0, :]
        initial_grid[-1, :] = border[-1, :]
        initial_grid[:, 0] = border[:, 0]
        initial_grid[:, -1] = border[:, -1]
    birth_list = rules.get("birth", [])
    survival_list = rules.get("survival", [])
    return _ca_core_v2(initial_grid, iterations, birth_list, survival_list)


CA_PRESET_CONFIGS = [
    (4, "Perlin", 0.3, "WalledCities"),
    (4, "normal", 0.1, "WalledCities"),
    (2, "normal", 0.6, "Conway"),
    (1, "Perlin", 0.6, "Conway"),
    (1, "Perlin", 0.4, "HighLife"),
    (4, "Perlin", 0.4, "HighLife"),
    (1, "normal", 0.05, "HighLife"),
    (2, "normal", 0.05, "HighLife"),
    (3, "normal", 0.05, "HighLife"),
    (4, "normal", 0.05, "HighLife"),
    (2, "normal", 0.2, "Day & Night"),
    (4, "normal", 0.2, "Day & Night"),
    (2, "Perlin", 0.3, "Day & Night"),
    (4, "Perlin", 0.3, "Day & Night"),
    (2, "normal", 0.35, "Maze"),
    (4, "normal", 0.35, "Maze"),
    (2, "Perlin", 0.35, "Maze"),
    (4, "Perlin", 0.35, "Maze"),
    (1, "normal", 0.4, "Mazectric"),
    (2, "normal", 0.4, "Mazectric"),
    (2, "normal", 0.15, "WalledCities"),
    (2, "normal", 0.4, "Diamoeba"),
    (2, "normal", 0.3, "Diamoeba"),
    (2, "normal", 0.4, "Cave"),
    (4, "normal", 0.4, "Cave"),
    (2, "normal", 0.3, "Cave"),
]
CA_PROB = np.ones(len(CA_PRESET_CONFIGS), dtype=float)


def cellular_automata_patch(
    height: int, width: int, border: np.ndarray = None
) -> np.ndarray:
    probs = CA_PROB.copy()
    probs /= probs.sum()
    rng = np.random.default_rng()
    strategy_index = rng.choice(len(CA_PRESET_CONFIGS), p=probs)
    strategy = CA_PRESET_CONFIGS[strategy_index]
    inner = _cellular_automata_patch_inner(
        height,
        width,
        obs_density=strategy[2],
        iterations=strategy[0],
        noise_type=strategy[1],
        rules=rules[strategy[3]],
        border=border,
    )
    if border is not None:
        for r in range(height):
            for c in range(width):
                if r == 0 or r == height - 1 or c == 0 or c == width - 1:
                    inner[r, c] = border[r, c]
    return inner
