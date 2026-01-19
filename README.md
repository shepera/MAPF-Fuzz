# MAPF-Fuzz: Metamorphic Fuzzing for Multi-Agent Path Finding

This is a code repository of the paper "Metamorphic Fuzzing for Multi-Agent Path Finding Algorithms" (ICSE '26). This repository implements **MAPF-Fuzz**, the metamorphic fuzzing framework designed to systematically test the failures and performance gap of MAPF solvers.

## Repository Structure
```
.
├── defects/                       # Discovered failure cases
├── mapf_test/                     # Core testing framework
│   ├── analysis/                  # Diversity scoring & feature extraction
│   ├── execution/                 # Test execution & logging
│   ├── mutation/                  # Map & agent mutation strategies
│   ├── maptype.py                 # Dataclasses defined in this project
│   ├── settings.py                # Global parameters & defaults
│   └── util.py                    # Shared helper functions
├── main.py                        # Single-process entry point
├── main_parallel.py               # Multi-process entry point
├── .gitignore
├── LICENSE
└── README.md                      # This file
```

## Quick Start

1. **Requirements**  
   - Python 3.11

   use the following command to install requirements:
    ```
    pip install -r requirements.txt
    ```

2. **Installation**  
    Clone this repo and tested MAPF algorithms. Currently MAPF-Fuzz supports testing of PIBT, HCA, PIBT+, LaCAM*. More can be added with ajustment to input file and solver paths. 

   ```bash
    git submodule update --init --recursive
   ```

    The MAPF Solvers need to build first. For more details, please refer to the algorithms' repos: [PIBT](https://github.com/Kei18/pibt2) and [LaCAM*](https://github.com/Kei18/lacam3).


3. **Initialization**
    Initial maps are needed. Some map files are already included in ./dataset/. More examples can be found in [MovingAiBenchmark](https://movingai.com/benchmarks/formats.html). Maps will be automatically resized according to the `settings.py`. 

3. **Usage**

    For single process, use
    ``` shell
    python main.py
    ```

    For a multi-process execution, use
    ``` shell
    python main_parallel.py
    ```

4. **Outputs**
    * Failure and suboptimal cases are stored under ./failures.
    * Detailed logs and statistics can be enabled to write into a logs/ directory.

## Module Overview

* mutation/
  
Implements map-level (mutate_map_fn.py) and agent-level (mutate_agent_fn.py, mutate_agent_no_guidance.py) mutation functions. The mutate.py script orchestrates mutation operations.

* analysis/

Extracts compact feature vectors, maintains an ANN index for novelty scoring, and computes energy values to guide seed selection.

* execution/
  
Wraps MAPF solver invocation, handles feasibility checks, logs results, and parses outcomes.

* maptype.py
  
Provides datatypes needed for MAPF-Fuzz.

* util.py
  
Shared utilities (e.g., path validation, distance metrics, argument parsing).

* main.py / main_parallel.py
  
Command-line interfaces for running MAPF-Fuzz in single or multi-process mode. Accepts solver choice, test duration, map size, agent count, etc.

## Configuration

All default parameters are defined in settings.py. Example defaults:
``` python
# settings.py
ALGORITHM='PIBT_PLUS'
METAMORPHIC_RELATIONS=[
    'exchange_start_end',
    'reverse_order',
    'fliplr',
    'flipud',
    'flip',
    'add_probe',
    'add_obstacle']
NUM_WORKERS=32
NUM_AGENT=100   
```

## Visualizer for Defects

The visualizer in the PIBT repository can be used to interactively inspect defects discovered by MAPF-Fuzz. Build the visualizer first (see the instructions in the [PIBT repo](https://github.com/Kei18/pibt2)). 

You’ll need both a map file and a result file:

1.	Put the corresponding map file into PIBT’s map/ directory.

2.	Run the viewer from the PIBT folder:

``` shell
cd build
../visualize.sh result.txt
```

As an alternative, we also recommend mapf-visualizer. Installation and usage details can be found in the [mapf-visualizer repository](https://github.com/Kei18/mapf-visualizer).


## Evaluation Highlights

Some of the scenarios that causes the defects are shown in `/defects`. These defect cases are aligned with the cases in questionnaire in the same order.

* In difficult for lacam-30s, we include a scenario that the first solution is found after 145s in parallel, which is pretty long for a 128x128 map and 100 agents and Non-Maze map. Consider that LaCAM3 in the original work was tested only up to a 30s execution budget. We observe that this behavior can be mitigated by modifying the parallel setting; see [Issue #13](https://github.com/Kei18/lacam3/issues/13) for details.

* For PIBT, HCA and PIBT+, we all includes some cases that either corresponding algorithms cannot solve or exbiting an optimal gap of more than 30% among equivalent metamorphic relations.


## Citation

If you find this work useful for your research, please cite our ICSE 2026 paper:
```
Luxia Lin, Xudong Zhang, Shihao Zhu, and Yan Cai. 2026. Metamorphic Fuzzing for Multi-Agent Path Finding Algorithms. In 2026 IEEE/ACM 48th International Conference on Software Engineering (ICSE ’26), April 12–18, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 13 pages. https://doi.org/10.1145/3744916.3787841
```
