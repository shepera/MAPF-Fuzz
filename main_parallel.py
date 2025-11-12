import mapf_test.execution.init_pool as init_pool
from mapf_test.maptype import *
from mapf_test.mutation.mutate import *
from mapf_test.execution.exec_instance import *
import time
from mapf_test import settings
import pickle
import os
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
    TimeoutError as FuturesTimeoutError,
)
import logging

log_filename = f"log/fuzz_run_{int(time.time())}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [PID:%(process)d] - %(message)s",
    filename=log_filename,
    filemode="w",
)


def task_processing_worker(
    original_task_from_pool, current_algorithm_id, current_experiment_id
):
    try:
        mutated_task_obj = mutate_task(original_task_from_pool)
        if mutated_task_obj is None:
            return None, "MutatedTaskIsNone"
        execution_result_status = metamorphic_test(
            mutated_task_obj, current_algorithm_id, current_experiment_id
        )
        return mutated_task_obj, execution_result_status
    except Exception as e:
        return None, f"ErrorInWorker: {str(e)}"


if __name__ == "__main__":
    print("Starting the main parallel execution script...")
    print(f"Fuzzing MAPF solver: {settings.ALGORITHM}")
    initial_setup_start_time = time.time()
    algorithm = settings.ALGORITHM
    exp_id = settings.EXP_ID
    exec_dict = {}
    mr_weight = np.ones(len(settings.METAMORPHIC_RELATIONS)).astype(float)
    mr_quota = np.ones(len(settings.METAMORPHIC_RELATIONS)) * settings.MR_QUOTA
    print("Initializing pool with", settings.INIT_MAP_NUM, "instances...")
    pool = init_pool.newPool()
    ann_raw_vector = init_pool.add_init_instance_concurrent(
        pool,
        algorithm,
        num_agent=settings.NUM_AGENT,
        size=settings.INIT_MAP_NUM,
        exp_id=exp_id,
        exec_dict=exec_dict,
    )
    if settings.ENABLE_DIVERSITY:
        if ann_raw_vector is None:
            exit(1)
        annoy = AnnoyFuzzIndex(len(ann_raw_vector[0]))
        annoy.build_from_scratch(ann_raw_vector)
    main_loop_start_time = time.time()
    time_limit = settings.TOTAL_RUNNING_TIME
    num_workers = settings.NUM_WORKERS
    active_futures = {}
    tasks_processed_count = 0
    stop_processing_flag = False
    total_error_num = 0
    try:
        print("Start fuzzing....")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            while not stop_processing_flag:
                current_time = time.time()
                if current_time - main_loop_start_time > time_limit:
                    stop_processing_flag = True
                    break
                while len(active_futures) < num_workers and not stop_processing_flag:
                    selected_task_from_pool = pool.select_best_task()
                    if selected_task_from_pool is None:
                        break
                    tasks_processed_count += 1
                    task_identifier = getattr(
                        selected_task_from_pool, "task_id", f"task_at_{time.time()}"
                    )
                    future = executor.submit(
                        task_processing_worker,
                        selected_task_from_pool,
                        algorithm,
                        exp_id,
                    )
                    active_futures[future] = task_identifier
                if not active_futures and selected_task_from_pool is None:
                    stop_processing_flag = True
                    break
                if not active_futures:
                    time.sleep(0.1)
                    if stop_processing_flag:
                        break
                    continue
                try:
                    for completed_future in as_completed(
                        list(active_futures.keys()), timeout=1.0
                    ):
                        original_task_id_context = active_futures.pop(completed_future)
                        try:
                            result_tuple = completed_future.result(timeout=1.0)
                            if result_tuple is None or len(result_tuple) != 2:
                                continue
                            processed_task_obj, exec_status = result_tuple
                        except FuturesTimeoutError:
                            active_futures[completed_future] = original_task_id_context
                            continue
                        except Exception as e:
                            continue
                        if (
                            exec_status is None
                            or isinstance(exec_status, str)
                            and exec_status.startswith("ErrorInWorker")
                        ):
                            task_id_info = (
                                getattr(processed_task_obj, "task_id", "N/A")
                                if processed_task_obj
                                else "N/A"
                            )
                            continue
                        elif processed_task_obj is None:
                            continue
                        current_fitness = processed_task_obj.fitness
                        current_diff = processed_task_obj.diff
                        current_score_list = processed_task_obj.score_list
                        task_display_id = getattr(
                            processed_task_obj, "task_id", "UnknownID"
                        )
                        succ = pass_check(current_diff)
                        if settings.ENABLE_DIVERSITY:
                            score = annoy.get_novelty_score(processed_task_obj.feat)
                            annoy.add_item(processed_task_obj.feat)
                        if succ:
                            total_error_num += 1
                            change_mr_weight(
                                current_score_list, current_diff, mr_weight, mr_quota
                            )
                        else:
                            current_fitness += cal_fitness(
                                current_score_list, current_diff, mr_weight, mr_quota
                            )
                            if (
                                settings.ENABLE_DIVERSITY
                                and settings.ENABLE_DIVERSITY_PUNISHMENT
                                and score is not None
                            ):
                                if score < annoy.low_novelty_threshold:
                                    current_fitness *= settings.NOVELTY_PUNISHMENT
                                elif score > annoy.high_novelty_threshold:
                                    current_fitness *= settings.NOVELTY_AWARD
                            length_punishment = length_multiplier(
                                current_score_list[0] / settings.NUM_AGENT
                            )
                            pool.add_task(
                                processed_task_obj, current_fitness * length_punishment
                            )
                        if current_diff is not None:
                            exec_dict[task_display_id] = GraphNode.new_node(
                                task_display_id,
                                score_list=current_score_list,
                                parent=exec_dict[task_identifier],
                                score=current_diff,
                                novelty_score=score,
                                feat=processed_task_obj.feat,
                                start_time=main_loop_start_time,
                            )
                        else:
                            exec_dict[task_display_id] = GraphNode.new_node(
                                task_display_id,
                                parent=exec_dict[task_identifier],
                                score=None,
                                score_list=None,
                                novelty_score=score,
                                feat=processed_task_obj.feat,
                                start_time=main_loop_start_time,
                            )
                        if stop_processing_flag:
                            break
                except FuturesTimeoutError:
                    pass
                if stop_processing_flag:
                    break
    except KeyboardInterrupt:
        stop_processing_flag = True
    finally:
        if stop_processing_flag and "executor" in locals() and executor is not None:
            for future_to_cancel in list(active_futures.keys()):
                future_to_cancel.cancel()

    run_path = settings.RUN_PATH
    run_config = {}
    for key in dir(settings):
        if key.isupper():
            run_config[key] = getattr(settings, key)
    os.makedirs(run_path, exist_ok=True)
    config_filepath = os.path.join(run_path, "config.json")
    with open(config_filepath, "w") as f:
        import json

        json.dump(run_config, f, indent=4)

    if settings.SHOW_JUPYTER:
        base_filename = settings.PICKLE_FILE
        output_filepath = os.path.join(run_path, base_filename)
        final_results_list = list(exec_dict.values())
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, "wb") as f:
            pickle.dump(final_results_list, f)
