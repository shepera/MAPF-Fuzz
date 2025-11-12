import pickle
import mapf_test.execution.init_pool as init_pool
from mapf_test.maptype import *
from mapf_test.execution.exec_instance import pass_check
from pprint import pprint
from mapf_test import settings
import mapf_test.execution.init_pool as init_pool
from mapf_test.mutation.mutate import *
from mapf_test.execution.exec_instance import *
import time

start_time = time.time()
algorithm = settings.ALGORITHM
exp_id = settings.EXP_ID
output_filename = settings.PICKLE_FILE
exec_dict = {}
t = Timer(name="main", logger=None)
t.start()
mr_weight = np.ones(len(settings.METAMORPHIC_RELATIONS)).astype(float)
mr_quota = np.ones(len(settings.METAMORPHIC_RELATIONS)) * settings.MR_QUOTA
pool = init_pool.newPool()
print("Initializing pool with", settings.INIT_MAP_NUM, "instances...")
ann_vector = init_pool.add_init_instance_single(
    pool,
    algorithm,
    num_agent=settings.NUM_AGENT,
    size=settings.INIT_MAP_NUM,
    exp_id=exp_id,
    exec_dict=exec_dict,
)
if settings.ENABLE_DIVERSITY:
    if ann_vector is None:
        exit(1)
    annoy = AnnoyFuzzIndex(len(ann_vector[0]))
    annoy.build_from_scratch(ann_vector)

start_time = time.time()
time_limit = settings.TOTAL_RUNNING_TIME
print("Start fuzzing....")
while True:

    task = pool.select_best_task()
    if task is None:
        print("no available tasks")
        break
    new_task = mutate_task(task)
    if new_task is None:
        continue
    res = metamorphic_test(new_task, algorithm, exp_id=exp_id)
    if res is None:
        break
    fitness = new_task.fitness
    diff = new_task.diff
    succ = pass_check(diff)
    print("Task ID:", new_task.task_id, "Diff:", diff, "Success:", succ)
    if succ:
        change_mr_weight(new_task.score_list, diff, mr_weight, mr_quota)
    else:
        pool.add_task(new_task, fitness)
    if settings.ENABLE_DIVERSITY:
        score = annoy.get_novelty_score(new_task.feat)
        annoy.add_item(new_task.feat)
    exec_dict[new_task.task_id] = GraphNode.new_node(
        new_task.task_id,
        parent=exec_dict[task.task_id],
        score=diff,
        novelty_score=score,
        start_time=start_time,
        feat=new_task.feat,
    )
    if time.time() - start_time > time_limit:
        break
t.stop()
pprint(Timer.timers)



if settings.SHOW_JUPYTER:
    with open(output_filename, "wb") as f:
        pickle.dump(list(exec_dict.values()), f)
