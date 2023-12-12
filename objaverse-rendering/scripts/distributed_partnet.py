import glob
import json
import multiprocessing
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional
import os

import boto3
import tyro
import wandb
import math
import random

@dataclass
class Args:
    # input_models_path: str
    # """Path to a json file containing a list of 3D object files"""

    upload_to_s3: bool = False
    """Whether to upload the rendered images to S3"""

    log_to_wandb: bool = False
    """Whether to log the progress to wandb"""

    num_gpus: int = 1
    """number of gpus to use. -1 means all available gpus"""

    workers_per_gpu: int = 1
    """number of workers per gpu"""

# ids = ['10383', '10306', '10626', '9992', '12073', '11242', '11586', '9968', '11477', '11429', '11156', '10885', '11395', '11075']

ids = new_ids = ['11581', '11586', '11691', '11778', '11854', '11876', '11945', '10040', '10098', '10101']
data_path = '/home/zubairirshad/SAPIEN/partnet-mobility-dataset'
def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
    s3: Optional[boto3.client],
) -> None:
    while True:
        item = queue.get()
        minimum = 1.5
        maximum = 2.2
        value = random.random()
        camera_dist = minimum + (value * (maximum - minimum))
        if item is None:
            break

        # view_path = os.path.join('.objaverse/hf-objaverse-v1/views_whole_sphere', item.split('/')[-1][:-4])
        path_parts = item.split('/')
        view_path = os.path.join('/home/zubairirshad/zero123/objaverse-rendering/partnet_mobility', path_parts[-4], path_parts[-2])
        if os.path.exists(view_path):
            queue.task_done()
            print('========', item, 'rendered', '========')
            continue
        else:
            os.makedirs(view_path, exist_ok = True)

        # Perform some operation on the item
        print(item, gpu)
        command = (
            # f"export DISPLAY=:0.{gpu} &&"
            # f" GOMP_CPU_AFFINITY='0-47' OMP_NUM_THREADS=48 OMP_SCHEDULE=STATIC OMP_PROC_BIND=CLOSE "
            f" CUDA_VISIBLE_DEVICES={gpu} "
            f" blender -b -P scripts/blender_script.py --"
            f" --object_path {item} --camera_dist {camera_dist} --engine CYCLES --scale 0.8 --num_images 12 --output_dir {view_path}"
        )
        print(command)
        subprocess.run(command, shell=True)

        with count.get_lock():
            count.value += 1

        queue.task_done()


if __name__ == "__main__":
    args = tyro.cli(Args)

    s3 = boto3.client("s3") if args.upload_to_s3 else None
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    if args.log_to_wandb:
        wandb.init(project="objaverse-rendering", entity="prior-ai2")

    # Start worker processes on each of the GPUs
    for gpu_i in range(args.num_gpus):
        for worker_i in range(args.workers_per_gpu):
            worker_i = gpu_i * args.workers_per_gpu + worker_i
            process = multiprocessing.Process(
                target=worker, args=(queue, count, gpu_i, s3)
            )
            process.daemon = True
            process.start()

    # Add items to the queue
    # with open(args.input_models_path, "r") as f:
    #     model_paths = json.load(f)

    # model_keys = list(model_paths.keys())

    # for item in model_keys:
    #     queue.put(os.path.join('.objaverse/hf-objaverse-v1', model_paths[item]))

    joint_angles_train = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    for id in ids:
        for angle in joint_angles_train:
            queue.put(os.path.join(data_path, id, 'textured_objs', str(angle), str(angle)+'.obj'))

    # update the wandb count
    if args.log_to_wandb:
        while True:
            time.sleep(5)
            wandb.log(
                {
                    "count": count.value,
                    "total": len(model_paths),
                    "progress": count.value / len(model_paths),
                }
            )
            if count.value == len(model_paths):
                break

    # Wait for all tasks to be completed
    queue.join()

    # Add sentinels to the queue to stop the worker processes
    for i in range(args.num_gpus * args.workers_per_gpu):
        queue.put(None)
