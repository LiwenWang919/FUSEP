import os
import time
import subprocess
import queue
import threading

# 任务队列（替换为你的命令列表）
TASKS = [
    "python main.py --config-file configs/train_config_det_epv.py",
    "python train.py --config=config2.yaml",
    "python train.py --config=config3.yaml",
]

# 指定显卡编号，例如 [0, 1, 2, 3]
AVAILABLE_GPUS = [1, 2, 3]

# 最小显存空闲阈值（单位：MB），只有显存空闲大于该值时才运行任务
MEMORY_THRESHOLD = 15000

def get_free_gpu():
    """返回单个空闲的 GPU ID（如果有的话）"""
    output = subprocess.check_output("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits", shell=True)
    lines = output.decode("utf-8").strip().split("\n")
    for line in lines:
        gpu_id, free_mem = map(int, line.split(", "))
        if gpu_id in AVAILABLE_GPUS and free_mem > MEMORY_THRESHOLD:
            return gpu_id
    return None

def worker(task_queue):
    """负责检查空闲 GPU 并执行任务，每个任务仅使用一张显卡"""
    while not task_queue.empty():
        gpu_id = get_free_gpu()
        if gpu_id is not None:
            command = task_queue.get()
            print(f"[INFO] Running on GPU {gpu_id}: {command}")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            subprocess.Popen(command, shell=True, env=env)
            time.sleep(5)  # 避免短时间内过度轮询
        else:
            time.sleep(10)  # 如果没有空闲 GPU，则等待 10 秒后再检查

if __name__ == "__main__":
    task_queue = queue.Queue()
    for task in TASKS:
        task_queue.put(task)
    
    threads = []
    for _ in range(len(AVAILABLE_GPUS)):
        t = threading.Thread(target=worker, args=(task_queue,))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    print("[INFO] 所有任务已完成。")
