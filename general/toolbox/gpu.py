from pynvml import *

import time
from functools import wraps

try:
    nvmlInit()
    cpu = False
except: 
    print("No GPU found")
    cpu = True

def timer():
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            elapsed = end - start
            print(f"{func.__name__:20s}: Execution time: {elapsed:.2f} seconds | {gpu_utilization()} | {gpu_free()}")
            return result
        return wrapper
    return decorator


def gpu_free():
    if cpu:
        return "CPU"
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    gb = info.free // 1042**3
    return f"GPU free: {gb} GB" if gb > 1 else f"GPU free: {gb*1024} MB"


def gpu_utilization():
    if cpu:
        return "CPU"
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    gb = info.used // 1042**3
    return f"GPU used: {gb} GB" if gb > 1 else f"GPU used: {gb*1024} MB"


def summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
