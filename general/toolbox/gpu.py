from pynvml import *


def gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    usegb = info.used // 1042 ** 3 > 1
    return (
        f"GPU used: {info.used//1024**3} GB"
        if usegb
        else f"GPU used: {info.used//1024**2} MB"
    )


def summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
