import time
import torch

import os

time.sleep(4*int(os.environ['RANK']))
import sys
print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION')
from subprocess import call

print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
print('Active CUDA Device: GPU', torch.cuda.current_device())
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())
call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])

for rank in range(torch.cuda.device_count()):
    x = torch.Tensor([1]).to(rank)
    print(f'send to cuda:{rank} ok')

""" ... couldnt install pycuda
import pycuda
from pycuda import compiler
import pycuda.driver as drv

drv.init()
print("%d device(s) found." % drv.Device.count())

for ordinal in range(drv.Device.count()):
    dev = drv.Device(ordinal)
    print (ordinal, dev.name())

from pycuda import gpuarray
from pycuda.curandom import rand as curand
# -- initialize the device
import pycuda.autoinit

height = 100
width = 200
X = curand((height, width), np.float32)
X.flags.c_contiguous 
print (type(X))
"""

print('\n'*3)
