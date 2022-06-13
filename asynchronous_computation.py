""" ASYNCHRONOUS COMPUTATION """
""" When you call a function that uses the GPU, the operations are enqueued to the particular device, but not necessarily executed until later. 
This allows us to execute more computations in parallel, including operations on the CPU or other GPUs. ASYNCHRNOUS COMPUTATION WITH PYTORCH """

import os
import subprocess
from sys import dllhandle
import numpy
import torch
from torch import nn
from d2l import torch as d2l

""" For a warmup consider the following toy problem: we want to generate a random matrix and multiply it. 
Let us do that both in NumPy and in PyTorch tensor to see the difference. Note that PyTorch tensor is defined on a GPU. """

device = d2l.try_gpu()
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a) # Performs a matrix multiplication of the matrices. It is multipling the matrix

with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000)) # A matrix 10000 per 10000
        b = numpy.dot(a, a) # return the dot product of two arrays

with d2l.Benchmark('torch'):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device) # A matrix 10000 per 10000
        b = torch.mm(a, a) 

""" for example, it will return the time execution: 
        numpy: 1.2775 sec 
        torch: 0.0010 sec 
        
    The benchmark output via PyTorch is orders of magnitude faster. NumPy dot product is executed on the CPU processor while PyTorch matrix multiplication 
    is executed on GPU and hence the latter is expected to be much faster. But the huge time difference suggests something else must be going on. 
    By default, GPU operations are asynchronous in PyTorch. Forcing PyTorch to finish all computation prior to returning shows what happened previously:
     computation is being executed by the backend while the frontend returns control to Python. """