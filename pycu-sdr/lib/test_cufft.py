#!/usr/bin/env python
# Copyright: (c) 2021, Edwin G. W. Peters


"""
Example of fast convolution in float32 GPU vs CPU

author = Edwin Peters
"""

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import time

import cufft

N = 4096 # fft len
filt_len = 100

# initialize arrays
a = np.random.randn(N)+1j*np.random.randn(N)
a = a.astype(np.complex64) # complex float32
filt = np.ones(filt_len).astype(np.complex64)
# pad filter
filt_pad = np.r_[filt,np.zeros(N-filt_len)]

# output array
res_conv = np.empty_like(a)


assert len(filt_pad) == len(a),'filt_pad and a should be of similar length'

# allocate GPU buffers
a_gpu = cuda.mem_alloc(a.nbytes)
filt_gpu = cuda.mem_alloc(filt_pad.nbytes)



# create fft plan

fftPlan = cufft.cufftPlan1d(N,cufft.CUFFT_C2C,1)

# Initialize GPU complex multiply and conjugate kernel
kernelStr = """
#include <math.h>
#include <cufft.h>
#include <cufftXt.h>

#define GRID_X (threadIdx.x + blockDim.x*blockIdx.x)


typedef float2 Complex;

static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b);
static __device__ __host__ inline Complex ComplexConj(Complex a);
static __device__ __host__ inline Complex ComplexScale(Complex a, float s);

__global__ void conjMul(Complex *out, Complex*  a, Complex* b, int len)
{

  const int x = GRID_X;

  if (x < len) // ensure that we don't operate out of array
    out[x] = ComplexScale(ComplexMul(a[x],ComplexConj(b[x])), (float) len);
}


static __device__ __host__ inline Complex ComplexScale(Complex a, float s)
{
  Complex c;
  c.x = a.x/s;
  c.y = a.y/s;

  return c;
}

static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b)
{
  // Internal complex multiplication
  Complex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}

static __device__ __host__ inline Complex ComplexConj(Complex a)
{
  // Internal complex conjugate
  a.y = -a.y;
  return a;
}


"""

# compile kernel with CUDA 
kernel = SourceModule(kernelStr)
conjMul = kernel.get_function('conjMul').prepare('PPPi')

## fast convolution GPU

t = time.time()

cuda.memcpy_htod(a_gpu,a.astype(np.complex64))
cuda.memcpy_htod(filt_gpu,filt_pad.astype(np.complex64))

# forward FFT
cufft.cufftExecC2C(fftPlan,int(a_gpu),int(a_gpu),cufft.CUFFT_FORWARD)
cufft.cufftExecC2C(fftPlan,int(filt_gpu),int(filt_gpu),cufft.CUFFT_FORWARD)

# run kernel
conjMul.prepared_call((int(np.ceil(N/1024)),1),(1024,1,1),a_gpu,a_gpu,filt_gpu,N) # (grid), (block)


# IFFT
cufft.cufftExecC2C(fftPlan,int(a_gpu),int(a_gpu),cufft.CUFFT_INVERSE)

cuda.memcpy_dtoh(res_conv,a_gpu)

print(f'time GPU {time.time()-t} s')

# cpu reference
t = time.time()
res_conv_cpu = np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(filt_pad)))
print(f'time CPU {time.time()-t} s')

# cleanup
cufft.cufftDestroy(fftPlan)

print(f'MSE fast convolution {np.mean(np.abs(res_conv-res_conv_cpu)**2)}')
