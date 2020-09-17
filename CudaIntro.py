from __future__ import print_function

"""
PyCuda in PyCharm installieren/gestartet bekommen
- erste online Introductions/Tutorials(?)
- kernel gestartet bekommen(?)
- Typkonvertierungen/Speicherreservierung
"""
import numpy
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule



# SourceModule compiles C code for CUDA
mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")

multiply_them = mod.get_function("multiply_them")

a = numpy.random.randn(400).astype(numpy.float32)
b = numpy.random.randn(400).astype(numpy.float32)
print(a)
print(b)
dest = numpy.zeros_like(a)
multiply_them(
    drv.Out(dest), drv.In(a), drv.In(b),
    block=(400, 1, 1), grid=(1, 1))

print(dest)

N = 1 << 20 #1M Elements -> 1 bit um 20 stellen nach links geshiftet
print(N)