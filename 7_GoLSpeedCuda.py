import sys
from time import time

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

np.random.seed(0)

GoL = SourceModule("""
#define _X  (threadIdx.x + blockIdx.x * blockDim.x)
#define _Y  (threadIdx.y + blockIdx.y * blockDim.y)
#define _width  (blockDim.x * gridDim.x )
#define _true(x)  ((x + _width) % _width )
#define _index(x,y)  (_true(x) + _true(y) * _width)

// return the number of living neighbors for a given cell                
__device__ int neighbors(int x, int y, int * in)
{
     return ( in[_index(x -1, y+1)] + in[_index(x-1, y)] + in[_index(x-1, y-1)] \
                   + in[_index(x, y+1)] + in[_index(x, y - 1)] \
                   + in[_index(x+1, y+1)] + in[_index(x+1, y)] + in[_index(x+1, y-1)]);
}

__global__ void gameoflife(int * grid_out, int * grid)
{
   int x = _X, y = _Y;

   int n = neighbors(x, y, grid);

    if (grid[_index(x,y)] == 1) {
        if (n == 2 || n == 3)  {
            grid_out[_index(x,y)] = 1;
        }
        else {
            grid_out[_index(x,y)] = 0;
        }
    }
    else if( grid[_index(x,y)] == 0 )
         if (n == 3)  {
            grid_out[_index(x,y)] = 1;
        }
        else {
            grid_out[_index(x,y)] = 0;
        }

}
""")

gameoflife = GoL.get_function("gameoflife")

X = [32, 64, 128, 256, 512, 1024, 2048]
Y = [1024, 2048, 4096, 8192, 16384]
iterations = 20
numDurchf = 20

for z in Y:
    Time = 0
    N = z
    print("Durchfuehrung mit Arraygroesse: " + str(N))
    for i in range(numDurchf):
        t_start = time()
        grid = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))
        grid_gpu = gpuarray.to_gpu(grid)
        emptygrid_gpu = gpuarray.empty_like(grid_gpu)
        for i in range(iterations):
            gameoflife(emptygrid_gpu, grid_gpu, block=(32, 32, 1), grid=(N/32, N/32, 1))
            grid_gpu[:] = emptygrid_gpu[:]
        grid = grid_gpu.get()
        t_end = time()
        Time += t_end - t_start
    print("end")
    print('Total time: %fs' % (Time / numDurchf))


"""
print("started with gridsize " + str(N) + " and " + str(iterations) + " iterations")
grid = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))
grid_gpu = gpuarray.to_gpu(grid)
emptygrid_gpu = gpuarray.empty_like(grid_gpu)

#print(grid)

X = 32
Y = N/32

if N < X:
    X = N
    Y = 1
else:
    if N % 32 != 0:
        raise Exception("N sollte ein vielfaches von 32 sein, sonst muss der Kernel manuell konfiguriert werden!")


for i in range(iterations):
    gameoflife(emptygrid_gpu, grid_gpu, block=(X, X, 1), grid=(Y, Y, 1))
    grid_gpu[:] = emptygrid_gpu[:]

grid = grid_gpu.get()
#print(grid)

t_end = time()

print("end")
print ('Total time: %fs' % (t_end - t_start))
"""
"""
mit Precompiling

iterations = 1
N = 256
Total time: 0.003000s

iterations = 200
N = 256
Total time: 0.014000s
"""