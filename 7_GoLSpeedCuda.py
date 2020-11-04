import sys
from time import time

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

np.random.seed(0)

ker = SourceModule("""
#define _X  ( threadIdx.x + blockIdx.x * blockDim.x )
#define _Y  ( threadIdx.y + blockIdx.y * blockDim.y )

#define _WIDTH  ( blockDim.x * gridDim.x )
#define _HEIGHT ( blockDim.y * gridDim.y  )

#define _XM(x)  ( (x + _WIDTH) % _WIDTH )
#define _YM(y)  ( (y + _HEIGHT) % _HEIGHT )

#define _INDEX(x,y)  ( _XM(x)  + _YM(y) * _WIDTH )

// return the number of living neighbors for a given cell
/*
A device function is a C function written in serial, which is called by an individual CUDA thread in kernel. 
this little function will be called in parallel by multiple threads from our kernel
*/                
__device__ int nbrs(int x, int y, int * in)
{
     return ( in[ _INDEX(x -1, y+1) ] + in[ _INDEX(x-1, y) ] + in[ _INDEX(x-1, y-1) ] \
                   + in[ _INDEX(x, y+1)] + in[_INDEX(x, y - 1)] \
                   + in[ _INDEX(x+1, y+1) ] + in[ _INDEX(x+1, y) ] + in[ _INDEX(x+1, y-1) ] );
}



__global__ void conway_ker(int * lattice_out, int * lattice  )
{
   // x, y are the appropriate values for the cell covered by this thread
   int x = _X, y = _Y;

   // count the number of neighbors around the current cell
   int n = nbrs(x, y, lattice);


    // if the current cell is alive, then determine if it lives or dies for the next generation.
    if ( lattice[_INDEX(x,y)] == 1)
       switch(n)
       {
          // if the cell is alive: it remains alive only if it has 2 or 3 neighbors.
          case 2:
          case 3: lattice_out[_INDEX(x,y)] = 1;
                  break;
          default: lattice_out[_INDEX(x,y)] = 0;                   
       }
    else if( lattice[_INDEX(x,y)] == 0 )
         switch(n)
         {
            // a dead cell comes to life only if it has 3 neighbors that are alive.
            case 3: lattice_out[_INDEX(x,y)] = 1;
                    break;
            default: lattice_out[_INDEX(x,y)] = 0;         
         }

}
""")

conway_ker = ker.get_function("conway_ker")


t_start = time()

iterations = 20
N = 16384
print("started with gridsize " + str(N) + " and " + str(iterations) + " iterations")
grid = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))
grid_gpu = gpuarray.to_gpu(grid)
newgrid_gpu = gpuarray.empty_like(grid_gpu)

#print(grid)

for i in range(iterations):
    conway_ker(newgrid_gpu, grid_gpu, block=(32, 32, 1), grid=(N / 32, N / 32, 1))
    grid_gpu[:] = newgrid_gpu[:]

grid = grid_gpu.get()
#print(grid)

t_end = time()

print("end")
print ('Total time: %fs' % (t_end - t_start))

"""
mit Precompiling

iterations = 1
N = 256
Total time: 0.003000s

iterations = 200
N = 256
Total time: 0.014000s
"""