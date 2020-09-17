import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Using shared memory:
# threads in the kernel can intercommunicate using arrays within the GPU's global memory;
# while it is possible to use global memory for most operations, we can speed things up by using shared memory
# ype of memory meant specifically for intercommunication of threads within a single CUDA block;
# much faster for pure interthread communication. In contrast to global memory, though,
# memory stored in shared memory cannot directly be accessed by the host
# shared memory must be copied back into global memory by the kernel itself first.

#when we normally declare variables in CUDA, they are by default local to each individual thread.
# Note that, even if we declare an array within a thread such as int a[10];,
# there will be an array of size 10 that is local to each thread.
"""
Local thread arrays (for example, a declaration of int a[10]; within the kernel) and pointers to global GPU memory 
(for example, a value passed as a kernel parameter of the form int * b) may look and act similarly, but are very different. 
For every thread in the kernel, there will be a separate a array that the other threads cannot read, 
yet there is a single b that will hold the same values and be equally accessible for all of the threads.
"""
# The story is obviously different for __shared__ variables

shared_ker = SourceModule("""    
#define _iters 1000000                       

#define _X  ( threadIdx.x + blockIdx.x * blockDim.x )
#define _Y  ( threadIdx.y + blockIdx.y * blockDim.y )

#define _WIDTH  ( blockDim.x * gridDim.x )
#define _HEIGHT ( blockDim.y * gridDim.y  )

#define _XM(x)  ( (x + _WIDTH) % _WIDTH )
#define _YM(y)  ( (y + _HEIGHT) % _HEIGHT )

#define _INDEX(x,y)  ( _XM(x)  + _YM(y) * _WIDTH )

// return the number of living neighbors for a given cell                
__device__ int nbrs(int x, int y, int * in)
{
     return ( in[ _INDEX(x -1, y+1) ] + in[ _INDEX(x-1, y) ] + in[ _INDEX(x-1, y-1) ] \
                   + in[ _INDEX(x, y+1)] + in[_INDEX(x, y - 1)] \
                   + in[ _INDEX(x+1, y+1) ] + in[ _INDEX(x+1, y) ] + in[ _INDEX(x+1, y-1) ] );
}

// bis hierhin wieder alles gleich
// wieder: nur noch ein Feld wird uebergeben, die Anzahl Iterationen fuer die for-Schleife
// neu: das Feld heisst jetzt p_lattice und wir benutzen noch ein "shared" Feld lattice intern in der Funktion
// auf dieses schreiben wir erstmal alle Positionen aus dem uebergebenen p_lattice, 
// hiernach werden die threads erstmal synchronisiert

__global__ void conway_ker_shared(int * p_lattice, int iters)
{
   // x, y are the appropriate values for the cell covered by this thread
   // we want to store these values in variables to reduce computation because directly using _X and _Y will 
   // recompute the x and y values every time these macros are referenced in our code)
   int x = _X, y = _Y;
   __shared__ int lattice[32*32];


   lattice[_INDEX(x,y)] = p_lattice[_INDEX(x,y)];
   __syncthreads();

   for (int i = 0; i < iters; i++)
   {

       // count the number of neighbors around the current cell
       int n = nbrs(x, y, lattice);

       int cell_value;


        // if the current cell is alive, then determine if it lives or dies for the next generation.
        if ( lattice[_INDEX(x,y)] == 1)
           switch(n)
           {
              // if the cell is alive: it remains alive only if it has 2 or 3 neighbors.
              case 2:
              case 3: cell_value = 1;
                      break;
              default: cell_value = 0;                   
           }
        else if( lattice[_INDEX(x,y)] == 0 )
             switch(n)
             {
                // a dead cell comes to life only if it has 3 neighbors that are alive.
                case 3: cell_value = 1;
                        break;
                default: cell_value = 0;         
             }

        __syncthreads();
        lattice[_INDEX(x,y)] = cell_value;
        __syncthreads();

    }
    // The rest of the kernel is exactly as before, only we have to copy from the shared lattice back into the GPU array. 
    // We do so as follows and then close off the inline code:
    __syncthreads();
    p_lattice[_INDEX(x,y)] = lattice[_INDEX(x,y)];
    __syncthreads();

}
""")

conway_ker_shared = shared_ker.get_function("conway_ker_shared")

if __name__ == '__main__':
    # set lattice size
    N = 32

    lattice = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))
    lattice_gpu = gpuarray.to_gpu(lattice)

    conway_ker_shared(lattice_gpu, np.int32(1000000), grid=(1, 1, 1), block=(32, 32, 1))

    fig = plt.figure(1)
    plt.imshow(lattice_gpu.get())
    plt.show()
