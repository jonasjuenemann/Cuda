import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt

# thread synchronization and thread intercommunication
# we may need to ensure that every single thread has reached the same exact line in the code
# before we continue with any further computation -> thread synch
# Synchronization works hand-in-hand with thread intercommunication, that is,
# different threads passing and reading input from each other
# CUDA __syncthreads device function, which is used for synchronizing a single block in a kernel

# We want to do multiple iterations of LIFE on the GPU that are fully synchronized;
# we also will want to use a single array of memory for the lattice.
# avoid race conditions by using a CUDA device function called __syncthreads()
# This function is a block level synchronization barrier, this means that every thread
# that is executing within a block will stop when it reaches a __syncthreads() instance
# and wait until each and every other thread within the same block reaches that same
# invocation of __syncthreads() before the the threads continue to execute the subsequent lines of code.
# (__syncthreads() can only synchronize threads within a single CUDA block, not all threads within a CUDA grid!)

ker = SourceModule("""
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

// bis hierhin alles gleich
// jetzt aber: nur noch ein Feld wird uebergeben, ausserdem neu: die Anzahl Iterationen fuer die for-Schleife

__global__ void conway_ker(int * lattice, int iters)
{
   // x, y are the appropriate values for the cell covered by this thread
   int x = _X, y = _Y;

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
        // bevor wir den ZellenWert tatsaechlich ueberschreiben, wird auf alle Threads gewartet, damit diese auch ihren neuen Wert haben.
        // bevor wir in die naechste Iteration der for-Schleife gehen, ebenso
        __syncthreads();
        lattice[_INDEX(x,y)] = cell_value;
        __syncthreads(); 
    }

}
""")

conway_ker = ker.get_function("conway_ker")

if __name__ == '__main__':
    # set lattice size
    N = 32
    # Leider nur noch size = 1024 (32x32), da __synchthreads nur ueber einen block funktioniert,
    # um das gleiche ueber mehrere Bloecke zu machen, recht kompliziert
    lattice = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))
    lattice_gpu = gpuarray.to_gpu(lattice)
    conway_ker(lattice_gpu, np.int32(100000), grid=(1, 1, 1), block=(32, 32, 1))
    fig = plt.figure(1)
    plt.imshow(lattice_gpu.get())
    plt.show()
