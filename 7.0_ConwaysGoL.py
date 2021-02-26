# coding=utf-8
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# a thread is a sequence of instructions that is executed on a single core of the GPU
# it is possible to launch kernels that use many more threads than there are cores on the GPU
# the operating system's scheduler can switch between these tasks rapidly,
# giving the appearance that they are running simultaneously. The GPU handles threads in a similar way
# Multiple threads are executed on the GPU in abstract units known as blocks.
# thread ID from threadIdx.x, threadIdx.y and threadIdx.z for up to 3 dimensions
# This is because you can index blocks over three dimensions, rather than just one dimension
# (for us generally 2 dimensions will be sufficient though)
# Blocks are further executed in abstract batches known as grids, which are best thought of as blocks of blocks.
# As with threads in a block, we can index each block in the grid in up to three dimensions
# with the constant values that are given by blockIdx.x , blockIdx.y, (and blockIdx.z)

"""Conway's GameOfLife in a GPU Implementation"""

# start by using the C language's #define directive to set up some constants and macros that we'll use throughout the kernel.
# Let's look at the first two we'll set up, _X and _Y
# #define -> it will literally replace any text of _X or _Y with the defined values
# (in the parentheses here) at compilation time, it creates macros for us

# Warum ueberhaupt hier mit mehreren Bloecken (also einem Grid) arbeiten?
# lediglich 1024 threads pro Block (also 32x32 Array). Das ist hier aber zu wenig, wir wollen 128x128
# wir brauchen also 4x4 (16mal) diese Groeße, also 16 blocks
"""
#define _X  ( threadIdx.x + blockIdx.x * blockDim.x ) 
//ermittelt die Position des Threads auf der x Achse, muss auch noch mit der Position auf der y kombiniert werden, fuer exakte Bestimmung im CUDA Code, s.u. bei index
#define _Y  ( threadIdx.y + blockIdx.y * blockDim.y )

#define _WIDTH  ( blockDim.x * gridDim.x ) 
//gesamte "Breite" des Arrays, Anzahl Spalten im Array (Groesse block) * wie oft diese Spalten hintereinander stehen (Anzahl Blocks)
#define _HEIGHT ( blockDim.y * gridDim.y )
//gesamte "Höhe" des Arrays, Anzahl Reihen im Array, bzw. Anzahl Arrays im Array * wie oft diese Reihen untereinander stehen (Anzahl Blocks)

//Dieser ganze Bereichnigungsschritt ist nur noetig, weil wir bei der Bestimmung der Nachbarn mitunter von x+1 bzw. y+1 bestimmen.
//wenn wir vom Ende des Grids machen, gaebe es aber keine Nachbarn mehr, da die entsprechende threadIdx.x nicht exisitert.
//So wird hier auf der anderen Seite des Arrayblocks wieder angesetzt
#define _XM(x)  ( (x + _WIDTH) % _WIDTH )
//Bereignigter threadIndex in seinem jeweiligen block
#define _YM(y)  ( (y + _HEIGHT) % _HEIGHT )

#define _INDEX(x,y)  ( _XM(x)  + _YM(y) * _WIDTH ) 
//erlaubt die ueberfuehrung des uebergebenen 2D Arrays in ein 1D Array
//(PyCUDA passes twodimensional arrays into CUDA C as onedimensional pointers; twodimensional arrays are passed in rowwise from Python into one dimensional C pointers.)
//WIDTH waere in unserem Bsp. 32*4 (=128), die fuere jeden threadIndex.y nochmal draufgerechnet werden muessen
//Solange das Array ein Quadrat ist, ist es egal, ob man hier _HEIGHT oder _WIDTH nimmt, grundsaetzlich ist aber WIDTH, da man in der 2.D pro Reihe vorgehen muss.
"""

# In dieser "simplen" Implementation, wird das Feld jedesmal bei Ausfuehrung des Kernels geupdated,
# keine SynchronisationsProbleme, da wir nur mit der vorherigen Iteration des Feldes arbeiten mussten,
# die immer verfügbar war.
# Wenn wir nun mehrere Iterationen bei Aufruf des Kernels durchfuehren wollen, koennten wir das mit einer for Schleife
# tun, dann kann es aber zu "race conditions" kommen (issue of multiple threads reading and writing to the same memory address)
# Bei einer Iteration ist das noch kein Problem, hier wird im Prinzip der Host als Synchronisation benutzt.
# s. ThreadSynchronisation fuer weiteres

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

# The frameNum parameter is just a value that is required by Matplotlib's animation module for
# update functions that we can ignore, while img will be the representative image of our
# cell lattice that is required by the module that will be iteratively displayed.
def update_gpu(frameNum, img, newLattice_gpu, lattice_gpu, N):
    conway_ker(newLattice_gpu, lattice_gpu, block=(32, 32, 1), grid=(N / 32, N / 32, 1)) # 16384 threads
    # set up the image data for our animation after grabbing the latest generated lattice from the GPU's memory with get()
    img.set_data(newLattice_gpu.get())
    # copy the new lattice data into the current data using the PyCUDA slice operator,
    # [:], which will copy over the previously allocated memory on the GPU so that we don't have to re-allocate:
    lattice_gpu[:] = newLattice_gpu[:]

    return img


if __name__ == '__main__':
    # set lattice size (has to be a multiple of 32 with our Setup, look @grid)
    N = 128
    # We'll populate a N x N graph of integers randomly with ones and zeros; generally,
    # if around 25% of the points are ones and the rest zeros, we can generate some interesting lattice animations:
    lattice = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N)) # 16384 Array, 128*128 groß
    lattice_gpu = gpuarray.to_gpu(lattice)

    newLattice_gpu = gpuarray.empty_like(lattice_gpu)

    fig, ax = plt.subplots()
    img = ax.imshow(lattice_gpu.get(), interpolation='nearest')
    ani = animation.FuncAnimation(fig, update_gpu, fargs=(img, newLattice_gpu, lattice_gpu, N,), interval=25,
                                  frames=1000, save_count=1000)

    plt.show()
