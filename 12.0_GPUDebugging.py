import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule

# printf is the standard function that prints a string to the standard output, and is
# really the equivalent in the C programming language of Python's print function.
"""
printf can also take a variable number of parameters in the case that we want to print any constants or variables from 
directly within C: if we want to print the 123integers to the output, we do this with printf("%d", 123)
Similarly, we use %f, %e, or %g to print floating-point values (where %f is the decimal notation, %e is the scientific 
notation, and %g is the shortest representation whether decimal or scientific)
"""

# we wrote \\n rather than \n. This is due to the fact that the triple quote in Python itself will interpret \n as a
# "new line", so we have to indicate that we mean this literally by using a double backslash so as to pass the \n
# directly into the CUDA compiler
ker = SourceModule('''
__global__ void hello_world_ker()
{
	printf("Hello world from thread %d, in block %d!\\n", threadIdx.x, blockIdx.x);

	__syncthreads();

	if(threadIdx.x == 0 && blockIdx.x == 0)
	{
		printf("-------------------------------------\\n");
		printf("This kernel was launched over a grid consisting of %d blocks,\\n", gridDim.x);
		printf("where each block has %d threads.\\n", blockDim.x);
	}
}
''')

hello_ker = ker.get_function("hello_world_ker")
hello_ker(block=(5, 1, 1), grid=(2, 1, 1))
