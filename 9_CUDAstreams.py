"""
We know that within a single kernel, there is one level of concurrency among its many threads;
however, there is another level of concurrency over multiple kernels and GPU memory operations
that is also available to us. This means that we can launch multiple memory and kernel operations at
once, without waiting for each operation to finish. However, on the other hand, we will have to be
somewhat organized to ensure that all inter-dependent operations are synchronized; this means that we
shouldn't launch a particular kernel until its input data is fully copied to the device memory, or we
shouldn't copy the output data of a launched kernel to the host until the kernel has finished execution.
"""
# - CUDAstreams - a stream is a sequence of operations that are run in order on the GPU
# - events, which are a feature of streams that are used to precisely time kernels and
# indicate to the host as to what operations have been completed within a given stream.
# - context can be thought of as analogous to a process in your operating system, in that the GPU
# keeps each context's data and kernel code walled off and encapsulated away from the other contexts
# currently existing on the GPU
