import numpy
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

mod = SourceModule("""
    struct DoubleOperation {
        int datalen, __padding; // so 64-bit ptrs can be aligned
        float *ptr;
    };

    __global__ void double_array(DoubleOperation *a) {
        a = &a[blockIdx.x];
        for (int idx = threadIdx.x; idx < a->datalen; idx += blockDim.x) {
            a->ptr[idx] *= 2;
        }
    }
    """)


class DoubleOpStruct:
    mem_size = 8 + numpy.intp(0).nbytes

    def __init__(self, array, struct_arr_ptr):
        self.data = drv.to_device(array)
        self.shape, self.dtype = array.shape, array.dtype
        drv.memcpy_htod(int(struct_arr_ptr), numpy.getbuffer(numpy.int32(array.size)))
        drv.memcpy_htod(int(struct_arr_ptr) + 8, numpy.getbuffer(numpy.intp(int(self.data))))

    def __str__(self):
        return str(drv.from_device(self.data, self.shape, self.dtype))


struct_arr = drv.mem_alloc(2 * DoubleOpStruct.mem_size)
do2_ptr = int(struct_arr) + DoubleOpStruct.mem_size

array1 = DoubleOpStruct(numpy.array([1, 2, 3], dtype=numpy.float32), struct_arr)
array2 = DoubleOpStruct(numpy.array([0, 4], dtype=numpy.float32), do2_ptr)
print("original arrays", array1, array2)

func = mod.get_function("double_array")
func(struct_arr, block=(32, 1, 1), grid=(2, 1))
print("doubled arrays", array1, array2)

func(numpy.intp(do2_ptr), block=(32, 1, 1), grid=(1, 1))
print("doubled second only", array1, array2, "\n")
