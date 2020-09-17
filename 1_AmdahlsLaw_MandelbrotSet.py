import numpy as np
from time import time
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

"""Amdahl's Law
Bsp. 100 + 100 / N -> 100 = konstante, kann nur von einer Person ausgefuehrt werden, 100/N parralellisierbare Aufgabe
-> .5 + .5 / N
-> 1 / (.5 + .5 / N) fuer den potential Speedup

Allg. Gleichung: speedup = 1/((1-p)+p/N), p -> program (originally serial), N number of cores available
"""

"""Mandelbrot set:
For a given complex number, c, we define a recursive sequence for
n >= 0 
z0 = 0
zn = (zn-1)**2+c
for
n >= 1
If |zn| remains bounded by 2 as n increases to infinity, then we will say that c is a member of the Mandelbrot set. 
"""


def simple_mandelbrot(width, height, real_low, real_high, imag_low, imag_high, max_iters):
    real_vals = np.linspace(real_low, real_high, width)
    imag_vals = np.linspace(imag_low, imag_high, height)

    # we will represent members as 1, non-members as 0.

    mandelbrot_graph = np.ones((height, width), dtype=np.float32)

    for x in range(width):

        for y in range(height):

            c = np.complex64(real_vals[x] + imag_vals[y] * 1j)
            z = np.complex64(0)

            for i in range(max_iters):

                z = z ** 2 + c

                if (np.abs(z) > 2):
                    mandelbrot_graph[y, x] = 0
                    break

    return mandelbrot_graph


if __name__ == '__main__':
    t1 = time()
    mandel = simple_mandelbrot(512, 512, -2, 2, -2, 2, 256)
    t2 = time()
    mandel_time = t2 - t1

    t1 = time()
    fig = plt.figure(1)
    plt.imshow(mandel, extent=(-2, 2, -2, 2))
    plt.savefig('mandelbrot.png', dpi=fig.dpi)
    t2 = time()

    dump_time = t2 - t1

    print 'It took {} seconds to calculate the Mandelbrot graph.'.format(mandel_time)
    print 'It took {} seconds to dump the image.'.format(dump_time)

# It took 16.5080001354 seconds to calculate the Mandelbrot graph.
# It took 0.128000020981 seconds to dump the image.
# p = 16.5080001354/(16.508 + 0.128) = ~99%

"""
As we have seen, we generate the Mandelbrot set point by point; there is no interdependence between the values of different points, and it is, 
therefore, an intrinsically parallelizable function. In contrast, the code to dump the image cannot be parallelized.
Nach Amdahls Law koennen wir mit einer gtx1050 mit 640 Kernen, also ein 
speedup = 1/(0.01+0.99/640) = ~86% erzielen
Keep in mind that Amdahl's Law only gives a very rough estimate! 
There will be additional considerations that will come into play when we offload computations onto the GPU, 
such as the additional time it takes for the CPU to send and receive data to and from the GPU; 
or the fact that algorithms that are offloaded to the GPU are only partially parallelizable.
"""

