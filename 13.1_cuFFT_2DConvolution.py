from __future__ import division
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import fft
from skcuda import linalg
from matplotlib import pyplot as plt

"""
do some basic fast Fourier transforms (FFT) with cuFFT.
Normally, computing a matrix-vector operation is of computational complexity O(N2) for a vector of length N. However, due to symmetries in the DFT matrix, 
this can always be reduced to O(N log N) by using an FFT
"""

"""1D FFT"""

x = np.asarray(np.random.rand(1000), dtype=np.float32 )
x_gpu = gpuarray.to_gpu(x)
x_hat = gpuarray.empty_like(x_gpu, dtype=np.complex64)

plan = fft.Plan(x_gpu.shape,np.float32,np.complex64)

inverse_plan = fft.Plan(x.shape, in_dtype=np.complex64, out_dtype=np.float32)

fft.fft(x_gpu, x_hat, plan)
fft.ifft(x_hat, x_gpu, inverse_plan, scale=True)

y = np.fft.fft(x)
print 'cuFFT matches NumPy FFT: %s' % np.allclose(x_hat.get(), y, atol=1e-6)
print 'cuFFT inverse matches original: %s' % np.allclose(x_gpu.get(), x, atol=1e-6)

#print 'cuFFT matches NumPy FFT: %s' % np.allclose(x_hat.get()[0:N//2], y[0:N//2], atol=1e-6)

"""2D Convolution"""

def cufft_conv(x, y):
    x = x.astype(np.complex64)
    y = y.astype(np.complex64)

    if (x.shape != y.shape):
        return -1

    plan = fft.Plan(x.shape, np.complex64, np.complex64)
    inverse_plan = fft.Plan(x.shape, np.complex64, np.complex64)

    x_gpu = gpuarray.to_gpu(x)
    y_gpu = gpuarray.to_gpu(y)

    x_fft = gpuarray.empty_like(x_gpu, dtype=np.complex64)
    y_fft = gpuarray.empty_like(y_gpu, dtype=np.complex64)
    out_gpu = gpuarray.empty_like(x_gpu, dtype=np.complex64)

    fft.fft(x_gpu, x_fft, plan)
    fft.fft(y_gpu, y_fft, plan)

    linalg.multiply(x_fft, y_fft, overwrite=True)

    fft.ifft(y_fft, out_gpu, inverse_plan, scale=True)

    conv_out = out_gpu.get()

    return conv_out


def conv_2d(ker, img):
    padded_ker = np.zeros((img.shape[0] + 2 * ker.shape[0], img.shape[1] + 2 * ker.shape[1])).astype(np.float32)

    padded_ker[:ker.shape[0], :ker.shape[1]] = ker

    padded_ker = np.roll(padded_ker, shift=-ker.shape[0] // 2, axis=0)
    padded_ker = np.roll(padded_ker, shift=-ker.shape[1] // 2, axis=1)

    padded_img = np.zeros_like(padded_ker).astype(np.float32)

    padded_img[ker.shape[0]:-ker.shape[0], ker.shape[1]:-ker.shape[1]] = img

    out_ = cufft_conv(padded_ker, padded_img)

    output = out_[ker.shape[0]:-ker.shape[0], ker.shape[1]:-ker.shape[1]]

    return output


gaussian_filter = lambda x, y, sigma: (1 / np.sqrt(2 * np.pi * (sigma ** 2))) * np.exp(
    -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))


def gaussian_ker(sigma):
    ker_ = np.zeros((2 * sigma + 1, 2 * sigma + 1))

    for i in range(2 * sigma + 1):
        for j in range(2 * sigma + 1):
            ker_[i, j] = gaussian_filter(i - sigma, j - sigma, sigma)

    total_ = np.sum(ker_.ravel())

    ker_ = ker_ / total_

    return ker_


if __name__ == '__main__':

    jonas = np.float32(plt.imread('jonas.jpg')) / 255
    jonas_blurred = np.zeros_like(jonas)
    ker = gaussian_ker(15)

    for k in range(3):
        jonas_blurred[:, :, k] = conv_2d(ker, jonas[:, :, k])

    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.suptitle('Gaussian Filtering', fontsize=20)
    ax0.set_title('Before')
    ax0.axis('off')
    ax0.imshow(jonas)
    ax1.set_title('After')
    ax1.axis('off')
    ax1.imshow(jonas_blurred)
    plt.tight_layout()
    plt.subplots_adjust(top=.85)
    plt.show()
