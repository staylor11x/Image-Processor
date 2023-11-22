import numpy as np
import cv2
from matplotlib import pyplot as plt

def create_kernel(m, n):
    '''create a mean kernel'''

    m2 = m // 2
    n2 = n // 2
    h = np.zeros((m, n))
    for j in range(-m2, m2+1):
        for k in range(-n2, n2+1):
            h[j+m2, k+n2] = 1 / (m*n)           
    return h

def create_gaussian(kernelXY):
    '''create a gaussian kernel'''

    kernel_size = (kernelXY,kernelXY)
    kernel = cv2.getGaussianKernel(kernel_size[0],-1)
    kernel = np.outer(kernel, kernel.transpose())

    #normalise the kernel
    kernel /= kernel.sum()

    return kernel


def convolve(f, h):
    '''convolve the image (f) with the kernel (h) 
    return the result of the convolution (g)'''

    m, n = h.shape
    m2 = m // 2
    n2 = n // 2
    g = np.zeros_like(f)
    for x in range(m2, f.shape[0]-m2):
        for y in range(n2, f.shape[1]-n2):
            sum = 0
            for j in range(-m2, m2+1):
                for k in range(-n2, n2+1):
                    sum += h[j+m2, k+n2] * f[x-j, y-k]
            g[x, y] = sum
    return g

#Load in test image
image = 'person.jpg'
img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

#create mean kernel
mean_kernel = create_kernel(3,3)

#create gaussian kernel
kernelXY = 3
gaussain_kernel = create_gaussian(kernelXY)

#convolve the image with the filter/kernel
blurred_img = convolve(img,gaussain_kernel)
   
#Sharpen Image
high_freq_detail = img - blurred_img        
new_image = img + high_freq_detail   

#display the results
plt.figure(1)
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(blurred_img, cmap='gray')
plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(new_image, cmap='gray')
plt.title('Sharpened Image'), plt.xticks([]), plt.yticks([])
plt.show()