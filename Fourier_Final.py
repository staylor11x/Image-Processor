import cv2
import numpy as np
from matplotlib import pyplot as plt

def CreateGuassianMask(grey_img, d):
    '''create a Gaussian mask'''

    rows, cols = grey_img.shape
    crow, ccol = rows // 2, cols // 2
    
    sigma = d / np.sqrt(2 * np.log(2))
    gaussian = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            gaussian[i, j] = np.exp(-((i - crow) ** 2 + (j - ccol) ** 2) / (2 * sigma ** 2))
    return gaussian

#read in img
image = 'cat.jpg'
grey_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

#fft image
img_fft = np.fft.fft2(grey_img)

#shift the zero-frequency components to the centre of the spectrum
fft_shift = np.fft.fftshift(img_fft)

#get filter mask
d = 30  #diameter of the filter
mask = CreateGuassianMask(grey_img, d)

#apply filter mask
filtered_img = fft_shift *mask

#this changes the filter to High-pass comment/uncomment as needed
high_frequency_detail = fft_shift - filtered_img
filtered_img = fft_shift + high_frequency_detail

#perform IFFT
fft_shift = np.fft.ifftshift(filtered_img)
img_back = np.fft.ifft2(fft_shift)

img_back = np.abs(img_back)

'''display the result'''
plt.figure(2)
plt.subplot(121), plt.imshow(grey_img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()


