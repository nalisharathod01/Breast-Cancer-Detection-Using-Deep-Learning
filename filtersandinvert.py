import os
import numpy as np
from PIL import Image
from PIL.ImageOps import grayscale
from numpy.fft import fft2 as fft2
from numpy.fft import fftshift as fftshift
from numpy import real as real
import matplotlib.pyplot as plt
from scipy import ndimage

print("Starting image conversion")


b = 'benign'
m = 'malignant'
kernelx3 = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])
kernelx5 = np.array([[-1, -1, -1, -1, -1],
                   [-1,  1,  2,  1, -1],
                   [-1,  2,  4,  2, -1],
                   [-1,  1,  2,  1, -1],
                   [-1, -1, -1, -1, -1]])

dirs = ['filterx5', 'filterx3', 'filter_gaussian']
for dir in dirs:
    os.mkdir(dir)
    os.mkdir(os.path.join(dir, 'benign'))
    os.mkdir(os.path.join(dir, 'malignant'))


for root, dirs, files in os.walk("images/"):
    for file in files:
        img = Image.open(os.path.join(root, file)).convert('L')

        high_pass_3 = ndimage.convolve(img, kernelx3)
        high_pass_3 = Image.fromarray(np.uint8(high_pass_3))
        if 'benign' in root:
            high_pass_3.save(os.path.join("filterx3", "benign", file))
        else:
            high_pass_3.save(os.path.join("filterx3", "malignant", file))

        high_pass_5 = ndimage.convolve(img, kernelx5)
        high_pass_5 = Image.fromarray(np.uint8(high_pass_5))
        if 'benign' in root:
            high_pass_5.save(os.path.join("filterx5", "benign", file))
        else:
            high_pass_5.save(os.path.join("filterx5", "malignant", file))

        high_pass_guass = img - ndimage.gaussian_filter(img, 3)
        high_pass_guass = Image.fromarray(np.uint8(high_pass_guass))
        if 'benign' in root:
            high_pass_guass.save(os.path.join("filter_gaussian", "benign", file))
        else:
            high_pass_guass.save(os.path.join("filter_gaussian", "malignant", file))


print("Completed")
