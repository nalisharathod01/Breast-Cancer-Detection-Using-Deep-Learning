## First use the link in the readme file to download the tar.gz file, then move the file to the same directory as this project and run this file

# Dont forget to make sure you have already pip installed the libraries listed in the include section so there are no errors.

import os
import glob
import numpy as np
from PIL import Image
from numpy.fft import fft2 as fft2
from numpy.fft import fftshift as fftshift
from numpy import real as real
import shutil


import tarfile

file = tarfile.open("BreakHis_v1.tar.gz")
file.extractall("extracted")
file.close()

print("Tar extraction complete")
print("Starting image directory cleaning")

os.mkdir("images")
os.mkdir(os.path.join("images", "benign"))
os.mkdir(os.path.join("images", "malignant"))
for root, dirs, files in os.walk(os.path.join("extracted", "BreakHis_v1", "histology_slides", "breast")):
    for file in files:
        if file.endswith(".png"):
            if(file[4] == 'B'):
                os.replace(os.path.join(root, file), os.path.join("images", "benign", file))
            else:
                os.replace(os.path.join(root, file), os.path.join("images", "malignant", file))
print("Finished image directory cleaning")
                
chris_stuff = False   # For my portion with the fourier transform
if (chris_stuff):
    os.mkdir("ft_images")
    os.mkdir(os.path.join("ft_images", "benign"))
    os.mkdir(os.path.join("ft_images", "malignant"))

    for root, dirs, files in os.walk("images/"):
        for file in files:
            image = Image.open(os.path.join(root, file))
            data = np.asarray(image)
            new = np.empty(shape=np.shape(data))
            for i in range(3):
                new[:,:,i] = fftshift(fft2(data[:,:,i]))
            img = Image.fromarray(np.uint8(new))
            if 'benign' in root:
                img.save(os.path.join("ft_images", "benign", file))
            else:
                img.save(os.path.join("ft_images", "malignant", file))


shutil.rmtree("extracted", ignore_errors=True)
print("Removed original extracted folder")
os.remove("BreakHis_v1.tar.gz")
print("Removed original tar.gz file")
