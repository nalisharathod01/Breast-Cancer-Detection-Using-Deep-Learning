import os
import numpy as np
from PIL import Image
from PIL.ImageOps import grayscale
from numpy.fft import fft2 as fft2
from numpy.fft import fftshift as fftshift
from numpy import real as real
import matplotlib.pyplot as plt

print("Starting image conversion")


#os.mkdir("gray_ft_images")
#os.mkdir(os.path.join("gray_ft_images", "benign"))
#os.mkdir(os.path.join("gray_ft_images", "malignant"))

for root, dirs, files in os.walk("images/"):
    for file in files:
        img = Image.open(os.path.join(root, file)).convert('L')
        img = np.asarray(img.getdata()).reshape(img.size)

        #img = img.resize((224, 224))

        fft = fft2(img)
        plt.imshow(np.abs(fftshift(fft))[125:350, 150:450], interpolation='nearest')
        plt.show()

        #o_image.show()
        #data = np.asarray(image)
        #new = np.empty(shape=np.shape(data))
        #new = fftshift(fft2(data))
        #img = Image.fromarray(np.uint8(image))
        #img.show()
        exit(0)

        if 'benign' in root:
            img.save(os.path.join("gray_ft_images", "benign", file))
        else:
            img.save(os.path.join("gray_ft_images", "malignant", file))


print("Completed")
