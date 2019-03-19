from skimage import io, util, color, data
import matplotlib.pyplot as plt

import numpy as np

import scipy.ndimage

image = io.imread("train-data/train/pos/00001.png", as_gray=True)

imagef = util.img_as_float(image)

image_camera = data.camera()
imagef_camera = util.img_as_float(image_camera)

# plt.show(io.imshow(imagef))

Gx = np.array([[-1, 0, 1]])

Gy = np.transpose(Gx)

line_conv = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
])

col_conv = np.array([
    [-1,-1,-1],
    [0,0,0],
    [1,1,1]
])

image_gx = scipy.ndimage.filters.convolve(imagef_camera, Gx)
image_gy = scipy.ndimage.filters.convolve(imagef_camera, Gy)

# plt.show(io.imshow(image_gx))

# plt.show(io.imshow(image_gy))

#image_line = scipy.ndimage.filters.convolve(imagef_camera, line_conv)
#plt.show(io.imshow(image_line))

#image_col = scipy.ndimage.filters.convolve(imagef_camera, col_conv)
#plt.show(io.imshow(image_col))

# plt.show(io.imshow(image_camera))

import math

image_gn = np.sqrt(image_gx ** 2 + image_gy ** 2)

image_gg = np.arctan2(image_gx, image_gy)


plt.figure()
plt.subplot(2, 2, 1)
io.imshow(image_gx)
plt.subplot(2, 2, 2)
io.imshow(image_gy)
plt.subplot(2, 2, 3)
io.imshow(image_gn)
plt.subplot(2, 2, 4)
io.imshow(image_gg * imagef_camera)
plt.show()

"""
normes = np.zeros(np.shape([len(imagef_camera) - 2, len(imagef_camera[0]) - 2]))
orientations = np.zeros(np.shape([len(imagef_camera) - 2, len(imagef_camera[0]) - 2]))

for xi in range(1, len(imagef_camera) - 1):
    for yi in range(1, len(imagef_camera[xi]) - 1):
        norme = math.sqrt(
            math.pow(imagef_camera[xi + 1][yi] - imagef_camera[xi-1][yi], 2) + 
            math.pow(imagef_camera[xi][yi + 1] - imagef_camera[xi][yi - 1], 2)
        )

        normes[xi-1][yi-1] = norme

        orientation = math.atan2(
            imagef_camera[xi + 1][yi] - imagef_camera[xi-1][yi],
            imagef_camera[xi][yi + 1] - imagef_camera[xi][yi - 1]
        )

        orientations[xi-1][yi-1] = orientation
"""
