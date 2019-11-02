import numpy as np
import matplotlib.pyplot as plt

from skimage import data, io
from skimage.transform import pyramid_gaussian


image = io.imread("../test/0069.jpg")
rows, cols, dim = image.shape
pyramid = tuple(pyramid_gaussian(image, downscale=np.sqrt(2), max_layer=5, multichannel=True))

composite_image = np.zeros((rows, cols + cols + cols, 3), dtype=np.double)

composite_image[:rows, :cols, :] = pyramid[0]

i_col = cols
for p in pyramid[1:]:
    n_rows, n_cols = p.shape[:2]
    print(n_rows, n_cols)
    composite_image[0:n_rows, i_col:i_col + n_cols] = p
    i_col += n_cols

fig, ax = plt.subplots()
ax.imshow(composite_image)
plt.show()