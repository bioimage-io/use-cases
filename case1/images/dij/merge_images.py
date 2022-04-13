import imageio
import numpy as np

i1 = imageio.imread("./input.png")
i2 = imageio.imread("./output.png")

print(i1.shape)
print(i2.shape)

im = i1.copy()
n, m = i1.shape[:-1]
indices = np.triu_indices(n=n, m=m, k=-30)
mask = np.zeros((n, m), dtype="bool")
mask[indices] = 1
mask = np.fliplr(mask)
im[mask] = i2[mask]

imageio.imwrite("./inout-fused.png", im)

# import napari
# napari.view_image(im)
# napari.run()
