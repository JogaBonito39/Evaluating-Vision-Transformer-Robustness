import numpy as np
import matplotlib.pyplot as plt

img_array = np.load('CIFAR-10-C/brightness.npy')

# fig, axes = plt.subplots(3, 3, figsize=(8, 8))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(img_array[i], interpolation='nearest')
#     ax.axis('off')

# plt.tight_layout()
# plt.show()


single_img = img_array[1]

plt.figure(figsize=(4, 4))
plt.imshow(single_img, interpolation='nearest')
plt.axis('off')
plt.show()