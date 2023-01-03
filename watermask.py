import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import ndimage


def spiral(X, Y):
    array = list([])
    x = y = 0
    dx = 0
    dy = -1
    for i in range(max(X, Y) ** 2):
        if (-X / 2 < x <= X / 2) and (-Y / 2 < y <= Y / 2):
            array.append(tuple((x, y)))

        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1 - y):
            dx, dy = -dy, dx
        x, y = x + dx, y + dy
    return array


N = 120
image = Image.open('SAA.jpeg')
color = Image.open('colorbar.jpeg')
color = color.resize((N, color.height))
colorbar = np.asarray(color)[0, :, :]
colorbar = colorbar.astype(np.uint64)
# print(colorbar.shape)
values_colorbar = np.linspace(22000 / 78000, 78000 / 78000, N)

mn_array = spiral(21, 21)

plt.figure()
data = np.asarray(image)
plt.imshow(data)
plt.draw()

threshold = 160

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if np.average(data[i, j]) < threshold:
            smoothing = True
            count = 0
            while smoothing:
                count += 1
                m, n = mn_array[count]
                if 0 <= i + m < data.shape[0] and 0 <= j + n < data.shape[1]:
                    if np.average(data[i + m, j + n]) >= threshold:
                        data[i, j] = data[i + m, j + n]
                        smoothing = False

                if count >= len(mn_array) - 1:
                    print(f'Smoothing failed for point {i, j}')
                    smoothing = False

smoothing_value = 7
data[:, :, 0] = ndimage.gaussian_filter(
    data[:, :, 0], smoothing_value, mode='nearest')
data[:, :, 1] = ndimage.gaussian_filter(
    data[:, :, 1], smoothing_value, mode='nearest')
data[:, :, 2] = ndimage.gaussian_filter(
    data[:, :, 2], smoothing_value, mode='nearest')

plt.figure()
plt.imshow(data)
plt.draw()
plt.show()

data = data.astype(np.uint64)
mask = np.zeros(data.shape)
print(data.dtype)
print(colorbar.dtype)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        error_min = 1e9
        value = 0
        for k in range(N):
            error = np.sqrt(np.mean(np.square(data[i, j] - colorbar[k, :])))

            if error < error_min:
                value = 1 - values_colorbar[k]
                error_min = error

        mask[i, j] = value

print(mask.shape)
mask = mask * np.array([1, 1, 1])
print(mask.shape)
print(mask)
plt.imshow(mask)
plt.imsave('SAAmask.png', mask)
plt.show()

saved_mask = Image.open('SAAmask.png')
saved_mask = saved_mask.resize((360, 180))
saved_mask.save('SAAmask.png')
