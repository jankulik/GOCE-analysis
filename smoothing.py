import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import os
from PIL import Image

np.seterr(divide='ignore')


def readData(directory):
    pos = np.array([])
    created = False

    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        ipos = np.genfromtxt(file, skip_header=1, delimiter=',')
        ipos = ipos.transpose()
        ipos = ipos[1:3, :]

        if not created:
            pos = ipos
            created = True
        else:
            pos = np.hstack((pos, ipos))

    return pos


def heatmap2(d, bins=(250, 750), smoothing=1.3, cmap='jet', index=1):
    x = list(np.hstack((d[0, :] - 360, d[0, :], d[0, :] + 360)))
    y = list(np.hstack((d[1, :], d[1, :], d[1, :])))

    heatmap, xedges, yedges = np.histogram2d(y, x, bins=bins)
    extent = [-180, 180, 90, -90]

    logheatmap = np.log(heatmap)
    logheatmap[np.isneginf(logheatmap)] = 0
    logheatmap = ndimage.gaussian_filter(logheatmap, smoothing, mode='nearest')
    logheatmap = logheatmap[:, bins[0]:2 * bins[0]]

    print(np.max(logheatmap))
    print(np.min(logheatmap))

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.plot(color='white', edgecolor='black', linewidth=0.3, zorder=0)
    plt.imshow(logheatmap, cmap=cmap, vmin=0,
               vmax=4.86, extent=extent, alpha=0.55)
    # plt.imsave('heatmap.png', logheatmap, cmap=cmap, dpi=1000)
    plt.axis('off')
    plt.gca().invert_yaxis()

    plt.savefig(f'map{index}.png', bbox_inches='tight', pad_inches=0, dpi=1000)

    size = (4096, 2048)
    img = Image.open(f'map{index}.png')
    img = img.resize(size, Image.ANTIALIAS)
    img.save(f'map{index}.png', 'PNG')


def heatmap(d1, d2, bins=(250, 750), smoothing=8.0, cmap='jet'):
    x1 = list(np.hstack((d1[0, :] - 360, d1[0, :], d1[0, :] + 360)))
    y1 = list(np.hstack((d1[1, :], d1[1, :], d1[1, :])))
    x2 = list(np.hstack((d2[0, :] - 360, d2[0, :], d2[0, :] + 360)))
    y2 = list(np.hstack((d2[1, :], d2[1, :], d2[1, :])))

    heatmap1, xedges1, yedges1 = np.histogram2d(y1, x1, bins=bins)
    heatmap2, xedges2, yedges2 = np.histogram2d(y2, x2, bins=bins)
    extent = [-180, 180, 90, -90]

    logheatmap1 = np.log(heatmap1)
    logheatmap1[np.isneginf(logheatmap1)] = 0
    logheatmap1 = ndimage.gaussian_filter(
        logheatmap1, smoothing, mode='nearest')
    logheatmap1 = logheatmap1[:, bins[0]:2 * bins[0]]

    logheatmap2 = np.log(heatmap2)
    logheatmap2[np.isneginf(logheatmap2)] = 0
    logheatmap2 = ndimage.gaussian_filter(
        logheatmap2, smoothing, mode='nearest')
    logheatmap2 = logheatmap2[:, bins[0]:2 * bins[0]]

    max = np.max(np.vstack((logheatmap1, logheatmap2)))
    min = np.min(np.vstack((logheatmap1, logheatmap2)))

    print(max)
    print(min)

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    for i in range(2):
        world.plot(color='white', edgecolor='black', linewidth=0.3, zorder=0)
        if i == 0:
            plt.imshow(logheatmap1, cmap=cmap, vmin=min,
                       vmax=max, extent=extent, alpha=0.55)
        elif i == 1:
            plt.imshow(logheatmap2, cmap=cmap, vmin=min,
                       vmax=max, extent=extent, alpha=0.55)

        plt.axis('off')
        plt.gca().invert_yaxis()
        plt.savefig(f'map{i + 1}.png', bbox_inches='tight',
                    pad_inches=0, dpi=1000)

        size = (4096, 2048)
        img = Image.open(f'map{i + 1}.png')
        img = img.resize(size, Image.ANTIALIAS)
        img.save(f'map{i + 1}.png', 'PNG')


pos1 = readData("Data losses")
pos2 = readData("Data losses redundant")

res = 350
bins = (res, 3 * res)
heatmap(pos1, pos2, bins)
