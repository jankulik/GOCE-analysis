import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import os
from PIL import Image

np.seterr(divide='ignore')


def readData(directory):
    pos = np.array([])
    created_L1 = False
    created_L2 = False

    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        ipos = np.genfromtxt(file, skip_header=1, delimiter=',')
        ipos = ipos.transpose()
        ipos = ipos[1:3, :]
        if file[-7:] == '_L1.csv':
            if not created_L1:
                pos_L1 = ipos
                created_L1 = True
            else:
                pos_L1 = np.hstack((pos_L1, ipos))
        elif file[-7:] == '_L2.csv':
            if not created_L2:
                pos_L2 = ipos
                created_L2 = True
            else:
                pos_L2 = np.hstack((pos_L2, ipos))
        else:
            print('Error could not identify frequency')

    return pos_L1, pos_L2


def readDataInterpolation(directory):
    pos = np.array([])
    created = False

    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        ipos = np.genfromtxt(file, skip_header=0, delimiter=',')
        ipos = ipos.transpose()
        ipos = ipos[0:2, 10800:97201]

        if not created:
            pos = ipos
            created = True
        else:
            pos = np.hstack((pos, ipos))

    return pos


def heatmap(d1, d1_L2, d2, d2_L2, posint, bins=(250, 750), smoothing=8.0, cmap='jet'):
    x1 = list(d1[0, :])
    y1 = list(d1[1, :])
    x2 = list(d2[0, :])
    y2 = list(d2[1, :])
    x1_L2 = list(d1_L2[0, :])
    y1_L2 = list(d1_L2[1, :])
    x2_L2 = list(d2_L2[0, :])
    y2_L2 = list(d2_L2[1, :])
    xint = list(posint[0, :])
    yint = list(posint[1, :])

    heatmap1, xedges1, yedges1 = np.histogram2d(y1, x1, bins=bins)
    heatmap2, xedges2, yedges2 = np.histogram2d(y2, x2, bins=bins)
    heatmap1_L2, xedges1_L2, yedges1_L2 = np.histogram2d(
        y1_L2, x1_L2, bins=bins)
    heatmap2_L2, xedges2_L2, yedges2_L2 = np.histogram2d(
        y2_L2, x2_L2, bins=bins)
    heatmapint, xedgesint, yedgesint = np.histogram2d(yint, xint, bins=bins)

    heatmapint[heatmapint == 0] = 1

    heatmap1 = heatmap1 / heatmapint
    heatmap2 = heatmap2 / heatmapint
    heatmap1_L2 = heatmap1_L2 / heatmapint
    heatmap2_L2 = heatmap2_L2 / heatmapint
    extent = [-180, 180, 90, -90]

    logheatmap1 = heatmap1
    logheatmap1[np.isneginf(logheatmap1)] = 0
    logheatmap1 = ndimage.gaussian_filter(
        logheatmap1, smoothing, mode='nearest')

    logheatmap2 = heatmap2
    logheatmap2[np.isneginf(logheatmap2)] = 0
    logheatmap2 = ndimage.gaussian_filter(
        logheatmap2, smoothing, mode='nearest')

    logheatmap1_L2 = heatmap1_L2
    logheatmap1_L2[np.isneginf(logheatmap1_L2)] = 0
    logheatmap1_L2 = ndimage.gaussian_filter(
        logheatmap1_L2, smoothing, mode='nearest')

    logheatmap2_L2 = heatmap2_L2
    logheatmap2_L2[np.isneginf(logheatmap2_L2)] = 0
    logheatmap2_L2 = ndimage.gaussian_filter(
        logheatmap2_L2, smoothing, mode='nearest')

    max = np.max(
        np.vstack((logheatmap1, logheatmap2, logheatmap1_L2, logheatmap2_L2)))
    min = np.min(
        np.vstack((logheatmap1, logheatmap2, logheatmap1_L2, logheatmap2_L2)))

    # print(logheatmap1)
    # print(logheatmap2)

    # average data losses primary receiver: 0.52497319
    # average data losses redundant receiver: 0.38309897

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    filenames = ['_nom_L1', '_red_L1', '_nom_L2', '_red_L2']
    for i in range(4):
        world.plot(color='white', edgecolor='black', linewidth=0.3, zorder=0)
        if i == 0:
            plt.imshow(logheatmap1, cmap=cmap, vmin=min,
                       vmax=max, extent=extent, alpha=0.55)
        elif i == 1:
            plt.imshow(logheatmap2, cmap=cmap, vmin=min,
                       vmax=max, extent=extent, alpha=0.55)
        elif i == 2:
            plt.imshow(logheatmap1_L2, cmap=cmap, vmin=min,
                       vmax=max, extent=extent, alpha=0.55)
        elif i == 3:
            plt.imshow(logheatmap2_L2, cmap=cmap, vmin=min,
                       vmax=max, extent=extent, alpha=0.55)

        plt.axis('off')
        plt.gca().invert_yaxis()
        plt.savefig(f'map{filenames[i]}.png',
                    bbox_inches='tight', pad_inches=0, dpi=1000)

        size = (4096, 2048)
        img = Image.open(f'map{filenames[i]}.png')
        img = img.resize(size, Image.ANTIALIAS)
        img.save(f'map{filenames[i]}.png', 'PNG')

    for i in range(4):
        world.plot(color='white', edgecolor='black', linewidth=0.3, zorder=0)
        if i == 0:
            plt.imshow(logheatmap1, cmap=cmap, vmin=min,
                       vmax=max, extent=extent, alpha=0.55)
        elif i == 1:
            plt.imshow(logheatmap2, cmap=cmap, vmin=min,
                       vmax=max, extent=extent, alpha=0.55)
        elif i == 2:
            plt.imshow(logheatmap1_L2, cmap=cmap, vmin=min,
                       vmax=max, extent=extent, alpha=0.55)
        elif i == 3:
            plt.imshow(logheatmap2_L2, cmap=cmap, vmin=min,
                       vmax=max, extent=extent, alpha=0.55)

        cbar = plt.colorbar()
        cbar.set_label('Number of tracking losses per pass')
        plt.gca().invert_yaxis()
        plt.xlabel('Longitude [deg]')
        plt.ylabel('Latitude [deg]')
        plt.savefig(f'figure{filenames[i]}.png', dpi=1000)
        plt.show()


pos1_L1, pos1_L2 = readData("Data losses")
pos2_L1, pos2_L2 = readData("Data losses redundant")
posint = readDataInterpolation("Interpolation results")

res = 180
bins = (res, 2 * res)
heatmap(pos1_L1, pos1_L2, pos2_L1, pos2_L2, posint, bins, smoothing=0.0)
