import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

np.seterr(divide='ignore')


def readData(directory):
    pos = np.array([])
    created = False
    indices = np.array([])

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
        indices = np.append(indices, int(pos.shape[1]))

    return pos, indices


def readDataInterpolation(directory):
    pos = np.array([])
    created = False
    indices = np.array([])

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
        indices = np.append(indices, int(pos.shape[1]))

    return pos, indices


def heatmap(d1, d2, posint, m, bins=(250, 750), smoothing=8.0, cmap='jet'):
    x1 = list(d1[0, :])
    y1 = list(d1[1, :])
    x2 = list(d2[0, :])
    y2 = list(d2[1, :])
    xint = list(posint[0, :])
    yint = list(posint[1, :])

    heatmap1, xedges1, yedges1 = np.histogram2d(y1, x1, bins=bins)
    heatmap2, xedges2, yedges2 = np.histogram2d(y2, x2, bins=bins)
    heatmapint, xedgesint, yedgesint = np.histogram2d(yint, xint, bins=bins)

    heatmapint[heatmapint == 0] = 1

    heatmap1 = heatmap1 / heatmapint
    heatmap2 = heatmap2 / heatmapint
    extent = [-180, 180, 90, -90]

    logheatmap1 = heatmap1
    logheatmap1[np.isneginf(logheatmap1)] = 0
    # logheatmap1 = ndimage.gaussian_filter(logheatmap1, smoothing, mode='nearest')

    logheatmap2 = heatmap2
    logheatmap2[np.isneginf(logheatmap2)] = 0
    # logheatmap2 = ndimage.gaussian_filter(logheatmap2, smoothing, mode='nearest')

    max = np.max(np.vstack((logheatmap1, logheatmap2)))
    min = np.min(np.vstack((logheatmap1, logheatmap2)))

    # print(logheatmap1)
    # print(logheatmap2)

    # average data losses primary receiver: 0.52497319
    # average data losses redundant receiver: 0.38309897

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
        img.save(f'Figures each day/map{i + 1}{m+1}.png', 'PNG')

    for i in range(2):
        world.plot(color='white', edgecolor='black', linewidth=0.3, zorder=0)
        if i == 0:
            plt.imshow(logheatmap1, cmap=cmap, vmin=min,
                       vmax=max, extent=extent, alpha=0.55)
        elif i == 1:
            plt.imshow(logheatmap2, cmap=cmap, vmin=min,
                       vmax=max, extent=extent, alpha=0.55)

        cbar = plt.colorbar()
        cbar.set_label('Number of tracking losses per pass')
        plt.gca().invert_yaxis()
        plt.xlabel('Longitude [deg]')
        plt.ylabel('Latitude [deg]')
        plt.savefig(f'Figures each day/figure{i + 1}{m+1}.png', dpi=1000)
        # plt.show()


pos1, ind1 = readData("Data losses")
pos2, ind2 = readData("Data losses redundant")
posint, indint = readDataInterpolation("Interpolation results")
ind1, ind2, indint = np.insert(ind1, 0, 0), np.insert(
    ind2, 0, 0), np.insert(indint, 0, 0)
print(pos1)
res = 90
bins = (res, 2 * res)
for m in range(len(ind1)-1):
    print(ind1)
    pos1a, pos2a, posinta = pos1[:, int(ind1[m]):int(ind1[m+1])], pos2[:, int(
        ind2[m]):int(ind2[m+1])], posint[:, int(indint[m]):int(indint[m+1])]
    print(pos1a, pos2a, posinta)
    heatmap(pos1a, pos2a, posinta, m=m, bins=bins, smoothing=0.0)
