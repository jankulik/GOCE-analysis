import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import os
from PIL import Image
import PIL
from scipy.stats.mstats import gmean
from xarray import DataArray

np.seterr(divide='ignore')


def area_grid(lat, lon):
    from numpy import meshgrid, deg2rad, gradient, cos

    xlon, ylat = meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=0))
    dlon = deg2rad(gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    area = dy * dx

    xda = DataArray(
        area,
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda


def earth_radius(lat):
    from numpy import deg2rad, sin, cos

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b ** 2 / a ** 2)

    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = deg2rad(lat)
    lat_gc = np.arctan((1 - e2) * np.tan(lat))

    # radius equation
    # see equation 3-107 in WGS84
    r = ((a * (1 - e2) ** 0.5) / (1 - (e2 * np.cos(lat_gc) ** 2)) ** 0.5)

    return r


grid_cell_area = area_grid(np.arange(-90, 90, 1), np.arange(-180, 180, 1))
plt.matshow(grid_cell_area)
plt.show()


def readWatermask(img_name):
    img = PIL.Image.open(img_name)
    image_array = np.array(img)
    image_array = image_array[:, :, 0]
    # print(image_array)
    image_array[np.where(image_array < 128)] = 0
    image_array[np.where(image_array >= 128)] = 1
    # print(image_array)
    return image_array


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


def heatmap(d1, d1_L2, d2, d2_L2, posint, bins=(250, 750), smoothing=8.0, cmap='jet', grid_cell_area=grid_cell_area):
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
    heatmap1_L2 = (heatmap1_L2 + heatmap1) / heatmapint
    heatmap2_L2 = (heatmap2_L2 + heatmap2) / heatmapint

    heatmap1_summed = np.sum(heatmap1) + np.sum(heatmap1_L2)
    heatmap2_summed = np.sum(heatmap2) + np.sum(heatmap2_L2)
    print(heatmap1_summed / np.sum(heatmapint))
    print(heatmap2_summed / np.sum(heatmapint))
    print((heatmap2_summed / np.sum(heatmapint) - heatmap1_summed / np.sum(heatmapint)) / (
        heatmap1_summed / np.sum(heatmapint)))
    print('---')
    print(np.sum(heatmap1_L2) / np.sum(heatmapint))
    print(np.sum(heatmap2_L2) / np.sum(heatmapint))

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
    heatmap1_max, heatmap2_max, heatmap1_L2_max, heatmap2_L2_max = np.average(heatmap1), np.average(
        heatmap2), np.average(heatmap1_L2), np.average(heatmap2_L2)

    print(
        f"Primary L1 average is {np.round(heatmap1_max, 4)} \ "
        f"Redundant L1 average is {np.round(heatmap2_max, 4)} \ "
        f"Primary L2 average is {np.round(heatmap1_L2_max, 4)} \ "
        f"Redundant L2 average is {np.round(heatmap2_L2_max, 4)}"
    )

    heatmap1_gmax, heatmap2_gmax, heatmap1_L2_gmax, heatmap2_L2_gmax = gmean(
        heatmap1.flatten()[np.where(heatmap1.flatten() != 0)]), gmean(
        heatmap2.flatten()[np.where(heatmap2.flatten() != 0)]), gmean(
        heatmap1_L2.flatten()[np.where(heatmap1_L2.flatten() != 0)]), gmean(
        heatmap2_L2.flatten()[np.where(heatmap2_L2.flatten() != 0)])
    print(
        f"Primary L1 average is {np.round(heatmap1_gmax, 4)} \ "
        f"Redundant L1 average is {np.round(heatmap2_gmax, 4)} \ "
        f"Primary L2 average is {np.round(heatmap1_L2_gmax, 4)} \ "
        f"Redundant L2 average is {np.round(heatmap2_L2_gmax, 4)}"
    )
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
        cbar.set_label('Number of ITLs per pass')
        plt.gca().invert_yaxis()
        plt.xlabel('Longitude [deg]')
        plt.ylabel('Latitude [deg]')
        plt.savefig(f'figure{filenames[i]}.png', dpi=1000)
        plt.show()

    return heatmap1, heatmap2, heatmap1_L2, heatmap2_L2, heatmapint


def ocean_counter(heatmapav, heatmap, heatmapint, ocean_index, land_index):
    heatmapav.append(
        (np.average((heatmap/heatmapint)[ocean_index]) / np.average((heatmap/heatmapint)[land_index])))

    return heatmapav


pos1_L1, pos1_L2 = readData("Data losses")
pos2_L1, pos2_L2 = readData("Data losses redundant")
posint = readDataInterpolation("Interpolation results")

res = 180
bins = (res, 2 * res)
heatmap1_orig, heatmap2_orig, heatmap1_L2_orig, heatmap2_L2_orig, heatmapint = heatmap(pos1_L1, pos1_L2, pos2_L1,
                                                                                       pos2_L2, posint,
                                                                                       bins, smoothing=0.0,
                                                                                       grid_cell_area=grid_cell_area)

watermask_orig = readWatermask('Watermask.png')
print(watermask_orig)
watermask_excl = readWatermask('Watermask_exclusion.png')
plt.matshow(watermask_excl)
plt.show()

heatmap1av, heatmap2av, heatmap1l2av, heatmap2l2av, heatmapmergav = [], [], [], [], []
grid_cell_area = DataArray.to_numpy(grid_cell_area)
for i in range(40):
    heatmap1, heatmap2, heatmap1_L2, heatmap2_L2 = heatmap1_orig[i:(180 - i), :], \
        heatmap2_orig[i:(180 - i), :], heatmap1_L2_orig[i:(180 - i), :], \
        heatmap2_L2_orig[i:(180 - i), :]
    heatmap_merg = heatmap1_orig[i:(180 - i), :] + heatmap2_orig[i:(180 - i), :] \
        + heatmap1_L2_orig[i:(180 - i), :] + heatmap2_L2_orig[i:(180 - i), :]
    watermask = watermask_orig[i:(180 - i), :]
    watermaske = watermask_excl[i:(180 - i), :]
    grid_cell_area_used = grid_cell_area[i:(180 - i), :]
    ocean_index = np.where(watermask == 0)
    land_index = np.where(watermask == 1)
    # ocean_index = np.where(np.logical_and(watermask == 0, watermaske == 0))
    # land_index = np.where(np.logical_and(watermask == 1, watermaske == 0))
    heatmapint_new = heatmapint[i:(180 - i), :]
    heatmap1av = ocean_counter(
        heatmap1av, heatmap1, heatmapint_new, ocean_index, land_index)
    heatmap2av = ocean_counter(
        heatmap2av, heatmap2, heatmapint_new, ocean_index, land_index)
    heatmap1l2av = ocean_counter(
        heatmap1l2av, heatmap1_L2, heatmapint_new, ocean_index, land_index)
    heatmap2l2av = ocean_counter(
        heatmap2l2av, heatmap2_L2, heatmapint_new, ocean_index, land_index)
    heatmapmergav = ocean_counter(
        heatmapmergav, heatmap_merg, heatmapint_new, ocean_index, land_index)


plt.clf()
plt.xlim(90, 50)
plt.ylim(0.6, 2.0)
plt.xlabel('Latitude range (symmetrical around the equator)')
plt.ylabel('Ratio of frequency of LoLs over the oceans and land')
plt.plot(list(range(90, 50, -1)), heatmap1av, label="Nominal L1")
plt.plot(list(range(90, 50, -1)), heatmap2av, label="Redundant L1")
plt.plot(list(range(90, 50, -1)), heatmap1l2av, label="Nominal L2")
plt.plot(list(range(90, 50, -1)), heatmap2l2av, label="Redundant L2")
# plt.plot(list(range(90, 50, -1)), heatmapmergav, label="Merged")
plt.plot(list(range(90, 50, -1)), np.ones(40), 'k--')
plt.legend()
plt.savefig('oceans.png', dpi=1000)
plt.show()
ocean_index = np.where(np.logical_and(
    watermask_orig == 0, watermask_excl == 0))
land_index = np.where(np.logical_and(watermask_orig == 1, watermask_excl == 0))
plt.hist(np.ndarray.flatten(
    heatmap1_L2_orig[ocean_index]-heatmap2_L2_orig[ocean_index]), bins=101)
print(
    f'average is {np.average(heatmap1_L2_orig[ocean_index]-heatmap2_L2_orig[ocean_index])}')
plt.show()
