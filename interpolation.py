import numpy as np
import itertools
import matplotlib.pyplot as plt


# linear interpolation every 1 s
def interpolation(positions):
    ipositions = np.zeros((10 * (len(positions) - 1) + 1, 3))

    for i in range(len(positions) - 1):
        ipositions[10 * i, :] = positions[i, :]

        slope = (positions[i + 1, :] - positions[i, :]) / 10
        # print(f'slope: {slope}, i: {i}')

        for j in range(1, 10):
            ipositions[10 * i + j, :] = positions[i, :] + slope * j
            # print(f'slope: {slope}, i: {i}, j: {j}')
            # print(f'positions: {ipositions[i + j, :]}')

    ipositions[10 * (len(positions) - 1), :] = positions[len(positions) - 1, :]

    return ipositions


def interpolation2(positions, velocities):
    ipositions = np.zeros((10 * (len(positions) - 1) + 1, 3))

    A = np.array([[0, 0, 0, 1], [1000, 100, 10, 1],
                 [0, 0, 1, 0], [300, 20, 1, 0]])
    Ainv = np.linalg.inv(
        np.array([[0, 0, 0, 1], [1000, 100, 10, 1], [0, 0, 1, 0], [300, 20, 1, 0]]))

    for i in range(len(positions) - 1):
        ipositions[10 * i, :] = positions[i, :]

        f = np.array([positions[i, :], positions[i + 1, :],
                     velocities[i, :], velocities[i + 1, :]])
        a = np.matmul(Ainv, f)

        for j in range(1, 10):
            t = np.array([j ** 3, j ** 2, j, 1])
            ipositions[10 * i + j, :] = np.matmul(t, a)

    ipositions[10 * (len(positions) - 1), :] = positions[len(positions) - 1, :]

    return ipositions


with open('GO_CONS_SST_PRD_2I_20130901T205944_20130903T025943_0001.IDF') as f_in:
    pos = np.genfromtxt(itertools.islice(f_in, 2, None, 3),
                        dtype=float, skip_header=7, skip_footer=0, autostrip=True)
with open('GO_CONS_SST_PRD_2I_20130901T205944_20130903T025943_0001.IDF') as f_in:
    vel = np.genfromtxt(itertools.islice(f_in, 0, None, 3),
                        dtype=float, skip_header=8, skip_footer=0, autostrip=True)
vel = vel / 10000

pos2 = pos[:10, 1:4]
vel2 = vel[:10, 1:4]

ipos1 = interpolation(pos2)
ipos2 = interpolation2(pos2, vel2)

x = np.arange(0, 91, 1)
plt.plot(x, ipos1[:, 0])
plt.plot(x, ipos2[:, 0])
plt.show()

print((ipos2 - ipos1) * 1000)
