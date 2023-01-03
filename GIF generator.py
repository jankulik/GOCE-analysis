import imageio
import os
import re


def natural_sort(l):
    def convert(text): return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


images = []
for filename in natural_sort(os.listdir('Figures')):
    if not filename.startswith('.'):
        images.append(imageio.imread(f'Figures/{filename}'))
imageio.mimsave('world.gif', images, duration=0.07)
