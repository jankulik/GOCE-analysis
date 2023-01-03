from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from PIL import Image

size = (480, 240)
img = Image.open('heatmap.png')
img = img.resize(size, Image.ANTIALIAS)
img = img.transpose(method=Image.FLIP_TOP_BOTTOM)
img.save('heatmap2.png', 'PNG')

step = 2
angle = 360
for i in range(int(angle / step)):
    map = Basemap(projection='ortho', lat_0=20, lon_0=(i * step))
    map.warpimage('heatmap2.png')
    map.drawcoastlines()
    plt.savefig(f'Figures/world_{i}.png', dpi=150)
    plt.show()
    plt.close()

    print(f'Progress: {round(i * step / angle * 100, 2)}%')
