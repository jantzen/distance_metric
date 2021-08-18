# file: generate_lvmap_figure.py

import numpy as np
import matplotlib.pyplot as plt
from eugene.src.tools.dmat_to_color_map import dmat_to_color_map
import os

data = np.load('./data/dmat.npy')
if os.path.isfile('./data/lv_color_map.npy'):
    print("Loading existing color map data...")
    color_map = np.load('./data/lv_color_map.npy')
else:
    color_map = dmat_to_color_map(data, order='F')
    np.save('./data/lv_color_map', color_map)
color_map = np.rot90(color_map)
fig, ax = plt.subplots()
ax.set_xlim([0., 1.2])
ax.set_ylim([-0.2, 1.2])
ax.set_aspect('equal')
ax.set_xlabel(r'$\alpha$', fontsize=10)
ax.set_ylabel(r'$\beta$', fontsize=10)
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.imshow(color_map, cmap='viridis', extent=(0.0, 1.2, -0.2, 1.2))
plt.savefig('lvmap.pdf', dpi=600)
plt.savefig('lvmap.eps', dpi=600)
