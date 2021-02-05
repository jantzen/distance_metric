# file: generate_kuramoto_figure.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

fig, axs = plt.subplots(3, 1)

# (a) Sample of system trajectory

df = pd.read_csv("./data/kuramoto_data.csv")
tmp = df[:5000]
tmp.plot('time', ['x1','x2','x3','y1','y2','y3'],
        legend=False, ax=axs[0], cmap='Greys') 
axs[0].set(title='(a)', xlabel='time', ylabel='intensity')

# (b) Matrix profile
df.plot('time', ['x1','x2','x3','y1','y2','y3'],
        color='lightgrey', legend=False, ax=axs[1]) 
axs[1].set(title='(b)', xlabel='time', ylabel='output intensity')
axs[1].set_ylim([-2., 2.])
mp_pl = axs[1].twinx()
mp = np.loadtxt('./data/mp.txt')
for row in mp:
    mp_pl.plot(row, 'k-')

# (c) Dynamical distance scan 
df.plot('time', ['x1','x2','x3','y1','y2','y3'],
        color='lightgrey', legend=False, ax=axs[2]) 
axs[2].set(title='(c)', xlabel='time', ylabel='output intensity')
axs[2].set_ylim([-2., 2.])
ra = axs[2].twinx()
results = pd.read_csv('./data/results.csv')
results.plot('time', ['rolling ave'], ax=ra, style=['k-'], legend=False)
t_pos = np.loadtxt('./data/t_pos.txt')
plt.axhline(t_pos, color='b', linestyle='-.')
crossings = np.loadtxt('./data/crossings.txt')
crossings = crossings.astype('int32')
cross_times = results['time'].iloc[crossings]
if type(cross_times) is np.float64:
    plt.sca(axs[2])
    plt.axvline(x=cross_times, color='r', linestyle='--')
else:
    for x in cross_times.values:
        plt.sca(axs[2])
        plt.axvline(x=x, color='r', linestyle='--')

plt.savefig('change.pdf', dpi=600)
plt.show()
