# file: generate_kuramoto_figure.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

fig, axs = plt.subplots(3, 1, figsize=(3.,6.), sharex=True)
fig.tight_layout(rect=[0.05, 0.05, 0.9, 0.95], h_pad=3., w_pad=2.7)

# (a) Sample of system trajectory

df = pd.read_csv("./data/kuramoto_data.csv")

# (b) Matrix profile
#df.plot('time', ['x1','x2','x3','y1','y2','y3'],
#        color='lightgrey', legend=False, ax=axs[1]) 
time = df['time'].to_numpy()
x = df[['x1','x2','x3','y1','y2','y3']].to_numpy()
for ii in range(x.shape[1]):
    axs[0].plot(time, x[:,ii], color='lightgrey', alpha=0.3)
    axs[1].plot(time, x[:,ii], color='lightgrey', alpha=0.3)
    axs[2].plot(time, x[:,ii], color='lightgrey', alpha=0.3)
axs[0].set_title('(a)', weight='bold', fontsize=10)
axs[0].set_ylabel(r'$x_i$', color='grey', fontsize=8)
axs[0].spines['left'].set_color('grey')
axs[0].yaxis.label.set_color('grey')
axs[0].tick_params(axis='y', colors='grey', labelsize=6)
axs[0].set_ylim([-2., 2.])
axs[0].set_xlim([0., np.max(time)])
mp_pl = axs[0].twinx()
mp_pl.set_xlabel('time')
mp_pl.set_ylabel('distance', fontsize=8)
mp_pl.tick_params(axis='y', labelsize=6)
mp_pl.set_xlim([0., np.max(time)])
mp = np.loadtxt('./data/mp.txt')
for ii in range(mp.shape[0]):
    mp_pl.plot(time[:mp.shape[1]], mp[ii, :], color='black')
# sample of system trajectory as inset
ax_inset = axs[0].inset_axes([0.6,0.5,0.4,0.5])
tmp = df[:5000]
inset_time = tmp['time'].to_numpy()
colors = ['whitesmoke', 'lightgrey', 'silver', 'darkgrey', 'grey', 'dimgrey']
for ii, var in enumerate(['x1','x2','x3','y1','y2','y3']):
    x = tmp[var].to_numpy()
    ax_inset.plot(inset_time, x, color=colors[ii])
    ax_inset.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
#tmp.plot('time', ['x1','x2','x3','y1','y2','y3'],
#        legend=False, ax=axs[0], cmap='Greys') 
#ax_inset.set(title='(a)', xlabel='time', ylabel='intensity')

# (c) Dynamical distance scan 
#df.plot('time', ['x1','x2','x3','y1','y2','y3'],
#        color='lightgrey', legend=False, ax=axs[2]) 
#time = df['time'].to_numpy()
#x = df[['x1','x2','x3','y1','y2','y3']].to_numpy()
#for ii in range(x.shape[1]):
#    axs[2].plot(time, x[:,ii], color='lightgrey', alpha=0.3)
#axs[1].set(title='(b)', ylabel='output intensity')
#axs[1].set_ylim([-2., 2.])
#axs[1].set_xlim([0., np.max(time)])
axs[1].set_title('(b)', weight='bold', fontsize=10)
axs[1].set_ylabel(r'$x_i$', color='grey', fontsize=8)
axs[1].spines['left'].set_color('grey')
axs[1].yaxis.label.set_color('grey')
axs[1].tick_params(axis='y', colors='grey', labelsize=6)
axs[1].set_ylim([-2., 2.])
axs[1].set_xlim([0., np.max(time)])
ra = axs[1].twinx()
ra.set_xlabel('time')
ra.set_ylabel('distance', fontsize=8)
ra.tick_params(axis='y', labelsize=6)
ra.set_xlim([0., np.max(time)])
results = pd.read_csv('./data/results.csv')
results_ra = results['rolling ave'].to_numpy()
results_time = results['time'].to_numpy()
#results.plot('time', ['rolling ave'], ax=ra, style=['k-'], legend=False)
ra.plot(results_time, results_ra, 'k-')
t_pos = np.loadtxt('./data/t_pos.txt')
plt.axhline(t_pos, color='b', linestyle='-.')
crossings = np.loadtxt('./data/crossings.txt')
crossings = crossings.astype('int32')
cross_times = results['time'].iloc[crossings]
if type(cross_times) is np.float64:
    plt.sca(axs[1])
    plt.axvline(x=cross_times, color='r', linestyle='--')
else:
    for x in cross_times.values:
        plt.sca(axs[1])
        plt.axvline(x=x, color='r', linestyle='--')

# (d) Dynamical distance scan, non-SSD
#df.plot('time', ['x1','x2','x3'],
#        color='lightgrey', legend=False, ax=axs[3]) 
#time = df['time'].to_numpy()
#x = df[['x1','x2','x3','y1','y2','y3']].to_numpy()
#for ii in range(x.shape[1]):
#    axs[2].plot(time, x[:,ii], color='lightgrey', alpha=0.3)
#axs[2].set(title='(d)', xlabel='time', ylabel='output intensity')
#axs[2].set_ylim([-2., 2.])
#axs[2].set_xlim([0., np.max(time)])
axs[2].set_title('(c)', weight='bold', fontsize=10)
axs[2].set_ylabel(r'$x_i$', color='grey', fontsize=8)
axs[2].spines['left'].set_color('grey')
axs[2].yaxis.label.set_color('grey')
axs[2].tick_params(axis='y', colors='grey', labelsize=6)
axs[2].tick_params(axis='x', labelsize=6)
axs[2].set_ylim([-2., 2.])
axs[2].set_xlim([0., np.max(time)])
axs[2].set_xlabel('time', fontsize=8)
ra = axs[2].twinx()
ra.set_xlabel('time')
ra.set_ylabel('distance', fontsize=8)
ra.tick_params(axis='y', labelsize=6)
ra.set_xlim([0., np.max(time)])
results = pd.read_csv('./data/results_nonssd.csv')
results_ra = results['rolling ave'].to_numpy()
results_time = results['time'].to_numpy()
#results.plot('time', ['rolling ave'], ax=ra, style=['k-'], legend=False)
ra.plot(results_time, results_ra, 'k-')
t_pos = np.loadtxt('./data/t_pos_nonssd.txt')
plt.axhline(t_pos, color='b', linestyle='-.')
crossings = np.loadtxt('./data/crossings_nonssd.txt')
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
