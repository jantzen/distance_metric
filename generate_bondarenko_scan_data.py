import pandas as pd
import numpy as np
import eugene as eu
import matplotlib.pyplot as plt
import pdb
""" Made reference to 
https://towardsdatascience.com/basic-time-series-manipulation-with-pandas-4432afee64ea
"""

df = pd.read_csv("./data/bond_data.csv")
df.set_index('time')
timeseries = df[['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10']].to_numpy()
lag = 100
width = 2 * 10**5
frags = 4500
reps = 10 
step_size=500
ds = eu.dd_scan.DiffScanner(timeseries.T, 
        step_size=step_size, lag=lag, window_width=width, steps=5)
ds.start_scan(frags=frags,reps=reps)
scan_data = ds._scan[:,1]
results = pd.DataFrame(data=scan_data, columns=['scan data'])
results['rolling ave'] = results['scan data'].ewm(span=50, adjust=False).mean()

# establish a threshold for declaring an event
# choose segment
ref_end = 3 * width + lag
#ref_series = df['v6'].to_numpy()[:ref_end].reshape(-1,1)
ref_series = df[['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10']].to_numpy()[:ref_end,:]
ds_ref = DiffScanner(ref_series.T, step_size=step_size, lag=lag, window_width=width, steps=5)
ds_ref.start_scan(frags=frags,reps=reps)
ref_data = ds_ref._scan[:,1]
mu = np.mean(ref_data)
sigma = np.std(ref_data)
t_pos = mu + 2. * sigma

# find the trigger points
print('Finding trigger points...')
ra = results['rolling ave'].to_numpy().reshape(-1,1)
below = (ra < t_pos)[:-1,:]
above = (ra >= t_pos)[1:,:]
crossings = np.flatnonzero(below * above)
print('Crossings: {}'.format(crossings))

print('Saving data...')
#np.savetxt('scan_data.txt', results['time','scan data'].to_numpy())
np.savetxt('./data/scan_data.txt', results['scan data'].to_numpy())
#np.savetxt('rolling_ave.txt', results['time','rolling ave'].to_numpy())
np.savetxt('./data/rolling_ave.txt', results['rolling ave'].to_numpy())
np.savetxt('./data/crossings.txt', crossings)
print('Plotting...')
# plot results
#for ii in range(timeseries.shape[1]):
#    tmp = timeseries[:,ii]
#    tmp = (tmp - np.mean(tmp))/(np.max(tmp)-np.min(tmp))
#    plt.plot(tmp, 'k-')
tmp = results['scan data']
#t = results['time']
#plt.plot(t, (tmp - np.mean(tmp))/(np.max(tmp)-np.min(tmp)), 'g.')
plt.plot((tmp - np.mean(tmp))/(np.max(tmp)-np.min(tmp)), 'g.')
tmp_pos = (t_pos - np.mean(tmp))/(np.max(tmp)-np.min(tmp))
#plt.axhline(t, tmp_pos,color='b')
plt.axhline(tmp_pos,color='b')
tmp = results['rolling ave']
#plt.plot(t, (tmp - np.mean(tmp))/(np.max(tmp)-np.min(tmp)), 'r-')
plt.plot((tmp - np.mean(tmp))/(np.max(tmp)-np.min(tmp)), 'r-')
#plt.savefig('out.png')
plt.show()
