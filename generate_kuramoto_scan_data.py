# file: generate_kuramoto_scan_data.py


import pandas as pd
import numpy as np
import eugene as eu
import matplotlib.pyplot as plt
import pdb
""" Made reference to 
https://towardsdatascience.com/basic-time-series-manipulation-with-pandas-4432afee64ea
"""

df = pd.read_csv("./data/kuramoto_data.csv")
timeseries = df[['x1','x2','x3','y1','y2','y3']].to_numpy()
lag = 100
width = int(2 * 10 **4)
frags = 200
reps = 10 
step_size=100
ds = eu.dd_scan.DiffScanner(timeseries.T, 
        step_size=step_size, lag=lag, window_width=width, steps=5)
ds.start_scan(frags=frags,reps=reps)
scan_data = ds._scan[:,1]
results = pd.DataFrame(data=scan_data, columns=['scan data'])
results['rolling ave'] = results['scan data'].ewm(span=50, adjust=False).mean()
index = width + int(round(lag / 2))
results['time'] = df['time'][index:-index:step_size].values

# establish a threshold for declaring an event
# choose segment
ref_end = 3 * width + lag
#ref_series = df['v6'].to_numpy()[:ref_end].reshape(-1,1)
ref_series = df[['x1','x2','x3','y1','y2','y3']].to_numpy()[:ref_end,:]
ds_ref = eu.dd_scan.DiffScanner(ref_series.T, step_size=step_size, lag=lag, window_width=width, steps=5)
ds_ref.start_scan(frags=frags,reps=reps)
ref_data = ds_ref._scan[:,1]
mu = np.mean(ref_data)
sigma = np.std(ref_data)
t_pos = mu + 3. * sigma # sets threshold of significance

# find the trigger points
print('Finding trigger points...')
ra = results['rolling ave'].to_numpy().reshape(-1,1)
below = (ra < t_pos)[:-1,:]
above = (ra >= t_pos)[1:,:]
crossings = np.flatnonzero(below * above)
print('Crossings: {}'.format(crossings))

# save the data
print('Saving data...')
results.to_csv('./data/results.csv')
np.savetxt('./data/crossings.txt', crossings)
np.savetxt('./data/t_pos.txt', [t_pos])

# REPEAT WITH NON-SSD DATA
timeseries = df[['x1','x2','x3']].to_numpy()
lag = 100
width = int(3 * 10 **4)
frags = 200
reps = 20 
step_size=100
ds = eu.dd_scan.DiffScanner(timeseries.T, 
        step_size=step_size, lag=lag, window_width=width, steps=5)
ds.start_scan(frags=frags,reps=reps)
scan_data = ds._scan[:,1]
results = pd.DataFrame(data=scan_data, columns=['scan data'])
results['rolling ave'] = results['scan data'].ewm(span=50, adjust=False).mean()
index = width + int(round(lag / 2))
results['time'] = df['time'][index:-index:step_size].values

# establish a threshold for declaring an event
# choose segment
ref_end = 3 * width + lag
#ref_series = df['v6'].to_numpy()[:ref_end].reshape(-1,1)
ref_series = df[['x1','x2','x3']].to_numpy()[:ref_end,:]
ds_ref = eu.dd_scan.DiffScanner(ref_series.T, step_size=step_size, lag=lag, window_width=width, steps=5)
ds_ref.start_scan(frags=frags,reps=reps)
ref_data = ds_ref._scan[:,1]
mu = np.mean(ref_data)
sigma = np.std(ref_data)
t_pos = mu + 2. * sigma # sets threshold of significance

# find the trigger points
print('Finding trigger points...')
ra = results['rolling ave'].to_numpy().reshape(-1,1)
below = (ra < t_pos)[:-1,:]
above = (ra >= t_pos)[1:,:]
crossings = np.flatnonzero(below * above)
print('Crossings: {}'.format(crossings))

# save the data
print('Saving data...')
results.to_csv('./data/results_nonssd.csv')
np.savetxt('./data/crossings_nonssd.txt', crossings)
np.savetxt('./data/t_pos_nonssd.txt', [t_pos])
