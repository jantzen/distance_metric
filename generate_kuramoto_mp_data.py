# file: generate_bondarenko_mp_data.py

from dask.distributed import Client
import pandas as pd
import numpy as np
import stumpy

df = pd.read_csv('./data/kuramoto_data.csv')
time = df['time'].to_numpy()
ts = df[['x1','x2','x3','y1','y2','y3']]

# run a matrix profile on the data
mp = stumpy.mstump(ts, m=200)

# save the matrix profile and indices
np.savetxt('./data/mp.txt', mp[0])
np.savetxt('./data/mp_indices.txt', mp[1])
