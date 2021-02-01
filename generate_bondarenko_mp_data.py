# file: generate_bondarenko_mp_data.py

from dask.distributed import Client
import pandas as pd
import numpy as np
import stumpy

df = pd.read_csv('./data/bond_data.csv')
time = df['time'].to_numpy()
ts = df[['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10']]


# run a matrix profile on the data
#dask_client = Client()
#mp = stumpy.mstumped(dask_client, ts, m=200)
mp = stumpy.mstump(ts, m=200)

# save the matrix profile and indices
np.savetxt('./data/mp.txt', mp[0])
np.savetxt('./data/mp_indices.txt', mp[1])
