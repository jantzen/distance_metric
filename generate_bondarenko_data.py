import pandas as pd
from bondarenko import *

# Using the paremterization in Hively & Protopopescu (2000), solve the
# Bondarenko equestions for N=10, allowing a 4 x 10^5 h time to reach stationarity
tt, yy = bondarenko(3. * 10**4, 0.01, changepoint=2. * 10**4)

# throw away the first 1/3 of the data points
cut = int(tt.shape[0] / 3.)

tt = tt[cut:]
yy = yy[cut:,:]

# merge and save
data = np.concatenate([tt.reshape(-1,1),yy],axis=1) 
df = pd.DataFrame(data,
        columns=['time','v1','v2','v3','v4','v5','v6','v7','v8','v9','v10'])
df.to_csv('./data/bond_data.csv')
#np.savetxt('./data/bond_data.csv', data, delimiter=',')
