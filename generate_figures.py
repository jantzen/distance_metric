# file: generate_figures.py

import matplotlib.pyplot as plt
import numpy as np

#Figure 1
fig, axs = plt.subplots(4, 2)

# (a) Difference in dynamical kind
data = np.loadtxt('./data/test_same_more_diff')
axs[0,0].plot(data[:,0], data[:,1], 'kx', label='parameter = carry capacity')
axs[0,0].plot(data[:,0], data[:,2], 'k+', label='parameter = growth rate')
axs[0,0].legend()
axs[0,0].set(xlabel='scale factor', ylabel='dynamical distance', title='(a)')

# (b) Difference in dynamical kind with sampling noise
data = np.loadtxt('./data/test_same_more_diff_noisy')
axs[0,1].plot(data[:,0], data[:,1], 'kx', label='parameter = carry capacity')
axs[0,1].plot(data[:,0], data[:,2], 'k+', label='parameter = growth rate')
axs[0,1].legend()
axs[0,1].set(xlabel='scale factor', ylabel='dynamical distance', title='(b)')

# (c) Same carrying capacity, more linear
data = np.loadtxt('./data/test_same_cap_more_linear')
axs[1,0].plot(data[:,0], data[:,1], 'kx')
axs[1,0].set(xlabel='scale factor', ylabel='dynamical distance', title='(c)')

# (d) Same carrying capacity, more linear with sampling noise
data = np.loadtxt('./data/test_same_cap_more_linear_noisy')
axs[1,1].plot(data[:,0], data[:,1], 'kx')
axs[1,1].set(xlabel='scale factor', ylabel='dynamical distance', title='(d)')

# (e) Same carrying capacity, more 2nd order
data = np.loadtxt('./data/test_same_cap_more_2nd_order')
axs[2,0].plot(data[:,0], data[:,1], 'kx')
axs[2,0].set(xlabel='scale factor', ylabel='dynamical distance', title='(e)')

# (f) Same carrying capacity, more 2nd order
data = np.loadtxt('./data/test_same_cap_more_2nd_order_noisy')
axs[2,1].plot(data[:,0], data[:,1], 'kx')
axs[2,1].set(xlabel='scale factor', ylabel='dynamical distance', title='(f)')

# (g) Chaos
data1 = np.loadtxt('./data/test_into_chaos_1')
data2 = np.loadtxt('./data/test_into_chaos_2')
data3 = np.loadtxt('./data/test_into_chaos_3')
axs[3,0].plot(data1[:,0], data1[:,1], 'kx', 
        label='chaotic transition, chaotic reference')
axs[3,0].plot(data2[:,0], data2[:,1], 'k+',
        label='non-chaotic variation, non-chaotic reference')
axs[3,0].plot(data3[:,0], data3[:,1], 'kX',
        label='chaotic transition, non-chaotic reference')
axs[3,0].set(xlabel='beta', ylabel='dynamical distance', title='(g)')

# (h) Chaos with sampling noise
data1_noisy = np.loadtxt('./data/test_into_chaos_noisy_1')
data2_noisy = np.loadtxt('./data/test_into_chaos_noisy_2')
data3_noisy = np.loadtxt('./data/test_into_chaos_noisy_3')
axs[3,1].plot(data1_noisy[:,0], data1_noisy[:,1], 'kx', 
        label='chaotic transition, chaotic reference')
axs[3,1].plot(data2_noisy[:,0], data2_noisy[:,1], 'k+',
        label='non-chaotic variation, non-chaotic reference')
axs[3,1].plot(data3_noisy[:,0], data3_noisy[:,1], 'kX',
        label='chaotic transition, non-chaotic reference')
axs[3,0].set(xlabel='beta', ylabel='dynamical distance', title='(h)')


plt.show()
