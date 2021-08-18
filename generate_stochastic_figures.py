# file: generate_stochastic_figures.py

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(3, 1, figsize=(3., 6.))
#plt.subplots_adjust(wspace=0.1,hspace=0.4,bottom=0.3)
#fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95], h_pad=3., w_pad=2.7)
fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95], h_pad=3., w_pad=2.7)

# (a) Difference in dynamical kind
data = np.loadtxt('./data/test_stochastic')
axs[0].plot(data[0,:], data[1,:], 'k+', label=r'varying $r_i$')
axs[0].plot(data[0,:], data[2,:], 'kx', label=r'varying $k_i$')
axs[0].legend(fontsize=7, bbox_to_anchor=(1., 1.), loc="upper left")
axs[0].set_xlabel('scale factor', fontsize=8)
axs[0].set_ylabel('dynamical distance', fontsize=8)
axs[0].set_title('(a)', weight='bold', fontsize=10, loc='center')
axs[0].tick_params(axis='x', labelsize=8)
axs[0].tick_params(axis='y', labelsize=8)

# (b) Difference in dynamical kind
data = np.loadtxt('./data/test_stochastic_nonssd')
axs[1].plot(data[0,:], data[1,:], 'k+', label=r'varying $r_i$')
axs[1].plot(data[0,:], data[2,:], 'kx', label=r'varying $k_i$')
axs[1].legend(fontsize=7, bbox_to_anchor=(1., 1.), loc="upper left")
axs[1].set_xlabel('scale factor', fontsize=8)
axs[1].set_ylabel('dynamical distance', fontsize=8)
axs[1].set_title('(b)', weight='bold', fontsize=10, loc='center')
axs[1].tick_params(axis='x', labelsize=8)
axs[1].tick_params(axis='y', labelsize=8)

# (c) Increasing stochasticity
# (b) Difference in dynamical kind
data = np.loadtxt('./data/test_stochastic_more_noise')
axs[2].plot(data[0,:], data[1,:], 'kP', label='same, SSD')
axs[2].plot(data[0,:], data[2,:], 'kX', label='different, SSD')
axs[2].plot(data[0,:], data[3,:], 'k+', label='same, non-SSD')
axs[2].plot(data[0,:], data[4,:], 'kx', label='different, non-SSD')
axs[2].legend(fontsize=7, bbox_to_anchor=(1., 1.), loc="upper left")
axs[2].set_xlabel(r'$\sigma$', fontsize=8)
axs[2].set_ylabel('dynamical distance', fontsize=8)
axs[2].set_title('(c)', weight='bold', fontsize=10, loc='center')
axs[2].tick_params(axis='x', labelsize=8)
axs[2].tick_params(axis='y', labelsize=8)

fig.savefig('distance_stochastic.pdf', dpi=600, bbox_inches='tight')
fig.savefig('distance_stochastic.eps', dpi=600, bbox_inches='tight')
