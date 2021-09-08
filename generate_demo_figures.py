# file: generate_demo_figures.py

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(3,1, figsize=(4,6))
fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95], h_pad=4., w_pad=2.7)

# Kuramoto with weakly coupled oscillators
data = np.loadtxt('./data/demo_theta1')
for ii in range(6):
    axs[0].plot(data[0, :], data[ii+1, :], 'k-', alpha=0.3)
axs[0].set_xlabel('time', fontsize=8, labelpad=0.0)
axs[0].set_ylabel('amplitude', fontsize=8, labelpad=0.0)
axs[0].tick_params(axis='x', labelsize=6)
axs[0].tick_params(axis='y', labelsize=6)
axs[0].set_title('A')

# Kuramoto with uncoupled oscillators
data = np.loadtxt('./data/demo_theta2')
for ii in range(6):
    axs[1].plot(data[0, :], data[ii+1, :], 'k-', alpha=0.3)
axs[1].set_xlabel('time', fontsize=8, labelpad=0.0)
axs[1].set_ylabel('amplitude', fontsize=8, labelpad=0.0)
axs[1].tick_params(axis='x', labelsize=6)
axs[1].tick_params(axis='y', labelsize=6)
axs[1].set_title('B')

# Kuramoto with weekly coupled oscillators, increased omega
data = np.loadtxt('./data/demo_theta3')
for ii in range(6):
    axs[2].plot(data[0, :], data[ii+1, :], 'k-', alpha=0.3)
axs[2].set_xlabel('time', fontsize=8, labelpad=0.0)
axs[2].set_ylabel('amplitude', fontsize=8, labelpad=0.0)
axs[2].tick_params(axis='x', labelsize=6)
axs[2].tick_params(axis='y', labelsize=6)
axs[2].set_title('C')

plt.savefig('demo.pdf', dpi=600)

