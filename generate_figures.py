# file: generate_figures.py

import matplotlib.pyplot as plt
import numpy as np

#Figure 1
fig, axs = plt.subplots(4, 2, figsize=(6, 7))
#plt.subplots_adjust(wspace=0.1,hspace=0.4,bottom=0.3)
fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95], h_pad=3., w_pad=2.7)

# (a) Difference in dynamical kind
data = np.loadtxt('./data/test_same_more_diff')
axs[0,0].plot(data[:,0], data[:,1], 'kx', label=r'varying $k_i$')
axs[0,0].plot(data[:,0], data[:,2], 'k+', label=r'varying $r_i$')
axs[0,0].legend(fontsize=7, bbox_to_anchor=(0.45, 0.75))
axs[0,0].set_xlabel('scale factor', fontsize=8)
axs[0,0].set_ylabel(r'$D_D$', fontsize=8)
axs[0,0].set_title('(a)', weight='bold', fontsize=10, loc='center')
axs[0,0].tick_params(axis='x', labelsize=8)
axs[0,0].tick_params(axis='y', labelsize=8)

# (b) Difference in dynamical kind with sampling noise
data = np.loadtxt('./data/test_same_more_diff_noisy')
axs[0,1].plot(data[:,0], data[:,1], 'kx', label=r'varying $k_i$')
axs[0,1].plot(data[:,0], data[:,2], 'k+', label=r'varying $r_i$')
axs[0,1].legend(fontsize=7, bbox_to_anchor=(0.45, 0.75))
axs[0,1].set_xlabel('scale factor', fontsize=8)
axs[0,1].set_ylabel(r'$D_D$', fontsize=8)
axs[0,1].set_title('(b)', weight='bold', fontsize=10, loc='center')
axs[0,1].tick_params(axis='x', labelsize=8)
axs[0,1].tick_params(axis='y', labelsize=8)

# (c) Same carrying capacity, more linear
data = np.loadtxt('./data/test_same_cap_more_linear')
axs[1,0].plot(data[:,0], data[:,1], 'kx')
axs[1,0].set_xlabel('scale factor', fontsize=8)
axs[1,0].set_ylabel(r'$D_D$', fontsize=8)
axs[1,0].set_title('(c)', weight='bold', fontsize=10, loc='center')
axs[1,0].tick_params(axis='x', labelsize=8)
axs[1,0].tick_params(axis='y', labelsize=8)

# (d) Same carrying capacity, more linear with sampling noise
data = np.loadtxt('./data/test_same_cap_more_linear_noisy')
axs[1,1].plot(data[:,0], data[:,1], 'kx')
axs[1,1].set_xlabel('scale factor', fontsize=8)
axs[1,1].set_ylabel(r'$D_D$', fontsize=8)
axs[1,1].set_title('(d)', weight='bold', fontsize=10, loc='center')
axs[1,1].tick_params(axis='x', labelsize=8)
axs[1,1].tick_params(axis='y', labelsize=8)

# (e) Same carrying capacity, more 2nd order
data = np.loadtxt('./data/test_same_cap_more_2nd_order')
axs[2,0].plot(np.log10(data[:,0]), data[:,1], 'kx')
axs[2,0].set_xlabel('log(scale factor)', fontsize=8)
axs[2,0].set_ylabel(r'$D_D$', fontsize=8)
axs[2,0].set_title('(e)', weight='bold', fontsize=10, loc='center')
axs[2,0].tick_params(axis='x', labelsize=8)
axs[2,0].tick_params(axis='y', labelsize=8)

# (f) Same carrying capacity, more 2nd order
data = np.loadtxt('./data/test_same_cap_more_2nd_order_noisy')
axs[2,1].plot(np.log10(data[:,0]), data[:,1], 'kx')
axs[2,1].set_xlabel('log(scale factor)', fontsize=8)
axs[2,1].set_ylabel(r'$D_D$', fontsize=8)
axs[2,1].set_title('(f)', weight='bold', fontsize=10, loc='center')
axs[2,1].tick_params(axis='x', labelsize=8)
axs[2,1].tick_params(axis='y', labelsize=8)

# (g) Chaos
data1 = np.loadtxt('./data/test_into_chaos_1')
data2 = np.loadtxt('./data/test_into_chaos_2')
data3 = np.loadtxt('./data/test_into_chaos_3')
l2data = np.loadtxt('./data/test_into_chaos_l2')
#axs[3,0].plot(data1[:,0], data1[:,1], 'kx', 
#        label='chaotic vs chaotic')
#axs[3,0].plot(data2[:,0], data2[:,1], 'k+',
#        label='non-chaotic vs non-chaotic')
#axs[3,0].plot(data3[:,0], data3[:,1], 'kX',
#        label='chaotic vs non-chaotic')
lns1 = axs[3,0].plot(data3[:,0], data3[:,1], 'k+',
        label=r'$D_D$')
tmp_ax = axs[3,0].twinx()
lns2 = tmp_ax.plot(l2data[:,0], l2data[:,1], 'k.',
        label='l2 norm')
#axs[3,0].legend(fontsize=7, bbox_to_anchor=(1.17, -0.45), ncol=3, loc="upper center")
#axs[3,0].legend(fontsize=7, bbox_to_anchor=(0.5, 0.5))
#tmp_ax.legend(fontsize=7, bbox_to_anchor=(0.5, 0.4))
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
axs[3,0].legend(lns, labs, loc=3, fontsize=7)
axs[3,0].set_xlabel(r'$\beta$', fontsize=8)
axs[3,0].set_ylim([0., 0.2])
axs[3,0].set_ylabel('distance', fontsize=8)
axs[3,0].set_title('(g)', weight='bold', fontsize=10, loc='center')
axs[3,0].tick_params(axis='x', labelsize=8)
axs[3,0].tick_params(axis='y', labelsize=8)
tmp_ax.tick_params(axis='y', labelsize=8)
tmp_ax.set_ylim([0., 25.])

# (h) Chaos with sampling noise
data1_noisy = np.loadtxt('./data/test_into_chaos_noisy_1')
data2_noisy = np.loadtxt('./data/test_into_chaos_noisy_2')
data3_noisy = np.loadtxt('./data/test_into_chaos_noisy_3')
l2data_noisy = np.loadtxt('./data/test_into_chaos_l2_noisy')
lns1 = axs[3,1].plot(data3_noisy[:,0], data3_noisy[:,1], 'k+',
        label=r'$D_D$')
tmp_ax = axs[3,1].twinx()
lns2 = tmp_ax.plot(l2data_noisy[:,0], l2data_noisy[:,1], 'k.',
        label='l2 norm')
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
axs[3,1].legend(lns, labs, loc=3, fontsize=7)
axs[3,1].set_xlabel(r'$\beta$', fontsize=8)
axs[3,1].set_ylim([0., 0.2])
axs[3,1].set_ylabel('distance', fontsize=8)
axs[3,1].set_title('(h)', weight='bold', fontsize=10, loc='center')
axs[3,1].tick_params(axis='x', labelsize=8)
axs[3,1].tick_params(axis='y', labelsize=8)
tmp_ax.tick_params(axis='y', labelsize=8)
tmp_ax.set_ylim([0., 25.])

plt.savefig('distance.pdf', dpi=600)
