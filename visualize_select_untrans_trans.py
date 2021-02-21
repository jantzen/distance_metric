# file: visualize_select_untrans_trans.py

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import eugene as eu
import pdb


def plot_initials(data1, data2, frags, reps):
    alpha = 0.8
    beta = 0.2
    fragmented = eu.fragment_timeseries.split_timeseries([data1, data2], frags)
    frag_len = np.max(fragmented[0][0].shape)
    fig, axs = plt.subplots(3, 1, figsize=(3,6))
    c1 = 6
    c2 = 23 
    axs[0].plot(data1[0, c1 * frag_len:(c1+4)*frag_len], 'k-', label=r'$x_1^A$')
    axs[0].plot(data1[1, c1 * frag_len:(c1+4)*frag_len], 'k-.', label=r'$x_2^A$')
    for ii in range(4):
        axs[0].scatter(ii * frag_len, data1[0, (ii + c1) * frag_len], s=50, alpha=0.4, color='tab:blue', edgecolors='none')
        axs[0].scatter(ii * frag_len, data1[1, (ii + c1) * frag_len], s=50, alpha=0.4, color='tab:blue', edgecolors='none')
        plt.sca(axs[0])
        plt.axvline(x=ii*frag_len, color='tab:blue', linestyle='--')
    axs[0].set_ylim(-100, 100)
#    axs[0].set_aspect('equal')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_xlabel(r'$t$')
    axs[0].set_ylabel(r'$\vec{x}^A$', rotation='horizontal', labelpad=10)
    axs[0].legend(bbox_to_anchor=[0.5, 0.84], loc="lower center", ncol=2, fontsize=6)

    colors = ['tab:blue', 'tab:purple']
    for ii, system in enumerate(fragmented):
        initials_x = []
        initials_y = []
        for fragment in system:
            initials_x.append(fragment[0,0])
            initials_y.append(fragment[1,0])
        initials_x = np.array(initials_x)
        initials_y = np.array(initials_y)
        axs[1].scatter(initials_x, initials_y, s=50, alpha=0.4,
                edgecolors='none', color=colors[ii])
    axs[1].set_aspect('equal')
    size = 40
    axs[1].set_xlim(-size, size)
    axs[1].set_ylim(-size, size)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_xlabel(r'$x_1$')
    axs[1].set_ylabel(r'$x_2$')
     
#    axs[2].plot(data2[0, :4000], 'k-', label=r'$x_1^B$')
#    axs[2].plot(data2[1, :4000], 'k-.', label=r'$x_2^B$')
    axs[2].plot(data2[0, c2 * frag_len:(c2+4)*frag_len], 'k-', label=r'$x_1^A$')
    axs[2].plot(data2[1, c2 * frag_len:(c2+4)*frag_len], 'k-.', label=r'$x_2^A$')
    for ii in range(4):
#        axs[2].scatter(ii * 1000, data2[0, ii * 1000], s=50, alpha=0.4,
#                color='tab:purple', edgecolors='none')
#        axs[2].scatter(ii * 1000, data2[1, ii * 1000], s=50, alpha=0.4,
#                color='tab:purple', edgecolors='none')
        axs[2].scatter(ii * frag_len, data2[0, (ii + c2) * frag_len], s=50, alpha=0.4, color='tab:purple', edgecolors='none')
        axs[2].scatter(ii * frag_len, data2[1, (ii + c2) * frag_len], s=50, alpha=0.4, color='tab:purple', edgecolors='none')
        plt.sca(axs[2])
        plt.axvline(x=ii*frag_len, color='tab:purple', linestyle='--')
    axs[2].set_ylim(-100, 100)
#    axs[2].set_aspect('equal')
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    axs[2].set_xlabel('$t$')
    axs[2].set_ylabel(r'$\vec{x}^B$', rotation='horizontal', labelpad=10)
    axs[2].legend(bbox_to_anchor=[0.5, 0.04], loc="lower center", ncol=2, fontsize=6)

    # compute and plot params used by choose_untrans_trans
    initials = np.concatenate([initials_x.reshape(1, -1), initials_y.reshape(1, -1)], axis=0)
    mu = np.mean(initials, axis=1).reshape(-1,1)
    cov = np.cov(initials)
    w = 5.
    corner = mu - np.array([w/2., w/2.]).reshape(-1,1)
    rect = plt.Rectangle(corner, w, w, edgecolor='black', fill=False)
    axs[1].add_patch(rect)

    # find the largest eigenvector-eigenvalue pair
    w, v = np.linalg.eig(cov)
    w_max = np.sqrt(np.max(w))
    e_vec = v[:,np.argsort(w)[-1]].reshape(-1,1)
    sc = 10.
    
    axs[1].arrow(float(mu[0]), float(mu[1]), float(e_vec[0]) * sc, float(e_vec[1]) * sc,
            facecolor='black', head_width=0.25*sc*np.linalg.norm(e_vec))

    # choose means and covariance matrix for distributions
    mu_untrans = mu - alpha * w_max * e_vec
    mu_trans = mu + alpha * w_max * e_vec
    u, s_full, v = np.linalg.svd(cov)
    s = beta * s_full
    cov_t = np.dot(np.dot(u, np.diag(s)), v)

    angle = 360. - (np.arcsin(e_vec[1] / np.linalg.norm(e_vec)) * 180. / np.pi)
    ell = Ellipse(mu, 
            width=np.sqrt(s_full[0]), 
            height=np.sqrt(s_full[1]),
            angle=angle,
            facecolor='black',
            alpha=0.2)
    axs[1].add_patch(ell)
    ell2 = Ellipse(mu_untrans, 
            width=np.sqrt(s[0]), 
            height=np.sqrt(s[1]),
            angle=angle,
            facecolor='orange',
            alpha=0.4)
    axs[1].add_patch(ell2)
    ell3 = Ellipse(mu_trans, 
            width=np.sqrt(s[0]), 
            height=np.sqrt(s[1]),
            angle=angle,
            facecolor='green',
            alpha=0.4)
    axs[1].add_patch(ell3)

    untrans, trans = eu.initial_conditions.choose_untrans_trans(fragmented, reps, 
            alpha=alpha,
            beta=beta)
    for ii, event in enumerate(untrans):
        initials_x = []
        initials_y = []
        for fragment in event:
            initials_x.append(fragment[0,0])
            initials_y.append(fragment[1,0])
        initials_x = np.array(initials_x)
        initials_y = np.array(initials_y)
        axs[1].scatter(initials_x, initials_y, s=50, alpha=1.0, marker='+',
                color='orange')
    for ii, event in enumerate(trans):
        initials_x = []
        initials_y = []
        for fragment in event:
            initials_x.append(fragment[0,0])
            initials_y.append(fragment[1,0])
        initials_x = np.array(initials_x)
        initials_y = np.array(initials_y)
        axs[1].scatter(initials_x, initials_y, s=50, alpha=1.0, marker='x',
                color='green')

    # add the arrows from upper and lower subplots to the middle plot
    axs[1].annotate('', xy=(0.36, 0.573), xytext=(0.16, 0.74), xycoords='figure fraction', arrowprops=dict(arrowstyle="->", color='grey', linestyle='-', linewidth=2))
    axs[1].annotate('', xy=(0.36, 0.573), xytext=(0.16, 0.805), xycoords='figure fraction', arrowprops=dict(arrowstyle="->", color='grey', linestyle='-', linewidth=2))
    axs[2].annotate('', xy=(0.655, 0.464), xytext=(0.332, 0.247), xycoords='figure fraction', arrowprops=dict(arrowstyle="->", color='grey', linestyle='-', linewidth=2))
    axs[2].annotate('', xy=(0.655, 0.464), xytext=(0.332, 0.205), xycoords='figure fraction', arrowprops=dict(arrowstyle="->", color='grey', linestyle='-', linewidth=2))

    plt.savefig('choose_ics.pdf', dpi=600)
#    plt.show()


def main():
    # load data
    df = pd.read_csv('./data/bond_data.csv')
#    data = df[['x1','x2','x3','y1','y2','y3']].to_numpy().T
    data1 = df[['v1','v2']].to_numpy().T[:, :100000]
    data2 = df[['v3','v4']].to_numpy().T[:, :100000]
    frags = 50
    reps = 5
    plot_initials(data1, data2, frags, reps)


if __name__ == '__main__':
    main()
