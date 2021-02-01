# file: generate_data.py

"""This module generates the data reported in ...

    It is based on test_robustness.py in the euegene package.
"""

import eugene as eu
import numpy as np
import matplotlib.pyplot as plt
from eugene.src.tools.LVDSim import *
from tqdm import trange
from os.path import join, isfile

import pdb


# HELPER FUNCTIONS

def no_overlay(x):
    return x

def pair_dist(data1, data2, min_len=10, steps=90):

    # convert to numpy
    untrans = [np.array(data1[0][0]).T, np.array(data2[0][0]).T]
    trans = [np.array(data1[0][1]).T, np.array(data2[0][1]).T]
    dmat = eu.dynamical_distance.distance_matrix(untrans, trans, min_len, steps=steps)
    return dmat[0,1]

def setup_params(same=True, stochastic=False, sigma = [0.1, 0.1]):
    r1 = np.array([1., 2.])
    k1 = np.array([100., 100.])

    alpha1 = np.array([[1., 0.5], [0.7, 1.]])
    alpha2 = alpha1

    if same:
        r2 = r1 * 1.5
        k2 = k1
    else:
        r2 = r1
        k2 = np.array([150., 150.])

    init1 = np.array([5., 5.])
    init2 = init1

    init_trans1 = np.array([8., 8.])
    init_trans2 = init_trans1

    if stochastic:
        params1 = [r1, k1, alpha1, sigma, init1, init_trans1]
        params2 = [r2, k2, alpha2, sigma, init2, init_trans2]
    else:
        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2]

    return [params1, params2]



################################################################################
# DATA GENERATION METHODS

def test_same_more_diff(data_dir="./data/"):
    print("test_same_more_diff")

    r1 = np.array([1., 2.])
    r2 = r1 * 1.5

    k1 = np.array([100., 100.])
    k2 = np.array([100., 100.])

    alpha1 = np.array([[1., 0.5], [0.7, 1.]])
    alpha2 = alpha1

    init1 = np.array([5, 5])
    init2 = init1

    init_trans1 = np.array([8, 8])
    init_trans2 = init_trans1

    params1 = [r1, k1, alpha1, init1, init_trans1]
    params2 = [r2, k2, alpha2, init2, init_trans2]
    params3 = [r1, k2, alpha2, init2, init_trans2]

    overlay = lambda x: np.mean(x, axis=1)

    dists = []
    dists2 = []
    dists_noisy = []
    dists2_noisy = []
    lens = []
    f = np.linspace(1., 5., 20)
    for x in trange(20):
        n = f[x]
        lens.append(n)
#        data_name = "test_same_more_diff-" + str(n)
#        data_path = join("test_data", data_name + ".npy")
        params2[1] = k2 * n
        params3[0] = r1 * n
        data1 = simData([params1], 15., 100, no_overlay, range_cover=False)
        data2 = simData([params2], 15., 100, no_overlay, range_cover=False)
        data3 = simData([params3], 15., 100, no_overlay, range_cover=False)
        data1_noisy = [[[],[]]]
        data2_noisy = [[[],[]]]
        data3_noisy = [[[],[]]]
        scale = 0.05 * k1[0]
        data1_noisy[0][0] = data1[0][0] + np.random.normal(loc=0., scale=scale, size=data1[0][0].shape)
        data2_noisy[0][0] = data2[0][0] + np.random.normal(loc=0., scale=scale, size=data1[0][0].shape)
        data3_noisy[0][0] = data3[0][0] + np.random.normal(loc=0., scale=scale, size=data1[0][0].shape)
        data1_noisy[0][1] = data1[0][1] + np.random.normal(loc=0., scale=scale, size=data1[0][1].shape)
        data2_noisy[0][1] = data2[0][1] + np.random.normal(loc=0., scale=scale, size=data1[0][1].shape)
        data3_noisy[0][1] = data3[0][1] + np.random.normal(loc=0., scale=scale, size=data1[0][1].shape)
        dist = pair_dist(data1, data2)
        dist2 = pair_dist(data1, data3)
        dist_noisy = pair_dist(data1_noisy, data2_noisy)
        dist2_noisy = pair_dist(data1_noisy, data3_noisy)
        dists.append(dist)
        dists2.append(dist2)
        dists_noisy.append(dist_noisy)
        dists2_noisy.append(dist2_noisy)

    # convert and save the data 
    lens = np.array(lens).reshape(-1, 1)
    dists = np.array(dists).reshape(-1,1)
    dists2 = np.array(dists2).reshape(-1,1)
    dists_noisy = np.array(dists_noisy).reshape(-1,1)
    dists2_noisy = np.array(dists2_noisy).reshape(-1,1)
    data = np.concatenate([lens, dists, dists2], axis=1)
    data_noisy = np.concatenate([lens, dists_noisy, dists2_noisy], axis=1)
    np.savetxt(join(data_dir, 'test_same_more_diff'), data)
    np.savetxt(join(data_dir, 'test_same_more_diff_noisy'), data_noisy)


def test_same_cap_more_linear(data_dir="./data/"):
    test_name = "test_same_cap_more_linear"
    print(test_name)

    params = setup_params()
    params1 = params[0]
    params1.append(0.)
    params2 = params[1]
    params2.append(0.)

    dists = []
    dists_noisy = []
    lens = []
    for x in trange(20):
        n = x/10
        lens.append(n)
        data_name = test_name + "-" + str(n)
        data_path = join("test_data", data_name + ".npy")
        params1[-1] = 1.
        params2[-1] = n
        data1 = simDataLin([params1], 10., 100, no_overlay)
        data2 = simDataLin([params2], 10., 100, no_overlay)
        data1_noisy = [[[],[]]]
        data2_noisy = [[[],[]]]
        scale = 0.05 * params1[1]
        data1_noisy[0][0] = data1[0][0] + np.random.normal(loc=0., scale=scale, size=data1[0][0].shape)
        data2_noisy[0][0] = data2[0][0] + np.random.normal(loc=0., scale=scale, size=data1[0][0].shape)
        data1_noisy[0][1] = data1[0][1] + np.random.normal(loc=0., scale=scale, size=data1[0][1].shape)
        data2_noisy[0][1] = data2[0][1] + np.random.normal(loc=0., scale=scale, size=data1[0][1].shape)
        dist = pair_dist(data1, data2)
        dist_noisy = pair_dist(data1_noisy, data2_noisy)
        dists.append(dist)
        dists_noisy.append(dist_noisy)

    # convert and save the data 
    lens = np.array(lens).reshape(-1, 1)
    dists = np.array(dists).reshape(-1,1)
    dists_noisy = np.array(dists_noisy).reshape(-1,1)
    data = np.concatenate([lens, dists], axis=1)
    data_noisy = np.concatenate([lens, dists_noisy], axis=1)
    np.savetxt(join(data_dir, 'test_same_cap_more_linear'), data)
    np.savetxt(join(data_dir, 'test_same_cap_more_linear_noisy'), data_noisy)


def test_same_cap_more_2nd_order(data_dir="./data/"):
    test_name = "test_same_cap_more_2nd_order"
    print(test_name)

    init_y = [0., 0.]
    params = setup_params()
    params1 = params[0]
    params1.append(init_y)
    params1.append(0.)
    params2 = params[1]
    params2.append(init_y)
    params2.append(0.)

    dists = []
    dists_noisy = []
    lens = []
    f = np.linspace(0., 3., 20)
    for x in trange(20):
        n = np.power(10, f[x])
        lens.append(n)
        data_name = test_name + "-" + str(n)
        data_path = join("test_data", data_name + ".npy")
        params1[-1] = 1.
        params2[-1] = n
        data1 = simData2OD([params1], 10., 100, no_overlay,
                       range_cover=False)
        data2 = simData2OD([params2], 10., 100, no_overlay,
                       range_cover=False)
        data1_noisy = [[[],[]]]
        data2_noisy = [[[],[]]]
        tmp = np.concatenate([data1[0][0][:,2:], data1[0][1][:,2:], data2[0][0][:,2:], data2[0][1][:,2:]])
        size = data1[0][0][:,:2].shape
        pop_scale = 0.05 * params1[1][0]
        deriv_scale = 0.05 * np.abs(np.max(tmp) - np.min(tmp))
        d10 = data1[0][0][:,:2] + np.random.normal(loc=0., scale=pop_scale, size=size)
        d11 = data1[0][0][:,2:] + np.random.normal(loc=0., scale=deriv_scale, size=size)
        data1_noisy[0][0] = np.concatenate([d10, d11], axis=1)
        d20 = data2[0][0][:,:2] + np.random.normal(loc=0., scale=pop_scale, size=size)
        d21 = data2[0][0][:,2:] + np.random.normal(loc=0., scale=deriv_scale, size=size)
        data2_noisy[0][0] = np.concatenate([d20, d21], axis=1)
        d10 = data1[0][1][:,:2] + np.random.normal(loc=0., scale=pop_scale, size=size)
        d11 = data1[0][1][:,2:] + np.random.normal(loc=0., scale=deriv_scale, size=size)
        data1_noisy[0][1] = np.concatenate([d10, d11], axis=1)
        d20 = data2[0][1][:,:2] + np.random.normal(loc=0., scale=pop_scale, size=size)
        d21 = data2[0][1][:,2:] + np.random.normal(loc=0., scale=deriv_scale, size=size)
        data2_noisy[0][1] = np.concatenate([d20, d21], axis=1)
        dist = pair_dist(data1, data2)
        dist_noisy = pair_dist(data1_noisy, data2_noisy)
        dists.append(dist)
        dists_noisy.append(dist_noisy)

    # convert and save the data 
    lens = np.array(lens).reshape(-1, 1)
    dists = np.array(dists).reshape(-1,1)
    dists_noisy = np.array(dists_noisy).reshape(-1,1)
    data = np.concatenate([lens, dists], axis=1)
    data_noisy = np.concatenate([lens, dists_noisy], axis=1)
    np.savetxt(join(data_dir, 'test_same_cap_more_2nd_order'), data)
    np.savetxt(join(data_dir, 'test_same_cap_more_2nd_order_noisy'), data_noisy)



def test_into_chaos(data_dir="./data/"):
    test_name = "test_into_chaos_interaction"
    print(test_name)
    
    R1 = np.array([1.7741, 1.0971, 1.5466, 4.4116])
    A1 = np.array([[1., 2.419, 2.248, 0.0023],
                       [0.001, 1., 0.001, 1.3142],
                       [2.3818, 0.001, 1., 0.4744],
                       [1.21, 0.5244, 0.001, 1.]])
    R2 = np.array([1., 0.1358, 1.4936, 4.8486])
    A2 = np.array([[1., 0.3064, 2.9141, 0.8668],
                    [0.125, 1., 0.3346, 1.854],
                    [1.9833, 3.5183, 1., 0.001],
                    [0.6986, 0.9653, 2.1232, 1.]])
    R3 = np.array([4.4208, 0.8150, 4.5068, 1.4172])
    A3 = np.array([[1., 3.6981, 1.4368, 0.0365],
                    [0., 1., 1.7781, 3.7306],
                    [0.5271, 4.1593, 1., 1.3645],
                    [0.8899, 0.2127, 3.4711, 1.]])

    def _r(alpha, beta):
        return R1 + alpha * (R2 - R1) + beta * (R3 - R1)

    def _A(alpha, beta):
        return A1 + alpha * (A2 - A1) + beta * (A3 - A1)

    r1 = _r(0., 1.)
    k1 = np.array([1., 1., 1., 1.])
    a1 = _A(0., 1.)
    init1 = np.array([0.1, 0.1, 0.1, 0.1])
    init_trans1 = np.array([0.2, 0.2, 0.2, 0.2])

    # params for system carried through the chaotic transition
    r2 = r1
    k2 = k1
    a2 = a1
    init2 = init1
    init_trans2 = init_trans1

    # params for baseline non-chaotic system
    r3 = _r(0.2, 1.0)
    k3 = k1
    a3 = _A(0.2, 1.0)
    init3 = init1
    init_trans3 = init_trans1

    # params for variable non-chaotic system
    r4 = r3
    k4 = k1
    a4 = a3
    init4 = init1
    init_trans4 = init_trans1

    # params for baseline non-chaotic system
    r5 = _r(0.1, 1.0)
    k5 = k1
    a5 = _A(0.1, 1.0)
    init5 = init1
    init_trans5 = init_trans1

    # params for variable non-chaotic system
    r6 = r5
    k6 = k1
    a6 = a5
    init6 = init1
    init_trans6 = init_trans1

    # alpha3 = alpha1 * np.power(alpha_s, 1.2)
    params1 = [r1, k1, a1, init1, init_trans1]
    params2 = [r2, k2, a2, init2, init_trans2]

    params3 = [r3, k3, a3, init3, init_trans3]
    params4 = [r4, k4, a4, init4, init_trans4]

    params5 = [r5, k5, a5, init5, init_trans5]
    params6 = [r6, k6, a6, init6, init_trans6]

    dists1 = []
    dists2 = []
    dists3 = []
    dists1_noisy = []
    dists2_noisy = []
    dists3_noisy = []
    lens = []
    points = 51
    f = np.linspace(0.95, 1.05, points)
    for x in trange(points):
        beta = f[x]
        lens.append(beta)
        data_name = test_name + "-" + str(beta)
        data_path = join("test_data", data_name + ".npy")

        params2[0] = _r(0., beta)
        params2[2] = _A(0., beta)

        params4[0] = _r(0.2, beta)
        params4[2] = _A(0.2, beta)

        params6[0] = _r(0.1, beta)
        params6[2] = _A(0.1, beta)

        data1 = simData([params1], 25., 1000, no_overlay, range_cover=False)
        data2 = simData([params2], 25., 1000, no_overlay, range_cover=False)
        data3 = simData([params3], 25., 1000, no_overlay, range_cover=False)
        data4 = simData([params4], 25., 1000, no_overlay, range_cover=False)
        data1_noisy = [[[],[]]]
        data2_noisy = [[[],[]]]
        data3_noisy = [[[],[]]]
        data4_noisy = [[[],[]]]
        # add noise to all four population time series (there are 4 species in the chaotic systems)
        scale = 0.05 
        data1_noisy[0][0] = data1[0][0] + np.random.normal(loc=0., scale=scale, size=data1[0][0].shape)
        data2_noisy[0][0] = data2[0][0] + np.random.normal(loc=0., scale=scale, size=data1[0][0].shape)
        data1_noisy[0][1] = data1[0][1] + np.random.normal(loc=0., scale=scale, size=data1[0][1].shape)
        data2_noisy[0][1] = data2[0][1] + np.random.normal(loc=0., scale=scale, size=data1[0][1].shape)
        data3_noisy[0][0] = data3[0][0] + np.random.normal(loc=0., scale=scale, size=data1[0][0].shape)
        data3_noisy[0][1] = data3[0][1] + np.random.normal(loc=0., scale=scale, size=data1[0][1].shape)
        data4_noisy[0][0] = data4[0][0] + np.random.normal(loc=0., scale=scale, size=data1[0][0].shape)
        data4_noisy[0][1] = data4[0][1] + np.random.normal(loc=0., scale=scale, size=data1[0][1].shape)

        dist1 = pair_dist(data1, data2, min_len=600)
        dist2 = pair_dist(data3, data4, min_len=600)
        dist3 = pair_dist(data3, data2, min_len=600)
        dist1_noisy = pair_dist(data1_noisy, data2_noisy, min_len=600)
        dist2_noisy = pair_dist(data3_noisy, data4_noisy, min_len=600)
        dist3_noisy = pair_dist(data3_noisy, data2_noisy, min_len=600)
        dists1.append(dist1)
        dists2.append(dist2)
        dists3.append(dist3)
        dists1_noisy.append(dist1_noisy)
        dists2_noisy.append(dist2_noisy)
        dists3_noisy.append(dist3_noisy)

    # convert and save the data 
    lens = np.array(lens).reshape(-1, 1)
    dists1 = np.array(dists1).reshape(-1,1)
    dists1_noisy = np.array(dists1_noisy).reshape(-1,1)
    data1 = np.concatenate([lens, dists1], axis=1)
    data1_noisy = np.concatenate([lens, dists1_noisy], axis=1)
    np.savetxt(join(data_dir, 'test_into_chaos_1'), data1)
    np.savetxt(join(data_dir, 'test_into_chaos_noisy_1'), data1_noisy)
    dists2 = np.array(dists2).reshape(-1,1)
    dists2_noisy = np.array(dists2_noisy).reshape(-1,1)
    data2 = np.concatenate([lens, dists2], axis=1)
    data2_noisy = np.concatenate([lens, dists2_noisy], axis=1)
    np.savetxt(join(data_dir, 'test_into_chaos_2'), data2)
    np.savetxt(join(data_dir, 'test_into_chaos_noisy_2'), data2_noisy)
    dists3 = np.array(dists3).reshape(-1,1)
    dists3_noisy = np.array(dists3_noisy).reshape(-1,1)
    data3 = np.concatenate([lens, dists3], axis=1)
    data3_noisy = np.concatenate([lens, dists3_noisy], axis=1)
    np.savetxt(join(data_dir, 'test_into_chaos_3'), data3)
    np.savetxt(join(data_dir, 'test_into_chaos_noisy_3'), data3_noisy)



if __name__ == '__main__':
    # process command line options
    experiments = [test_same_more_diff, test_same_cap_more_linear, test_same_cap_more_2nd_order,test_into_chaos]
#    experiments = [test_into_chaos]
    # run experiments
    for ex in experiments:
        ex()

    plt.show()
