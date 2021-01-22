# file: generate_data.py

"""This module generates the data reported in ...
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
    lens = []
    f = np.linspace(1., 5., 20)
    for x in trange(20):
#        n = x + 1
        n = f[x]
        lens.append(n)
        data_name = "test_same_more_diff-" + str(n)
        data_path = join("test_data", data_name + ".npy")
        params2[1] = k2 * n
        params3[0] = r1 * n
        data = None
        if isfile(data_path):
            # data = np.load(data_path)
            pass
        else:
            data1 = simData([params1], 15., 100, no_overlay, range_cover=False)
            data2 = simData([params2], 15., 100, no_overlay, range_cover=False)
            data3 = simData([params3], 15., 100, no_overlay, range_cover=False)
            # np.save(data_path, data)
#        dist = self.pair_dist(data, reps=False, clip=True)
#        dist2 = self.pair_dist(data2, reps=False, clip=True)
        dist = pair_dist(data1, data2)
        dist2 = pair_dist(data1, data3)
        dists.append(dist)
        dists2.append(dist2)

    print(dists)
    fig, ax = plt.subplots()
    ax.plot(lens, dists, 'bo', label='parameter = capacities')
    ax.plot(lens, dists2, 'ro', label='parameter = growth rates')
    ax.legend()
    ax.set(xlabel='x * parameter', ylabel='distance',
           title='Distance Between Two Systems')
#    plt.savefig(data_name + ".pdf")


def test_same_cap_more_linear(data_dir="./data/"):
    test_name = "test_same_cap_more_linear"
    print(test_name)

    params = setup_params()
    params1 = params[0]
    params1.append(0.)
    params2 = params[1]
    params2.append(0.)

    dists = []
    lens = []
#    dl = np.linspace(0., 2., 10)
    for x in trange(20):
        n = x/10
#         n = np.tan(np.power(x, 2))
#        n = dl[x]
        lens.append(n)
        data_name = test_name + "-" + str(n)
        data_path = join("test_data", data_name + ".npy")
        params1[-1] = 1.
        params2[-1] = n
        data1 = simDataLin([params1], 10., 100, no_overlay)
        data2 = simDataLin([params2], 10., 100, no_overlay)
        # np.save(data_path, data)
        dist = pair_dist(data1, data2)
        dists.append(dist)

    print(dists)
    fig, ax = plt.subplots()
    ax.plot(lens, dists, 'bo')
    ax.set(xlabel='linearity factor', ylabel='distance',
           title='Two Systems with Same Capacity and Different Linearity')
#    plt.savefig(test_name + ".pdf")


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
    lens = []
    f = np.linspace(1., 5., 20)
    for x in trange(20):
        n = np.power(10, f[x])
        lens.append(n)
        data_name = test_name + "-" + str(n)
        data_path = join("test_data", data_name + ".npy")
        params1[-1] = 1.
        params2[-1] = n
#        params = [params1, params2]
        data = None
        data1 = simData2OD([params1], 10., 100, no_overlay,
                       range_cover=False)
        data2 = simData2OD([params2], 10., 100, no_overlay,
                       range_cover=False)
            # np.save(data_path, data)
#        dist = self.pair_dist(data, reps=False, clip=True)
        dist = pair_dist(data1, data2)
        dists.append(dist)

    print(dists)
    fig, ax = plt.subplots()
    ax.plot(lens, dists, 'bo')
    ax.set(xlabel='order factor', ylabel='distance', xscale='log',
           title='Two Systems with Same Capacity and Different Effective Order')
#    plt.savefig(test_name + ".pdf")



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

    # params for baseline chaotic system
#    r1 = np.array([1.7741, 1.0971, 1.5466, 4.4116])
#    k1 = np.array([100., 100., 100., 100.])
#    k1 = np.array([1., 1., 1., 1.])
#    a1 = np.array([[1., 2.419, 2.248, 0.0023],
#                       [0.001, 1., 0.001, 1.3142],
#                       [2.3818, 0.001, 1., 0.4744],
#                       [1.21, 0.5244, 0.001, 1.]])
#    init1 = np.array([5., 5., 5., 5.])
#    init_trans1 = np.array([8., 8., 8., 8.])
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
#    alpha_s = np.array([[1., 0.9, 1., 1.],
#                       [1., 1., 1., 1.],
#                       [1., 1., 1., 1.],
#                       [1., 1., 1., 1.]])
#    a2 = np.array([[1., 2.419, 2.248, 0.0023],
#                       [0.001, 1., 0.001, 1.3142],
#                       [2.3818, 0.001, 1., 0.4744],
#                       [1.21, 0.5244, 0.001, 1.]])

    # params for baseline non-chaotic system
    r3 = _r(0.2, 1.0)
    k3 = k1
    init3 = init1
    init_trans3 = init_trans1
#    a3 = np.array([[1., 2.5, 2.5, 0.1],
#                        [0.1, 1., 0.1, 1.5],
#                        [2.5, 0.1, 1., 0.5],
#                        [1.5, 0.5, 0.1, 1.]])
    a3 = _A(0.2, 1.0)

    # params for variable non-chaotic system
    r4 = r3
    k4 = k1
    a4 = a3
    init4 = init1
    init_trans4 = init_trans1
#    a4 = np.array([[1., 2.5, 2.5, 0.1],
#                        [0.1, 1., 0.1, 1.5],
#                        [2.5, 0.1, 1., 0.5],
#                        [1.5, 0.5, 0.1, 1.]])

    # params for baseline non-chaotic system
    r5 = _r(0.1, 1.0)
    k5 = k1
    init5 = init1
    init_trans5 = init_trans1
#    a3 = np.array([[1., 2.5, 2.5, 0.1],
#                        [0.1, 1., 0.1, 1.5],
#                        [2.5, 0.1, 1., 0.5],
#                        [1.5, 0.5, 0.1, 1.]])
    a5 = _A(0.1, 1.0)

    # params for variable non-chaotic system
    r6 = r5
    k6 = k1
    a6 = a5
    init6 = init1
    init_trans6 = init_trans1
#    a4 = np.array([[1., 2.5, 2.5, 0.1],
#                        [0.1, 1., 0.1, 1.5],
#                        [2.5, 0.1, 1., 0.5],
#                        [1.5, 0.5, 0.1, 1.]])



    # alpha3 = alpha1 * np.power(alpha_s, 1.2)
    params1 = [r1, k1, a1, init1, init_trans1]
    params2 = [r2, k2, a2, init2, init_trans2]

#    params3 = [r1, k1, alpha3, init1, init_trans1]
    params3 = [r3, k3, a3, init3, init_trans3]
#    params4 = [r2, k2, alpha3, init2, init_trans2]
    params4 = [r4, k4, a4, init4, init_trans4]

    params5 = [r5, k5, a5, init5, init_trans5]
    params6 = [r6, k6, a6, init6, init_trans6]

    dists1 = []
    dists2 = []
    dists3 = []
    lens = []
    points = 101
    f = np.linspace(0.95, 1.05, points)
    for x in trange(points):
#        n = (x/50 - 1)*0.1
        beta = f[x]
        lens.append(beta)
        data_name = test_name + "-" + str(beta)
        data_path = join("test_data", data_name + ".npy")
#        tmp_val = 2.419 + n
#        tmp_a = np.array([[1., tmp_val, 2.248, 0.0023],
#                           [0.001, 1., 0.001, 1.3142],
#                           [2.3818, 0.001, 1., 0.4744],
#                           [1.21, 0.5244, 0.001, 1.]])
#        params2[2][0,1] = 2.419 + n
#        params4[2][0,1] = 2.5 + n
        params2[0] = _r(0., beta)
        params2[2] = _A(0., beta)

        params4[0] = _r(0.2, beta)
        params4[2] = _A(0.2, beta)

        params6[0] = _r(0.1, beta)
        params6[2] = _A(0.1, beta)
#        temp_a = alpha1 * np.power(alpha_s, n)
#        params2[2] = tmp_a
        # lens.append(tmp_val)
#        params = [params1, params2]

#        tmp_val2 = 2.5 + n
#        tmp_a2 = np.array([[1., tmp_val2, 2.5, 0.1],
#                           [0.1, 1., 0.1, 1.5],
#                           [2.5, 0.1, 1., 0.5],
#                           [1.5, 0.5, 0.1, 1.]])
#        params4[2] = tmp_a2
#        params0 = [params3, params2]
#        data = None
#        data2 = None
#        if isfile(data_path):
#            data = np.load(data_path)
#        else:
##            data = simData(params, 100., 200, no_overlay, range_cover=False)
##            data2 = simData(params0, 100., 200, no_overlay, range_cover=False)
#            data1 = simData([params1, 100., 200, no_overlay, range_cover=False)
#            data2 = simData([params0, 100., 200, no_overlay, range_cover=False)
#            data3 = simData([params0, 100., 200, no_overlay, range_cover=False)
#            data4 = simData([params0, 100., 200, no_overlay, range_cover=False)
        data1 = simData([params1], 25., 1000, no_overlay, range_cover=False)
        data2 = simData([params2], 25., 1000, no_overlay, range_cover=False)
        data3 = simData([params3], 25., 1000, no_overlay, range_cover=False)
        data4 = simData([params4], 25., 1000, no_overlay, range_cover=False)
#        data5 = simData([params5], 25., 1000, no_overlay, range_cover=False)
#        data6 = simData([params6], 25., 1000, no_overlay, range_cover=False)
#            # np.save(data_path, data)
        dist1 = pair_dist(data1, data2, min_len=600)
        dist2 = pair_dist(data3, data4, min_len=600)
        dist3 = pair_dist(data3, data2, min_len=600)
#        dist3 =  pair_dist(data5, data6, min_len=600)
        dists1.append(dist1)
        dists2.append(dist2)
        dists3.append(dist3)

    d1 = simData([params1], 25., 1000, no_overlay, range_cover=False)
    params2[0] = _r(0., 0.9)
    params2[2] = _A(0., 0.9)
    d2 = simData([params2], 25., 1000, no_overlay, range_cover=False)
    d3 = simData([params3], 25., 1000, no_overlay, range_cover=False)
    params4[0] = _r(0.2, 0.9)
    params4[2] = _A(0.2, 0.9)
    d4 = simData([params4], 25., 1000, no_overlay, range_cover=False)
    d5 = simData([params5], 25., 1000, no_overlay, range_cover=False)
    params6[0] = _r(0.1, 0.9)
    params6[2] = _A(0.1, 0.9)
    d6 = simData([params6], 25., 1000, no_overlay, range_cover=False)

    plt.figure()
    for ii in range(4):
        plt.plot(d1[0][0][:,ii])
    plt.figure()
    for ii in range(4):
        plt.plot(d2[0][0][:,ii])
    plt.figure()
    for ii in range(4):
        plt.plot(d3[0][0][:,ii])
    plt.figure()
    for ii in range(4):
        plt.plot(d4[0][0][:,ii])
    plt.figure()
    for ii in range(4):
        plt.plot(d5[0][0][:,ii])
    plt.figure()
    for ii in range(4):
        plt.plot(d6[0][0][:,ii])

    fig, ax = plt.subplots()
    ax.plot(lens, dists1, 'r-', label='chaotic transition relative to chaotic base')
#    ax.plot(lens, dists3, 'r+', label='chaotic transition relative to non-chaotic base')
    ax.plot(lens, dists2, 'k-', label='non-chaotic variation relative to non-chaotic base, alpha=0.2')
    ax.plot(lens, dists3, 'b+', label='chaotic variation relative to non-chaotic base, alpha=0.2')
    ax.legend()
    ax.set(xlabel='beta', ylabel='distance',
           title='Two Systems with Different Interaction Term through Chaos')
#    plt.savefig(test_name + "-long.pdf")

#    dists1 = np.array(dists[0:3])
#    dists2 = np.array(dists[-4:-1])
#    self.assertTrue(dists1.var() > dists2.var())



if __name__ == '__main__':
    # process command line options
#    experiments = [test_same_more_diff,test_same_cap_more_linear, test_same_cap_more_2nd_order,test_into_chaos_interaction]
    experiments = [test_into_chaos]
    # run experiments
    for ex in experiments:
        ex()

    plt.show()
