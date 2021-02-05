# generate_stochastic_data.py


import eugene as eu
import numpy as np
import matplotlib.pyplot as plt
from eugene.src.tools.LVDSim import *
from tqdm import trange
from os.path import join, isfile

import pdb

def no_overlay(x):
    return x

def mean_overlay(x):
    return np.mean(x, axis=1)

def pair_dist(data1, data2, min_len=10, steps=90):

    # convert to numpy
    data1T = []
    data2T = []
    for dataset in data1:
        datasetT = []
        for curve in dataset:
            datasetT.append(np.array(curve))
        data1T.append(datasetT)
    for dataset in data2:
        datasetT = []
        for curve in dataset:
            datasetT.append(np.array(curve))
        data2T.append(datasetT)
            
    untrans = [data1T[0], data2T[0]]
    trans = [data1T[1], data2T[1]]
    try:
        dmat = eu.dynamical_distance.distance_matrix(untrans, trans, min_len, steps=steps)
    except:
        print("dmat computation error!")
        print("Length of untrans[0]: {}".format(len(untrans[0])))
        print("Shape of untrans[0][0]: {}".format(untrans[0][0].shape))
        dmat = np.nan * np.ones((2,2))
    return dmat[0,1]


def stochastic_generation_loop(test_name, params, max_time, n_time,
                                     overlay, reps, data_dir, overlay_tag=False):
    if overlay_tag:
        data_name = test_name + 'r_' + str(params[1][0]) + '-k_' + str(params[1][1]) \
                + '_overlay'
    else:
        data_name = test_name + 'r_' + str(params[1][0]) + '-k_' + str(params[1][1])
    data_path = join(data_dir, data_name + ".npy")
    if isfile(data_path):
        data = np.load(data_path)
    else:
        data = simData(params, max_time, n_time, overlay,
                       stochastic_reps=reps, range_cover=False)
        np.save(data_path, data)
    dist = pair_dist(data[0], data[1])
    return [dist, params[0][3][0]]


def stochastic_generation_loop_sigma(test_name, params, max_time, n_time,
                                     overlay, reps, data_dir, overlay_tag=False):
    if overlay_tag:
        data_name = test_name + 'r_' + str(params[1][0]) + '-k_' + str(params[1][1]) \
                + '_overlay'
    else:
        data_name = test_name + 'r_' + str(params[1][0]) + '-k_' + str(params[1][1])
    data_path = join(data_dir, data_name + ".npy")
    if isfile(data_path):
        data = np.load(data_path)
    else:
        data = simData(params, max_time, n_time, overlay,
                       stochastic_reps=reps, range_cover=False)
        np.save(data_path, data)
    dist = pair_dist(data[0], data[1])
    return [dist, params[0][3][0]]




################################################################################
# DATA GENERATION METHODS


def test_stochastic(data_dir="./data/"):
    print ("test_stochastic")

    sigma = np.array([0.1, 0.1])

    r = np.array([1., 2.])

    k = np.array([100., 100.])

    alpha = np.array([[1., 0.5], [0.7, 1.]])

    init1 = np.array([5., 5.])
    init2 = init1

    init_trans1 = np.array([8., 8.])
    init_trans2 = init_trans1

    cpus = max(cpu_count() - 2, 1)
    test_name = "test_stochastic-"
    f = np.linspace(1., 5., 20)
    combined1 = Parallel(n_jobs=cpus)(
        delayed(stochastic_generation_loop)(test_name,
                                                 [[r, k, alpha,
                                                   sigma,
                                                   init1, init_trans1],
                                                  [r * f[x], k, alpha,
                                                   sigma,
                                                   init2, init_trans2]],
                                                 5., 500, no_overlay,
                                                 10, data_dir) for x in range(20))
    dists1 = np.array(combined1)[:, 0]
    combined2 = Parallel(n_jobs=cpus)(
        delayed(stochastic_generation_loop)(test_name,
                                                 [[r, k, alpha,
                                                   sigma,
                                                   init1, init_trans1],
                                                  [r, k * f[x], alpha,
                                                   sigma,
                                                   init2, init_trans2]],
                                                 5., 500, no_overlay,
                                                 10, data_dir) for x in range(20))
    dists2 = np.array(combined2)[:, 0]
 
    data = np.concatenate([f.reshape(1,-1), dists1.reshape(1,-1), dists2.reshape(1,-1)], axis=0)
    np.savetxt('./data/test_stochastic', data)
#    fig, ax = plt.subplots()
#    ax.plot(f, dists1, 'bo')
#    ax.plot(f, dists2, 'ro')
#    ax.set(xlabel='scale factor', ylabel='distance',
#           title='Distance Between Two Stochastic Systems of the Same Kind')


def test_stochastic_nonssd(data_dir="./data/"):
    print ("test_stochastic_nonssd")

    sigma = np.array([0.1, 0.1])

    r = np.array([1., 2.])

    k = np.array([100., 100.])

    alpha = np.array([[1., 0.5], [0.7, 1.]])

    init1 = np.array([5., 5.])
    init2 = init1

    init_trans1 = np.array([8., 8.])
    init_trans2 = init_trans1

    cpus = max(cpu_count() - 2, 1)
    test_name = "test_stochastic_nonssd-"
    f = np.linspace(1., 5., 20)
    combined1 = Parallel(n_jobs=cpus)(
        delayed(stochastic_generation_loop)(test_name,
                                                 [[r, k, alpha,
                                                   sigma,
                                                   init1, init_trans1],
                                                  [r * f[x], k, alpha,
                                                   sigma,
                                                   init2, init_trans2]],
                                                 5., 500, mean_overlay,
                                                 10, data_dir, overlay_tag=True) for x in range(20))
    dists1 = np.array(combined1)[:, 0]
    combined2 = Parallel(n_jobs=cpus)(
        delayed(stochastic_generation_loop)(test_name,
                                                 [[r, k, alpha,
                                                   sigma,
                                                   init1, init_trans1],
                                                  [r, k * f[x], alpha,
                                                   sigma,
                                                   init2, init_trans2]],
                                                 5., 500, mean_overlay,
                                                 10, data_dir, overlay_tag=True) for x in range(20))
    dists2 = np.array(combined2)[:, 0]
    data = np.concatenate([f.reshape(1,-1), dists1.reshape(1,-1), dists2.reshape(1,-1)], axis=0)
    np.savetxt('./data/test_stochastic_nonssd', data)

#    fig, ax = plt.subplots()
#    ax.plot(f, dists1, 'bo')
#    ax.plot(f, dists2, 'ro')
#    ax.set(xlabel='scale factor', ylabel='distance',
#           title='Distance Between Two Stochastic Systems of the Same Kind')


def test_stochastic_more_noise(data_dir="./data/"):
    print ("test_stochastic_more_noise")

    sigma = np.array([0.1, 0.1])

    r1 = np.array([1., 2.])
    r2 = 2. * r1 

    k1 = np.array([100., 100.])
    k2 = 2. * k1

    alpha = np.array([[1., 0.5], [0.7, 1.]])

    init1 = np.array([5., 5.])
    init2 = init1

    init_trans1 = np.array([8., 8.])
    init_trans2 = init_trans1

    cpus = max(cpu_count() - 2, 1)
    test_name = "test_stochastic_more_noise-"
    combined1 = Parallel(n_jobs=cpus)(
        delayed(stochastic_generation_loop_sigma)(test_name,
                                                 [[r1, k1, alpha,
                                                   x * sigma,
                                                   init1, init_trans1],
                                                  [r2, k1, alpha,
                                                   x * sigma,
                                                   init2, init_trans2]],
                                                 5., 500, no_overlay,
                                                 10, data_dir) for x in range(10))
    dists1 = np.array(combined1)[:, 0]

    combined2 = Parallel(n_jobs=cpus)(
        delayed(stochastic_generation_loop_sigma)(test_name,
                                                 [[r1, k1, alpha,
                                                   x * sigma,
                                                   init1, init_trans1],
                                                  [r1, k2, alpha,
                                                   x * sigma,
                                                   init2, init_trans2]],
                                                 5., 500, no_overlay,
                                                 10, data_dir) for x in range(10))
    dists2 = np.array(combined2)[:, 0]

    combined3 = Parallel(n_jobs=cpus)(
        delayed(stochastic_generation_loop_sigma)(test_name,
                                                 [[r1, k1, alpha,
                                                   x * sigma,
                                                   init1, init_trans1],
                                                  [r2, k1, alpha,
                                                   x * sigma,
                                                   init2, init_trans2]],
                                                 5., 500, mean_overlay,
                                                 10, data_dir, overlay_tag=True) for x in range(10))
    dists3 = np.array(combined3)[:, 0]

    combined4 = Parallel(n_jobs=cpus)(
        delayed(stochastic_generation_loop_sigma)(test_name,
                                                 [[r1, k1, alpha,
                                                   x * sigma,
                                                   init1, init_trans1],
                                                  [r1, k2, alpha,
                                                   x * sigma,
                                                   init2, init_trans2]],
                                                 5., 500, mean_overlay,
                                                 10, data_dir, overlay_tag=True) for x in range(10))
    dists4 = np.array(combined4)[:, 0]
    sigma = np.array(combined4)[:, 1] 


    data = np.concatenate([sigma.reshape(1,-1), dists1.reshape(1,-1), 
        dists2.reshape(1,-1), dists3.reshape(1,-1), dists4.reshape(1,-1)], axis=0)
    np.savetxt('./data/test_stochastic_more_noise', data)
#    fig, ax = plt.subplots()
#    ax.plot(sigma, dists1, 'bo', label="same kind, SSD")
#    ax.plot(sigma, dists2, 'ro', label="different kind, SSD")
#    ax.plot(sigma, dists3, 'b+', label="same kind, non-SSD")
#    ax.plot(sigma, dists4, 'r+', label="different kind, non-SSD")
#    ax.set(xlabel='sigma', ylabel='distance',
#           title='Distance Between Two Stochastic Systems of the Same Kind')
 


if __name__ == '__main__':
    # process command line options
    experiments = [test_stochastic, test_stochastic_nonssd, test_stochastic_more_noise]

    # run experiments
    for ex in experiments:
        ex()

    plt.show()
