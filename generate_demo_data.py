import numpy as np
from scipy.integrate import ode
from scipy import zeros_like
import matplotlib.pyplot as plt
import eugene as eu
from eugene.src.data_prep.fragment_timeseries import *
from eugene.src.data_prep.initial_conditions import *
import dtw


# set up Kuramoto systems

N = 3
K1 = 0.
K2 = 1.0

omega= np.linspace(np.pi, 3. * np.pi / 2., N)

def model(t, theta, arg1):
    K = arg1[0]
    omega = arg1[1]
    dtheta = zeros_like(theta)
    sum1 = [0.] * N
    for ii in range(N):
        for jj in range(N):
            sum1[ii] += np.sin(theta[jj] - theta[ii])
            
        dtheta[ii] = omega[ii] + (K / N) * sum1[ii] 
    
    return dtheta

# initial condition
theta0 = np.linspace(0.1, 2., N)
theta1 = np.linspace(0.3, 6, N)

# time points
t0 = 0.
#t1 = 5.
t1 = 25.
resolution = 25 * 10 **4
dt = (t1 - t0) / resolution

# solve ODE_A at each timestep
r = ode(model).set_integrator('lsoda')
r.set_initial_value(theta0, t0).set_f_params([K1, omega])
x = []
t = []
while r.successful() and r.t < t1:
    t.append(r.t)
    tmp = r.integrate(r.t+dt)
    x.append(np.array([np.cos(tmp), np.sin(tmp)]).reshape(-1,1))

t = np.array(t).reshape(1,-1)
theta1 = np.concatenate(x, axis=1)
theta1 = np.concatenate([t, theta1], axis=0)

## trans
#r = ode(model).set_integrator('lsoda')
#r.set_initial_value(theta1, t0).set_f_params([K1, omega])
#x = []
#t = []
#while r.successful() and r.t < t1:
#    t.append(r.t)
#    tmp = r.integrate(r.t+dt)
#    x.append(np.array([np.cos(tmp), np.sin(tmp)]).reshape(-1,1))
#
#theta1_trans = np.concatenate(x, axis=1)

# solve ODE_B at each timestep
r = ode(model).set_integrator('lsoda')
r.set_initial_value(theta0, t0).set_f_params([K2, omega])
x = []
t = []
while r.successful() and r.t < t1:
    t.append(r.t)
    tmp = r.integrate(r.t+dt)
    x.append(np.array([np.cos(tmp), np.sin(tmp)]).reshape(-1,1))

t = np.array(t).reshape(1,-1)
theta2 = np.concatenate(x, axis=1)
theta2 = np.concatenate([t, theta2], axis=0)

## trans
#r = ode(model).set_integrator('lsoda')
#r.set_initial_value(theta1, t0).set_f_params([K2, omega])
#x = []
#t = []
#while r.successful() and r.t < t1:
#    t.append(r.t)
#    tmp = r.integrate(r.t+dt)
#    x.append(np.array([np.cos(tmp), np.sin(tmp)]).reshape(-1,1))
#
#theta2_trans = np.concatenate(x, axis=1)

# solve ODE_C at each timestep
r = ode(model).set_integrator('lsoda')
r.set_initial_value(theta0, t0).set_f_params([K1, omega * 2.])
x = []
t = []
while r.successful() and r.t < t1:
    t.append(r.t)
    tmp = r.integrate(r.t+dt)
    x.append(np.array([np.cos(tmp), np.sin(tmp)]).reshape(-1,1))

t = np.array(t).reshape(1,-1)
theta3 = np.concatenate(x, axis=1)
theta3 = np.concatenate([t, theta3], axis=0)

### Save the data
np.savetxt('./data/demo_theta1', theta1)
np.savetxt('./data/demo_theta2', theta2)
np.savetxt('./data/demo_theta3', theta3)

## trans
#r = ode(model).set_integrator('lsoda')
#r.set_initial_value(theta1, t0).set_f_params([K1, omega * 2.])
#x = []
#t = []
#while r.successful() and r.t < t1:
#    t.append(r.t)
#    tmp = r.integrate(r.t+dt)
#    x.append(np.array([np.cos(tmp), np.sin(tmp)]).reshape(-1,1))
#
#theta3_trans = np.concatenate(x, axis=1)


#c = 0.2

#theta1_untrans += c * np.random.random_sample(theta1_untrans.shape)
#theta1_trans += c * np.random.random_sample(theta1_trans.shape)
#theta2_untrans += c * np.random.random_sample(theta2_untrans.shape)
#theta2_trans += c * np.random.random_sample(theta2_trans.shape)
#theta3_untrans += c * np.random.random_sample(theta3_untrans.shape)
#theta3_trans += c * np.random.random_sample(theta3_trans.shape)

## plot results
#fig, ax = plt.subplots(3, 2, sharex = 'col')
#for ii in range(2 * N):
#    ax[0][0].plot(t, theta1_untrans[ii,:], 'b-')
#ax[0][0].set_ylabel('System 1')
#ax[0][0].set_title('Initial Conditions 1')
#for ii in range(2 * N):
#    ax[0][1].plot(t, theta1_trans[ii,:], 'r-')
#ax[0][1].set_title('Initial Conditions 2')
#for ii in range(2 * N):
#    ax[1][0].plot(t, theta2_untrans[ii,:], 'b-')
#ax[1][0].set_ylabel('System 2')
#for ii in range(2 * N):
#    ax[1][1].plot(t, theta2_trans[ii,:], 'r-')
#for ii in range(2 * N):
#    ax[2][0].plot(t, theta3_untrans[ii,:], 'b-')
#ax[2][0].set_ylabel('System 3')
#for ii in range(2 * N):
#    ax[2][1].plot(t, theta3_trans[ii,:], 'r-')
#ax[2][0].set_xlabel('time')
#ax[2][1].set_xlabel('time')

## plot results
#fig, ax = plt.subplots(3, 1, sharex = 'col')
#for ii in range(2 * N):
#    ax[0].plot(t, theta1[ii,:], 'b-')
#ax[0].set_ylabel('System 1')
##ax[0][0].set_title('Initial Conditions 1')
##for ii in range(2 * N):
##    ax[0][1].plot(t, theta1_trans[ii,:], 'r-')
##ax[0][1].set_title('Initial Conditions 2')
#for ii in range(2 * N):
#    ax[1].plot(t, theta2[ii,:], 'b-')
#ax[1].set_ylabel('System 2')
##for ii in range(2 * N):
##    ax[1][1].plot(t, theta2_trans[ii,:], 'r-')
#for ii in range(2 * N):
#    ax[2].plot(t, theta3[ii,:], 'b-')
#ax[2].set_ylabel('System 3')
#for ii in range(2 * N):
#    ax[2][1].plot(t, theta3_trans[ii,:], 'r-')
#ax[2][0].set_xlabel('time')
#ax[2][1].set_xlabel('time')


#untrans = [[theta1_untrans], [theta2_untrans], [theta3_untrans]]
#trans = [[theta1_trans], [theta2_untrans], [theta3_trans]]

#dmat = eu.dynamical_distance.distance_matrix(untrans, trans, 10, 100)
#x1 = np.concatenate([theta1_untrans, theta1_trans], axis=0).T
#x2 = np.concatenate([theta2_untrans, theta2_trans], axis=0).T
#x3 = np.concatenate([theta3_untrans, theta3_trans], axis=0).T
#
#d12 = EnergyDistance(x1, x2)
#d13 = EnergyDistance(x1, x3)
#d23 = EnergyDistance(x2, x3)

# print([d12, d13, d23])
#print("D_D matrix:")
#print(dmat) 

split_data = split_timeseries([theta1, theta2, theta3], 10**4)
untrans, trans = choose_untrans_trans(split_data, 3)
dmat = eu.dynamical_distance.distance_matrix(untrans, trans, 10, 100)
print("D_D matrix:")
print(dmat) 
np.savetxt('./data/demo_D_D', dmat)

### Compute differences in means
mu1 = np.mean(theta1, axis=1)
mu2 = np.mean(theta2, axis=1)
mu3 = np.mean(theta3, axis=1)

means = np.zeros_like(dmat)
for ii, xx in enumerate([mu1, mu2, mu3]):
    for jj, yy in enumerate([mu1, mu2, mu3]):
        means[ii, jj] = np.linalg.norm(xx - yy)
print("difference of means matrix:")
print(means)
np.savetxt('./data/demo_diff_of_means', means)

### Compute DTW distances
dtw_d = np.zeros_like(dmat)
for ii, xx in enumerate([theta1, theta2, theta3]):
    for jj, yy in enumerate([theta1, theta2, theta3]):
        alignment = dtw.dtw(xx, yy)
#        dtw_d[ii, jj] = alignment.distance
        dtw_d[ii, jj] = alignment.normalizedDistance
print(dtw_d)
print("DTW matrix:")
np.savetxt('./data/demo_dtw', dtw_d)

#plt.show()

