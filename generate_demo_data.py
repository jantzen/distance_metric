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

