from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
import eugene as eu
from scipy.io import matlab
from scipy import zeros_like
import pandas as pd

def f(t, theta, args):
    """ Kuramoto system.
        args[0]: K
        args[1]: list of frequencies, omega
    """
    dtheta =zeros_like(theta)
    summand = [0] * N
    for ii in range(N):
        for jj in range(N):
            summand[ii] += np.sin(theta[jj] - theta[ii])
        dtheta[ii] = args[1][ii] + (args[0] / N) * summand[ii]
    return dtheta

def f_transition(t, theta, args):
    """ Kuramoto system.
        args[0]: K0
        args[1]: list of frequencies, omega
        args[2]: dk/dt
	args[3]: transition time
    """
    dtheta =zeros_like(theta)
    summand = [0] * N
    for ii in range(N):
        for jj in range(N):
            summand[ii] += np.sin(theta[jj] - theta[ii])
        dtheta[ii] = args[1][ii] + ((args[0] + args[2] * (t - args[3])) / N) * summand[ii]
    return dtheta


# parameters
N = 3
K = 1.0

#omega = np.linspace(np.pi * 1.1, 2.2 * np.pi, N)
#omega = np.linspace(1.5, 2., N)
omega= np.linspace(np.pi, 3. * np.pi / 2., N)
add_noise = True
std = 0.05
alpha = 1.0

# initial condition
t0 = 0.
theta0 =  np.linspace(0.1, 2., N)

#dt = 5. * 10**(-4)
dt = 5. * 10**(-4)

#t1 = 1000. # onset of first transition
#t2 = 2000. # onset of second transition
#tf = 3000. # final time

# properties of the transition
transition_period = 10.

t1 = 100. - transition_period / 2 # onset of transition
t2 = 200. 

#transition_period = dt 
#dK = -1.0 / (transition_period / dt)
#domega = omega / (transition_period / dt)
dK = -1.0 
dKdt = dK / transition_period
domega = omega 


times = []
theta_vals = []

times.append(t0)
theta_vals.append(theta0.reshape(1,-1))

#while r.successful() and r.t < t1:
#    times.append(r.t+dt)
#    theta_vals.append(r.integrate(r.t+dt).reshape(1,-1))
#
#while r.successful() and t1 <= r.t <  t1 + transition_period:
#    K += dK
#    r.set_f_params([K, omega])
#    times.append(r.t+dt)
#    theta_vals.append(r.integrate(r.t+dt).reshape(1,-1))
#
#while r.successful() and t1 + transition_period <= r.t < t2:
#    times.append(r.t+dt)
#    theta_vals.append(r.integrate(r.t+dt).reshape(1,-1))
#
#while r.successful() and t2 <= r.t <  t2 + transition_period:
#    omega += domega
#    r.set_f_params([K, omega])
#    times.append(r.t+dt)
#    theta_vals.append(r.integrate(r.t+dt).reshape(1,-1))
#
#while r.successful() and t2 + transition_period <= r.t < tf:
#    times.append(r.t+dt)
#    theta_vals.append(r.integrate(r.t+dt).reshape(1,-1))

r1 = ode(f).set_integrator('lsoda')
r1.set_initial_value(theta0, t0).set_f_params([K, omega])

print("Integrating first segment...")
while r1.successful() and r1.t < t1:
    times.append(r1.t+dt)
    theta_vals.append(r1.integrate(r1.t+dt).reshape(1,-1))

r2 = ode(f_transition).set_integrator('lsoda')
r2.set_initial_value(r1._y, r1.t).set_f_params([K, omega, dKdt, t1])

print("Integrating transition zone...")
while r2.successful() and t1 <= r2.t < t1 + transition_period:
    times.append(r2.t+dt)
    theta_vals.append(r2.integrate(r2.t+dt).reshape(1,-1))

K += dK
r3 = ode(f).set_integrator('lsoda')
r3.set_initial_value(r2._y, r2.t).set_f_params([K, omega])

print("Integrating final segment...")
while r3.successful() and t1 + transition_period <= r3.t < t2:
    times.append(r3.t+dt)
    theta_vals.append(r3.integrate(r3.t+dt).reshape(1,-1))

print("Finished integrating.")
#omega += domega
#r3 = ode(f).set_integrator('lsoda')
#r3.set_initial_value(r2._y, r2.t).set_f_params([K, omega])
#
#while r3.successful() and t2 <= r3.t < tf:
#    times.append(r3.t+dt)
#    theta_vals.append(r3.integrate(r3.t+dt).reshape(1,-1))

theta_vals = np.concatenate(theta_vals, axis=0)

x_vals = np.cos(theta_vals)
y_vals = np.sin(theta_vals)

# add noise
if add_noise:
    x_vals = x_vals + np.random.normal(0., std, size = x_vals.shape)
    y_vals = y_vals + np.random.normal(0., std, size = y_vals.shape)

xy_vals = np.concatenate([np.array(times).reshape(-1,1), x_vals, y_vals], axis=1)

df = pd.DataFrame(data=xy_vals, 
        columns=['time', 'x1', 'x2', 'x3', 'y1', 'y2', 'y3'])
df.to_csv('./data/kuramoto_data.csv')

#plt.figure()
#for ii in range(N):
#    plt.plot(times, x_vals[:,ii], times, y_vals[:,ii])


#width_times = np.asarray([250.])
#lag_times = np.asarray([50.])
#step_time = 23.
#
#step_size = int(round(step_time / dt)) 
#widths = np.round(width_times / dt).astype('int')
#lags = np.round(lag_times / dt).astype('int')
#
#plot_rows = len(widths)
#plot_cols = len(lags)
#
#i = 1
#
#plt.figure()
#for window_width in widths:
#    for lag in lags:
#        data = xy_vals.T
#        np.save('raw_data', np.concatenate([np.array(times).reshape(-1,1),
#            xy_vals], axis=1))
#        diff = eu.dd_scan.DiffScanner(data, window_width=window_width, step_size=step_size, lag=lag)
#        print('starting diff scan...')
#        frags = int(window_width / 200)
#        reps = int(frags / 10)
#        diff.start_scan(frags=frags, reps=reps, free_cores=4, alpha=alpha)
##       diff.start_scan(frags=1000, reps=100, free_cores=2)
##       with open('./test_diff_kuramoto_scan_window_' + str(window_width) + '_lag_' + str(lag) + '.pkl', 'wb+') as f:
##           pickle.dump(diff._scan, f)
#        diff_times = []
#        for ii in diff._scan[:,0]:
#            index = int(ii + window_width + lag / 2 - 1)
#            diff_times.append(times[index])
#
#        plt.subplot(plot_rows,plot_cols,i)
#        plt.plot(diff_times, diff._scan[:,1])
#        title = 'width =' + str(window_width) + '; lag = ' + str(lag)
#        plt.title(title)
#        i += 1
#
#plt.figure()
#
#i = 1
#
#for window_width in widths:
#    for lag in lags:
##       data = np.concatenate([theta_vals[:,0].reshape(1,-1), np.random.rand(1, theta_vals.shape[0])], axis=0)
##       data = np.concatenate([xy_vals[:,0].reshape(1,-1), xy_vals[:,0].reshape(1,-1)], axis=0)
#        data = xy_vals[:,:3].T
#        diff = eu.dd_scan.DiffScanner(data, window_width=window_width, step_size=step_size, lag=lag)
#        print('starting diff scan...')
##       diff.start_scan(frags=500)
#        frags = (window_width / 200)
#        reps = (frags / 10)
#        diff.start_scan(frags=frags, reps=reps, free_cores=4, alpha=alpha)
##       with open('./test_diff_scan_kuramoto_partial_info_window_' + str(window_width) + '_lag_' + str(lag) + '.pkl', 'wb+') as f:
##           pickle.dump(diff._scan, f)
#        diff_times = []
#        for ii in diff._scan[:,0]:
#            index = int(ii + window_width + lag / 2 - 1)
#            diff_times.append(times[index])
#
#        plt.subplot(plot_rows,plot_cols,i)
#        plt.plot(diff_times, diff._scan[:,1])
#        title = 'Partial information. width =' + str(window_width) + '; lag = ' + str(lag)
#        plt.title(title)
#        i += 1

plt.show()
