import numpy as np
from ddeint import ddeint
import pdb


def initials(t, M):
    if t == 0.:
        return np.random.rand(M)
    else:
        return np.zeros(M)


def nnmodel(Y, t, tau, a, c, e, omega_e):
    yvec = Y(t)
    yvec = np.array(yvec).reshape(-1,1)
    yvec_delay = Y(t - tau)
    yvec_delay = np.array(yvec_delay).reshape(-1,1)

    return -yvec + np.dot(a, c * np.tanh(yvec_delay)) + e * np.sin(omega_e * t)


def nnmodel_changepoint(Y, t, tau, a, c, e, omega_e, changepoint):
    yvec = Y(t)
    yvec = np.array(yvec).reshape(-1,1)
    yvec_delay = Y(t - tau)
    yvec_delay = np.array(yvec_delay).reshape(-1,1)
    if changepoint is not None and t > changepoint:
        c = 2. * c

    return -yvec + np.dot(a, c * np.tanh(yvec_delay)) + e * np.sin(omega_e * t)


def bondarenko(tmax, dt, tau=10., M=10, c=11.5, e=0., omega_e=0., changepoint=None):
    a = 4. * np.random.rand(M,M) - 2.   # matching range in Hively + Protopopescu
                                        # rather than Bondarekno
    g = lambda t: initials(t, M)

    tt = np.arange(0., tmax, dt)
    if changepoint is None:
        yy = ddeint(nnmodel, g, tt, fargs=(tau, a, c, e, omega_e,))
    else:
        yy = ddeint(nnmodel_changepoint, g, tt, fargs=(tau, a, c, e,
            omega_e, changepoint,))

    return tt, yy

    
