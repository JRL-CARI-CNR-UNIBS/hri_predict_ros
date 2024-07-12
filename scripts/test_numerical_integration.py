import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, KalmanFilter
from filterpy.common import Q_discrete_white_noise, runge_kutta4
from numpy.random import randn

np.set_printoptions(precision=5, suppress=False)

def dynamics(x, t):
    # double integrator dynamics
    x_dot = np.zeros(3)
    x_dot[0] = x[1]    # p_dot = v
    x_dot[1] = x[2]    # v_dot = a
    x_dot[2] = 0       # a_dot = u
    return x_dot


def fx(x, dt, t):
    # state transition function - predict next state based
    # on constant velocity model x = vt + x_0
    # F = np.array([[1, dt, 0],
    #               [0, 1, dt],
    #               [0, 0, 0]], dtype=float)
    # return np.dot(F, x) + np.array([0, 0, 1])

    return runge_kutta4(x, t, dt, dynamics)

def hx(x):
   # measurement function - convert state into a measurement
   # where measurements are [x_pos]
   return np.array([x[0]])

def F(dt):
    return np.array([[1, dt, 0],
                      [0, 1, dt],
                      [0, 0, 1]], dtype=float)

def H():
    return np.array([[1, 0, 0]], dtype=float)

def B():
    return np.array([[0],
                      [0],
                      [0]], dtype=float)


dt1 = 0.01
dt2 = 0.5

n_steps1 = 50
n_steps2 = 1

n_states = 3
init_state = np.array([-1., 1., 1.])

# create sigma points to use in the filter. This is standard for Gaussian processes
points = MerweScaledSigmaPoints(n_states, alpha=.1, beta=2., kappa=-1)

kf1 = UnscentedKalmanFilter(dim_x=n_states, dim_z=n_states, dt=dt1, fx=fx, hx=hx, points=points)
kf1.x = init_state # initial state
kf1.P *= 0.2 # initial uncertainty
z_std = 0.1
kf1.R = np.eye(n_states) * z_std**2 # 1 standard
kf1.Q = Q_discrete_white_noise(dim=n_states, dt=dt1, var=0.01**2, block_size=1)
# kf1.Q = np.zeros((n_states, n_states))

kf2 = UnscentedKalmanFilter(dim_x=n_states, dim_z=n_states, dt=dt2, fx=fx, hx=hx, points=points)
kf2.x = init_state # initial state
kf2.P *= 0.2 # initial uncertainty
z_std = 0.1
kf2.R = np.eye(n_states) * z_std**2 # 1 standard
kf2.Q = Q_discrete_white_noise(dim=n_states, dt=dt2, var=1**2, block_size=1)
# kf2.Q = np.zeros((n_states, n_states))

kf_lin = KalmanFilter(dim_x=n_states, dim_z=n_states)
kf_lin.F = F(dt2)
kf_lin.H = H()
kf_lin.x = init_state
kf_lin.P *= 0.2
kf_lin.R = np.eye(n_states) * z_std**2
kf_lin.Q = Q_discrete_white_noise(dim=n_states, dt=dt1, var=0.01**2, block_size=1)
# kf_lin.Q = np.zeros((n_states, n_states))

print("kf1.Q : ", kf1.Q)
print("kf2.Q : ", kf2.Q)

print("\n")

t = 0
print("kf1.x : ", kf1.x)
for _ in range(n_steps1):
    fx_args = {'t': t}
    kf1.predict(**fx_args)
    t += dt1
    print("kf1.x : ", kf1.x)

print("kf1.P : ", kf1.P)

print("\n")

t = 0
print("kf2.x : ", kf2.x)
for _ in range(n_steps2):
    fx_args = {'t': t}
    kf2.predict(**fx_args)
    t += dt2
    print("kf2.x : ", kf2.x)

print("kf2.P : ", kf2.P)

print("\n")

t = 0
print("kf_lin.x : ", kf_lin.x)
for _ in range(n_steps2):
    kf_lin.predict()
    t += dt2
    print("kf_lin.x : ", kf_lin.x)

