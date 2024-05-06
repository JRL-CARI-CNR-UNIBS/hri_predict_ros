def runge_kutta(dynamics, x0, u0, dt):
    k1 = dynamics(x0, u0)
    k2 = dynamics(x0+0.5*dt*k1, u0)
    k3 = dynamics(x0+0.5*dt*k2, u0)
    k4 = dynamics(x0+dt*k3, u0)
    return x0 + dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)