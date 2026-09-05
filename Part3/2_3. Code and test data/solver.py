import numpy as np

def rk4_step(func, t, y, dt):
    """
    Fourth-order Runge-Kutta (RK4) integrator.

    Chosen for:
    - O(dt^4) local truncation error
    - Good stability for smooth ODEs
    - Balance between accuracy and computational cost

    With DT = 0.02, RK4 remains stable for the chosen gains
    (K_P, K_V, K_REP) without requiring implicit methods.
    """
    k1 = func(t, y)
    k2 = func(t + 0.5*dt, y + 0.5*dt*k1)
    k3 = func(t + 0.5*dt, y + 0.5*dt*k2)
    k4 = func(t + dt, y + dt*k3)
    
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
