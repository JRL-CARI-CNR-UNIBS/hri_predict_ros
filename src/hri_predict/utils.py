import numpy as np

def runge_kutta(dynamics, x0, u0, dt):
    k1 = dynamics(x0, u0)
    k2 = dynamics(x0+0.5*dt*k1, u0)
    k3 = dynamics(x0+0.5*dt*k2, u0)
    k4 = dynamics(x0+dt*k3, u0)
    return x0 + dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)


def runge_kutta4(y, x, u, dx, f):
    """computes 4th order Runge-Kutta for dy/dx.

    Parameters
    ----------

    y : scalar
        Initial/current value for y
    x : scalar
        Initial/current value for x
    dx : scalar
        difference in x (e.g. the time step)
    f : ufunc(y,x)
        Callable function (y, x) that you supply to compute dy/dx for
        the specified values.

    """

    k1 = dx * f(y, x, u)
    k2 = dx * f(y + 0.5*k1, x + 0.5*dx, u)
    k3 = dx * f(y + 0.5*k2, x + 0.5*dx, u)
    k4 = dx * f(y + k3, x + dx, u)

    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.


def get_near_psd(P, max_iter=10):

    eps = 1e-3  # Small positive jitter for regularization
    increment_factor = 10  # Factor to increase eps if needed
        
    def is_symmetric(A):
        return np.allclose(A, A.T)
                    
    def is_positive_definite(A):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
        
    for _ in range(max_iter):
        if is_symmetric(P) and is_positive_definite(P):
            return P  # The matrix is suitable for Cholesky
    
        # Make P symmetric
        P = (P + P.T) / 2
    
        # Set negative eigenvalues to zero
        eigval, eigvec = np.linalg.eig(P)
        eigval[eigval < 0] = 0
        # add a jitter for strictly positive
        eigval += eps
    
        # Reconstruct the matrix
        P = eigvec.dot(np.diag(eigval)).dot(eigvec.T)

        # Force P to be real
        P = np.real(P)
    
        # Check if P is now positive definite
        if is_positive_definite(P):
            return P
    
        # Increase regularization factor for the next iteration
        eps *= increment_factor
    
    raise ValueError("Unable to convert the matrix to positive definite within max iterations.")