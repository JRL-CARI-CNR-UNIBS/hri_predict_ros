import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, unscented_transform
from filterpy.common import Q_discrete_white_noise
from numpy.random import randn

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
    
        # Check if P is now positive definite
        if is_positive_definite(P):
            return P
    
        # Increase regularization factor for the next iteration
        eps *= increment_factor
    
    raise ValueError("Unable to convert the matrix to positive definite within max iterations.")


def k_step_predict(kf, k) -> tuple:    
    # calculate sigma points for current mean and covariance
    sigmas = kf.points_fn.sigma_points(kf.x, kf.P)
    
    sigmas_f = np.zeros((len(sigmas), kf._dim_x))               # sigma points after passing through dynamics function
    xx = np.zeros((k, kf._dim_x))                               # mean after passing through dynamics function
    variances = np.zeros((k, kf._dim_x))                        # covariance after passing through dynamics function      

    for _ in range(k):
        # transform sigma points through the dynamics function
        sigmas_f = np.array([kf.fx(s, kf._dt) for s in sigmas])

        # pass sigmas through the unscented transform to compute prior
        x, P = unscented_transform(sigmas_f,
                                    kf.Wm,
                                    kf.Wc,
                                    kf.Q,
                                    kf.x_mean,
                                    kf.residual_x)

        # check for semi-positive definitness of P matrix and correct if needed
        P = get_near_psd(P)

        # update sigma points to reflect the new variance of the points
        sigmas = kf.points_fn.sigma_points(x, P)

        # store state means and variances for each step
        xx[_] = x
        variances[_] = np.diag(P)
    
    return xx, variances

def f(x, dt):
    block = np.array([[1, dt,  0],
                      [0,  1, dt],
                      [0,  0,  1]], dtype=float)
    return block @ x + 1.0


def h(x):
    block = np.array([[1, 0, 0]], dtype=float)
    return block @ x


def main():
    sigma_points = MerweScaledSigmaPoints(n=3, alpha=0.1, beta=2.0, kappa=1.0)
    kalman_filter1 = UnscentedKalmanFilter(dim_x=3, dim_z=1, dt=0.01, hx=h, fx=f, points=sigma_points)
    kalman_filter1.x = np.array([0., 0.6, 0.95])
    kalman_filter1.Q = Q_discrete_white_noise(dim=3, dt=0.01, var=0.01)
    # kalman_filter1.Q = np.zeros(3)

    kalman_filter2 = UnscentedKalmanFilter(dim_x=3, dim_z=1, dt=0.1, hx=h, fx=f, points=sigma_points)
    kalman_filter2.x = np.array([0., 0.6, 0.95])
    kalman_filter2.Q = Q_discrete_white_noise(dim=3, dt=0.1, var=0.01)
    # kalman_filter2.Q = np.zeros(3)

    x1, P1 = k_step_predict(kalman_filter1, 50)
    x2, P2 = k_step_predict(kalman_filter2, 5)

    for _ in range(50):
        kalman_filter1.predict()
    x1_baseline = kalman_filter1.x

    for _ in range(5):
        kalman_filter2.predict()
    x2_baseline = kalman_filter2.x


    time1 = np.arange(0, 0.5, 0.01)
    time2 = np.arange(0, 0.5, 0.1)

    print("x1:\t\t ", x1[-1])
    print("x1_baseline:\t ", x1_baseline)
    print("x2:\t\t ", x2[-1])
    print("x2_baseline:\t ", x2_baseline)
    # print("P1: ", P1[-1])
    # print("P2: ", P2[-1])
 
    import matplotlib.pyplot as plt
    # Plotting x1 (position)
    plt.plot(time1, x1[:, 0], label='x1')
    # Plotting x2 (position)
    plt.plot(time2, x2[:, 0], label='x2')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()