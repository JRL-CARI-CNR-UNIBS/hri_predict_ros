from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter, unscented_transform
from filterpy.common import Q_discrete_white_noise
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.linalg import block_diag
from .HumanRobotSystem import HumanRobotSystem
from . import HumanModel as HM
from . import RobotModel as RM
from .utils import get_near_psd

@dataclass
class KalmanPredictor:
    dt:             float = field(default=0.01, init=True, repr=True) # time step

    model:          HumanRobotSystem = field(init=False, repr=True)
    kalman_filter:  UnscentedKalmanFilter = field(init=False, repr=True)

    sigma_points:   MerweScaledSigmaPoints = field(init=False, repr=True)
    alpha:          float = field(default=0.3, init=True, repr=True)
    beta:           float = field(default=2.0, init=True, repr=True)
    kappa:          float = field(default=0.1, init=True, repr=True)

    p_idx:          np.ndarray = field(default=np.array([]), init=False, repr=False)
    v_idx:          np.ndarray = field(default=np.array([]), init=False, repr=False)
    a_idx:          np.ndarray = field(default=np.array([]), init=False, repr=False) 


    def __init__(self,
                 dt: float=0.01,
                 human_control_law: HM.ControlLaw=HM.ControlLaw.CONST_ACC,
                 human_kynematic_model: HM.KynematicModel=HM.KynematicModel.KEYPOINTS,
                 human_noisy_model: bool=False,
                 human_noisy_measure: bool=False,
                 human_R: dict={},
                 human_W: dict={},
                 human_n_kpts: int=18,
                 human_n_dof: int=3,
                 human_Kp: float=1.0,
                 human_Kd: float=1.0,
                 human_K_repulse: float=1.0,
                 robot_control_law: RM.ControlLaw=RM.ControlLaw.TRAJ_FOLLOW,
                 robot_n_dof: int=6,
                 alpha: float=0.3,
                 beta: float=2.,
                 kappa: float=0.1,
                 u_min_human: float=-100,
                 u_max_human: float=100,
                 a_min_human: float=-50,
                 a_max_human: float=50,
                 v_min_human: float=-5,
                 v_max_human: float=5) -> None:
        self.dt = dt

        self.model = HumanRobotSystem(dt=dt,
                                      human_control_law=human_control_law,
                                      human_kynematic_model=human_kynematic_model,
                                      human_noisy_model=human_noisy_model,
                                      human_noisy_measure=human_noisy_measure,
                                      human_R=human_R,
                                      human_W=human_W,
                                      human_n_kpts=human_n_kpts,
                                      human_n_dof=human_n_dof,
                                      human_Kp=human_Kp,
                                      human_Kd=human_Kd,
                                      human_K_repulse=human_K_repulse,
                                      robot_control_law=robot_control_law,
                                      robot_n_dof=robot_n_dof,
                                      u_min_human=u_min_human,
                                      u_max_human=u_max_human,
                                      a_min_human=a_min_human,
                                      a_max_human=a_max_human,
                                      v_min_human=v_min_human,
                                      v_max_human=v_max_human)   

       
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        self.sigma_points = MerweScaledSigmaPoints(n=self.model.n_states,
                                                   alpha=alpha,
                                                   beta=beta,
                                                   kappa=kappa)
        
        self.kalman_filter = UnscentedKalmanFilter(dim_x=self.model.n_states,
                                                   dim_z=self.model.n_outs,
                                                   fx=self.model.f,
                                                   hx=self.model.h,
                                                   dt=dt,
                                                   points=self.sigma_points)
        
        # Set the indices for position, velocity and acceleration for the human model
        self.p_idx = np.arange(0, self.model.human_model.n_states, 3)
        self.v_idx = np.arange(1, self.model.human_model.n_states, 3)
        self.a_idx = np.arange(2, self.model.human_model.n_states, 3)


    def initialize(self, P0_human: dict, P0_robot: dict,
                   x0_human: Optional[np.ndarray]=None,
                   x0_robot: Optional[np.ndarray]=None) -> None:
        # Initialize the STATE of the model and the Kalman filter
        if x0_human is not None and x0_robot is not None:
            self.model.set_state(x0_human, x0_robot)
        self.kalman_filter.x = self.model.get_state()

        # Initialize the MEASUREMENT VECTOR with nan values
        self.kalman_filter.z = np.zeros(self.model.n_outs)

        # Initialize the STATE COVARIANCE matrix
        P0_val_human = [value for value in P0_human.values()] # [pos, vel, acc] for the single DoF
        P0_val_human = np.tile(P0_val_human, self.model.human_model.n_dof) # replicate for each DoF
        P0_val_robot = [value for value in P0_robot.values()] # [pos, vel, acc] for the single DoF
        P0_val_robot = np.tile(P0_val_robot, self.model.robot_model.n_dof) # replicate for each DoF

        P_human = np.diag(P0_val_human)
        P_robot = np.diag(P0_val_robot)
        P = block_diag(P_human, P_robot)
        self.kalman_filter.P = P

        # Initialize the MODEL UNCERTAINTY matrix
        # Q_human = self.model.human_model.W
        # Q_robot = np.zeros((self.model.robot_model.n_states, self.model.robot_model.n_states))
        # Q = block_diag(Q_human, Q_robot)

        n_dof = self.model.human_model.n_dof+self.model.robot_model.n_dof
        # Q_variances_human = np.diag(self.model.human_model.W) # [pos, vel, acc] for the single DoF

        # print("Q_variances_human: ", Q_variances_human, ", shape: ", Q_variances_human.shape)

        # Q_variances_robot = np.zeros(self.model.robot_model.n_dof)

        # print("Q_variances_robot: ", Q_variances_robot, ", shape: ", Q_variances_robot.shape)
        # Q_variances = np.concatenate((Q_variances_human, Q_variances_robot))

        # print("Q_variances: ", Q_variances, ", shape: ", Q_variances.shape)

        var = 0.0
        self.kalman_filter.Q = Q_discrete_white_noise(dim=3, dt=self.dt, var=var, block_size=n_dof)
        
        # Initialize the MEASUREMENT NOISE matrix
        R_human = self.model.human_model.R
        R_robot = np.zeros((self.model.robot_model.n_outs, self.model.robot_model.n_outs))
        R = block_diag(R_human, R_robot)
        self.kalman_filter.R = R

        print("\n======================================")
        print("    Human state ", self.model.human_model.get_state(), ", shape: ", self.model.human_model.get_state().shape)
        print("    Robot state ", self.model.robot_model.get_state(), ", shape: ", self.model.robot_model.get_state().shape)
        print("    Initial STATE: ", self.kalman_filter.x, ", shape: ", self.kalman_filter.x.shape)
        print("    Initial STATE COVARIANCE: ", self.kalman_filter.P, ", shape: ", self.kalman_filter.P.shape)
        print("    MODEL UNCERTAINTY matrix: ", self.kalman_filter.Q, ", shape: ", self.kalman_filter.Q.shape)
        print("    MEASUREMENT NOISE matrix: ", self.kalman_filter.R, ", shape: ", self.kalman_filter.R.shape)
        print("======================================\n")


    def predict(self, **predict_args):
        # Check for semi-positive definitness of P matrix and correct if needed
        self.kalman_filter.P = get_near_psd(self.kalman_filter.P)

        # Compute average magnitude of eigenvalues of P matrix
        eigenvalues = np.linalg.eigvals(self.kalman_filter.P)
        average_magnitude = np.mean(np.abs(eigenvalues))
        print("[KalmanPredictor::predict] Average magnitude of the eigenvalues of P: ", average_magnitude)

        self.kalman_filter.predict(**predict_args)
        self.model.set_state(self.kalman_filter.x[:self.model.human_model.n_states],
                             self.kalman_filter.x[self.model.human_model.n_states:])


    def update(self, z: np.ndarray):
        self.kalman_filter.update(z)
        self.model.set_state(self.kalman_filter.x[:self.model.human_model.n_states],
                             self.kalman_filter.x[self.model.human_model.n_states:])


    def k_step_predict(self, k: int, **predict_args) -> tuple:
        # calculate sigma points for current mean and covariance
        sigmas = self.kalman_filter.points_fn.sigma_points(self.kalman_filter.x,
                                                           self.kalman_filter.P)
        
        sigmas_f = np.zeros((len(sigmas), self.kalman_filter._dim_x))               # sigma points after passing through dynamics function
        xx = np.zeros((k, self.kalman_filter._dim_x))                               # mean after passing through dynamics function
        variances = np.zeros((k, self.kalman_filter._dim_x))                        # covariance after passing through dynamics function      

        for _ in range(k):
            # transform sigma points through the dynamics function
            sigmas_f = np.array([self.kalman_filter.fx(s, self.kalman_filter._dt, **predict_args) for s in sigmas])

            # pass sigmas through the unscented transform to compute prior
            x, P = unscented_transform(sigmas_f,
                                       self.kalman_filter.Wm,
                                       self.kalman_filter.Wc,
                                       self.kalman_filter.Q,
                                       self.kalman_filter.x_mean,
                                       self.kalman_filter.residual_x)

            # check for semi-positive definitness of P matrix and correct if needed
            P = get_near_psd(P)

            # update sigma points to reflect the new variance of the points
            sigmas = self.kalman_filter.points_fn.sigma_points(x, P)

            # store state means and variances for each step
            xx[_] = x
            variances[_] = np.diag(P)
        
        return xx, variances
        
    
