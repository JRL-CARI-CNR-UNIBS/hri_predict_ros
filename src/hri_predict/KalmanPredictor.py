from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter, unscented_transform
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import scipy.linalg
from .HumanRobotSystem import HumanRobotSystem
from . import HumanModel as HM
from . import RobotModel as RM


@dataclass
class KalmanPredictor:
    dt:             float = field(default=0.01, init=True, repr=True) # time step

    model:          HumanRobotSystem = field(init=False, repr=True)
    kalman_filter:  UnscentedKalmanFilter = field(init=False, repr=True)

    sigma_points:   MerweScaledSigmaPoints = field(init=False, repr=True)
    alpha:          float = field(default=0.3, init=True, repr=True)
    beta:           float = field(default=2.0, init=True, repr=True)
    kappa:          float = field(default=0.1, init=True, repr=True)


    def __init__(self,
                 dt: float=0.01,
                 human_control_law: HM.ControlLaw=HM.ControlLaw.CONST_VEL,
                 human_kynematic_model: HM.KynematicModel=HM.KynematicModel.KEYPOINTS,
                 human_noisy_model: bool=False,
                 human_noisy_measure: bool=False,
                 human_W: np.ndarray=np.array([], dtype=float),
                 human_R: np.ndarray=np.array([], dtype=float),
                 human_n_dof: int=3,
                 human_Kp: float=1.0,
                 human_Kd: float=1.0,
                 human_K_repulse: float=1.0,
                 robot_control_law: RM.ControlLaw=RM.ControlLaw.TRAJ_FOLLOW,
                 robot_n_dof: int=6,
                 alpha: float=0.3,
                 beta: float=2.,
                 kappa: float=0.1) -> None:
        self.dt = dt

        self.model = HumanRobotSystem(dt=dt,
                                      human_control_law=human_control_law,
                                      human_kynematic_model=human_kynematic_model,
                                      human_noisy_model=human_noisy_model,
                                      human_noisy_measure=human_noisy_measure,
                                      human_W=human_W,
                                      human_R=human_R,
                                      human_n_dof=human_n_dof,
                                      human_Kp=human_Kp,
                                      human_Kd=human_Kd,
                                      human_K_repulse=human_K_repulse,
                                      robot_control_law=robot_control_law,
                                      robot_n_dof=robot_n_dof)   

       
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


    def initialize(self, P0_human: np.ndarray, P0_robot: np.ndarray,
                   x0_human: Optional[np.ndarray]=None,
                   x0_robot: Optional[np.ndarray]=None) -> None:
        # Initialize the STATE of the model and the Kalman filter
        if x0_human is not None and x0_robot is not None:
            self.model.set_state(x0_human, x0_robot)
        self.kalman_filter.x = self.model.get_state()

        # Initialize the STATE COVARIANCE matrix
        P_human = np.diag(P0_human)
        P_robot = np.diag(P0_robot)
        P = scipy.linalg.block_diag(P_human, P_robot)
        self.kalman_filter.P = P

        # Initialize the MODEL UNCERTAINTY matrix
        Q_human = self.model.human_model.W
        Q_robot = np.zeros((self.model.robot_model.n_states, self.model.robot_model.n_states))
        Q = scipy.linalg.block_diag(Q_human, Q_robot)
        self.kalman_filter.Q = Q

        # Initialize the MEASUREMENT NOISE matrix
        R_human = self.model.human_model.R
        R_robot = np.zeros((self.model.robot_model.n_outs, self.model.robot_model.n_outs))
        R = scipy.linalg.block_diag(R_human, R_robot)
        self.kalman_filter.R = R


    def predict(self):
        self.kalman_filter.predict()
        self.model.set_state(self.kalman_filter.x[:self.model.human_model.n_states],
                             self.kalman_filter.x[self.model.human_model.n_states:])


    def update(self, z: np.ndarray):
        self.kalman_filter.update(z)
        self.model.set_state(self.kalman_filter.x[:self.model.human_model.n_states],
                             self.kalman_filter.x[self.model.human_model.n_states:])


    def k_step_predict(self, k: int) -> tuple:
        # calculate sigma points for current mean and covariance
        sigmas = self.kalman_filter.points_fn.sigma_points(self.kalman_filter.x,
                                                           self.kalman_filter.P)
        
        sigmas_f = np.zeros((len(sigmas), self.kalman_filter._dim_x))               # sigma points after passing through dynamics function
        xx = np.zeros((k, self.kalman_filter._dim_x))                               # mean after passing through dynamics function
        PP = np.zeros((k, self.kalman_filter._dim_x, self.kalman_filter._dim_x))    # covariance after passing through dynamics function
        for _ in range(k):
            # transform sigma points through the dynamics function
            for i, s in enumerate(sigmas):
                sigmas_f[i] = self.kalman_filter.fx(s, self.kalman_filter._dt)

            # pass sigmas through the unscented transform to compute prior
            x, P = unscented_transform(sigmas_f,
                                       self.kalman_filter.Wm,
                                       self.kalman_filter.Wc,
                                       self.kalman_filter.Q,
                                       self.kalman_filter.x_mean,
                                       self.kalman_filter.residual_x)

            # update sigma points to reflect the new variance of the points
            sigmas_f = self.kalman_filter.points_fn.sigma_points(x, P)

            # store x (mean) and P (covariance) for each step
            xx[_] = x
            PP[_] = P
        
        return xx, PP
        
    
