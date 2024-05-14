from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy.linalg import block_diag
from .utils import runge_kutta

class ControlLaw(Enum):
    CONST_VEL = 'CONST_VEL'
    CONST_ACC = 'CONST_ACC'
    PD = 'PD'
    PD_REPULSE = 'PD_REPULSE'
    IOC = 'IOC'

class KynematicModel(Enum):
    KEYPOINTS = 'KEYPOINTS'
    KYN_CHAIN = 'KYN_CHAIN'

@dataclass
class HumanModel:
    kynematic_model:    KynematicModel = field(default=KynematicModel.KEYPOINTS, init=True, repr=True)
    control_law:        ControlLaw = field(default=ControlLaw.CONST_VEL, init=True, repr=True)
    
    noisy_model:        bool = field(default=False, init=True, repr=True)
    noisy_measure:      bool = field(default=False, init=True, repr=True)
      
    n_kpts:     int = field(default=18, init=True, repr=True)     # number of keypoints
    n_dof:      int = field(default=54, init=True, repr=True)     # n_dof = 3*n_kpts if KEYPOINTS, n_joints if KYN_CHAIN
    n_states:   int = field(init=False, repr=True)                # total number of state variables
    n_outs:     int = field(init=False, repr=True)                # number of output variables
  
    p_idx:      np.ndarray = field(init=False, repr=False)        # position indices
    v_idx:      np.ndarray = field(init=False, repr=False)        # velocity indices
    a_idx:      np.ndarray = field(init=False, repr=False)        # acceleration indices
  
    x:          np.ndarray = field(init=False, repr=False)                                       # state 
    W:          np.ndarray = field(default=np.array([], dtype=float), init=True, repr=True)      # model uncertainty matrix
    R:          np.ndarray = field(default=np.array([], dtype=float), init=True, repr=True)      # measurement noise matrix
  
    dt:         float = field(default=0.01, init=True, repr=True) # time step

    Kp:         float = field(default=1.0, init=True, repr=True)  # proportional gain for control law
    Kd:         float = field(default=1.0, init=True, repr=True)  # derivative gain for control law
    K_repulse:  float = field(default=1.0, init=True, repr=True)  # repulsion gain for control law


    def __init__(self,
                 control_law: ControlLaw=ControlLaw.CONST_VEL,
                 kynematic_model: KynematicModel=KynematicModel.KEYPOINTS,
                 noisy_model: bool=False,
                 noisy_measure: bool=False,
                 R: dict={},
                 W: dict={},
                 n_kpts: int=18,
                 n_dof: int=3,
                 dt: float=0.01,
                 Kp: float=1.0,
                 Kd: float=1.0,
                 K_repulse: float=1.0) -> None:
        self.control_law = control_law
        self.kynematic_model = kynematic_model
        self.noisy_model = noisy_model
        self.noisy_measure = noisy_measure

        # Override n_dof for "KEYPOINTS" kinematic model: n_DoF = (x, y, z) for each keypoint
        if self.kynematic_model == KynematicModel.KEYPOINTS:
            self.n_dof = 3 * n_kpts
        elif self.kynematic_model == KynematicModel.KYN_CHAIN:
            self.n_dof = n_dof
        else:
            raise ValueError('Invalid kynematic model')
        
        self.n_states = 3 * self.n_dof # (position, velocity, acceleration) for each DoF
        self.n_outs = self.n_dof       # x, y, z, position for each keypoint

        self.x = np.zeros(self.n_states)

        self.dt = dt
        self.Kp = Kp
        self.Kd = Kd
        self.K_repulse = K_repulse

        # State indices
        self.p_idx = np.arange(0, self.n_states, 3)
        self.v_idx = np.arange(1, self.n_states, 3)
        self.a_idx = np.arange(2, self.n_states, 3)
        self.pv_idx = np.sort(np.concatenate((self.p_idx, self.v_idx)))

        # Model uncertainty matrix
        W_values = [value for value in W.values()] # [pos, vel, acc] for the single DoF
        if self.noisy_model:
            W_block = np.diag(W_values)
        else:
            W_block = np.zeros((len(W_values), len(W_values))) 
        self.W = block_diag(*[W_block for _ in range(self.n_dof)]) # replicate the block for each DoF

        # Measurement noise matrix
        R_values = [v for v in R.values()] # [pos_x, pos_y, pos_z] for the single keypoint
        if self.noisy_measure:
            R_block = np.diag(R_values)
        else:
            R_block = np.zeros((len(R_values), len(R_values)))
        self.R = block_diag(*[R_block for _ in range(self.n_kpts)]) # replicate the block for each keypoint

        # DEBUG: Print the model parameters
        print("\n======================================")
        print("    HumanModel:")
        print("    self.x: ", self.x)
        print("    self.n_dof: ", self.n_dof)
        print("    self.n_states: ", self.n_states)
        print("    self.n_outs: ", self.n_outs)
        print("    self.R: ", self.R, ", shape: ", self.R.shape)
        print("======================================\n")


    def set_state(self, x0: np.ndarray) -> None:
        self.x = x0


    def get_state(self) -> np.ndarray:
        return self.x


    # double integrator dynamics
    def dynamics(self, x0: np.ndarray, u0: np.ndarray) -> np.ndarray: 
        x_dot = np.zeros(self.n_states)
        x_dot[self.p_idx] = x0[self.v_idx]    # p_dot = v
        x_dot[self.v_idx] = x0[self.a_idx]    # v_dot = a
        x_dot[self.a_idx] = u0                # a_dot = u
        return x_dot

    
    def f(self, x: np.ndarray, dt: float) -> np.ndarray:
        block = np.array([[1, dt,  0],
                          [0,  1, dt],
                          [0,  0,  1]], dtype=float)
        F = block_diag(*[block for _ in range(self.n_dof)])
        return F @ x
       

    def step(self) -> None:
        self.x = runge_kutta(self.dynamics,                                         # explicit RK4 integration
                             self.x,
                             self.compute_control_action(),
                             self.dt) \
                 + np.random.multivariate_normal(np.zeros(self.n_states), self.W)   # gaussian noise


    def compute_control_action(self,
                               x_target: np.ndarray=np.array([], dtype=float),
                               x_obstacle: np.ndarray=np.array([], dtype=float)) -> np.ndarray:
        if self.control_law == ControlLaw.CONST_VEL:
            u = np.zeros(self.n_dof)

        elif self.control_law == ControlLaw.CONST_ACC:
            u = self.x[self.a_idx]

        elif self.control_law == ControlLaw.PD:
            u = - self.Kp * (self.x[self.p_idx] - x_target) \
                - self.Kd * self.x[self.v_idx]
        
        elif self.control_law == ControlLaw.PD_REPULSE:
            u = - self.Kp * (self.x[self.p_idx] - x_target) \
                - self.Kd * self.x[self.v_idx] \
                - self.K_repulse * (self.x[self.p_idx] - x_obstacle)
        
        elif self.control_law == ControlLaw.IOC:
            pass # TODO
        
        else:
            raise ValueError('Invalid control law')

        return u
    

    def output(self) -> np.ndarray:
        if self.kynematic_model == KynematicModel.KEYPOINTS:
            return self.x[self.p_idx]
        elif self.kynematic_model == KynematicModel.KYN_CHAIN:
            return self.fwd_kin()
        else:
            raise ValueError('Invalid kynematic model')
        

    def h(self, x: np.ndarray) -> np.ndarray:
        if self.kynematic_model == KynematicModel.KEYPOINTS:
            block = np.array([[1, 0, 0]], dtype=float)
            H = block_diag(*[block for _ in range(self.n_dof)])
            return H @ x
        elif self.kynematic_model == KynematicModel.KYN_CHAIN:
            return x # TODO
        else:
            raise ValueError('Invalid kynematic model')
    

    # forward kinematics for the kinematic chain model
    def fwd_kin(self) -> np.ndarray:
        return self.x # TODO
        

    # inverse kinematics
    def inv_kin(self, keypts: tuple) -> tuple:
        if self.kynematic_model == KynematicModel.KEYPOINTS:
            return keypts[0], keypts[1]                             # position, velocity  
        elif self.kynematic_model == KynematicModel.KYN_CHAIN:
            return keypts[0], keypts[1] # TODO                      # position, velocity
        else:
            raise ValueError('Invalid kynematic model')