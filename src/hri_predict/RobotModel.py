from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from .utils import runge_kutta

class ControlLaw(Enum):
    TRAJ_FOLLOW = 'TRAJ_FOLLOW'
    SAFE_TRAJ_FOLLOW = 'SAFE_TRAJ_FOLLOW'
    ELASTIC_STRIP = 'ELASTIC_STRIP'
    SCALED_ELASTIC_STRIP = 'SCALED_ELASTIC_STRIP'

@dataclass
class RobotModel:
    control_law: ControlLaw = field(default=ControlLaw.TRAJ_FOLLOW, init=True, repr=True)

    n_dof:      int = field(default=6, init=True, repr=True)         # DoF (e.g., 6 for a 6-DoF robot)     
    n_states:   int = field(init=False, repr=True)                   # total number of state variables
    n_outs:     int = field(init=False, repr=True)                   # number of output variables

    p_idx:      np.ndarray = field(init=False, repr=False)           # position indices
    v_idx:      np.ndarray = field(init=False, repr=False)           # velocity indices
    a_idx:      np.ndarray = field(init=False, repr=False)           # acceleration indices
  
    x:          np.ndarray = field(init=False, repr=False)           # state

    dt:         float = field(default=0.01, init=True, repr=True)    # time step
    t_nom:      float = field(default=0.0, init=False, repr=False)   # nominal time of the trajectory

    K_repulse:  float = field(default=1.0, init=True, repr=True)     # repulsion gain for control law


    def __init__(self,
                 control_law: ControlLaw=ControlLaw.TRAJ_FOLLOW,
                 n_dof: int=3,
                 dt: float=0.01,
                 K_repulse: float=1.0) -> None:
        self.control_law = control_law
        self.n_dof = n_dof

        self.n_states = 3 * self.n_dof      # number of state variables for each DoF (e.g., if 3: position, velocity, acceleration)
        self.x = np.zeros(self.n_states)

        self.n_outs = 2 * self.n_dof        # number of output variables for each DoF (e.g., if 2: position, velocity)

        self.p_idx = np.arange(0, self.n_states, 3)
        self.v_idx = np.arange(1, self.n_states, 3)
        self.a_idx = np.arange(2, self.n_states, 3)
        self.pv_idx = np.sort(np.concatenate((self.p_idx, self.v_idx)))

        self.dt = dt

        self.K_repulse = K_repulse


    def initialize(self, x0: np.ndarray) -> None:
        self.x = x0
        

    # double integrator dynamics
    def dynamics(self, x0: np.ndarray, u0: np.ndarray) -> np.ndarray: 
        x_dot = np.zeros(self.n_states)
        x_dot[self.p_idx] = x0[self.v_idx]    # p_dot = v
        x_dot[self.v_idx] = x0[self.a_idx]    # v_dot = a
        x_dot[self.a_idx] = u0                # a_dot = u
        return x_dot
    
    
    def f(self, x: np.ndarray, dt: float) -> np.ndarray:
        F = np.array([[1, dt,  0],
                      [0,  1, dt],
                      [0,  0,  1]], dtype=float)
        return F @ x


    def step(self, scaling: float=0.0) -> None:
        self.x = runge_kutta(self.dynamics,                    # explicit RK4 integration
                             self.x,
                             self.compute_control_action(),
                             self.dt)
        self.t_nom += scaling*self.dt


    def compute_control_action(self,
                               acc_target: np.ndarray=np.array([], dtype=float),
                               x_obstacle: np.ndarray=np.array([], dtype=float),
                               scaling: float=0.0) -> np.ndarray:
        if self.control_law == ControlLaw.TRAJ_FOLLOW:
            u = acc_target
        elif self.control_law == ControlLaw.SAFE_TRAJ_FOLLOW:
            u = scaling**2 * acc_target
        elif self.control_law == ControlLaw.ELASTIC_STRIP:
            u = acc_target + self.K_repulse * (self.x[self.p_idx] - x_obstacle)
        elif self.control_law == ControlLaw.SCALED_ELASTIC_STRIP:
            u = scaling**2 * acc_target + self.K_repulse * (self.x[self.p_idx] - x_obstacle)
        else:
            raise ValueError('Invalid control law')

        return u
    

    def output(self) -> np.ndarray:
        return self.x[self.pv_idx]
    

    def h(self, x: np.ndarray) -> np.ndarray:
        H = np.array([[1, 0, 0],
                      [0, 1, 0]], dtype=float)
        return H @ x


    # forward kinematics
    def fwd_kin(self):
        return self.x # TODO


    # inverse kinematics
    def inv_kin(self, keypts: tuple) -> tuple:
        return keypts[0], keypts[1] # TODO                      # position, velocity