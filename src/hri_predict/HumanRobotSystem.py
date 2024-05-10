from dataclasses import dataclass, field
import numpy as np
from . import HumanModel as HM
from . import RobotModel as RM

@dataclass
class HumanRobotSystem:
    human_model: HM.HumanModel = field(init=False, repr=True)
    robot_model: RM.RobotModel = field(init=False, repr=True)

    n_states:   int = field(init=False, repr=True)                 # total number of state variables
    n_outs:     int = field(init=False, repr=True)                 # number of output variables
    dt:         float = field(default=0.01, init=True, repr=True)  # time step


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
                 robot_n_dof: int=6) -> None:
        self.dt = dt

        self.human_model = HM.HumanModel(control_law=human_control_law,
                                         kynematic_model=human_kynematic_model,
                                         noisy_model=human_noisy_model,
                                         noisy_measure=human_noisy_measure,
                                         W=human_W,
                                         R=human_R,
                                         n_dof=human_n_dof,
                                         dt=dt,
                                         Kp=human_Kp,
                                         Kd=human_Kd,
                                         K_repulse=human_K_repulse)
        
        self.robot_model = RM.RobotModel(control_law=robot_control_law,
                                         n_dof=robot_n_dof,
                                         dt=dt)
        
        self.n_states = self.human_model.n_states + self.robot_model.n_states
        self.n_outs = self.human_model.n_outs + self.robot_model.n_outs

        # DEBUG: Print the number of states and outputs
        print("\n======================================")
        print("    self.human_model.n_states: ", self.human_model.n_states)
        print("    self.robot_model.n_states: ", self.robot_model.n_states)
        print("    self.n_states: ", self.n_states)
        print("    self.human_model.n_outs: ", self.human_model.n_outs)
        print("    self.robot_model.n_outs: ", self.robot_model.n_outs)
        print("    self.n_outs: ", self.n_outs)
        print("======================================\n")


    def set_state(self, x0_human: np.ndarray, x0_robot: np.ndarray) -> None:
        self.human_model.set_state(x0_human)
        self.robot_model.set_state(x0_robot)


    def get_state(self) -> np.ndarray:
        return np.concatenate((self.human_model.x, self.robot_model.x))
    

    def dynamics(self, x0: np.ndarray, u0: np.ndarray) -> np.ndarray: 
        human_dyn = self.human_model.dynamics(x0, u0)
        robot_dyn = self.robot_model.dynamics(x0, u0)
        return np.concatenate((human_dyn, robot_dyn))
    

    def f(self, x: np.ndarray, dt: float) -> np.ndarray:
        human_f = self.human_model.f(x[:self.human_model.n_states], dt)
        robot_f = self.robot_model.f(x[self.human_model.n_states:], dt)
        return np.concatenate((human_f, robot_f))


    def step(self, scaling: float=0.0) -> None:
        self.human_model.step()
        self.robot_model.step(scaling)


    def compute_control_action(self,
                               x_target: np.ndarray=np.array([], dtype=float),
                               acc_target: np.ndarray=np.array([], dtype=float),
                               x_obstacle: np.ndarray=np.array([], dtype=float),
                               scaling: float=0.0) -> np.ndarray:
        human_u = self.human_model.compute_control_action(x_target, x_obstacle)
        robot_u = self.robot_model.compute_control_action(acc_target, x_obstacle, scaling)
        return np.concatenate((human_u, robot_u))
    

    def output(self) -> np.ndarray:
        human_out = self.human_model.output()
        robot_out = self.robot_model.output()
        return np.concatenate((human_out, robot_out))
    

    def h(self, x: np.ndarray) -> np.ndarray:
        human_h = self.human_model.h(x[:self.human_model.n_states])
        robot_h = self.robot_model.h(x[self.human_model.n_states:])
        return np.concatenate((human_h, robot_h))
    

    # forward kinematics
    def fwd_kin(self) -> np.ndarray:
        human_fk = self.human_model.fwd_kin()
        robot_fk = self.robot_model.fwd_kin()
        return np.concatenate((human_fk, robot_fk))


    # inverse kinematics
    def inv_kin(self, keypts) -> np.ndarray:
        human_ik = self.human_model.inv_kin(keypts)
        robot_ik = self.robot_model.inv_kin(keypts)
        return np.concatenate((human_ik, robot_ik))