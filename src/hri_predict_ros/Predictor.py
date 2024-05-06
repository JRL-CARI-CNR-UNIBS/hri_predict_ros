from dataclasses import dataclass, field
import numpy as np

import hri_predict.HumanModel as HM
import hri_predict.RobotModel as RM
from hri_predict.HumanRobotSystem import HumanRobotSystem
from hri_predict.KalmanPredictor import KalmanPredictor

import rospy
from zed_msgs.msg import ObjectsStamped
from sensor_msgs.msg import JointState


@dataclass
class Predictor:
    human_robot_system:     HumanRobotSystem = field(init=True, repr=True)
    kalman_predictor:       KalmanPredictor = field(init=True, repr=True)
    skeleton_kpts:          np.ndarray = field(default=np.array([]), init=False, repr=False)
    robot_state:            np.ndarray = field(default=np.array([]), init=False, repr=False)
    skeleton_sub:           rospy.Subscriber = field(init=True, repr=False)
    robot_state_sub:        rospy.Subscriber = field(init=True, repr=False)


    def __init__(self,
                 dt: float=0.01,
                 human_control_law: str='CONST_VEL',
                 human_kynematic_model: str='KEYPOINTS',
                 human_noisy: bool=False,
                 human_W: np.ndarray=np.array([], dtype=float),
                 human_n_dof: int=3,
                 human_Kp: float=1.0,
                 human_Kd: float=1.0,
                 human_K_repulse: float=1.0,
                 robot_control_law: str='TRAJ_FOLLOW',
                 robot_n_dof: int=6,
                 alpha: float=0.3,
                 beta: float=2.,
                 kappa: float=0.1,
                 skeleton_topic: str="",
                 robot_js_topic: str="") -> None:
        
        human_control_law = HM.ControlLaw[human_control_law]
        human_kynematic_model = HM.KynematicModel[human_kynematic_model]
        robot_control_law = RM.ControlLaw[robot_control_law]

        # Instantiate HumanRobotSystem
        self.human_robot_system = HumanRobotSystem(
            dt=dt,
            human_control_law=human_control_law,
            human_kynematic_model=human_kynematic_model,
            human_noisy=human_noisy,
            human_W=human_W,
            human_n_dof=human_n_dof,
            human_Kp=human_Kp,
            human_Kd=human_Kd,
            human_K_repulse=human_K_repulse,
            robot_control_law=robot_control_law,
            robot_n_dof=robot_n_dof
        )

        # Instantiate KalmanPredictor
        self.kalman_predictor = KalmanPredictor(
            dt=dt,
            human_control_law=human_control_law,
            human_kynematic_model=human_kynematic_model,
            human_noisy=human_noisy,
            human_W=human_W,
            human_n_dof=human_n_dof,
            human_Kp=human_Kp,
            human_Kd=human_Kd,
            human_K_repulse=human_K_repulse,
            robot_control_law=robot_control_law,
            robot_n_dof=robot_n_dof,
            alpha=alpha,
            beta=beta,
            kappa=kappa
        )

        self.skeleton_sub = rospy.Subscriber(skeleton_topic, ObjectsStamped, self.read_skeleton_cb) # "/zed/zed_node/body_trk/skeletons"
        self.robot_state_sub = rospy.Subscriber(robot_js_topic, JointState, self.read_robot_js_cb)


    def read_skeleton_cb(self, msg: ObjectsStamped) -> None:
        self.skeleton_kpts = np.array(msg.objects[0].data, dtype=float) # TODO: check if this is the correct data type


    def read_robot_js_cb(self, robot_state: np.ndarray) -> None:
        self.robot_state = robot_state # TODO: check if this is the correct data type

    