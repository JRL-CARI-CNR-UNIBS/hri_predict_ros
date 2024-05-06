from dataclasses import dataclass, field
import numpy as np

import hri_predict.HumanModel as HM
import hri_predict.RobotModel as RM
from hri_predict.KalmanPredictor import KalmanPredictor

import rospy
from zed_msgs.msg import ObjectsStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


@dataclass
class Predictor:
    kalman_predictor:       KalmanPredictor = field(init=True, repr=True)
    skeleton_kpts:          np.ndarray = field(default=np.array([]), init=False, repr=False)
    human_state:            np.ndarray = field(default=np.array([]), init=False, repr=False)
    robot_state:            np.ndarray = field(default=np.array([]), init=False, repr=False)
    skeleton_sub:           rospy.Subscriber = field(init=True, repr=False)
    robot_state_sub:        rospy.Subscriber = field(init=True, repr=False)


    def __init__(self,
                 dt: float=0.01,
                 human_control_law: str='CONST_VEL',
                 human_kynematic_model: str='KEYPOINTS',
                 human_noisy_model: bool=False,
                 human_noisy_measure: bool=False,
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
                 robot_js_topic: str="",
                 predicted_hri_state_topic: str="",
                 predicted_hri_cov_topic: str="") -> None:
        
        human_control_law = HM.ControlLaw[human_control_law]
        human_kynematic_model = HM.KynematicModel[human_kynematic_model]
        robot_control_law = RM.ControlLaw[robot_control_law]
        human_w = np.diag(human_W)

        # Instantiate KalmanPredictor
        self.kalman_predictor = KalmanPredictor(
            dt=dt,
            human_control_law=human_control_law,
            human_kynematic_model=human_kynematic_model,
            human_noisy_model=human_noisy_model,
            human_noisy_measure=human_noisy_measure,
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

        self.skeleton_sub = rospy.Subscriber(skeleton_topic, ObjectsStamped, self.read_skeleton_cb)
        self.robot_state_sub = rospy.Subscriber(robot_js_topic, JointState, self.read_robot_js_cb)

        self.state_pub = rospy.Publisher(predicted_hri_state_topic, Float64MultiArray, queue_size=10)
        self.covariance_pub = rospy.Publisher(predicted_hri_cov_topic, Float64MultiArray, queue_size=10)


    def read_skeleton_cb(self, msg: ObjectsStamped) -> None:
        for obj in msg.objects:
            self.skeleton_kpts = np.array(obj.skeleton_3d.keypoints) # TODO correct this: if multiple skeletons, selecting one and overwriting previous
            rospy.loginfo(f"Received skeleton keypoints: {self.skeleton_kpts}")
        if self.kalman_predictor.model.human_model.kynematic_model == HM.KynematicModel.KEYPOINTS:
            self.human_state = np.reshape(self.skeleton_kpts, (1, -1))
        elif self.kalman_predictor.model.human_model.kynematic_model == HM.KynematicModel.KYN_CHAIN:
            self.human_state = np.array(msg.joints)
        else:
            raise ValueError("Invalid kynematic model for the human agent.")

    def read_robot_js_cb(self, msg: JointState) -> None:
        pos = msg.position
        vel = msg.velocity
        eff = msg.effort
        self.robot_state = np.concatenate((pos, vel, eff), axis=1)
        np.reshape(self.robot_state, (1, -1))
        rospy.loginfo(f"Received robot state: {self.robot_state}")

    def publish_state(self, state):
        state_msg = Float64MultiArray()
        state_msg.data = state.flatten().tolist()
        self.state_pub.publish(state_msg)
        
    def publish_covariance(self, covariance):
        covariance_msg = Float64MultiArray()
        covariance_msg.data = covariance.flatten().tolist()
        self.covariance_pub.publish(covariance_msg)

    