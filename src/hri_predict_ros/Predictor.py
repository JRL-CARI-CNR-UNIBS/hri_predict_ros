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
    n_kpts:                 int = field(default=0, init=False, repr=True)
    skeleton_kpts:          np.ndarray = field(default=np.array([]), init=False, repr=False)
    skeleton_kpts_prev:     np.ndarray = field(default=np.array([]), init=False, repr=False)
    skeleton_time:          float = field(default=0.0, init=False, repr=False)
    skeleton_time_prev:     float = field(default=0.0, init=False, repr=False)
    human_state:            np.ndarray = field(default=np.array([]), init=False, repr=False)
    robot_state:            np.ndarray = field(default=np.array([]), init=False, repr=False)
    skeleton_sub:           rospy.Subscriber = field(init=True, repr=False)
    robot_state_sub:        rospy.Subscriber = field(init=True, repr=False)
    pred_state_pub:         rospy.Publisher = field(init=True, repr=False)
    covariance_pub:         rospy.Publisher = field(init=True, repr=False)
    human_state_pub:        rospy.Publisher = field(init=True, repr=False)


    def __init__(self,
                 dt: float=0.01,
                 human_control_law: str='CONST_VEL',
                 human_kynematic_model: str='KEYPOINTS',
                 human_noisy_model: bool=False,
                 human_noisy_measure: bool=False,
                 human_W: np.ndarray=np.array([], dtype=float),
                 human_n_dof: int=3,
                 human_n_kpts: int=18,
                 human_Kp: float=1.0,
                 human_Kd: float=1.0,
                 human_K_repulse: float=1.0,
                 robot_control_law: str='TRAJ_FOLLOW',
                 robot_n_dof: int=6,
                 alpha: float=0.3,
                 beta: float=2.,
                 kappa: float=0.1,
                 node_name: str="",
                 skeleton_topic: str="",
                 robot_js_topic: str="",
                 predicted_hri_state_topic: str="",
                 predicted_hri_cov_topic: str="",
                 human_state_topic: str="") -> None:
        
        human_control_law_      = HM.ControlLaw[human_control_law]
        human_kynematic_model_  = HM.KynematicModel[human_kynematic_model]
        robot_control_law_      = RM.ControlLaw[robot_control_law]
        human_W_                = np.diag(human_W)

        # Instantiate KalmanPredictor
        self.kalman_predictor = KalmanPredictor(
            dt=dt,
            human_control_law=human_control_law_,
            human_kynematic_model=human_kynematic_model_,
            human_noisy_model=human_noisy_model,
            human_noisy_measure=human_noisy_measure,
            human_W=human_W_,
            human_n_dof=human_n_dof,
            human_Kp=human_Kp,
            human_Kd=human_Kd,
            human_K_repulse=human_K_repulse,
            robot_control_law=robot_control_law_,
            robot_n_dof=robot_n_dof,
            alpha=alpha,
            beta=beta,
            kappa=kappa
        )

        self.n_kpts = human_n_kpts

        # Initialize skeleton keypoints ([x, y, z] for each kpt)
        self.skeleton_kpts = np.zeros((self.n_kpts, 3), dtype=float)
        self.skeleton_kpts_prev = np.zeros((self.n_kpts, 3), dtype=float)

        # Initialize ROS subscribers and publishers
        self.skeleton_sub       = rospy.Subscriber(skeleton_topic, ObjectsStamped, self.read_skeleton_cb)
        self.robot_state_sub    = rospy.Subscriber(robot_js_topic, JointState, self.read_robot_js_cb)
        self.pred_state_pub     = rospy.Publisher(node_name + predicted_hri_state_topic, Float64MultiArray, queue_size=10)
        self.covariance_pub     = rospy.Publisher(node_name + predicted_hri_cov_topic, Float64MultiArray, queue_size=10)
        self.human_state_pub    = rospy.Publisher(node_name + human_state_topic, JointState, queue_size=10)


    def read_skeleton_cb(self, msg: ObjectsStamped) -> None:
        if msg.objects:
            for obj in msg.objects: # TODO correct this: if multiple skeletons, selecting one and overwriting previous
                # Extract skeleton keypoints from message ([x, y, z] for each kpt)
                kpts = np.array([[kp.kp] for kp in obj.skeleton_3d.keypoints])
                kpts = kpts[:self.n_kpts] # select only the first n_kpts
                self.skeleton_kpts = np.reshape(kpts, (self.n_kpts, 3)) # reshape to (n_kpts, 3)
                self.skeleton_time = msg.header.stamp.to_sec()
                # rospy.loginfo(f"Received skeleton keypoints:\n{self.skeleton_kpts}\nat time: {self.skeleton_time}")
            
            if self.kalman_predictor.model.human_model.kynematic_model == HM.KynematicModel.KEYPOINTS:
                # Compute velocity for each keypoint
                pos = np.reshape(self.skeleton_kpts, (1, -1))
                pos_prev = np.reshape(self.skeleton_kpts_prev, (1, -1))
                vel = (pos - pos_prev) / (self.skeleton_time - self.skeleton_time_prev)
                
                # Update current human state
                self.human_state = np.dstack((pos, vel)).ravel()
                np.reshape(self.human_state, (1, -1))

            elif self.kalman_predictor.model.human_model.kynematic_model == HM.KynematicModel.KYN_CHAIN:
                pass # TODO: implement
            
            else:
                raise ValueError("Invalid kynematic model for the human agent.")
        
        else:
            rospy.logwarn("No skeleton keypoints received.")
            self.human_state = np.array([])
            self.skeleton_kpts = np.full(self.skeleton_kpts.shape, np.nan)
            self.skeleton_time = np.nan
            pos = np.full(self.skeleton_kpts.shape, np.nan)
            vel = np.full(self.skeleton_kpts.shape, np.nan)

        # Update previous state
        self.skeleton_kpts_prev = self.skeleton_kpts
        self.skeleton_time_prev = self.skeleton_time
        
        rospy.loginfo(f"Received human state:\n{self.human_state}")

        # Publish current human state
        human_state_msg = JointState()
        human_state_msg.header.stamp = rospy.Time.now()
        human_state_msg.position = pos.ravel().tolist()
        human_state_msg.velocity = vel.ravel().tolist()
        human_state_msg.effort = np.full(pos.shape, np.nan).ravel().tolist() # otherwise Plotjuggler does not display the message correctly [Bug fixed on 2024-02-04]
        self.human_state_pub.publish(human_state_msg)


    def read_robot_js_cb(self, msg: JointState) -> None:
        pos = msg.position
        vel = msg.velocity
        self.robot_state = np.dstack((pos, vel)).ravel()
        np.reshape(self.robot_state, (1, -1))
        # rospy.loginfo(f"Received robot state: {self.robot_state}")

    def publish_state(self, state):
        state_msg = Float64MultiArray()
        state_msg.data = state.flatten().tolist()
        self.pred_state_pub.publish(state_msg)
        
    def publish_covariance(self, covariance):
        covariance_msg = Float64MultiArray()
        covariance_msg.data = covariance.flatten().tolist()
        self.covariance_pub.publish(covariance_msg)

    