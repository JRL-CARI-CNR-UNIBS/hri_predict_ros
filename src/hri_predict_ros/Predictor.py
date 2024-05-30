from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import os

import hri_predict.HumanModel as HM
import hri_predict.RobotModel as RM
from hri_predict.KalmanPredictor import KalmanPredictor

import rospy
import tf
import tf.transformations
from zed_msgs.msg import ObjectsStamped
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray, Pose, PointStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


@dataclass
class Predictor:
    kalman_predictor:           KalmanPredictor = field(init=False, repr=True)
    n_kpts:                     int = field(default=0, init=False, repr=True)
    skeleton_kpts:              np.ndarray = field(default=np.array([]), init=False, repr=False)
    human_meas:                 np.ndarray = field(init=False, repr=False)
    robot_meas:                 np.ndarray = field(init=False, repr=False)
    skeleton_sub:               rospy.Subscriber = field(init=True, repr=False)
    robot_state_sub:            rospy.Subscriber = field(init=True, repr=False)
    pred_state_pub:             rospy.Publisher = field(init=True, repr=False)
    pred_state_pub_pj:          rospy.Publisher = field(init=True, repr=False) # PLOTJUGGLER
    pred_variance_pub:          rospy.Publisher = field(init=True, repr=False)
    pred_variance_pub_pj:       rospy.Publisher = field(init=True, repr=False) # PLOTJUGGLER
    pred_state_lcl_pub:         rospy.Publisher = field(init=True, repr=False)
    pred_state_lcl_pub_pj:      rospy.Publisher = field(init=True, repr=False) # PLOTJUGGLER
    pred_state_ucl_pub:         rospy.Publisher = field(init=True, repr=False)
    pred_state_ucl_pub_pj:      rospy.Publisher = field(init=True, repr=False) # PLOTJUGGLER
    human_meas_pub:             rospy.Publisher = field(init=True, repr=False)
    tf_listener:                tf.TransformListener = field(init=False, repr=False)
    camera_frame:               str = field(default="", init=True, repr=True)
    world_frame:                str = field(default="world", init=True, repr=True)
    cam_to_world_matrix:        np.ndarray = field(default=np.eye(4), init=False, repr=True)
    skeleton_absent_count:      int = field(default=0, init=False, repr=False)
    max_steps_skeleton_absent:  int = field(default=10, init=True, repr=True)


    def __init__(self,
                 dt: float=0.01,
                 human_control_law: str='CONST_ACC',
                 human_kynematic_model: str='KEYPOINTS',
                 human_noisy_model: bool=False,
                 human_noisy_measure: bool=False,
                 human_R: dict={},
                 human_Q: dict={},
                 human_n_kpts: int=18,
                 human_n_dof: int=3,
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
                 human_meas_topic: str="",
                 human_filt_pos_topic: str="",   
                 human_filt_vel_topic: str="",
                 human_filt_acc_topic: str="",   
                 camera_frame: str="",
                 world_frame: str="world",
                 TF_world_camera: list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                 u_min_human: float=-100,
                 u_max_human: float=100,
                 a_min_human: float=-50,
                 a_max_human: float=50,
                 v_min_human: float=-5,
                 v_max_human: float=5,
                 predict_steps: int=5,
                 max_steps_skeleton_absent: int=10) -> None:

        human_control_law_      = HM.ControlLaw[human_control_law]
        human_kynematic_model_  = HM.KynematicModel[human_kynematic_model]
        robot_control_law_      = RM.ControlLaw[robot_control_law]

        # Instantiate KalmanPredictor
        self.kalman_predictor = KalmanPredictor(
            dt=dt,
            human_control_law=human_control_law_,
            human_kynematic_model=human_kynematic_model_,
            human_noisy_model=human_noisy_model,
            human_noisy_measure=human_noisy_measure,
            human_R=human_R,
            human_Q=human_Q,
            human_n_kpts=human_n_kpts,
            human_n_dof=human_n_dof,
            human_Kp=human_Kp,
            human_Kd=human_Kd,
            human_K_repulse=human_K_repulse,
            robot_control_law=robot_control_law_,
            robot_n_dof=robot_n_dof,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            u_min_human=u_min_human,
            u_max_human=u_max_human,
            a_min_human=a_min_human,
            a_max_human=a_max_human,
            v_min_human=v_min_human,
            v_max_human=v_max_human,
            predict_steps=predict_steps
            )

        self.n_kpts = human_n_kpts

        # Initialize skeleton keypoints ([x, y, z] for each kpt)
        self.skeleton_kpts = np.zeros((self.n_kpts, 3), dtype=float)

        # Initialize human measurement vector ([x, y, z] position for each kpt)
        self.human_meas = np.full(3*human_n_kpts, np.nan, dtype=float)
        
        # Initialize robot measurement vector ([pos, vel] for each DoF)
        self.robot_meas = np.full(2*robot_n_dof, np.nan, dtype=float)

        # Maximum number of steps to tolerate skeleton absence
        self.max_steps_skeleton_absent = max_steps_skeleton_absent

        # Initialize ROS subscribers and publishers
        self.skeleton_sub       = rospy.Subscriber(skeleton_topic, ObjectsStamped, self.read_skeleton_cb)
        self.robot_state_sub    = rospy.Subscriber(robot_js_topic, JointState, self.read_robot_js_cb)
        
        self.human_meas_pub     = rospy.Publisher(node_name + human_meas_topic, PoseArray, queue_size=10)                  # current human measured position
        self.human_filt_pos_pub = rospy.Publisher(node_name + human_filt_pos_topic, PoseArray, queue_size=10)              # current human estimated position
        self.human_filt_vel_pub = rospy.Publisher(node_name + human_filt_vel_topic, PoseArray, queue_size=10)              # current human estimated velocity
        self.human_filt_acc_pub = rospy.Publisher(node_name + human_filt_acc_topic, PoseArray, queue_size=10)              # current human estimated acceleration
        
        self.pred_state_pub     = rospy.Publisher(node_name + predicted_hri_state_topic, JointTrajectory, queue_size=10)   # k-step ahead predicted state
        self.pred_variance_pub  = rospy.Publisher(node_name + predicted_hri_cov_topic, JointTrajectory, queue_size=10)     # k-step ahead predicted variance
        self.pred_state_pub_pj  = rospy.Publisher(node_name + predicted_hri_state_topic + '/pj', JointTrajectory, queue_size=10)   # PLOTJUGGLER
        self.pred_variance_pub_pj  = rospy.Publisher(node_name + predicted_hri_cov_topic + '/pj', JointTrajectory, queue_size=10)   # PLOTJUGGLER
        
        self.pred_state_lcl_pub = rospy.Publisher(node_name + predicted_hri_state_topic + '/lcl', JointTrajectory, queue_size=10)   # k-step ahead predicted state
        self.pred_state_ucl_pub = rospy.Publisher(node_name + predicted_hri_state_topic + '/ucl', JointTrajectory, queue_size=10)   # k-step ahead predicted variance
        self.pred_state_lcl_pub_pj  = rospy.Publisher(node_name + predicted_hri_state_topic + '/pj/lcl', JointTrajectory, queue_size=10)   # PLOTJUGGLER
        self.pred_state_ucl_pub_pj  = rospy.Publisher(node_name + predicted_hri_state_topic + '/pj/ucl', JointTrajectory, queue_size=10)   # PLOTJUGGLER

        # Initialize camera and world frames
        self.camera_frame = camera_frame
        self.world_frame = world_frame

        # Initialize TF listener
        self.tf_listener = tf.TransformListener()

        # === ENABLE TO READ TRANSFORMATION FROM TF AT RUNTIME ===
        # # Wait for the transform to be available
        # while not self.tf_listener.canTransform(camera_frame, world_frame, rospy.Time(0)) \
        #       and not rospy.is_shutdown():
        #     rospy.sleep(0.1)

        # # # Get the transformation
        # (trans, rot) = self.tf_listener.lookupTransform(camera_frame, world_frame, rospy.Time(0))
 
        # # Compose the transformation matrix from the world frame to the camera frame
        # self.cam_to_world_matrix = tf.transformations.concatenate_matrices(
        #     tf.transformations.translation_matrix(trans),
        #     tf.transformations.quaternion_matrix(rot)
        # )
        # =========================================================

        # Assuming TF_world_camera is a list with the translation [tx, ty, tz] and quaternion [qx, qy, qz, qw]
        translation_world_camera = np.array(TF_world_camera[0:3])
        quaternion_world_camera = np.array(TF_world_camera[3:7])

        rospy.loginfo(f"Translation from world to camera: {translation_world_camera}")
        rospy.loginfo(f"Quaternion from world to camera: {quaternion_world_camera}")

        # Convert the quaternion to a rotation matrix
        rotation_matrix_world_camera = tf.transformations.quaternion_matrix(quaternion_world_camera)

        # Create a translation matrix
        translation_matrix_world_camera = tf.transformations.translation_matrix(translation_world_camera)

        # Combine the rotation and translation to get the transformation matrix from the world frame to the camera frame
        self.cam_to_world_matrix = tf.transformations.concatenate_matrices(
            translation_matrix_world_camera,
            rotation_matrix_world_camera
        )


    def read_skeleton_cb(self, msg: ObjectsStamped) -> None:
        if msg.objects:
            self.skeleton_absent_count = 0
            for obj in msg.objects: # TODO correct this: if multiple skeletons, selecting one and overwriting previous
                # Extract skeleton keypoints from message ([x, y, z] for each kpt)
                kpts = np.array([[kp.kp] for kp in obj.skeleton_3d.keypoints])
                kpts = kpts[:self.n_kpts] # select only the first n_kpts

                self.skeleton_kpts = np.reshape(kpts, (self.n_kpts, 3)) # reshape to (n_kpts, 3)

                # Convert keypoints to world frame
                for i in range(self.n_kpts):
                    # Create a homogeneous coordinate for the keypoint position
                    kpt = np.array([self.skeleton_kpts[i][0],
                                    self.skeleton_kpts[i][1],
                                    self.skeleton_kpts[i][2],
                                    1])

                    # Transform the keypoint to the world frame using the transformation matrix
                    kpt_world = np.dot(self.cam_to_world_matrix, kpt)

                    self.skeleton_kpts[i][0] = kpt_world[0]
                    self.skeleton_kpts[i][1] = kpt_world[1]
                    self.skeleton_kpts[i][2] = kpt_world[2]
            
        else:
            self.skeleton_kpts = np.full(self.skeleton_kpts.shape, np.nan)
            self.skeleton_absent_count += 1

        # Update current human measurement vector
        self.human_meas = self.skeleton_kpts.flatten()

        # Publish current human measurement vector
        # Initialize PoseArray message
        human_meas_msg = PoseArray()

        # Set the header
        human_meas_msg.header.stamp = rospy.Time.now()
        human_meas_msg.header.frame_id = self.world_frame
        human_meas_msg.poses = []

        # Add the positions of all keypoints
        for i in range(self.n_kpts):
            pose = Pose()
            pose.position.x = self.skeleton_kpts[i][0]
            pose.position.y = self.skeleton_kpts[i][1]
            pose.position.z = self.skeleton_kpts[i][2]
            pose.orientation.w = 1.0 # dummy value (orientation.x, y, z are 0.0 by default)

            human_meas_msg.poses.append(pose)

        # Publish the message
        self.human_meas_pub.publish(human_meas_msg)


    def skeleton_present(self) -> bool:
        return self.skeleton_absent_count < self.max_steps_skeleton_absent


    def read_robot_js_cb(self, msg: JointState) -> None:
        pos = msg.position
        vel = msg.velocity
        self.robot_meas = np.dstack((pos, vel)).ravel()
        np.reshape(self.robot_meas, (1, -1))


    def publish_future_traj(self,
                            state: np.ndarray,
                            pub: rospy.Publisher,
                            pub_pj: rospy.Publisher, # PLOTJUGGLER
                            timestamp: rospy.Time,
                            n_points: int=1) -> None:
        state_msg = JointTrajectory()
        state_msg.header.stamp = timestamp
        state_msg.header.frame_id = self.world_frame

        state_msg.joint_names = []
        if self.kalman_predictor.model.human_model.kynematic_model == HM.KynematicModel.KEYPOINTS:
            for i in range(self.kalman_predictor.model.human_model.n_kpts):
                state_msg.joint_names.append(f'kpt_{i}_x')
                state_msg.joint_names.append(f'kpt_{i}_y')
                state_msg.joint_names.append(f'kpt_{i}_z')
        elif self.kalman_predictor.model.human_model.kynematic_model == HM.KynematicModel.KYN_CHAIN:
            for i in range(self.kalman_predictor.model.human_model.n_dof):
                state_msg.joint_names.append(f'human_joint_{i}')
        else:
            raise ValueError('Invalid kynematic model')

        state_msg.points = []
        for i in range(n_points):
            point = JointTrajectoryPoint()

            pos = state[i][self.kalman_predictor.p_idx]
            vel = state[i][self.kalman_predictor.v_idx]
            acc = state[i][self.kalman_predictor.a_idx]

            point.positions = pos.tolist()
            point.velocities = vel.tolist()
            point.accelerations = acc.tolist()
            point.time_from_start = rospy.Duration.from_sec(i*self.kalman_predictor.dt)

            state_msg.points.append(point)

        # PLOTJUGGLER (only plot the last point of the trajectory, the one furthest in the future)
        state_msg_pj = JointTrajectory()
        state_msg_pj.header.stamp = timestamp + rospy.Duration.from_sec(n_points*self.kalman_predictor.dt)
        state_msg_pj.header.frame_id = self.world_frame
        state_msg_pj.joint_names = state_msg.joint_names
        state_msg_pj.points = []

        point = JointTrajectoryPoint()

        pos = state[-1][self.kalman_predictor.p_idx]
        vel = state[-1][self.kalman_predictor.v_idx]
        acc = state[-1][self.kalman_predictor.a_idx]

        point.positions = pos.tolist()
        point.velocities = vel.tolist()
        point.accelerations = acc.tolist()
        point.time_from_start = rospy.Duration.from_sec(0.0)

        state_msg_pj.points.append(point)

        pub.publish(state_msg)
        pub_pj.publish(state_msg_pj)


    def publish_future_traj_stdDev(self,
                                   state: np.ndarray,
                                   var: np.ndarray,
                                   pub_lcl: rospy.Publisher,
                                   pub_ucl: rospy.Publisher,
                                   pub_lcl_pj: rospy.Publisher, # PLOTJUGGLER
                                   pub_ucl_pj: rospy.Publisher, # PLOTJUGGLER
                                   timestamp: rospy.Time,
                                   n_points: int=1) -> None:
        # Lower confidence limit
        state_msg_lcl = JointTrajectory()
        state_msg_lcl.header.stamp = timestamp # + rospy.Duration.from_sec(n_points*self.kalman_predictor.dt)
        state_msg_lcl.header.frame_id = self.world_frame

        state_msg_lcl.joint_names = []
        if self.kalman_predictor.model.human_model.kynematic_model == HM.KynematicModel.KEYPOINTS:
            for i in range(self.kalman_predictor.model.human_model.n_kpts):
                state_msg_lcl.joint_names.append(f'kpt_{i}_x')
                state_msg_lcl.joint_names.append(f'kpt_{i}_y')
                state_msg_lcl.joint_names.append(f'kpt_{i}_z')
        elif self.kalman_predictor.model.human_model.kynematic_model == HM.KynematicModel.KYN_CHAIN:
            for i in range(self.kalman_predictor.model.human_model.n_dof):
                state_msg_lcl.joint_names.append(f'human_joint_{i}')
        else:
            raise ValueError('Invalid kynematic model')
        
        state_msg_lcl.points = []
        for i in range(n_points):
            point = JointTrajectoryPoint()

            pos = state[i][self.kalman_predictor.p_idx]
            vel = state[i][self.kalman_predictor.v_idx]
            acc = state[i][self.kalman_predictor.a_idx]
            pos_std = np.sqrt(var[i][self.kalman_predictor.p_idx])
            vel_std = np.sqrt(var[i][self.kalman_predictor.v_idx])
            acc_std = np.sqrt(var[i][self.kalman_predictor.a_idx])

            point.positions = (pos - 1*pos_std).tolist()
            point.velocities = (vel - 1*vel_std).tolist()
            point.accelerations = (acc - 1*acc_std).tolist()
            point.time_from_start = rospy.Duration.from_sec(i*self.kalman_predictor.dt)

            state_msg_lcl.points.append(point)

        # Upper confidence limit
        state_msg_ucl = state_msg_lcl

        state_msg_ucl.points = []
        for i in range(n_points):
            point = JointTrajectoryPoint()

            pos = state[i][self.kalman_predictor.p_idx]
            vel = state[i][self.kalman_predictor.v_idx]
            acc = state[i][self.kalman_predictor.a_idx]
            pos_std = np.sqrt(var[i][self.kalman_predictor.p_idx])
            vel_std = np.sqrt(var[i][self.kalman_predictor.v_idx])
            acc_std = np.sqrt(var[i][self.kalman_predictor.a_idx])

            point.positions = (pos + 1*pos_std).tolist()
            point.velocities = (vel + 1*vel_std).tolist()
            point.accelerations = (acc + 1*acc_std).tolist()
            point.time_from_start = rospy.Duration.from_sec(i*self.kalman_predictor.dt)

            state_msg_ucl.points.append(point)

        # PLOTJUGGLER - LCL
        state_msg_lcl_pj = JointTrajectory()
        state_msg_lcl_pj.header.stamp = timestamp + rospy.Duration.from_sec(n_points*self.kalman_predictor.dt)
        state_msg_lcl_pj.header.frame_id = self.world_frame
        state_msg_lcl_pj.joint_names = state_msg_lcl.joint_names
        state_msg_lcl_pj.points = []

        point = JointTrajectoryPoint()

        pos = state[-1][self.kalman_predictor.p_idx]
        vel = state[-1][self.kalman_predictor.v_idx]
        acc = state[-1][self.kalman_predictor.a_idx]

        point.positions = (pos - 1*pos_std).tolist()
        point.velocities = (vel - 1*vel_std).tolist()
        point.accelerations = (acc - 1*acc_std).tolist()
        point.time_from_start = rospy.Duration.from_sec(0.0)

        state_msg_lcl_pj.points.append(point)

        # PLOTJUGGLER - UCL
        state_msg_ucl_pj = JointTrajectory()
        state_msg_ucl_pj.header.stamp = timestamp + rospy.Duration.from_sec(n_points*self.kalman_predictor.dt)
        state_msg_ucl_pj.header.frame_id = self.world_frame
        state_msg_ucl_pj.joint_names = state_msg_ucl.joint_names
        state_msg_ucl_pj.points = []

        point = JointTrajectoryPoint()

        point.positions = (pos + 1*pos_std).tolist()
        point.velocities = (vel + 1*vel_std).tolist()
        point.accelerations = (acc + 1*acc_std).tolist()
        point.time_from_start = rospy.Duration.from_sec(0.0)

        state_msg_ucl_pj.points.append(point)

        # Publish the messages
        pub_lcl.publish(state_msg_lcl)
        pub_ucl.publish(state_msg_ucl)
        pub_lcl_pj.publish(state_msg_lcl_pj)
        pub_ucl_pj.publish(state_msg_ucl_pj)
        

    def publish_human_filt_pos(self, timestamp):
        # Initialize PoseArray message
        human_filt_pos_msg = PoseArray()

        # Set the header
        human_filt_pos_msg.header.stamp = timestamp
        human_filt_pos_msg.header.frame_id = self.world_frame
        human_filt_pos_msg.poses = []

        # Add the positions of all keypoints
        for i in range(self.n_kpts):
            pose = Pose()

            if self.kalman_predictor.model.human_model.kynematic_model == HM.KynematicModel.KEYPOINTS:
                x_idx = self.kalman_predictor.p_idx[i*3]
                y_idx = self.kalman_predictor.p_idx[i*3 + 1]
                z_idx = self.kalman_predictor.p_idx[i*3 + 2]
                pose.position.x = self.kalman_predictor.kalman_filter.x[x_idx]
                pose.position.y = self.kalman_predictor.kalman_filter.x[y_idx]
                pose.position.z = self.kalman_predictor.kalman_filter.x[z_idx]
                pose.orientation.w = 1.0 # dummy value (orientation.x, y, z are 0.0 by default)
            elif self.kalman_predictor.model.human_model.kynematic_model == HM.KynematicModel.KYN_CHAIN:
                pass # TODO: Implement this for KYN_CHAIN model
            else:
                raise ValueError('Invalid kynematic model')

            human_filt_pos_msg.poses.append(pose)

        self.human_filt_pos_pub.publish(human_filt_pos_msg)


    def publish_human_filt_vel(self, timestamp):
        # Initialize PoseArray message
        human_filt_vel_msg = PoseArray() # TODO: Change the message type to TwistArray (not available in geometry_msgs)

        # Set the header
        human_filt_vel_msg.header.stamp = timestamp
        human_filt_vel_msg.header.frame_id = self.world_frame
        human_filt_vel_msg.poses = []

        # Add the velocities of all keypoints
        for i in range(self.n_kpts):
            pose = Pose()

            if self.kalman_predictor.model.human_model.kynematic_model == HM.KynematicModel.KEYPOINTS:
                x_idx = self.kalman_predictor.v_idx[i*3]
                y_idx = self.kalman_predictor.v_idx[i*3 + 1]
                z_idx = self.kalman_predictor.v_idx[i*3 + 2]
                pose.position.x = self.kalman_predictor.kalman_filter.x[x_idx]
                pose.position.y = self.kalman_predictor.kalman_filter.x[y_idx]
                pose.position.z = self.kalman_predictor.kalman_filter.x[z_idx]
                pose.orientation.w = 1.0
            elif self.kalman_predictor.model.human_model.kynematic_model == HM.KynematicModel.KYN_CHAIN:
                pass # TODO: Implement this for KYN_CHAIN model
            else:
                raise ValueError('Invalid kynematic model')

            human_filt_vel_msg.poses.append(pose)

        self.human_filt_vel_pub.publish(human_filt_vel_msg)


    def publish_human_filt_acc(self, timestamp):
        # Initialize PoseArray message
        human_filt_acc_msg = PoseArray()

        # Set the header
        human_filt_acc_msg.header.stamp = timestamp
        human_filt_acc_msg.header.frame_id = self.world_frame
        human_filt_acc_msg.poses = []

        # Add the positions of all keypoints
        for i in range(self.n_kpts):
            pose = Pose()

            if self.kalman_predictor.model.human_model.kynematic_model == HM.KynematicModel.KEYPOINTS:
                x_idx = self.kalman_predictor.a_idx[i*3]
                y_idx = self.kalman_predictor.a_idx[i*3 + 1]
                z_idx = self.kalman_predictor.a_idx[i*3 + 2]
                pose.position.x = self.kalman_predictor.kalman_filter.x[x_idx]
                pose.position.y = self.kalman_predictor.kalman_filter.x[y_idx]
                pose.position.z = self.kalman_predictor.kalman_filter.x[z_idx]
                pose.orientation.w = 1.0 # dummy value (orientation.x, y, z are 0.0 by default)
            elif self.kalman_predictor.model.human_model.kynematic_model == HM.KynematicModel.KYN_CHAIN:
                pass # TODO: Implement this for KYN_CHAIN model
            else:
                raise ValueError('Invalid kynematic model')

            human_filt_acc_msg.poses.append(pose)

        self.human_filt_acc_pub.publish(human_filt_acc_msg)

    
    def predict_update_step(self,
                            iter: int,
                            t: float,
                            logs_dir: str="",
                            dump_to_file: bool=False,
                            plot_covariance: bool=False,
                            human_init_variance: dict={},
                            robot_init_variance: dict={}) -> None:

        predict_args = {'t': t, 'u': self.kalman_predictor.model.human_model.compute_control_action()}
        timestamp=rospy.Time.now()

        # PREDICT human_robot_system NEXT state using kalman_predictor
        if self.skeleton_absent_count < self.max_steps_skeleton_absent:
            if np.isnan(self.kalman_predictor.kalman_filter.x).all() and not np.isnan(self.human_meas).any():
                rospy.logwarn("Filter state is NaN. Resetting the filter with the current measurement.")

                # Initialize the human state with the initial measurements
                human_init_state = np.zeros_like(self.kalman_predictor.model.human_model.x)
                pos_idx = self.kalman_predictor.model.human_model.p_idx
                human_init_state[pos_idx] = self.human_meas
                human_init_state[np.isnan(human_init_state)] = 0.0 # set NaNs to 0.0

                self.kalman_predictor.initialize(x0_human=human_init_state,
                                                 P0_human=human_init_variance,
                                                 P0_robot=robot_init_variance)
                self.kalman_predictor.model.human_model.x = human_init_state
            
            elif np.isnan(self.kalman_predictor.kalman_filter.x).all():
                return
            
            else:
                # 1-step predict if skeleton is present for the last max_steps_skeleton_absent steps
                self.kalman_predictor.predict(**predict_args)

                # Publish filtered human position, velocity, and acceleration
                self.publish_human_filt_pos(timestamp)
                self.publish_human_filt_vel(timestamp)
                self.publish_human_filt_acc(timestamp)

                err_flag = 0
                # Check if human measurements are available. If not, skip model and kalman filter update
                if (np.isnan(self.human_meas)).all():
                    err_flag = 1
                
                # Check if human measurements contain NaNs. If so, skip model and kalman filter update
                if (np.isnan(self.human_meas)).any() and err_flag == 0:
                    err_flag = 2
                
                # Get current measurement vector (concatenation of human and robot measurements)
                current_meas = np.concatenate((self.human_meas, self.robot_meas))
                # rospy.loginfo(f"Current measurement: {current_meas}")

                # Check if the new measurement vector (current_meas) is different
                # from the previous one (self.kalman_predictor.kalman_filter.z).
                # To do so, only check if the non-nan values in both arrays are close
                non_nan_mask = ~np.isnan(current_meas) & ~np.isnan(self.kalman_predictor.kalman_filter.z)
                values_are_close = np.allclose(current_meas[non_nan_mask], self.kalman_predictor.kalman_filter.z[non_nan_mask])
                if values_are_close and err_flag == 0:
                    err_flag = 3

                current_meas[-12:] = 0.0 # zero out robot measurements # ELIMINATE
                # rospy.loginfo(f"Current measurement: {current_meas}") # ELIMINATE

                if err_flag == 0:
                    # UPDATE human_robot_system CURRENT measurement using kalman_predictor
                    self.kalman_predictor.update(current_meas)
                elif err_flag == 1:
                    rospy.logwarn("Human measurements are not available. Skipping update step.")
                elif err_flag == 2:
                    rospy.logwarn("Human measurements contain NaNs. Skipping update step.")
                elif err_flag == 3:
                    rospy.logwarn("Human-Robot measurements have not changed. Skipping update step.")
                else:
                    rospy.logwarn("Unknown error. Skipping update step.")
                
                # k-step ahead prediction of human_robot_system state
                pred_x_mean, pred_x_var = self.kalman_predictor.k_step_predict(**predict_args)

                # Select the sequence of predicted states and variances ONLY for the HUMAN agent
                human_state_traj = pred_x_mean[:self.kalman_predictor.model.human_model.n_states]
                human_var_traj = pred_x_var[:self.kalman_predictor.model.human_model.n_states]

                # Plot the covariance matrix P as a heatmap
                if plot_covariance:
                    self.plot_cov_matrix(iter)

                # Dump the sequence of predicted states and variances for the human agent to a file
                if dump_to_file:
                    np.savez_compressed(
                        os.path.join(logs_dir, f'iter_{iter}.npz'),
                        timestamp=timestamp.to_sec(),
                        human_meas_pos=self.human_meas,
                        human_filt_x=self.kalman_predictor.kalman_filter.x[:self.kalman_predictor.model.human_model.n_states],
                        human_filt_var=np.diag(self.kalman_predictor.kalman_filter.P[:self.kalman_predictor.model.human_model.n_states]),
                        pred_human_x=human_state_traj,
                        pred_human_var=human_var_traj,
                    )

                # Publish k-step ahead predicted state and variance
                self.publish_future_traj(human_state_traj,
                                        self.pred_state_pub,
                                        self.pred_state_pub_pj,
                                        timestamp,
                                        self.kalman_predictor.predict_steps)
                self.publish_future_traj(human_var_traj,
                                        self.pred_variance_pub,
                                        self.pred_variance_pub_pj,
                                        timestamp,
                                        self.kalman_predictor.predict_steps)
                self.publish_future_traj_stdDev(human_state_traj,
                                                human_var_traj,
                                                self.pred_state_lcl_pub,
                                                self.pred_state_ucl_pub,
                                                self.pred_state_lcl_pub_pj,
                                                self.pred_state_ucl_pub_pj,
                                                timestamp,
                                                self.kalman_predictor.predict_steps)

        else:
            rospy.logwarn(f"Skeleton is absent for more than {self.max_steps_skeleton_absent} steps.")

            # set the filter state to NaN if the skeleton is absent for more than max_steps_skeleton_absent steps
            self.kalman_predictor.kalman_filter.x = np.full_like(self.kalman_predictor.kalman_filter.x, np.nan)
            self.kalman_predictor.model.human_model.x = np.full_like(self.kalman_predictor.model.human_model.x, np.nan)


    def plot_cov_matrix(self, iter):
        plt.imshow(self.kalman_predictor.kalman_filter.P, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title('Covariance Matrix P at iteration ' + str(iter))
        plt.xlabel('State Dimension')
        plt.ylabel('State Dimension')
        plt.show(block=False)
        plt.pause(0.01)
        plt.clf()
