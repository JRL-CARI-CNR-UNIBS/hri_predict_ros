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
    kalman_predictor:       KalmanPredictor = field(init=False, repr=True)
    n_kpts:                 int = field(default=0, init=False, repr=True)
    skeleton_kpts:          np.ndarray = field(default=np.array([]), init=False, repr=False)
    human_meas:             np.ndarray = field(init=False, repr=False)
    robot_meas:             np.ndarray = field(init=False, repr=False)
    skeleton_sub:           rospy.Subscriber = field(init=True, repr=False)
    robot_state_sub:        rospy.Subscriber = field(init=True, repr=False)
    pred_state_pub:         rospy.Publisher = field(init=True, repr=False)
    pred_variance_pub:      rospy.Publisher = field(init=True, repr=False)
    pred_state_lcl_pub:     rospy.Publisher = field(init=True, repr=False) # DEBUG: state - 3*std_dev
    pred_state_ucl_pub:     rospy.Publisher = field(init=True, repr=False) # DEBUG: state + 3*std_dev
    test_publisher:         rospy.Publisher = field(init=True, repr=False) # DEBUG
    human_meas_pub:         rospy.Publisher = field(init=True, repr=False)
    tf_listener:            tf.TransformListener = field(init=False, repr=False)
    camera_frame:           str = field(default="", init=True, repr=True)
    world_frame:            str = field(default="world", init=True, repr=True)
    cam_to_world_matrix:    np.ndarray = field(default=np.eye(4), init=False, repr=True)


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
                 camera_frame: str="",
                 world_frame: str="world",
                 TF_world_camera: list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                 u_min_human: float=-100,
                 u_max_human: float=100,
                 a_min_human: float=-50,
                 a_max_human: float=50,
                 v_min_human: float=-5,
                 v_max_human: float=5) -> None:

        
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
            v_max_human=v_max_human
            )

        self.n_kpts = human_n_kpts

        # Initialize skeleton keypoints ([x, y, z] for each kpt)
        self.skeleton_kpts = np.zeros((self.n_kpts, 3), dtype=float)

        # Initialize human measurement vector ([x, y, z] position for each kpt)
        self.human_meas = np.full(3*human_n_kpts, np.nan, dtype=float)
        
        # Initialize robot measurement vector ([pos, vel] for each DoF)
        self.robot_meas = np.full(2*robot_n_dof, np.nan, dtype=float)

        # Initialize ROS subscribers and publishers
        self.skeleton_sub       = rospy.Subscriber(skeleton_topic, ObjectsStamped, self.read_skeleton_cb)
        self.robot_state_sub    = rospy.Subscriber(robot_js_topic, JointState, self.read_robot_js_cb)
        self.pred_state_pub     = rospy.Publisher(node_name + predicted_hri_state_topic, JointTrajectory, queue_size=10)   # k-step ahead predicted state
        self.pred_variance_pub  = rospy.Publisher(node_name + predicted_hri_cov_topic, JointTrajectory, queue_size=10)     # k-step ahead predicted variance
        self.pred_state_lcl_pub = rospy.Publisher(node_name + predicted_hri_state_topic + '/lcl', JointTrajectory, queue_size=10)   # k-step ahead predicted state
        self.pred_state_ucl_pub = rospy.Publisher(node_name + predicted_hri_state_topic + '/ucl', JointTrajectory, queue_size=10)   # k-step ahead predicted variance
        self.human_meas_pub     = rospy.Publisher(node_name + human_meas_topic, PoseArray, queue_size=10)                  # current human measured position
        self.human_filt_pos_pub = rospy.Publisher(node_name + human_filt_pos_topic, PoseArray, queue_size=10)              # current human estimated position
        self.human_filt_vel_pub = rospy.Publisher(node_name + human_filt_vel_topic, PoseArray, queue_size=10)              # current human estimated velocity

        # # DEBUG
        # self.righthand_publisher = rospy.Publisher(node_name + predicted_hri_state_topic + '/right_hand', PointStamped, queue_size=10)
        # self.pred_righthand_publisher = rospy.Publisher(node_name + predicted_hri_state_topic + '/pred_right_hand', PointStamped, queue_size=10)
        # self.head_publisher = rospy.Publisher(node_name + predicted_hri_state_topic + '/head', PointStamped, queue_size=10)
        # self.pred_head_publisher = rospy.Publisher(node_name + predicted_hri_state_topic + '/pred_head', PointStamped, queue_size=10)


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


    def read_robot_js_cb(self, msg: JointState) -> None:
        pos = msg.position
        vel = msg.velocity
        self.robot_meas = np.dstack((pos, vel)).ravel()
        np.reshape(self.robot_meas, (1, -1))


    def publish_future_traj(self, state: np.ndarray, pub: rospy.Publisher, n_points: int=1) -> None:
        time = rospy.Time.now()
        state_msg = JointTrajectory()
        state_msg.header.stamp = time # + rospy.Duration.from_sec(n_points*self.kalman_predictor.dt) # REMOVE ADDITIVE TERM
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

        pub.publish(state_msg)

        # # DEBUG
        # test_msg = PointStamped()
        # test_msg.header.stamp = time + rospy.Duration.from_sec(n_points*self.kalman_predictor.dt)
        # test_msg.header.frame_id = self.world_frame
        # test_msg.point.x = state[-1][self.kalman_predictor.p_idx[0]]
        # test_msg.point.y = state[-1][self.kalman_predictor.p_idx[1]]
        # test_msg.point.z = state[-1][self.kalman_predictor.p_idx[2]]

        # self.test_publisher.publish(test_msg)

        # # DEBUG - right hand filtered
        # righthand_msg = PointStamped()
        # righthand_msg.header.stamp = time + rospy.Duration.from_sec(n_points*self.kalman_predictor.dt)
        # righthand_msg.header.frame_id = self.world_frame
        # righthand_msg.point.x = self.kalman_predictor.kalman_filter.x[self.kalman_predictor.p_idx[12]]
        # righthand_msg.point.y = self.kalman_predictor.kalman_filter.x[self.kalman_predictor.p_idx[13]]
        # righthand_msg.point.z = self.kalman_predictor.kalman_filter.x[self.kalman_predictor.p_idx[14]]
                                         
        # self.righthand_publisher.publish(righthand_msg)

        # # DEBUG - head filtered
        # head_msg = PointStamped()
        # head_msg.header.stamp = time + rospy.Duration.from_sec(n_points*self.kalman_predictor.dt)
        # head_msg.header.frame_id = self.world_frame
        # head_msg.point.x = self.kalman_predictor.kalman_filter.x[self.kalman_predictor.p_idx[0]]
        # head_msg.point.y = self.kalman_predictor.kalman_filter.x[self.kalman_predictor.p_idx[1]]
        # head_msg.point.z = self.kalman_predictor.kalman_filter.x[self.kalman_predictor.p_idx[2]]
                                                                 
        # self.head_publisher.publish(head_msg)

        # # DEBUG - right hand prediction
        # righthand_msg = PointStamped()
        # righthand_msg.header.stamp = time + rospy.Duration.from_sec(n_points*self.kalman_predictor.dt)
        # righthand_msg.header.frame_id = self.world_frame
        # righthand_msg.point.x = state[-1][self.kalman_predictor.p_idx[12]]
        # righthand_msg.point.y = state[-1][self.kalman_predictor.p_idx[13]]
        # righthand_msg.point.z = state[-1][self.kalman_predictor.p_idx[14]]
                                         
        # self.pred_righthand_publisher.publish(righthand_msg)

        # # DEBUG - head prediction
        # head_msg = PointStamped()
        # head_msg.header.stamp = time + rospy.Duration.from_sec(n_points*self.kalman_predictor.dt)
        # head_msg.header.frame_id = self.world_frame
        # head_msg.point.x = state[-1][self.kalman_predictor.p_idx[0]]
        # head_msg.point.y = state[-1][self.kalman_predictor.p_idx[1]]
        # head_msg.point.z = state[-1][self.kalman_predictor.p_idx[2]]

        # self.pred_head_publisher.publish(head_msg)


    def publish_future_traj_stdDev(self,
                                   state: np.ndarray,
                                   var: np.ndarray,
                                   pub_lcl: rospy.Publisher,
                                   pub_ucl: rospy.Publisher,
                                   n_points: int=1) -> None:
        # Lower confidence limit
        state_msg_lcl = JointTrajectory()
        state_msg_lcl.header.stamp = rospy.Time.now() # + rospy.Duration.from_sec(n_points*self.kalman_predictor.dt)
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

        pub_lcl.publish(state_msg_lcl)

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

        pub_ucl.publish(state_msg_ucl)
        

    def publish_human_filt_pos(self):
        # Initialize PoseArray message
        human_filt_pos_msg = PoseArray()

        # Set the header
        human_filt_pos_msg.header.stamp = rospy.Time.now()
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


    def publish_human_filt_vel(self):
        # Initialize PoseArray message
        human_filt_vel_msg = PoseArray() # TODO: Change the message type to TwistArray (not available in geometry_msgs)

        # Set the header
        human_filt_vel_msg.header.stamp = rospy.Time.now()
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

    
    def predict_update_step(self,
                            iter: int,
                            t: float,
                            logs_dir: str,
                            num_steps: int,
                            dump_to_file: bool=False,
                            plot_covariance: bool=False) -> None:

        predict_args = {'t': t, 'u': self.kalman_predictor.model.human_model.compute_control_action()}

        # PREDICT human_robot_system NEXT state using kalman_predictor
        self.kalman_predictor.predict(**predict_args)

        # k-step ahead prediction of human_robot_system state
        pred_x_mean, pred_x_var = self.kalman_predictor.k_step_predict(num_steps, **predict_args)

        # Select the sequence of predicted states and variances ONLY for the HUMAN agent
        human_state_traj = pred_x_mean[:self.kalman_predictor.model.human_model.n_states]
        human_var_traj = pred_x_var[:self.kalman_predictor.model.human_model.n_states]

        # Plot the covariance matrix P as a heatmap
        if plot_covariance:
            self.plot_cov_matrix(iter)

        # Dump the sequence of predicted states and variances for the human agent to a file
        if dump_to_file:
            np.savez(os.path.join(logs_dir, 'npz', f'human_traj_iter_{iter}.npz'),
                     timestamp=rospy.Time.now().to_sec(),
                     pred_human_x=human_state_traj,
                     pred_human_var=human_var_traj,
                     filt_cov_mat=self.kalman_predictor.kalman_filter.P)

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
        
        # Publish filtered human position and velocity
        self.publish_human_filt_pos()
        self.publish_human_filt_vel()

        # Publish k-step ahead predicted state and variance
        self.publish_future_traj(human_state_traj, self.pred_state_pub, num_steps)
        self.publish_future_traj(human_var_traj, self.pred_variance_pub, num_steps)
        self.publish_future_traj_stdDev(human_state_traj,
                                        human_var_traj,
                                        self.pred_state_lcl_pub,
                                        self.pred_state_ucl_pub,
                                        num_steps)


    def plot_cov_matrix(self, iter):
        plt.imshow(self.kalman_predictor.kalman_filter.P, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title('Covariance Matrix P at iteration ' + str(iter))
        plt.xlabel('State Dimension')
        plt.ylabel('State Dimension')
        plt.show(block=False)
        plt.pause(0.01)
        plt.clf()
