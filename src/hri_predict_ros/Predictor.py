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
from geometry_msgs.msg import PointStamped, PoseArray, Pose
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
    human_meas_pub:         rospy.Publisher = field(init=True, repr=False)
    tf_listener:            tf.TransformListener = field(init=False, repr=False)
    camera_frame:           str = field(default="", init=True, repr=True)
    world_frame:            str = field(default="world", init=True, repr=True)
    cam_to_world_matrix:    np.ndarray = field(default=np.eye(4), init=False, repr=True)


    def __init__(self,
                 dt: float=0.01,
                 human_control_law: str='CONST_VEL',
                 human_kynematic_model: str='KEYPOINTS',
                 human_noisy_model: bool=False,
                 human_noisy_measure: bool=False,
                 human_R: dict={},
                 human_W: dict={},
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
                 TF_world_camera: list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) -> None:
        
        human_control_law_      = HM.ControlLaw[human_control_law]
        human_kynematic_model_  = HM.KynematicModel[human_kynematic_model]
        robot_control_law_      = RM.ControlLaw[robot_control_law]

        # Override the number of DoF for the human agent if the kynematic model is KEYPOINTS
        if human_kynematic_model_ == HM.KynematicModel.KEYPOINTS:
            human_n_dof = human_n_kpts * 3 # n_dof = 3*n_keypoints (x, y, z for each keypoint)

        # Instantiate KalmanPredictor
        self.kalman_predictor = KalmanPredictor(
            dt=dt,
            human_control_law=human_control_law_,
            human_kynematic_model=human_kynematic_model_,
            human_noisy_model=human_noisy_model,
            human_noisy_measure=human_noisy_measure,
            human_R=human_R,
            human_W=human_W,
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

        # Initialize human measurement vector ([x, y, z] position for each kpt)
        self.human_meas = np.full(3*human_n_kpts, np.nan, dtype=float)
        
        # Initialize robot measurement vector ([pos, vel] for each DoF)
        self.robot_meas = np.full(2*robot_n_dof, np.nan, dtype=float)

        # Initialize ROS subscribers and publishers
        self.skeleton_sub       = rospy.Subscriber(skeleton_topic, ObjectsStamped, self.read_skeleton_cb)
        self.robot_state_sub    = rospy.Subscriber(robot_js_topic, JointState, self.read_robot_js_cb)
        self.pred_state_pub     = rospy.Publisher(node_name + predicted_hri_state_topic, JointTrajectory, queue_size=10)   # k-step ahead predicted state
        self.pred_variance_pub  = rospy.Publisher(node_name + predicted_hri_cov_topic, JointTrajectory, queue_size=10)     # k-step ahead predicted variance
        self.human_meas_pub     = rospy.Publisher(node_name + human_meas_topic, PoseArray, queue_size=10)                  # current human measured position
        self.human_filt_pos_pub = rospy.Publisher(node_name + human_filt_pos_topic, PoseArray, queue_size=10)              # current human estimated position
        self.human_filt_vel_pub = rospy.Publisher(node_name + human_filt_vel_topic, PoseArray, queue_size=10)              # current human estimated velocity

        # Initialize camera and world frames
        self.camera_frame = camera_frame
        self.world_frame = world_frame

        # Initialize TF listener
        self.tf_listener = tf.TransformListener()

        # ONLY FOR OFFLINE TESTING
        # # Wait for the transform to be available
        # while not self.tf_listener.canTransform(camera_frame, world_frame, rospy.Time(0)) \
        #       and not rospy.is_shutdown():
        #     rospy.sleep(0.1)

        # # Get the transformation
        # (trans, rot) = self.tf_listener.lookupTransform(camera_frame, world_frame, rospy.Time(0))

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
        TF_matrix_world_camera = tf.transformations.concatenate_matrices(translation_matrix_world_camera, rotation_matrix_world_camera)

        # Take the inverse of the transformation matrix to get the transformation matrix from the camera frame to the world frame
        self.cam_to_world_matrix = np.linalg.inv(TF_matrix_world_camera)


    def read_skeleton_cb(self, msg: ObjectsStamped) -> None:
        if msg.objects:
            for obj in msg.objects: # TODO correct this: if multiple skeletons, selecting one and overwriting previous
                # Extract skeleton keypoints from message ([x, y, z] for each kpt)
                kpts = np.array([[kp.kp] for kp in obj.skeleton_3d.keypoints])
                kpts = kpts[:self.n_kpts] # select only the first n_kpts

                self.skeleton_kpts = np.reshape(kpts, (self.n_kpts, 3)) # reshape to (n_kpts, 3)

                # Convert keypoints to world frame
                for i in range(self.n_kpts):
                    # # Create a PointStamped message for the keypoint position
                    # kpt = PointStamped()
                    # kpt.header.stamp = rospy.Time.now()
                    # kpt.header.frame_id = self.camera_frame
                    # kpt.point.x = self.skeleton_kpts[i][0]
                    # kpt.point.y = self.skeleton_kpts[i][1]
                    # kpt.point.z = self.skeleton_kpts[i][2]

                    # # Transform the point to the world frame
                    # try:
                    #     kpt_world = self.tf_listener.transformPoint(self.world_frame, kpt)
                    # except tf.ExtrapolationException as e:
                    #     rospy.logwarn(f"TF Exception: {e}")
                    #     continue

                    # # Update the keypoint position in the world frame
                    # self.skeleton_kpts[i][0] = kpt_world.point.x
                    # self.skeleton_kpts[i][1] = kpt_world.point.y
                    # self.skeleton_kpts[i][2] = kpt_world.point.z

                    # ONLY FOR OFFLINE TESTING
                    # Create a homogeneous coordinate for the keypoint position
                    kpt = np.array([self.skeleton_kpts[i][0], self.skeleton_kpts[i][1], self.skeleton_kpts[i][2], 1])

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
        state_msg = JointTrajectory()
        state_msg.header.stamp = rospy.Time.now()
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
            x_idx = self.kalman_predictor.p_idx[i*3]
            y_idx = self.kalman_predictor.p_idx[i*3 + 1]
            z_idx = self.kalman_predictor.p_idx[i*3 + 2]
            pose.position.x = self.kalman_predictor.kalman_filter.x[x_idx]
            pose.position.y = self.kalman_predictor.kalman_filter.x[y_idx]
            pose.position.z = self.kalman_predictor.kalman_filter.x[z_idx]
            pose.orientation.w = 1.0 # dummy value (orientation.x, y, z are 0.0 by default)

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
            x_idx = self.kalman_predictor.v_idx[i*3]
            y_idx = self.kalman_predictor.v_idx[i*3 + 1]
            z_idx = self.kalman_predictor.v_idx[i*3 + 2]
            pose.position.x = self.kalman_predictor.kalman_filter.x[x_idx]
            pose.position.y = self.kalman_predictor.kalman_filter.x[y_idx]
            pose.position.z = self.kalman_predictor.kalman_filter.x[z_idx]
            pose.orientation.w = 1.0

            human_filt_vel_msg.poses.append(pose)

        self.human_filt_vel_pub.publish(human_filt_vel_msg)

    
    def predict_update_step(self, iter, logs_dir, num_steps, dump_to_file: bool=False, plot_covariance: bool=False) -> None:
        # Publish filtered human position and velocity
        self.publish_human_filt_pos()
        self.publish_human_filt_vel()

        # PREDICT human_robot_system NEXT state using kalman_predictor
        self.kalman_predictor.predict()

        # k-step ahead prediction of human_robot_system state
        pred_x_mean, pred_x_var = self.kalman_predictor.k_step_predict(num_steps)

        # Publish the sequence of predicted states and variances for the human agent
        human_state_traj = pred_x_mean[:self.kalman_predictor.model.human_model.n_states]
        human_var_traj = pred_x_var[:self.kalman_predictor.model.human_model.n_states]
        self.publish_future_traj(human_state_traj, self.pred_state_pub, num_steps)
        self.publish_future_traj(human_var_traj, self.pred_variance_pub, num_steps)

        if plot_covariance:
            # Plot the covariance matrix P as a heatmap
            self.plot_cov_matrix(iter)

        if dump_to_file:
            # Dump the sequence of predicted states and variances for the human agent to a file
            np.savez(os.path.join(logs_dir, f'human_traj_iter_{iter}.npz'),
                     timestamp=rospy.Time.now().to_sec(),
                     pred_human_x=human_state_traj,
                     pred_human_var=human_var_traj,
                     filt_cov_mat=self.kalman_predictor.kalman_filter.P)

        # Check if human measurements are available. If not, skip model and kalman filter update
        if (np.isnan(self.human_meas)).all():
            rospy.logwarn("Human measurements are not available. Skipping update step.")
            return
        
        # Check if human measurements contain NaNs. If so, skip model and kalman filter update
        if (np.isnan(self.human_meas)).any():
            rospy.logwarn("Human measurements contain NaNs. Skipping update step.")
            return
        
        current_meas = np.concatenate((self.human_meas, self.robot_meas))
        # rospy.loginfo(f"Current measurement: {current_meas}")

        # Check if the new measurement vector (current_meas) is different
        # from the previous one (self.kalman_predictor.kalman_filter.z).
        # To do so, only check if the non-nan values in both arrays are close
        non_nan_mask = ~np.isnan(current_meas) & ~np.isnan(self.kalman_predictor.kalman_filter.z)
        values_are_close = np.allclose(current_meas[non_nan_mask], self.kalman_predictor.kalman_filter.z[non_nan_mask])
        if values_are_close:
            rospy.logwarn("Human-Robot measurements have not changed. Skipping update step.")
            return

        # UPDATE human_robot_system CURRENT measurement using kalman_predictor
        self.kalman_predictor.update(current_meas)


    def write_cov_matrix(self, logs_dir, iter):
        with open(os.path.join(logs_dir, f'P_{iter}.csv'), 'wb') as f:
            np.savetxt(f, self.kalman_predictor.kalman_filter.P, delimiter=",")
            rospy.loginfo(f"Saved state covariance matrix to 'P_{iter}.csv'")


    def plot_cov_matrix(self, iter):
        plt.imshow(self.kalman_predictor.kalman_filter.P, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title('Covariance Matrix P at iteration ' + str(iter))
        plt.xlabel('State Dimension')
        plt.ylabel('State Dimension')
        plt.show(block=False)
        plt.pause(0.01)
        plt.clf()
