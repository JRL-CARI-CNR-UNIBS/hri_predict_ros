from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import os

import hri_predict.HumanModel as HM
import hri_predict.RobotModel as RM
from hri_predict.KalmanPredictor import KalmanPredictor

import rospy
import tf
from zed_msgs.msg import ObjectsStamped
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped, PoseArray, Pose
from std_msgs.msg import Float64MultiArray


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
    covariance_pub:         rospy.Publisher = field(init=True, repr=False)
    human_meas_pub:        rospy.Publisher = field(init=True, repr=False)
    tf_listener:            tf.TransformListener = field(init=False, repr=False)
    camera_frame:           str = field(default="", init=True, repr=True)
    world_frame:            str = field(default="world", init=True, repr=True)


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
                 human_state_topic: str="",
                 camera_frame: str="",
                 world_frame: str="world") -> None:
        
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
        self.pred_state_pub     = rospy.Publisher(node_name + predicted_hri_state_topic, Float64MultiArray, queue_size=10)
        self.covariance_pub     = rospy.Publisher(node_name + predicted_hri_cov_topic, Float64MultiArray, queue_size=10)
        self.human_meas_pub    = rospy.Publisher(node_name + human_state_topic, PoseArray, queue_size=10)

        # Initialize camera and world frames
        self.camera_frame = camera_frame
        self.world_frame = world_frame

        # Initialize TF listener
        self.tf_listener = tf.TransformListener()


    def read_skeleton_cb(self, msg: ObjectsStamped) -> None:
        if msg.objects:
            for obj in msg.objects: # TODO correct this: if multiple skeletons, selecting one and overwriting previous
                # Extract skeleton keypoints from message ([x, y, z] for each kpt)
                kpts = np.array([[kp.kp] for kp in obj.skeleton_3d.keypoints])
                kpts = kpts[:self.n_kpts] # select only the first n_kpts

                self.skeleton_kpts = np.reshape(kpts, (self.n_kpts, 3)) # reshape to (n_kpts, 3)

                # Convert keypoints to world frame
                for i in range(self.n_kpts):
                    # Create a PointStamped message for the keypoint position
                    kpt = PointStamped()
                    kpt.header.stamp = rospy.Time.now()
                    kpt.header.frame_id = self.camera_frame
                    kpt.point.x = self.skeleton_kpts[i][0]
                    kpt.point.y = self.skeleton_kpts[i][1]
                    kpt.point.z = self.skeleton_kpts[i][2]

                    # Transform the point to the world frame
                    try:
                        kpt_world = self.tf_listener.transformPoint(self.world_frame, kpt)
                    except tf.ExtrapolationException as e:
                        rospy.logwarn(f"TF Exception: {e}")
                        continue

                    # Update the keypoint position in the world frame
                    self.skeleton_kpts[i][0] = kpt_world.point.x
                    self.skeleton_kpts[i][1] = kpt_world.point.y
                    self.skeleton_kpts[i][2] = kpt_world.point.z
            
        else:
            # rospy.logwarn("No skeleton keypoints received.")
            self.skeleton_kpts = np.full(self.skeleton_kpts.shape, np.nan)

        # Update current human measurement vector
        self.human_meas = self.skeleton_kpts.flatten()

        # DEBUG: Print the detected skeleton keypoints
        # rospy.loginfo(f"Received skeleton keypoints:\n{self.skeleton_kpts}, shape: {self.skeleton_kpts.shape}")
        # rospy.loginfo(f"Received human state:\n{self.human_meas}, shape: {self.human_meas.shape}")

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
        # rospy.loginfo(f"Received robot state: {self.robot_state}")


    def publish_state(self, state):
        state_msg = Float64MultiArray()
        # Each row corresponds to an element and each column corresponds to a future value
        state_msg.data = state.flatten('F').tolist() # flatten the 2D array into a 1D array in column-major (Fortran-style) order, which means it concatenates the columns together
        self.pred_state_pub.publish(state_msg)
        

    def publish_covariance(self, covariance):
        covariance_msg = Float64MultiArray()
        covariance_msg.data = covariance.flatten().tolist()
        self.covariance_pub.publish(covariance_msg)

    
    def predict_update_step(self, iter, logs_dir, num_steps) -> None:
        # Write the state covariance matrix to a new file 'P_i.csv'
        # self.write_cov_matrix(logs_dir, iter)

        # Plot the covariance matrix P as a heatmap
        self.plot_cov_matrix(iter)

        # PREDICT human_robot_system NEXT state using kalman_predictor
        self.kalman_predictor.predict()

        ######################################################################### TODO: K-STEP PREDICT IS VERY SLOW
        # # k-step ahead prediction of human_robot_system state
        # pred_state, pred_cov = self.kalman_predictor.k_step_predict(num_steps)

        # # DUBUG: Print the predicted state and covariance
        # # rospy.loginfo(f"Predicted State: {pred_state}\n")
        # # rospy.loginfo(f"Predicted Covariance: {pred_cov}\n\n")

        ######################################################################### TODO: FIX THE PUBLISHING OF PREDICTED STATE AND COVARIANCE OR CHANGE WITH SAVING TO FILE
        # # Publish the sequence of predicted states along with their covariances
        # self.publish_state(pred_state)
        # self.publish_covariance(pred_cov)

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