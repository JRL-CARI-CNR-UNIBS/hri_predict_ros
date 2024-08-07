#!/usr/bin/env python3

import sys, os
import rospy, rospkg
import numpy as np
from hri_predict_ros.Predictor import Predictor
import matplotlib.pyplot as plt


# Define global variables
node_name = "hri_prediction_node"
pred_horizon: float=0.5
dt: float=0.01
human_control_law = None
human_kynematic_model = None
human_noisy_model = None
human_noisy_measure = None
human_meas_variance = None
human_model_variance = None
human_init_variance = None
robot_init_variance = None
human_n_dof = None
human_n_kpts = None
human_Kp = None
human_Kd = None
human_K_repulse = None
robot_control_law = None
robot_n_dof = None
alpha = None
beta = None
kappa = None
skeleton_topic = "/zed/zed_node/body_trk/skeletons"
robot_js_topic = "/robot/joint_states"
predicted_hri_state_topic = "/predicted_hri_state"
predicted_hri_cov_topic = "/predicted_hri_variance"
human_meas_topic = "/human_meas_pos"
human_filt_pos_topic = "/human_filt_pos"
human_filt_vel_topic = "/human_filt_vel"
human_filt_acc_topic = "/human_filt_acc"
camera_frame = "zed_camera_link"  # if sl::REFERENCE_FRAME::WORLD for the ZED camera is selected
world_frame = "world"
dump_to_file = True
plot_covariance = False
TF_world_camera = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] # [x y z qx qy qz qw]
u_min_human = -100
u_max_human = 100
a_min_human = -50
a_max_human = 50
v_min_human = -5
v_max_human = 5
offline = False
sub_number = 0
max_time_skeleton_absent = 1


def read_params():
    global  dt, \
            pred_horizon, \
            human_control_law, \
            human_kynematic_model, \
            human_noisy_model, \
            human_noisy_measure, \
            human_meas_variance, \
            human_model_variance, \
            human_init_variance, \
            robot_init_variance, \
            human_n_dof, \
            human_n_kpts, \
            human_Kp, \
            human_Kd, \
            human_K_repulse, \
            robot_control_law, \
            robot_n_dof, \
            alpha, \
            beta, \
            kappa, \
            skeleton_topic, \
            robot_js_topic, \
            predicted_hri_state_topic, \
            predicted_hri_cov_topic, \
            human_meas_topic, \
            human_filt_pos_topic, \
            human_filt_vel_topic, \
            human_filt_acc_topic, \
            camera_frame, \
            world_frame, \
            dump_to_file, \
            plot_covariance, \
            TF_world_camera, \
            u_min_human, \
            u_max_human, \
            a_min_human, \
            a_max_human, \
            v_min_human, \
            v_max_human, \
            offline, \
            sub_number, \
            max_time_skeleton_absent

    try:
        dt =                                 rospy.get_param(node_name + '/dt')
        pred_horizon =                       rospy.get_param(node_name + '/pred_horizon')
        human_control_law =                  rospy.get_param(node_name + '/human_control_law')
        human_kynematic_model =              rospy.get_param(node_name + '/human_kynematic_model')
        human_noisy_model =                  rospy.get_param(node_name + '/human_noisy_model')
        human_noisy_measure =                rospy.get_param(node_name + '/human_noisy_measure')

        human_meas_variance = {} # [pos, vel] for each axis (x, y, z)
        for axis in ['x', 'y', 'z']:
            param_name = f'/human_meas_variance/{axis}'
            human_meas_variance[axis] = rospy.get_param(node_name + param_name)

        human_model_variance = {} # [pos, vel, acc] for each DoF
        for var_type in ['pos', 'vel', 'acc']:
                param_name = f'/human_model_variance/{var_type}'
                human_model_variance[var_type] = rospy.get_param(node_name + param_name)

        human_init_variance = {} # [pos, vel, acc] for each DoF
        for var_type in ['pos', 'vel', 'acc']:
                param_name = f'/human_init_variance/{var_type}'
                human_init_variance[var_type] = rospy.get_param(node_name + param_name)

        robot_init_variance = {} # [pos, vel, acc] for each DoF
        for var_type in ['pos', 'vel', 'acc']:
                param_name = f'/robot_init_variance/{var_type}'
                robot_init_variance[var_type] = rospy.get_param(node_name + param_name)

        human_n_dof =                        rospy.get_param(node_name + '/human_n_dof')
        human_n_kpts =                       rospy.get_param(node_name + '/human_n_kpts')
        human_Kp =                           rospy.get_param(node_name + '/human_Kp')
        human_Kd =                           rospy.get_param(node_name + '/human_Kd')
        human_K_repulse =                    rospy.get_param(node_name + '/human_K_repulse')
        robot_control_law =                  rospy.get_param(node_name + '/robot_control_law')
        robot_n_dof =                        rospy.get_param(node_name + '/robot_n_dof')
        alpha =                              rospy.get_param(node_name + '/alpha')
        beta =                               rospy.get_param(node_name + '/beta')
        kappa =                              rospy.get_param(node_name + '/kappa')
        skeleton_topic =                     rospy.get_param(node_name + '/skeleton_topic', skeleton_topic)
        robot_js_topic =                     rospy.get_param(node_name + '/robot_js_topic', robot_js_topic)
        predicted_hri_state_topic =          rospy.get_param(node_name + '/predicted_hri_state_topic', predicted_hri_state_topic)
        predicted_hri_cov_topic =            rospy.get_param(node_name + '/predicted_hri_cov_topic', predicted_hri_cov_topic)
        human_meas_topic =                   rospy.get_param(node_name + '/human_meas_topic', human_meas_topic)
        human_filt_pos_topic =               rospy.get_param(node_name + '/human_filt_pos_topic', human_filt_pos_topic)
        human_filt_vel_topic =               rospy.get_param(node_name + '/human_filt_vel_topic', human_filt_vel_topic)
        human_filt_acc_topic =               rospy.get_param(node_name + '/human_filt_acc_topic', human_filt_acc_topic)
        camera_frame =                       rospy.get_param(node_name + '/camera_frame', camera_frame)
        world_frame =                        rospy.get_param(node_name + '/world_frame', world_frame)
        dump_to_file =                       rospy.get_param(node_name + '/dump_to_file', dump_to_file)
        plot_covariance =                    rospy.get_param(node_name + '/plot_covariance', plot_covariance)
        TF_world_camera =                    rospy.get_param(node_name + '/TF_world_camera', TF_world_camera)
        u_min_human =                        rospy.get_param(node_name + '/u_min_human', u_min_human)
        u_max_human =                        rospy.get_param(node_name + '/u_max_human', u_max_human)
        a_min_human =                        rospy.get_param(node_name + '/a_min_human', a_min_human)
        a_max_human =                        rospy.get_param(node_name + '/a_max_human', a_max_human)
        v_min_human =                        rospy.get_param(node_name + '/v_min_human', v_min_human)
        v_max_human =                        rospy.get_param(node_name + '/v_max_human', v_max_human)
        offline =                            rospy.get_param(node_name + '/offline')
        sub_number =                         rospy.get_param(node_name + '/sub_number')
        max_time_skeleton_absent =           rospy.get_param(node_name + '/max_time_skeleton_absent')

        if bool(offline):
            print()
            rospy.logwarn("========================")
            rospy.logwarn("Running in offline mode.")
            rospy.logwarn("========================\n")
            skeleton_topic = '/offline' + str(skeleton_topic)
            robot_js_topic = '/offline' + str(robot_js_topic)

    except KeyError:
        rospy.logerr(f"Some parameters are not set. Exiting.")
        rospy.signal_shutdown("Parameters not set.")
        sys.exit(1)  # exit the program

    rospy.loginfo(
        f"Loaded parameters: \n\
        node_name={node_name}, \n\
        dt={dt}, \n\
        pred_horizon={pred_horizon}, \n\
        human_control_law={human_control_law}, \n\
        human_kynematic_model={human_kynematic_model}, \n\
        human_noisy_model={human_noisy_model}, \n\
        human_noisy_measure={human_noisy_measure}, \n\
        human_meas_variance={human_meas_variance}, \n\
        human_model_variance={human_model_variance}, \n\
        human_init_variance={human_init_variance}, \n\
        robot_init_variance={robot_init_variance}, \n\
        human_n_dof={human_n_dof}, \n\
        human_n_kpts={human_n_kpts}, \n\
        human_Kp={human_Kp}, \n\
        human_Kd={human_Kd}, \n\
        human_K_repulse={human_K_repulse}, \n\
        robot_control_law={robot_control_law}, \n\
        robot_n_dof={robot_n_dof}, \n\
        alpha={alpha}, \n\
        beta={beta}, \n\
        kappa={kappa}, \n\
        skeleton_topic={skeleton_topic}, \n\
        robot_js_topic={robot_js_topic}, \n\
        predicted_hri_state_topic={predicted_hri_state_topic}, \n\
        predicted_hri_cov_topic={predicted_hri_cov_topic}, \n\
        human_meas_topic={human_meas_topic}, \n\
        human_filt_pos_topic={human_filt_pos_topic}, \n\
        human_filt_vel_topic={human_filt_vel_topic}, \n\
        human_filt_acc_topic={human_filt_acc_topic}, \n\
        camera_frame={camera_frame}, \n\
        world_frame={world_frame}, \n\
        dump_to_file={dump_to_file}, \n\
        plot_covariance={plot_covariance}, \n\
        TF_world_camera={TF_world_camera}, \n\
        u_min_human={u_min_human}, \n\
        u_max_human={u_max_human}, \n\
        a_min_human={a_min_human}, \n\
        a_max_human={a_max_human}, \n\
        v_min_human={v_min_human}, \n\
        v_max_human={v_max_human}, \n\
        offline={offline}, \n\
        sub_number={sub_number}, \n\
        max_time_skeleton_absent={max_time_skeleton_absent}"
    )


def shutdown_hook():
    print("\n===== Shutting down node. =====\n")


def main():
    rospy.init_node(node_name, log_level=rospy.INFO)
    
    read_params()

    rospack = rospkg.RosPack()
    package_path = rospack.get_path('hri_predict_ros')
    log_dir = os.path.join(package_path, 'logs', 'npz', ('sub_'+str(sub_number)))

    # Check if the logs directory exists, if not, create it
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    rospy.loginfo(f"Dumping the logs to: {log_dir}")

    # Set the rate of the node
    rate = rospy.Rate(1/dt)

    # Number of steps to predict
    predict_steps = int(pred_horizon / dt)

    max_steps_skeleton_absent = int(max_time_skeleton_absent / dt)

    # Create Predictor object to interface with ROS
    predictor = Predictor(
        dt=dt,
        human_control_law=human_control_law,
        human_kynematic_model=human_kynematic_model,
        human_noisy_model=human_noisy_model,
        human_noisy_measure=human_noisy_measure,
        human_R=human_meas_variance,
        human_Q=human_model_variance,
        human_n_kpts=human_n_kpts,
        human_n_dof=human_n_dof,
        human_Kp=human_Kp,
        human_Kd=human_Kd,
        human_K_repulse=human_K_repulse,
        robot_control_law=robot_control_law,
        robot_n_dof=robot_n_dof,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        node_name=node_name,
        skeleton_topic=skeleton_topic,
        robot_js_topic=robot_js_topic,
        predicted_hri_state_topic=predicted_hri_state_topic,
        predicted_hri_cov_topic=predicted_hri_cov_topic,
        human_meas_topic=human_meas_topic,
        human_filt_pos_topic=human_filt_pos_topic,
        human_filt_vel_topic=human_filt_vel_topic,
        human_filt_acc_topic=human_filt_acc_topic,
        camera_frame=camera_frame,
        world_frame=world_frame,
        TF_world_camera=TF_world_camera,
        u_min_human=u_min_human,
        u_max_human=u_max_human,
        a_min_human=a_min_human,
        a_max_human=a_max_human,
        v_min_human=v_min_human,
        v_max_human=v_max_human,
        predict_steps=predict_steps,
        max_steps_skeleton_absent=max_steps_skeleton_absent
    )

    while np.isnan(predictor.human_meas).any():
        rospy.logwarn("Initialization: waiting for human measurements.")
        rate.sleep()

    # Initialize the human state with the initial measurements
    human_init_state = predictor.human_meas
    rospy.loginfo(f"Human initial measurements: {human_init_state}, shape: {human_init_state.shape}")

    # Initialize the kalman_predictor
    human_init_state = predictor.kalman_predictor.model.human_model.x
    pos_idx = predictor.kalman_predictor.model.human_model.p_idx
    human_init_state[pos_idx] = predictor.human_meas

    predictor.kalman_predictor.initialize(x0_human=human_init_state,
                                          P0_human=human_init_variance,
                                          P0_robot=robot_init_variance)

    # Main loop
    i = 0
    t = 0
    plt.figure()
    while not rospy.is_shutdown():
        try:
            predictor.predict_update_step(i, t, log_dir, dump_to_file, plot_covariance, human_init_variance, robot_init_variance)
        except np.linalg.LinAlgError as e:
            rospy.logerr(f"LinAlgError: {e}")
        
        i += 1
        t += dt
        rate.sleep()

    rospy.on_shutdown(shutdown_hook)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass