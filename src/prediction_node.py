#!/usr/bin/env python3

import sys
import rospy
import numpy as np
from hri_predict_ros.Predictor import Predictor


# Define global variables
node_name = "hri_prediction_node"
dt = 0.01
human_control_law = None
human_kynematic_model = None
human_noisy_model = None
human_noisy_measure = None
human_model_variances = None
human_init_variances = None
robot_init_variances = None
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
predicted_hri_cov_topic = "/predicted_hri_cov"
human_state_topic = "/human_state"
camera_frame = "zed_camera_link"  # if sl::REFERENCE_FRAME::WORLD for the ZED camera is selected
world_frame = "world"
hz = 100
num_steps = 10


def read_params():
    global  dt, \
            human_control_law, \
            human_kynematic_model, \
            human_noisy_model, \
            human_noisy_measure, \
            human_model_variances, \
            human_init_variances, \
            robot_init_variances, \
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
            human_state_topic, \
            camera_frame, \
            world_frame, \
            hz, \
            num_steps
    
    try:
        dt =                                 rospy.get_param(node_name + '/dt')
        human_control_law =                  rospy.get_param(node_name + '/human_control_law')
        human_kynematic_model =              rospy.get_param(node_name + '/human_kynematic_model')
        human_noisy_model =                  rospy.get_param(node_name + '/human_noisy_model')
        human_noisy_measure =                rospy.get_param(node_name + '/human_noisy_measure')
        human_model_variances =     np.array(rospy.get_param(node_name + '/human_model_variances'))
        human_init_variances =      np.array(rospy.get_param(node_name + '/human_init_variances'))
        robot_init_variances =      np.array(rospy.get_param(node_name + '/robot_init_variances'))
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
        human_state_topic =                  rospy.get_param(node_name + '/human_state_topic', human_state_topic)
        camera_frame =                       rospy.get_param(node_name + '/camera_frame', camera_frame)
        world_frame =                        rospy.get_param(node_name + '/world_frame', world_frame)
        hz =                                 rospy.get_param(node_name + '/hz', hz)
        num_steps =                          rospy.get_param(node_name + '/num_steps', num_steps)

        rospy.loginfo(f"Loaded parameters: \n\
            node_name={node_name}, \n\
            dt={dt}, \n\
            human_control_law={human_control_law}, \n\
            human_kynematic_model={human_kynematic_model}, \n\
            human_noisy_model={human_noisy_model}, \n\
            human_noisy_measure={human_noisy_measure}, \n\
            human_model_variances={human_model_variances}, \n\
            human_init_variances={human_init_variances}, \n\
            robot_init_variances={robot_init_variances}, \n\
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
            human_state_topic={human_state_topic}, \n\
            camera_frame={camera_frame}, \n\
            world_frame={world_frame}, \n\
            hz={hz}, \n\
            num_steps={num_steps}"
        )

    except KeyError:
        rospy.logerr(f"Some parameters are not set. Exiting.")
        rospy.signal_shutdown("Parameters not set.")
        sys.exit(1)  # exit the program


def shutdown_hook():
    print("\n===== Shutting down node. =====\n")


def main():
    rospy.init_node(node_name, log_level=rospy.INFO)
    
    read_params()

    # Create Predictor object to interface with ROS
    predictor = Predictor(
        dt=dt,
        human_control_law=human_control_law,
        human_kynematic_model=human_kynematic_model,
        human_noisy_model=human_noisy_model,
        human_noisy_measure=human_noisy_measure,
        human_W=human_model_variances,
        human_n_dof=human_n_dof,
        human_n_kpts=human_n_kpts,
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
        human_state_topic=human_state_topic,
        camera_frame=camera_frame,
        world_frame=world_frame
    )

    rospy.loginfo("CREATED 'PREDICTOR' OBJECT:" + 
                  "\n======================================================================" +
                  f"\n{predictor}" +
                  "\n======================================================================\n")

    # Initialize the kalman_predictor
    predictor.kalman_predictor.initialize(x0_human=predictor.human_state,
                                          x0_robot=predictor.robot_state,
                                          P0_human=human_init_variances,
                                          P0_robot=robot_init_variances)
    

    # Set the rate of the node
    rate = rospy.Rate(hz)

    # Main loop
    while not rospy.is_shutdown():
        # UPDATE human_robot_system CURRENT state using kalman_predictor
        current_meas = np.concatenate((predictor.human_state, predictor.robot_state))
        predictor.kalman_predictor.update(current_meas)

        # PREDICT human_robot_system NEXT state using kalman_predictor
        predictor.kalman_predictor.predict()

        # k-step ahead prediction of human_robot_system state
        pred_state, pred_cov = predictor.kalman_predictor.k_step_predict(num_steps)

        # DUBUG: Print the predicted state and covariance
        rospy.logerr(f"Predicted State: {pred_state}\n")
        rospy.logerr(f"Predicted Covariance: {pred_cov}\n\n")

        # Publish the sequence of predicted states along with their covariances
        predictor.publish_predicted_state(pred_state)
        predictor.publish_predicted_cov(pred_cov)

        rospy.spin()
        rate.sleep()

    rospy.on_shutdown(shutdown_hook)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass