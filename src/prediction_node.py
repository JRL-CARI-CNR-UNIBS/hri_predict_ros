#!/usr/bin/env python3

import rospy
import numpy as np
import hri_predict.HumanModel as HM
import hri_predict.RobotModel as RM
from hri_predict.HumanRobotSystem import HumanRobotSystem
from hri_predict.KalmanPredictor import KalmanPredictor

def main():
    rospy.init_node('kalman_predictor_node')

    # Instantiate HumanRobotSystem
    human_robot_system = HumanRobotSystem(
        dt=0.01,
        human_control_law=HM.ControlLaw.CONST_VEL,
        human_kynematic_model=HM.KynematicModel.KEYPOINTS,
        human_noisy=False,
        human_W=np.array([], dtype=float),
        human_n_dof=3,
        human_Kp=1.0,
        human_Kd=1.0,
        human_K_repulse=1.0,
        robot_control_law=RM.ControlLaw.TRAJ_FOLLOW,
        robot_n_dof=6
    )

    # Instantiate KalmanPredictor
    kalman_predictor = KalmanPredictor(
        dt=0.01,
        human_control_law=HM.ControlLaw.CONST_VEL,
        human_kynematic_model=HM.KynematicModel.KEYPOINTS,
        human_noisy=False,
        human_W=np.array([], dtype=float),
        human_n_dof=3,
        human_Kp=1.0,
        human_Kd=1.0,
        human_K_repulse=1.0,
        robot_control_law=RM.ControlLaw.TRAJ_FOLLOW,
        robot_n_dof=6,
        alpha=0.3,
        beta=2.0,
        kappa=0.1
    )

    rate = rospy.Rate(100)  # 100 Hz

    while not rospy.is_shutdown():
        # Read person keypoint from camera

        # Read robot joint coordinates from robot

        # UPDATE human_robot_system CURRENT state using kalman_predictor

        # PREDICT human_robot_system NEXT state using kalman_predictor

        # 0

        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass