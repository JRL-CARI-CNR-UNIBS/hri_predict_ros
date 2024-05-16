cmake_minimum_required(VERSION 3.0.2)
project(hri_predict_ros)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  zed_msgs
  sensor_msgs
  geometry_msgs
  trajectory_msgs
  # message_generation
)

# Install modules and global scripts declared therein
# See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

# Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   message.msg
# )

# Generate added messages and services with any dependencies listed here
# generate_messages(
  # DEPENDENCIES
  # std_msgs
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package()

#############
## Install ##
#############
## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
  src/prediction_node.py
  src/${PROJECT_NAME}/Predictor.py
  src/hri_predict/HumanModel.py
  src/hri_predict/RobotModel.py
  src/hri_predict/HumanRobotSystem.py
  src/hri_predict/KalmanPredictor.py
  src/hri_predict/utils.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark other files for installation (e.g. launch and bag files, etc.)
install(FILES
  config/params.yaml
  launch/predictor_bringup.launch
  launch/record_data_online.launch
  launch/record_data_offline.launch
  launch/execute_trajectories.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############
## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
