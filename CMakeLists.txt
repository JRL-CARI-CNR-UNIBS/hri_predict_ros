cmake_minimum_required(VERSION 3.0.2)
project(hri_predict_ros)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
)

# Install modules and global scripts declared therein
# See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

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
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############
## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
