from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['hri_predict_ros'],
    scripts=['src/prediction_node.py',
             'src/hri_predict/HumanModel.py',
             'src/hri_predict/RobotModel.py',
             'src/hri_predict/HumanRobotSystem.py',
             'src/hri_predict/KalmanPredictor.py',
             'src/hri_predict/utils.py',
             'src/hri_predict_ros/Predictor.py'],
    package_dir={'': 'src'}
)

setup(**d)