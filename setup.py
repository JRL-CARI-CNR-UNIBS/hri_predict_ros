from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['hri_predict_ros'],
    scripts=['src/prediction_node.py'],
    package_dir={'': 'src'}
)

setup(**d)