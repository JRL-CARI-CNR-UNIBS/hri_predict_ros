import numpy as np
import pandas as pd
from utils import plot_covariance_cone
import os, pickle

SRC = {'FOLDER': 'scripts', 'SUBFOLDER': 'data_analysis'}                               # source folder
DATA = {'FOLDER': 'data',
        'SUBFOLDER_OUTPUT': 'output',
        'SUBFOLDER_PLOTS': 'plots',
}
# Define the number of keypoints and joints
N_KPTS = 13
N_JOINTS = 28
N_PARAM = 8                                     # (shoulder_distance, chest_hip_distance, hip_distance,
                                                # upper_arm_length, lower_arm_length,
                                                # upper_leg_length, lower_leg_length,
                                                # head_distance)
N_VAR_PER_JOINT = 3                             # position, velocity, acceleration
N_VAR_PER_KPT = 3                               # position, velocity, acceleration
N_DIM_PER_KPT = 3                               # x, y, z


SAMPLING_TIME = 0.1 # seconds
PREDICT_K_STEPS = True
PREDICTION_STEPS = 5 # steps

TRAINORTEST = 'train'

# Define the time step to uniformly sample the time range with covariance cones
CONE_STEP = 1 # seconds

DIM_TYPE = 'pos'

# Plot time series with covariance cones on the predicted estimates [PAPER VERSION]
FILTER_TYPES = ['CA', 'IMM']
VELOCITIES = ['FAST']#['SLOW', 'FAST']
TASKS = ['PICK-&-PLACE']#['PICK-&-PLACE', 'WALKING']
SPACES_COMPUTE = ['cartesian']#['cartesian', 'joint']
SPACES_EVAL = ['cartesian']#['cartesian', 'joint']
SUBJECT = 'sub_9'
INSTRUCTION = 1

# Get the current working directory
cwd = os.getcwd()

# Split the path to get the package directory
pkg_dir = cwd.split(SRC['FOLDER'])[0].split(SRC['SUBFOLDER'])[0]

# Define the data and plot directories
data_dir = os.path.join(pkg_dir, DATA['FOLDER'], DATA['SUBFOLDER_OUTPUT'])
plot_dir = os.path.join(pkg_dir, DATA['FOLDER'], DATA['SUBFOLDER_PLOTS'])
os.makedirs(plot_dir, exist_ok=True)

# Plot the time series with covariance cones for each combination of filter type, velocity, task, and space
for space_c in SPACES_COMPUTE:

    if space_c == 'cartesian':
        dim_x = N_VAR_PER_KPT * N_DIM_PER_KPT * N_KPTS
        p_idx = np.arange(0, dim_x, N_VAR_PER_KPT) # position indices
        param_idx = np.array([]) # no body params are states tracked by the filter
    elif space_c == 'joint':
        dim_x = N_VAR_PER_JOINT * N_JOINTS
        p_idx = np.arange(0, dim_x, N_VAR_PER_JOINT) # position indices
        param_idx = np.array(range(dim_x, dim_x + N_PARAM))
        dim_x += N_PARAM # the body params are states tracked by the filter
    else:
        ValueError("SPACE must be either 'cartesian' or 'joint'.")

    for space_e in SPACES_EVAL:
        assert space_e == 'cartesian' if space_c == 'cartesian' else True, \
            'The evaluation space must be CARTESIAN if the computation space is CARTESIAN'

        for filter_type in FILTER_TYPES:
            for velocity in VELOCITIES:
                for task in TASKS:

                    if space_e == 'cartesian':
                        # Load the measurements
                        file = os.path.join(data_dir,
                                            f'{TRAINORTEST}_measurements_{space_c}_.pkl') #TODO
                        
                        with open(file, 'rb') as f:
                            print(f"\nLoading file: {file}...")
                            measurements = pickle.load(f)      

                        # Load the filtering results
                        file = os.path.join(data_dir,
                                            f'{TRAINORTEST}_filtering_results_{PREDICTION_STEPS}_steps_{space_c}_.pkl')
                        
                        with open(file, 'rb') as f:
                            print(f"\nLoading file: {file}...")
                            filtering_results = pickle.load(f)

                        # Load the prediction results
                        file = os.path.join(data_dir,
                                            f'{TRAINORTEST}_prediction_results_{PREDICTION_STEPS}_steps_{space_c}_.pkl')
                        
                        with open(file, 'rb') as f:
                            print(f"Loading file: {file}...")
                            prediction_results = pickle.load(f)

                        # Define y_axes limits for the plots and the first and last timestamps of the measurements
                        if velocity == 'SLOW':
                            if task == 'PICK-&-PLACE':
                                y_axes_lim = [0.12, 0.85]
                                start_meas = 10.0
                                end_meas = 30.0
                                kpt = 4
                                dim = 'x'
                            elif task == 'WALKING':
                                y_axes_lim = [-1.0, 1.5]
                                start_meas = 5.0
                                end_meas = 25.0
                                kpt = 0
                                dim = 'y'
                        if velocity == 'FAST':
                            if task == 'PICK-&-PLACE':
                                y_axes_lim = [-0.3, 1.15]
                                start_meas = 0.0
                                end_meas = 20.0
                                kpt = 4
                                dim = 'x'
                            elif task == 'WALKING':
                                y_axes_lim = [-1, 1.25]
                                start_meas = 0.0
                                end_meas = 15.0
                                kpt = 0
                                dim = 'y'
                    elif space_e == 'JOINT':
                        pass
                    else:
                        raise ValueError('Invalid space for evaluation')

                    # Define a list of pd.Timedelta to uniformly sample the time range
                    selected_timestamps = [pd.Timedelta(seconds=s)
                                           for s in np.arange(start_meas+SAMPLING_TIME*PREDICTION_STEPS,
                                                              end_meas-CONE_STEP,
                                                              CONE_STEP)]

                    # Define the time range to plot the time series
                    lower_bound = pd.Timedelta(seconds=start_meas)
                    upper_bound = pd.Timedelta(seconds=end_meas)
                    selected_range = [lower_bound, upper_bound]

                    plot_covariance_cone(measurements, filtering_results, prediction_results,
                                         SUBJECT, velocity, task, INSTRUCTION, kpt, dim, DIM_TYPE,
                                         dim_x, N_VAR_PER_KPT, N_DIM_PER_KPT, SAMPLING_TIME, PREDICTION_STEPS,
                                         PREDICTION_STEPS, selected_timestamps, selected_range, filter_type, y_axes_lim,
                                         plot_dir)