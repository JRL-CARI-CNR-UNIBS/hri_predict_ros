import numpy as np
import pandas as pd
from utils import plot_covariance_cone
import os, pickle

SRC = {'FOLDER': 'scripts', 'SUBFOLDER': 'data_analysis'}                               # source folder
DATA = {'FOLDER': 'data',
        'SUBFOLDER_OUTPUT': 'output',
}
RESULTS = {'FOLDER': 'results', 'SUBFOLDER_PLOTS': 'plots'}

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

SELECTED_KEYPOINTS_FOR_KINEMATIC_MODEL = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]    # select which keypoints to consider for the kinematic model

NUM_SIGMAS = 1 # number of standard deviations for the covariance cones
PREDICT_K_STEPS = True
SAMPLING_TIME = 0.1 # seconds
PREDICTION_HORIZON = 0.5 # seconds
prediction_steps = int(np.round(PREDICTION_HORIZON / SAMPLING_TIME))

TRAINORTEST = 'train'

# Define the time step to uniformly sample the time range with covariance cones
CONE_STEP = 0.1 # seconds

DIM_TYPE = 'pos'

# Plot time series with covariance cones on the predicted estimates [PAPER VERSION]
FILTER_TYPES = ['CA', 'IMM']
VELOCITIES = ['FAST']#['SLOW', 'FAST']
TASKS = ['PICK-&-PLACE']#['PICK-&-PLACE', 'WALKING']
SPACES_COMPUTE = ['joint']#['cartesian', 'joint']
SPACES_EVAL = ['cartesian']#['cartesian', 'joint']
SUBJECT = 'sub_11'
INSTRUCTION = 3

# Get the current working directory
cwd = os.getcwd()

# Split the path to get the package directory
pkg_dir = cwd.split(SRC['FOLDER'])[0].split(SRC['SUBFOLDER'])[0]

# Define the data and plot directories
data_dir = os.path.join(pkg_dir, DATA['FOLDER'], DATA['SUBFOLDER_OUTPUT'])
plot_dir = os.path.join(pkg_dir, RESULTS['FOLDER'], RESULTS['SUBFOLDER_PLOTS'])
os.makedirs(plot_dir, exist_ok=True)

# Plot the time series with covariance cones for each combination of filter type, velocity, task, and space
for space_c in SPACES_COMPUTE:

    if space_c == 'cartesian':
        dim_x = N_VAR_PER_KPT * N_DIM_PER_KPT * N_KPTS
        param_idx = np.array([]) # no body params are states tracked by the filter
    elif space_c == 'joint':
        dim_x = N_VAR_PER_JOINT * N_JOINTS
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
                                            f'{TRAINORTEST}_measurements_{space_c}_.pkl')
                        
                        with open(file, 'rb') as f:
                            print(f"\nLoading file: {file}...")
                            measurements = pickle.load(f)      

                        # Load the filtering results
                        file = os.path.join(data_dir,
                                            f'{TRAINORTEST}_filtering_results_{prediction_steps}_steps_{space_c}_.pkl')
                        
                        with open(file, 'rb') as f:
                            print(f"\nLoading file: {file}...")
                            filtering_results = pickle.load(f)

                        # Load the prediction results
                        file = os.path.join(data_dir,
                                            f'{TRAINORTEST}_prediction_results_{prediction_steps}_steps_{space_c}_.pkl')
                        
                        with open(file, 'rb') as f:
                            print(f"Loading file: {file}...")
                            prediction_results = pickle.load(f)

                        # Define y_axes limits for the plots and the first and last timestamps of the measurements
                        if velocity == 'SLOW':
                            if task == 'PICK-&-PLACE':
                                y_axes_lim = [0.12, 0.85]
                                start_meas = 0.0
                                end_meas = 10.0
                                if INSTRUCTION == 1:
                                    kpt = 4 # right wrist
                                elif INSTRUCTION == 3:
                                    kpt = 7 # left wrist
                                else:
                                    kpt = 4 # by default choose right wrist
                                    raise ValueError('CHECK')
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
                                end_meas = 10.0
                                if INSTRUCTION == 1:
                                    kpt = 4 # right wrist
                                elif INSTRUCTION == 3:
                                    kpt = 7 # left wrist
                                else:
                                    kpt = 4 # by default choose right wrist
                                    raise ValueError('CHECK')
                                dim = 'x'
                            elif task == 'WALKING':
                                y_axes_lim = [-1, 1.25]
                                start_meas = 0.0
                                end_meas = 15.0
                                kpt = 0
                                dim = 'y'
                    elif space_e == 'JOINT':
                        pass #TODO: implement the evaluation in joint space
                    else:
                        raise ValueError('Invalid space for evaluation')
                    
                    # Clip start and end times to the range of the measurements
                    start_meas = max(start_meas, measurements[(SUBJECT, velocity, task, INSTRUCTION)].index[0].total_seconds())
                    end_meas = min(end_meas, measurements[(SUBJECT, velocity, task, INSTRUCTION)].index[-1].total_seconds())

                    assert start_meas < end_meas, 'The start time must be less than the end time.'

                    # Define a list of pd.Timedelta to uniformly sample the time range
                    selected_timestamps = [pd.Timedelta(seconds=round(s, 1))
                                           for s in np.arange(start_meas,
                                                              end_meas-PREDICTION_HORIZON,
                                                              CONE_STEP)]

                    # Define the time range to plot the time series
                    lower_bound = pd.Timedelta(seconds=start_meas)
                    upper_bound = pd.Timedelta(seconds=end_meas)
                    selected_range = [lower_bound, upper_bound]

                    # Find the positional index of kpt
                    kpt_idx = SELECTED_KEYPOINTS_FOR_KINEMATIC_MODEL.index(kpt)

                    plot_covariance_cone(measurements, filtering_results, prediction_results,
                                         SUBJECT, velocity, task, INSTRUCTION, kpt, kpt_idx, dim, DIM_TYPE,
                                         dim_x, N_VAR_PER_KPT, N_DIM_PER_KPT, SAMPLING_TIME, PREDICT_K_STEPS,
                                         prediction_steps, selected_timestamps, selected_range, filter_type, y_axes_lim,
                                         plot_dir, num_sigmas=NUM_SIGMAS)