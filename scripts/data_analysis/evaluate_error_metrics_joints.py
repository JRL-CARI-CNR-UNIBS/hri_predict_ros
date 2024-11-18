import os, time, json, pickle
import numpy as np
import pandas as pd
import nstep_ukf_imm_estimator as ukf_predictor
from data_loading_utils import filter_upper_body_joints, select_trajectory_dataset

pd.options.mode.chained_assignment = None  # default='warn'

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")


# ====================================================================================================================================
## DEFINE PARAMETERS ##

SRC = {'FOLDER': 'scripts', 'SUBFOLDER': 'data_analysis'}                               # source folder
MODELS = {'FOLDER': 'models'}                                                           # models folder
DATA = {'FOLDER': 'data',
        'SUBFOLDER_PREPROC': 'preprocessed',
        'SUBFOLDER_FILT': 'filtered',
        'SUBFOLDER_PRED': 'predicted',
        'CSV_FILE': 'dataset_pickplace.csv', 'JSON_FILE': 'column_names.json'}          # data folder and files
RESULTS = {'FOLDER': 'results'}                                                         # results folder

COLUMN_NAMES_IDX = ['conf_names', 'conf_names_filt', 'conf_names_vel',
                    'conf_names_acc', 'conf_names_jerk', 'param_names',
                    'kpt_names', 'kpt_names_filt', 'kpt_names_vel', 'kpt_names_acc']    # data column names to load from the JSON file

SELECTED_INSTRUCTIONS = [1, 3, 7]                                                       # select which instructions to consider
SELECTED_VELOCITIES = ['SLOW', 'FAST', 'MEDIUM']                                                # select which velocities to consider
SELECTED_TASK_NAMES = ['PICK-&-PLACE'] #, 'WALKING', 'PASSING-BY']                     # select which tasks to consider

TRAIN_SUBJECTS = ['sub_9', 'sub_4', 'sub_11', 'sub_7', 'sub_8', 'sub_10', 'sub_6']      # select which subjects to use for training
TEST_SUBJECTS = ['sub_13', 'sub_12', 'sub_3']                                           # select which subjects to use for testing

# Define which keypoints to consider for each task in the metrics evaluation
SELECTED_KEYPOINTS = {'PICK-&-PLACE': [4, 7], # [2, 3, 4, 5, 6, 7],
                      'WALKING': [0, 1], # [0, 1, 2, 5, 8, 11],
                      'PASSING-BY': [0, 1]} # [0, 1, 2, 5, 8, 11]}

DIMENSIONS_PER_KEYPOINT = {0:  ['y'],
                           1:  ['y'],
                           2:  ['y'],
                           3:  ['x', 'y', 'z'],
                           4:  ['x', 'y', 'z'],
                           5:  ['y'],
                           6:  ['x', 'y', 'z'],
                           7:  ['x', 'y', 'z'],
                           8:  ['y'],
                           11: ['y']}

# Define possible prediction horizons (in steps)
PRED_HORIZONS = [1, 3, 5]

# Define the space in which the filter operates
SPACE = 'joint'                             # 'cartesian' or 'joint'

if SPACE == 'cartesian':
    N_KPTS = 18
elif SPACE == 'joint':
    N_KPTS = 13
    N_JOINTS = 28
else:
    ValueError("SPACE must be either 'cartesian' or 'joint'.")


# ONLY_USE_UPPER_BODY = False                                                              # select whether to use only upper body joints
# if ONLY_USE_UPPER_BODY:
#     UPPER_BODY_FRAMES = ['arm', 'elbow', 'head']
#     UPPER_BODY_KPTS = ['0','1','2','3','4','5','6','7','14','15','16','17']
#     N_KPTS = len(UPPER_BODY_KPTS)
#     N_JOINTS = 10
# else:
#     N_KPTS = 18
#     N_JOINTS = 28


# Filter parameters
DT = 0.1
PREDICT_K_STEPS = True
N_VAR_PER_JOINT = 3                             # position, velocity, acceleration
N_PARAM = 8                                     # (shoulder_distance, chest_hip_distance, hip_distance,
                                                # upper_arm_length, lower_arm_length,
                                                # upper_leg_length, lower_leg_length,
                                                # head_distance)
N_VAR_PER_KPT = 3                               # position, velocity, acceleration
N_DIM_PER_KPT = 3                               # x, y, z
MAX_TIME_NO_MEAS = pd.Timedelta(seconds= 0.1)   # maximum time without measurements before resetting the filter

# Initial uncertainty parameters for the filters
VAR_MEAS_KPT = 0.0025           # [paper: r_y] Hip: 3-sigma (99.5%) = 0.15 m ==> sigma 0.05 m ==> var = (0.05)^2 m^2
VAR_Q_ACC = 0.01                # [paper: q_a] a_dot = u (u = 0 is very uncertain ==> add variance here)

if SPACE == 'cartesian':
    INIT_VAR_POS = VAR_MEAS_KPT     # [paper: p_y] Set equal to the measurement noise since the state is initialized with the measurement
    INIT_VAR_VEL = 0.02844          # [paper: p_v] Hip: no keypoint moves faster than 1.6 m/s ==> 3-sigma (99.5%) = 1.6 m/s ==> var = (1.6/3)^2 m^2/s^2
    INIT_VAR_ACC = 1.1111           # [paper: p_a] Hip: no keypoint accelerates faster than 10 m/s^2 ==> 3-sigma (99.5%) = 10 m/s^2 ==> var = (10/3)^2 m^2/s^4
elif SPACE == 'joint':
    INIT_VAR_POS = 0.01 #TODO: to tune -> should be related to VAR_MEAS_KPT, but the link is the NONLINEAR inverse kinematics => not trivial, requires MonteCarlo?
                        # Moreover: which subject to use for the MonteCarlo? All subjects? Only the training subjects? The body parameters are different for each subject.
    INIT_VAR_VEL = 0.01 #TODO: to tune
    INIT_VAR_ACC = 0.01 #TODO: to tune
    INIT_VAR_PARAM = 0.1 #TODO: to tune
else:
    ValueError("SPACE must be either 'cartesian' or 'joint'.")

# Parameters for the IMM estimator
USE_SINDY_MODEL = False
NUM_FILTERS_IN_BANK = 3
M = np.array([[0.55, 0.15, 0.30], # transition matrix for the IMM estimator
              [0.15, 0.75, 0.10],
              [0.60, 0.30, 0.10]])

INIT_MU = np.array([0.55, 0.40, 0.05]) # initial mode probabilities for the IMM estimator

# Define the column names for the IMM probabilities
if USE_SINDY_MODEL:
    PROB_IMM_COLUMN_NAMES = ['prob_ca', 'prob_sindy', 'prob_cv']
else:
    PROB_IMM_COLUMN_NAMES = ['prob_ca', 'prob_ca_no', 'prob_cv']


# ====================================================================================================================================
print("\n1 / 5. Load preprocessed data...")

# Get the current working directory
cwd = os.getcwd()

# Split the path to get the package directory
pkg_dir = cwd.split(SRC['FOLDER'])[0].split(SRC['SUBFOLDER'])[0]

# Define the path to the preprocessed, filtered, and predicted data
preprocessed_dir = os.path.join(pkg_dir, DATA['FOLDER'], DATA['SUBFOLDER_PREPROC'])
filtered_dir = os.path.join(pkg_dir, DATA['FOLDER'], DATA['SUBFOLDER_FILT'])
predicted_dir = os.path.join(pkg_dir, DATA['FOLDER'], DATA['SUBFOLDER_PRED'])
models_dir = os.path.join(pkg_dir, MODELS['FOLDER'])

# Define directory to store csv results
results_dir = os.path.join(pkg_dir, RESULTS['FOLDER'])

# Ensure the directories exist
os.makedirs(preprocessed_dir, exist_ok=True)
os.makedirs(filtered_dir, exist_ok=True)
os.makedirs(predicted_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Load the dataset
df = pd.read_csv(os.path.join(preprocessed_dir, DATA['CSV_FILE']))

# Load the column names
with open(os.path.join(preprocessed_dir, DATA['JSON_FILE']), 'r') as f:
    data = json.load(f)

# Extract the column names
column_names = {}
for col in COLUMN_NAMES_IDX:
    column_names[col] = data[col]

# Filter the dataset to keep only the selected instructions and velocities
df_subset = df[(df['Instruction_id'].isin(SELECTED_INSTRUCTIONS)) & (df['Velocity'].isin(SELECTED_VELOCITIES))]

# Split the dataset into training and testing
train_df = df_subset[df_subset['Subject'].isin(TRAIN_SUBJECTS)]
test_df = df_subset[df_subset['Subject'].isin(TEST_SUBJECTS)]

print('Training dataset shape:', train_df.shape)
print('Testing dataset shape:', test_df.shape)


# ====================================================================================================================================
# if ONLY_USE_UPPER_BODY:
#     print("\n1[b] / 5. [Optional] Select only the upper body...")

#     for col in COLUMN_NAMES_IDX:
#         print(f"Column names: {col}")
#         if 'conf' in col:
#             column_names[col] = filter_upper_body_joints(column_names[col], UPPER_BODY_FRAMES)
#             assert len(column_names[col]) == N_JOINTS, f"Number of joints must be equal to {N_JOINTS}."


# ====================================================================================================================================
print("\n2 / 5. Build identification and validation datasets...")

# Define the training data
X_train_list, time_train_list, train_traj_idx = \
    select_trajectory_dataset(
        train_df,
        TRAIN_SUBJECTS,
        SELECTED_INSTRUCTIONS,
        SELECTED_VELOCITIES,
        SELECTED_TASK_NAMES,
        column_names['kpt_names_filt'] # measurement data
    )

print("\n==========================================\n")

# Define the testing data
X_test_list, time_test_list, test_traj_idx = \
    select_trajectory_dataset(
        test_df,
        TEST_SUBJECTS,
        SELECTED_INSTRUCTIONS,
        SELECTED_VELOCITIES,
        SELECTED_TASK_NAMES,
        column_names['kpt_names_filt'] # measurement data
    )


# ====================================================================================================================================
print("\n3 / 5. Initialize the filter...")

# Define the dimensionality of the state space for the filter
if SPACE == 'cartesian':
    dim_x = N_VAR_PER_KPT * N_DIM_PER_KPT * N_KPTS
    p_idx = np.arange(0, dim_x, N_VAR_PER_KPT) # position indices
elif SPACE == 'joint':
    dim_x = N_VAR_PER_JOINT * N_JOINTS
    p_idx = np.arange(0, dim_x, N_VAR_PER_JOINT) # position indices
    param_idx = np.array(range(dim_x, dim_x + N_PARAM))
    dim_x += N_PARAM # the body params are states tracked by the filter
else:
    ValueError("SPACE must be either 'cartesian' or 'joint'.")

# Define the dimensionality of the measurement space for the filter
dim_z = N_DIM_PER_KPT * N_KPTS

# Initialize the state covariance matrix
if SPACE == 'cartesian':
    init_P = ukf_predictor.initialize_P(N_VAR_PER_KPT, N_DIM_PER_KPT, N_KPTS, INIT_VAR_POS, INIT_VAR_VEL, INIT_VAR_ACC)
elif SPACE == 'joint':
    init_P = ukf_predictor.initialize_P(N_VAR_PER_KPT, N_DIM_PER_KPT, N_KPTS, INIT_VAR_POS, INIT_VAR_VEL, INIT_VAR_ACC,
                                        space=SPACE, var_P_param=INIT_VAR_PARAM, n_param=N_PARAM, n_joints=N_JOINTS)
else:
    ValueError("SPACE must be either 'cartesian' or 'joint'.")

# Define the column names for the filtered and predicted data
# CARTESIAN SPACE: always done
filtered_column_names = ['{}_kp{}_{}'.format(filt_type, i, suffix)
                         for filt_type in ['ca', 'cv', 'imm']
                         for i in range(N_KPTS)
                         for suffix in ['x', 'xd', 'xdd', 'y', 'yd', 'ydd', 'z', 'zd', 'zdd']]
filtered_pred_column_names = ['{}_kp{}_{}'.format(filt_type, i, suffix)
                              for filt_type in ['ca', 'imm']
                              for i in range(N_KPTS)
                              for suffix in ['x', 'xd', 'xdd', 'y', 'yd', 'ydd', 'z', 'zd', 'zdd']]

# JOINT SPACE: only done if the flag is set
if SPACE == 'joint':
    filtered_joint_column_names = ['{}_{}_{}'.format(filt_type, joint, suffix)
                                   for filt_type in ['ca', 'cv', 'imm']
                                   for joint in column_names['conf_names_filt']
                                   for suffix in ['pos', 'vel', 'acc']]
    filtered_pred_joint_column_names = ['{}_{}_{}'.format(filt_type, joint, suffix)
                                        for filt_type in ['ca', 'imm']
                                        for joint in column_names['conf_names_filt']
                                        for suffix in ['pos', 'vel', 'acc']]
    filtered_param_names = column_names['param_names']
    filtered_pred_param_names = column_names['param_names']
    
# Define column names for the filtered and predicted data
if SPACE == 'cartesian':
    output_column_names = {
        'filtered_column_names': filtered_column_names,
        'filtered_pred_column_names': filtered_pred_column_names
    }
elif SPACE == 'joint':
    output_column_names = {
        'filtered_column_names': filtered_column_names,
        'filtered_pred_column_names': filtered_pred_column_names,
        'filtered_joint_column_names': filtered_joint_column_names,
        'filtered_pred_joint_column_names': filtered_pred_joint_column_names,
        'filtered_param_names': filtered_param_names,
        'filtered_pred_param_names': filtered_pred_param_names

    }
else:
    raise ValueError("SPACE must be either 'cartesian' or 'joint'.")


# ====================================================================================================================================
print("\n4 / 5. Evaluate error metrics for all filters for the IDENTIFICATION subjects...")

loading_results = False

tic = time.time()

if not loading_results:

    sindy_model = None
    if USE_SINDY_MODEL:
        # Load SINDy model
        sindy_model = {}
        import dill # requires dill to load LAMBDA functions

        with open(os.path.join(models_dir, 'sindy_model_chest_pos_legs.pkl'), 'rb') as f:
            sindy_model['chest_pos_legs'] = dill.load(f)
        with open(os.path.join(models_dir, 'sindy_model_chest_rot_left_arm.pkl'), 'rb') as f:
            sindy_model['chest_rot_left_arm'] = dill.load(f)
        with open(os.path.join(models_dir, 'sindy_model_chest_rot_right_arm.pkl'), 'rb') as f:
            sindy_model['chest_rot_right_arm'] = dill.load(f)
        with open(os.path.join(models_dir, 'sindy_model_upper_body.pkl'), 'rb') as f:
            sindy_model['upper_body'] = dill.load(f)

    _, train_filtering_results, train_prediction_results = ukf_predictor.run_filtering_loop_joints(
        X_train_list, time_train_list, train_traj_idx, PRED_HORIZONS, PREDICT_K_STEPS,
        dim_x, dim_z, p_idx, DT,
        N_VAR_PER_KPT, N_DIM_PER_KPT, N_KPTS,
        init_P, VAR_MEAS_KPT, VAR_Q_ACC,
        INIT_MU, M, NUM_FILTERS_IN_BANK,
        ukf_predictor.custom_inv, MAX_TIME_NO_MEAS, PROB_IMM_COLUMN_NAMES, output_column_names,
        space=SPACE,
        param_idx=param_idx,
        sindy_model=sindy_model
    )

    # Save the results
    with open(os.path.join(filtered_dir, 'train_filtering_results.pkl'), 'wb') as f:
        pickle.dump(train_filtering_results, f)
    with open(os.path.join(predicted_dir, 'train_prediction_results.pkl'), 'wb') as f:
        pickle.dump(train_prediction_results, f)

else:
    with open(os.path.join(filtered_dir, 'train_filtering_results.pkl'), 'rb') as f:
        train_filtering_results = pickle.load(f)
    with open(os.path.join(predicted_dir, 'train_prediction_results.pkl'), 'rb') as f:
        train_prediction_results = pickle.load(f)


# Evaluate the metrics for the identification subjects
for horizon in PRED_HORIZONS:
    ukf_predictor.evaluate_metrics(
        TRAIN_SUBJECTS, SELECTED_VELOCITIES, SELECTED_TASK_NAMES,
        SELECTED_KEYPOINTS, DIMENSIONS_PER_KEYPOINT,
        N_VAR_PER_KPT, N_DIM_PER_KPT, dim_x, horizon,
        train_filtering_results, train_prediction_results, results_dir
    )

toc = time.time()
minutes, seconds = divmod(toc - tic, 60)
print(f"[IDENTIFICATION] Metrics evaluation took {minutes:.0f} minutes and {seconds:.2f} seconds.")


# ====================================================================================================================================
print("\n5 / 5. Evaluate error metrics for all filters for the VALIDATION subjects...")

tic = time.time()

if not loading_results:
    _, test_filtering_results, test_prediction_results = ukf_predictor.run_filtering_loop_joints(
        X_test_list, time_test_list, test_traj_idx, PRED_HORIZONS, PREDICT_K_STEPS,
        dim_x, dim_z, p_idx, DT,
        N_VAR_PER_KPT, N_DIM_PER_KPT, N_KPTS,
        init_P, VAR_MEAS_KPT, VAR_Q_ACC,
        INIT_MU, M, NUM_FILTERS_IN_BANK,
        ukf_predictor.custom_inv, MAX_TIME_NO_MEAS, PROB_IMM_COLUMN_NAMES, output_column_names,
        space=SPACE,
        param_idx=param_idx
    )

    # Save the results
    with open(os.path.join(filtered_dir, 'test_filtering_results.pkl'), 'wb') as f:
        pickle.dump(test_filtering_results, f)
    with open(os.path.join(predicted_dir, 'test_prediction_results.pkl'), 'wb') as f:
        pickle.dump(test_prediction_results, f)

else:
    with open(os.path.join(filtered_dir, 'test_filtering_results.pkl'), 'rb') as f:
        test_filtering_results = pickle.load(f)
    with open(os.path.join(predicted_dir, 'test_prediction_results.pkl'), 'rb') as f:
        test_prediction_results = pickle.load(f)


# Evaluate the metrics for the validation subjects
for horizon in PRED_HORIZONS:
    ukf_predictor.evaluate_metrics(
        TEST_SUBJECTS, SELECTED_VELOCITIES, SELECTED_TASK_NAMES,
        SELECTED_KEYPOINTS, DIMENSIONS_PER_KEYPOINT,
        N_VAR_PER_KPT, N_DIM_PER_KPT, dim_x, horizon,
        test_filtering_results, test_prediction_results, results_dir
    )
    
toc = time.time()
minutes, seconds = divmod(toc - tic, 60)
print(f"[VALIDATION] Metrics evaluation took {minutes:.0f} minutes and {seconds:.2f} seconds.")