import os, time, json, pickle, glob
import numpy as np
import pandas as pd
import nstep_ukf_imm_estimator as ukf_predictor
from utils import select_trajectory_dataset

# Fix warnings due to Pandas
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
pd.options.mode.chained_assignment = None  # default='warn'


# ====================================================================================================================================
## DEFINE PARAMETERS ##

start_time = time.time()

SRC = {'FOLDER': 'scripts', 'SUBFOLDER': 'data_analysis'}                               # source folder
MODELS = {'FOLDER': 'models'}                                                           # models folder
DATA = {'FOLDER': 'data',
        'SUBFOLDER_PREPROC': 'preprocessed',
        'SUBFOLDER_OUTPUT': 'output',
        'SUBFOLDER_TUNING': 'tuning',
        'CSV_FILES': {'PICK-&-PLACE': 'dataset_PICK-&-PLACE.csv',
                      'WALKING': 'dataset_WALKING.csv',
                      'PASSING-BY': 'dataset_PASSING-BY.csv'},
        'JSON_FILE': 'column_names.json'}                                               # data folder and files
RESULTS = {'FOLDER': 'results'}                                                         # results folder

COLUMN_NAMES_IDX = ['conf_names', 'conf_names_filt', 'conf_names_vel',
                    'conf_names_acc', 'conf_names_jerk', 'param_names',
                    'kpt_names', 'kpt_names_filt', 'kpt_names_vel', 'kpt_names_acc']    # data column names to load from the JSON file

# Select which instructions to consider
SELECTED_INSTRUCTIONS = {
    'PICK-&-PLACE': [1, 2, 3, 4, 5, 6, 7, 8], # discard instruction_id=0, as the subject stands still  (for quick test: [3])
    'WALKING':      [1, 2, 3],                # discard instruction_id=0, as the subject stands still  (for quick test: [1])
    'PASSING-BY':   [1, 2] # do not select instruction_id=0, as the subject is outside the camera view (for quick test: [1])
}                                                 
SELECTED_VELOCITIES = ['FAST'] #'SLOW', 'MEDIUM', 'FAST']                # select which velocities to consider
SELECTED_TASK_NAMES = ['PICK-&-PLACE'] #, 'WALKING', 'PASSING-BY']          # select which tasks to consider

# select which keypoints to consider for the kinematic model
SELECTED_KEYPOINTS_FOR_KINEMATIC_MODEL = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                          16, 17]   # added left and right ear keypoints (16, 17) -> used to compute the head center
SELECTED_KPT_NAMES = {
    0: "head",
    5: "left_shoulder",
    6: "left_elbow",
    7: "left_wrist",
    11: "left_hip",
    12: "left_knee",
    13: "left_ankle",
    2: "right_shoulder",
    3: "right_elbow",
    4: "right_wrist",
    8: "right_hip",
    9: "right_knee",
    10: "right_ankle",
}

TRAIN_SUBJECTS = ['sub_9', 'sub_4', 'sub_11', 'sub_7', 'sub_8', 'sub_10', 'sub_6']      # select which subjects to use for training
TEST_SUBJECTS = ['sub_13', 'sub_12', 'sub_3']                                           # select which subjects to use for testing

# Define which keypoints to consider for each task in the metrics evaluation
SELECTED_KEYPOINTS_FOR_EVALUATION = {
    'PICK-&-PLACE': [2, 3, 4, 5, 6, 7],
    'WALKING':      [0, 2, 5, 8, 11],
    'PASSING-BY':   [0, 2, 5, 8, 11]
} 

DIMENSIONS_PER_KEYPOINT = {0:  ['y'],
                           2:  ['y'],
                           3:  ['x', 'y', 'z'],
                           4:  ['x', 'y', 'z'],
                           5:  ['y'],
                           6:  ['x', 'y', 'z'],
                           7:  ['x', 'y', 'z'],
                           8:  ['y'],
                           11: ['y']}

# Sampling time
DT = 0.1

# Define possible prediction horizons (in seconds)
PRED_HORIZONS = [0.1, 0.3, 0.5]

# Define the number of steps based on PRED_HORIZONS and DT
num_pred_steps = [int(np.round(hor / DT)) for hor in PRED_HORIZONS]

# Define whether to load the results from the pickle files
LOADING_RESULTS = False

# Define whether to only tune the parameters or to also run the filtering loop
ONLY_TUNING = True

# Define whether to tune the variance parameters
TUNE_INIT_VARIANCE_JOINTS = True     # initial variance
NUM_MONTECARLO_SAMPLES = 10          # initial variance (Monte Carlo samples. With 10k similar results compared to 30k)
PLOT_DISTRIBUTIONS = False           # plot the distributions of the Monte Carlo samples
TUNE_Q_ACC = False                   # process variance on acceleration
if TUNE_Q_ACC:
    TOLERANCE_PERCENT_IN_BAND = 2   # tolerance for the percentage of points in the band
                                    # [68% - TOLERANCE_PERCENT_IN_BAND%, 68% + TOLERANCE_PERCENT_IN_BAND%]
    Q_INF = 0.005
    Q_SUP = 0.05
    NUM_Q_VALUES = 20
    max_ahead_steps = [np.max(num_pred_steps)]          # worst case scenario: max prediction horizon
    IMPOSED_VEL_TUNING = 'FAST'                         # worst case scenario: fastest velocity
    IMPOSED_TASKS_TUNING = ['PICK-&-PLACE']             # arbitrary choice

    assert IMPOSED_VEL_TUNING in SELECTED_VELOCITIES, \
        "IMPOSED_VEL_TUNING must be one of the SELECTED_VELOCITIES."
    assert all(task in SELECTED_TASK_NAMES for task in IMPOSED_TASKS_TUNING), \
        "IMPOSED_TASKS_TUNING must be a subset of SELECTED_TASK_NAMES."


# Define the space in which the filter operates
SPACE = 'joint'                             # 'cartesian' or 'joint'
# Define the space in which the error metrics are evaluated
SPACE_FOR_EVALUATION = 'joint'              # 'cartesian' or 'joint'

assert SPACE_FOR_EVALUATION == 'cartesian' if SPACE == 'cartesian' else True, \
    "SPACE_FOR_EVALUATION must be 'cartesian' if SPACE is 'cartesian'."

assert SPACE == 'joint' and not LOADING_RESULTS if TUNE_INIT_VARIANCE_JOINTS else True, \
    "SPACE must be 'joint' and LOADING_RESULTS must be False if TUNE_INIT_VARIANCE_JOINTS is True."

# Define the number of keypoints and joints
N_KPTS = 13
if SPACE == 'cartesian':
    N_JOINTS = 0
    N_PARAM = 0
elif SPACE == 'joint':
    N_JOINTS = 28
    N_PARAM = 8 # (shoulder_distance, chest_hip_distance, hip_distance,
                # upper_arm_length, lower_arm_length,
                # upper_leg_length, lower_leg_length,
                # head_distance)
else:
    ValueError("SPACE must be either 'cartesian' or 'joint'.")


# Filter parameters
PREDICT_K_STEPS = True
N_VAR_PER_JOINT = 3                             # position, velocity, acceleration                                 
N_VAR_PER_KPT = 3                               # position, velocity, acceleration
N_DIM_PER_KPT = 3                               # x, y, z
MAX_TIME_NO_MEAS = pd.Timedelta(seconds= 0.1)   # maximum time without measurements before resetting the filter

# Initial uncertainty parameters for the filters
VAR_MEAS_KPT = 0.05**2           # [paper: r_y] Hip: 3-sigma (99.5%) = 0.15 m ==> sigma 0.05 m ==> var = (0.05)^2 m^2

if SPACE == 'cartesian': # parameters are defined ISOTROPICALLY for all keypoints
    INIT_VAR_POS = VAR_MEAS_KPT     # [paper: p_y] Set equal to the measurement noise since the state is initialized with the measurement
    INIT_VAR_VEL = (1.6/3.0)**2     # [paper: p_v] Hip: no keypoint moves faster than 1.6 m/s ==> 3-sigma (99.5%) = 1.6 m/s ==> var = (1.6/3)^2 m^2/s^2
    INIT_VAR_ACC = (10.0/3.0)**2     # [paper: p_a] Hip: no keypoint accelerates faster than 10 m/s^2 ==> 3-sigma (99.5%) = 10 m/s^2 ==> var = (10/3)^2 m^2/s^4
    init_var_pos = INIT_VAR_POS * np.ones(N_DIM_PER_KPT * N_KPTS)
    init_var_vel = INIT_VAR_VEL * np.ones(N_DIM_PER_KPT * N_KPTS)
    init_var_acc = INIT_VAR_ACC * np.ones(N_DIM_PER_KPT * N_KPTS)

    VAR_Q_ACC = 0.04                 # [paper: q_a] [TUNABLE] a_dot = u (u = 0 is very uncertain ==> add variance here)
                                     # 0.04 -> tuned on cartesian, FAST, PICK-&-PLACE (id=3)
                                     # 0.06833333 -> tuned on cartesian, FAST, PICK-&-PLACE (id=3) + WALKING (id=1)
                                     # -> but % for PICK-&-PLACE is ~80%
                                     # 10.0 -> tuned on cartesian, FAST, PICK-&-PLACE (id=3) + WALKING (id=1) + PASSING-BY (id=1) 
                                     # -> but % for PICK-&-PLACE is ~85% and WALKING is ~90%
    var_q_acc = VAR_Q_ACC * np.ones(N_DIM_PER_KPT * N_KPTS)

elif SPACE == 'joint': # parameters are defined ANISOTROPICALLY for all joints after TUNING -> these values using 5000 samples
    INIT_VAR_POS = np.array([ # [TUNABLE] Related to VAR_MEAS_KPT through the NONLINEAR inverse kinematics (MonteCarlo)
        1.24878814e-03, 1.25025499e-03, 1.24938023e-03, 3.15882166e-03,
        3.19423892e-03, 1.11517969e-02, 1.40733355e-03, 5.92206854e-02,
        2.32764736e-01, 1.17150559e-01, 1.73681043e-01, 1.72518177e-01,
        1.02796480e+00, 1.28433843e-01, 1.13192918e-01, 1.37330454e-01,
        9.22987824e-01, 1.46677344e-01, 4.30215461e-02, 9.37398164e-02,
        1.40016761e+00, 4.57995704e-02, 1.36780340e-01, 2.08892053e-01,
        1.19624003e+00, 5.63690636e-02, 1.36172951e-01, 1.10386859e-01,
    ])
    INIT_VAR_PARAM = np.array([ # [TUNABLE] Related to VAR_MEAS_KPT through the NONLINEAR inverse kinematics (MonteCarlo)
        0.00476110, 0.00246833, 0.00418179, 0.00230106,
        0.00228468, 0.00242246, 0.00240519, 0.00231500,
    ])
    
    assert len(INIT_VAR_POS) == N_JOINTS, "INIT_VAR_POS must have the same length as the number of joints."
    assert len(INIT_VAR_PARAM) == N_PARAM, "INIT_VAR_PARAM must have the same length as the number of body parameters."
    
    UNCERTAINTY_FACTOR = 10.0                         #TODO: check, arbitrary choice
    INIT_VAR_VEL = UNCERTAINTY_FACTOR * INIT_VAR_POS  #TODO: check, arbitrary choice
    INIT_VAR_ACC = UNCERTAINTY_FACTOR * INIT_VAR_VEL  #TODO: check, arbitrary choice
    init_var_pos = INIT_VAR_POS
    init_var_vel = INIT_VAR_VEL
    init_var_acc = INIT_VAR_ACC
    init_var_param = INIT_VAR_PARAM

    VAR_Q_ACC = np.array([
        8.44170670e-06, 8.47534206e-06, 8.45810259e-06, 4.11114506e-05,
        3.69733710e-05, 2.68022674e-03, 6.16536135e-05, 4.69242265e-04,
        2.89087155e-03, 1.46875575e-03, 9.78714941e-04, 8.80769026e-04,
        5.26042456e-03, 3.31590275e-03, 8.65553052e-04, 8.69487377e-04,
        4.08258903e-03, 2.84582653e-03, 8.09989411e-04, 1.47666091e-03,
        5.56546895e-03, 1.22001770e-03, 7.15604987e-04, 1.45833908e-03,
        5.54223470e-03, 1.21912156e-03, 4.05655006e-03, 1.00089631e-03,
        3.21908819e-05, 1.65665736e-05, 2.66588312e-05, 1.56006772e-05,
        1.54874003e-05, 1.64894635e-05, 1.63873816e-05, 2.22579787e-05,
    ])                 # [paper: q_a] [TUNABLE] a_dot = u (u = 0 is very uncertain ==> add variance here)
    var_q_acc = VAR_Q_ACC
    
else:
    ValueError("SPACE must be either 'cartesian' or 'joint'.")

# Parameters for the IMM estimator
USE_SINDY_MODEL = False

assert SPACE == 'joint' if USE_SINDY_MODEL else True, \
    "SPACE must be 'joint' if USE_SINDY_MODEL is True."

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
print("\nDefine parameters and load column names...")

# Get the current working directory
cwd = os.getcwd()

# Split the path to get the package directory
pkg_dir = cwd.split(SRC['FOLDER'])[0].split(SRC['SUBFOLDER'])[0]

# Define the path to the preprocessed, filtered, predicted data + tuning and models directories
preprocessed_dir = os.path.join(pkg_dir, DATA['FOLDER'], DATA['SUBFOLDER_PREPROC'])
output_dir = os.path.join(pkg_dir, DATA['FOLDER'], DATA['SUBFOLDER_OUTPUT'])
tuning_dir = os.path.join(pkg_dir, DATA['FOLDER'], DATA['SUBFOLDER_TUNING'])
models_dir = os.path.join(pkg_dir, MODELS['FOLDER'])

# Define directory to store csv results
results_dir = os.path.join(pkg_dir, RESULTS['FOLDER'])

# Ensure the directories exist
os.makedirs(preprocessed_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(tuning_dir, exist_ok=True)

# Load the column names
with open(os.path.join(preprocessed_dir, DATA['JSON_FILE']), 'r') as f:
    data = json.load(f)

# Extract the column names
column_names = {}
for col in COLUMN_NAMES_IDX:
    column_names[col] = data[col]

# Define the column names for the filtered and predicted data
# CARTESIAN SPACE: always done
filtered_column_names = ['{}_kp{}_{}'.format(filt_type, i, suffix)
                         for filt_type in ['ca', 'cv', 'imm']
                         for i in SELECTED_KEYPOINTS_FOR_KINEMATIC_MODEL
                         for suffix in ['x', 'xd', 'xdd', 'y', 'yd', 'ydd', 'z', 'zd', 'zdd']]
filtered_pred_column_names = ['{}_kp{}_{}'.format(filt_type, i, suffix)
                              for filt_type in ['ca', 'imm']
                              for i in SELECTED_KEYPOINTS_FOR_KINEMATIC_MODEL
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

conf_names = output_column_names['filtered_joint_column_names'] \
    if SPACE == 'joint' else output_column_names['filtered_column_names']

# Define the dimensionality of the state space for the filter
if SPACE == 'cartesian':
    dim_x = N_VAR_PER_KPT * N_DIM_PER_KPT * N_KPTS
    p_idx = np.arange(0, dim_x, N_VAR_PER_KPT) # position indices
    param_idx = np.array([]) # no body params are states tracked by the filter
elif SPACE == 'joint':
    dim_x = N_VAR_PER_JOINT * N_JOINTS
    p_idx = np.arange(0, dim_x, N_VAR_PER_JOINT) # position indices
    param_idx = np.array(range(dim_x, dim_x + N_PARAM))
    dim_x += N_PARAM # the body params are states tracked by the filter
else:
    ValueError("SPACE must be either 'cartesian' or 'joint'.")

# Define the dimensionality of the measurement space for the filter
dim_z = N_DIM_PER_KPT * N_KPTS


# Load the dataset
if not LOADING_RESULTS:
    # Filter the dataset to keep only the selected tasks, velocities, and instructions
    global_df = pd.DataFrame()
    for task in SELECTED_TASK_NAMES:
        print(f"\nTask: {task}")
        print(f"Selected keypoints: {SELECTED_KEYPOINTS_FOR_EVALUATION[task]}")
        print(f"Loading dataset: {DATA['CSV_FILES'][task]}")

        # Load the datasets
        df = pd.read_csv(os.path.join(preprocessed_dir, DATA['CSV_FILES'][task]))

        # Filter the dataset to keep only the selected instructions and velocities
        df_subset = df[(df['Instruction_id'].isin(SELECTED_INSTRUCTIONS[task])) & (df['Velocity'].isin(SELECTED_VELOCITIES))]

        # Append the filtered dataset to the global dataframe
        global_df = pd.concat([global_df, df_subset])


    # Split the dataset into training and testing
    train_df = global_df[global_df['Subject'].isin(TRAIN_SUBJECTS)]
    test_df = global_df[global_df['Subject'].isin(TEST_SUBJECTS)]

    print('\nTraining dataset shape:', train_df.shape)
    print('Testing dataset shape:', test_df.shape)


    print("\nBuild identification and validation datasets...")

    selected_columns = [column for column in column_names['kpt_names_filt'] \
                        if any(str(kpt) in column.split('kp')[1].split('_') \
                            for kpt in SELECTED_KEYPOINTS_FOR_KINEMATIC_MODEL)]

    # Define the training data
    print("\nTraining dataset:")
    X_train_list, time_train_list, train_traj_idx = \
        select_trajectory_dataset(
            train_df,
            TRAIN_SUBJECTS,
            SELECTED_INSTRUCTIONS,
            SELECTED_VELOCITIES,
            SELECTED_TASK_NAMES,
            selected_columns # measurement data
        )

    print("\n==========================================\n")

    # Define the testing data
    print("\nTesting dataset:")
    X_test_list, time_test_list, test_traj_idx = \
        select_trajectory_dataset(
            test_df,
            TEST_SUBJECTS,
            SELECTED_INSTRUCTIONS,
            SELECTED_VELOCITIES,
            SELECTED_TASK_NAMES,
            selected_columns # measurement data
        )
    

    # Tune the initial variance parameters in the joint space
    tic = time.time()
    if SPACE == 'joint' and TUNE_INIT_VARIANCE_JOINTS:
        print("\nTuning the initial variance using the IDENTIFICATION subjects...")
        ukf_predictor.tune_init_variance_joints(
            X_train_list, time_train_list, train_traj_idx,
            DT, SELECTED_KPT_NAMES, SELECTED_KEYPOINTS_FOR_KINEMATIC_MODEL,
            N_JOINTS, N_PARAM, NUM_MONTECARLO_SAMPLES, VAR_MEAS_KPT,
            tuning_dir, plot_distributions=PLOT_DISTRIBUTIONS
        )

        # Load the results from the tuning directory
        with open(os.path.join(tuning_dir, 'init_cov_q.pkl'), 'rb') as f:
            init_var_pos = pickle.load(f)
        with open(os.path.join(tuning_dir, 'init_cov_param.pkl'), 'rb') as f:
            init_var_param = pickle.load(f)

        print(f"\ninit_var_pos:\n{init_var_pos}")
        print(f"\ninit_var_param:\n{init_var_param}")

        init_var_vel = UNCERTAINTY_FACTOR * init_var_pos #TODO: check, arbitrary choice
        init_var_acc = UNCERTAINTY_FACTOR * init_var_vel #TODO: check, arbitrary choice

        toc = time.time() - tic
        minutes, seconds = divmod(toc, 60)
        print(f"\nVariance parameters tuning took {minutes:.0f} minutes and {seconds:.2f} seconds.")


    # Initialize the state covariance matrix
    print("\nInitialize the filter...")
    if SPACE == 'cartesian':
        init_P = ukf_predictor.initialize_P(N_VAR_PER_KPT, N_DIM_PER_KPT, N_KPTS, init_var_pos, init_var_vel, init_var_acc)
    elif SPACE == 'joint':
        init_P = ukf_predictor.initialize_P(N_VAR_PER_KPT, N_DIM_PER_KPT, N_KPTS, init_var_pos, init_var_vel, init_var_acc,
                                            space=SPACE, var_P_param=init_var_param, n_param=N_PARAM, n_joints=N_JOINTS)
    else:
        ValueError("SPACE must be either 'cartesian' or 'joint'.")


    # Possibly initialize the SINDy model
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

        toc = time.time() - tic
        minutes, seconds = divmod(toc, 60)
        print(f"SINDy initialization took {minutes:.0f} minutes and {seconds:.2f} seconds.")


    # Tune the process variance on acceleration
    tic = time.time()
    if TUNE_Q_ACC:
        q_sweep = np.linspace(Q_INF, Q_SUP, NUM_Q_VALUES)

        q_sweep_values = []
        for i in range(NUM_Q_VALUES):
            if SPACE == 'cartesian':
                q_sweep_values.append(np.ones(N_KPTS*N_DIM_PER_KPT) * q_sweep[i])

            elif SPACE == 'joint':
                max_init_var = np.max(np.concatenate((init_var_pos, init_var_param)))
                init_var_pos_normalized = init_var_pos / max_init_var
                init_var_param_normalized = init_var_param / max_init_var

                init_var_pos_normalized_swept = init_var_pos_normalized * q_sweep[i]
                init_var_param_normalized_swept = init_var_param_normalized * q_sweep[i]

                init_var = np.concatenate((init_var_pos_normalized_swept, init_var_param_normalized_swept))
            
                q_sweep_values.append(init_var)

        print("\nTuning the process variance on acceleration using the IDENTIFICATION subjects...")
        var_q_acc = ukf_predictor.tune_q_acc(
            X_train_list, time_train_list, train_traj_idx,
            max_ahead_steps, IMPOSED_VEL_TUNING, IMPOSED_TASKS_TUNING, PREDICT_K_STEPS,
            dim_x, dim_z, p_idx, DT,
            N_VAR_PER_KPT, N_DIM_PER_KPT, N_KPTS,
            init_P, VAR_MEAS_KPT, q_sweep_values,
            INIT_MU, M, NUM_FILTERS_IN_BANK,
            ukf_predictor.custom_inv, MAX_TIME_NO_MEAS,
            PROB_IMM_COLUMN_NAMES, output_column_names,
            SELECTED_KPT_NAMES, SELECTED_KEYPOINTS_FOR_KINEMATIC_MODEL,
            tuning_dir,
            'train',
            SELECTED_KEYPOINTS_FOR_EVALUATION, DIMENSIONS_PER_KEYPOINT,
            space_compute=SPACE,
            space_eval=SPACE_FOR_EVALUATION,
            conf_names=conf_names,
            n_joints=N_JOINTS,
            n_params=N_PARAM,
            param_idx=param_idx,
            sindy_model=sindy_model,
            tolerance_percent_in_band=TOLERANCE_PERCENT_IN_BAND
        )
        print(f"Tuned var_q_acc: {var_q_acc}")

        toc = time.time() - tic
        minutes, seconds = divmod(toc, 60)
        print(f"\nProcess variance tuning took {minutes:.0f} minutes and {seconds:.2f} seconds.")

    if ONLY_TUNING:
        print("\nOnly tuning the parameters. Exiting...")
        raise SystemExit(0)

    print("\nRun the filtering loop for all subjects, identification and validation...")
    tic = time.time()

    # Run the loop for the IDENTIFICATION subjects
    print("\nTraining dataset:")
    ukf_predictor.run_filtering_loop(
        X_train_list, time_train_list, train_traj_idx, num_pred_steps, PREDICT_K_STEPS,
        dim_x, dim_z, p_idx, DT,
        N_VAR_PER_KPT, N_DIM_PER_KPT, N_KPTS,
        init_P, VAR_MEAS_KPT, var_q_acc,
        INIT_MU, M, NUM_FILTERS_IN_BANK,
        ukf_predictor.custom_inv, MAX_TIME_NO_MEAS,
        PROB_IMM_COLUMN_NAMES, output_column_names,
        SELECTED_KPT_NAMES, SELECTED_KEYPOINTS_FOR_KINEMATIC_MODEL,
        output_dir,
        'train',
        space=SPACE,
        n_joints=N_JOINTS,
        n_params=N_PARAM,
        param_idx=param_idx,
        sindy_model=sindy_model
    )

    # Run the loop for the VALIDATION subjects
    print("\n\nTesting dataset:")
    ukf_predictor.run_filtering_loop(
        X_test_list, time_test_list, test_traj_idx, num_pred_steps, PREDICT_K_STEPS,
        dim_x, dim_z, p_idx, DT,
        N_VAR_PER_KPT, N_DIM_PER_KPT, N_KPTS,
        init_P, VAR_MEAS_KPT, var_q_acc,
        INIT_MU, M, NUM_FILTERS_IN_BANK,
        ukf_predictor.custom_inv, MAX_TIME_NO_MEAS,
        PROB_IMM_COLUMN_NAMES, output_column_names,
        SELECTED_KPT_NAMES, SELECTED_KEYPOINTS_FOR_KINEMATIC_MODEL,
        output_dir,
        'test',
        space=SPACE,
        n_joints=N_JOINTS,
        n_params=N_PARAM,
        param_idx=param_idx,
        sindy_model=sindy_model
    )

    toc = time.time()
    minutes, seconds = divmod(toc - tic, 60)
    print(f"Running filtering loops took {minutes:.0f} minutes and {seconds:.2f} seconds.")


# ====================================================================================================================================
print("\nEvaluate error metrics for all subjects, identification and validation...")

tic = time.time()
 
# Load the IDENTIFICATION and VALIDATION results (all pkl files in the output directory)
pickle_files = glob.glob(os.path.join(output_dir, '*.pkl'))

# Olny select files that contain the string SPACE
pickle_files = [file for file in pickle_files if SPACE in file]

# Print the files that will be loaded line by line
print("\nPickle files available:")
for file in pickle_files:
    print(file)

for horizon in num_pred_steps:
    # Load the results for the IDENTIFICATION subjects
    for file in pickle_files:
        if f'train_filtering_results_{horizon}_steps' in file:
            with open(file, 'rb') as f:
                print(f"\nLoading file: {file}...")
                train_filtering_results = pickle.load(f)
        elif f'train_prediction_results_{horizon}_steps' in file:
            with open(file, 'rb') as f:
                print(f"Loading file: {file}...")
                train_prediction_results = pickle.load(f)

    # Evaluate the metrics for the IDENTIFICATION subjects
    print('\nIDENTIFICATION')
    ukf_predictor.evaluate_metrics(
        TRAIN_SUBJECTS, SELECTED_VELOCITIES, SELECTED_TASK_NAMES, SELECTED_INSTRUCTIONS,
        SELECTED_KEYPOINTS_FOR_EVALUATION, DIMENSIONS_PER_KEYPOINT,
        N_VAR_PER_KPT, N_DIM_PER_KPT, dim_x, horizon,
        train_filtering_results, train_prediction_results, results_dir,
        space_compute=SPACE, space_eval=SPACE_FOR_EVALUATION,
        conf_names=conf_names
    )

    # Free up memory
    del train_filtering_results, train_prediction_results

    # Load the results for the VALIDATION subjects
    for file in pickle_files:
        if f'test_filtering_results_{horizon}_steps' in file:
            with open(file, 'rb') as f:
                print(f"\nLoading file: {file}...")
                test_filtering_results = pickle.load(f)
        elif f'test_prediction_results_{horizon}_steps' in file:
            with open(file, 'rb') as f:
                print(f"Loading file: {file}...")
                test_prediction_results = pickle.load(f)

    print('\nVALIDATION')
    # Evaluate the metrics for the VALIDATION subjects
    ukf_predictor.evaluate_metrics(
        TEST_SUBJECTS, SELECTED_VELOCITIES, SELECTED_TASK_NAMES, SELECTED_INSTRUCTIONS,
        SELECTED_KEYPOINTS_FOR_EVALUATION, DIMENSIONS_PER_KEYPOINT,
        N_VAR_PER_KPT, N_DIM_PER_KPT, dim_x, horizon,
        test_filtering_results, test_prediction_results, results_dir,
        space_compute=SPACE, space_eval=SPACE_FOR_EVALUATION,
        conf_names=conf_names
    )

    # Free up memory
    del test_filtering_results, test_prediction_results

    print('\n')
    
toc = time.time()
minutes, seconds = divmod(toc - tic, 60)
print(f"Metrics evaluation took {minutes:.0f} minutes and {seconds:.2f} seconds.")

total_time = time.time() - start_time
minutes, seconds = divmod(total_time, 60)
print(f"\nTOTAL TIME: {minutes:.0f} minutes and {seconds:.2f} seconds.")