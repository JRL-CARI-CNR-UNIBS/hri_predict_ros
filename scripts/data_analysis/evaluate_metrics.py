import os
import rospkg
import numpy as np
import tf

print("Define parameters and load data...")

# Create a RosPack object
rospack = rospkg.RosPack()

# Get the path to the package this script is in
package_path = rospack.get_path('hri_predict_ros')

# Define the path to the plots directory
plot_dir = os.path.join(package_path, 'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Specify which topics to read from the rosbag file
topic_names = ['/offline/zed/zed_node/body_trk/skeletons']

n_kpts = 18
TF_world_camera = [0.100575, -0.9304, 2.31042, 0.180663, 0.516604, 0.119341, 0.828395]

translation_world_camera = np.array(TF_world_camera[0:3])
quaternion_world_camera = np.array(TF_world_camera[3:7])

# Convert the quaternion to a rotation matrix
rotation_matrix_world_camera = tf.transformations.quaternion_matrix(quaternion_world_camera)

# Create a translation matrix
translation_matrix_world_camera = tf.transformations.translation_matrix(translation_world_camera)

# Combine the rotation and translation to get the transformation matrix from the world frame to the camera frame
cam_to_world_matrix = tf.transformations.concatenate_matrices(
    translation_matrix_world_camera,
    rotation_matrix_world_camera
)

human_meas_names = ['human_kp{}_{}'.format(i, suffix)
                    for i in range(n_kpts)
                    for suffix in ['x', 'y', 'z']]

# Define the frequency of the measurements for resampling
f = 20 # Hz
meas_dt = 1/f
freq_str = f'{meas_dt}S' # seconds



print("Load rosbag data...")

import rosbag

# Define the path to the bag directory
bag_dir = os.path.join(package_path, 'logs', 'bag')

bag_files = os.listdir(bag_dir)
bag_files = [os.path.join(bag_dir, bag_file) for bag_file in bag_files]

bag_data = {}
for bag_file in bag_files:
    with rosbag.Bag(bag_file, 'r') as bag:
        rows_list = []
        for topic, msg, t in bag.read_messages(topics=topic_names):
            row_dict = {}

            timestamp = t.to_sec()

            human_meas = np.full((1, n_kpts*3), np.nan)
            if topic == '/offline/zed/zed_node/body_trk/skeletons':
                skeleton_kpts = np.full((n_kpts, 3), np.nan)
                if msg.objects:                
                    for obj in msg.objects:
                        # Extract skeleton keypoints from message ([x, y, z] for each kpt)
                        kpts = np.array([[kp.kp] for kp in obj.skeleton_3d.keypoints])
                        kpts = kpts[:n_kpts] # select only the first n_kpts

                        skeleton_kpts = np.reshape(kpts, (n_kpts, 3)) # reshape to (n_kpts, 3)

                        # Convert keypoints to world frame
                        for i in range(n_kpts):
                            # Create a homogeneous coordinate for the keypoint position
                            kpt = np.array([skeleton_kpts[i][0],
                                            skeleton_kpts[i][1],
                                            skeleton_kpts[i][2],
                                            1])

                            # Transform the keypoint to the world frame using the transformation matrix
                            kpt_world = np.dot(cam_to_world_matrix, kpt)

                            skeleton_kpts[i][0] = kpt_world[0]
                            skeleton_kpts[i][1] = kpt_world[1]
                            skeleton_kpts[i][2] = kpt_world[2]
                    
                else:
                    skeleton_kpts = np.full(skeleton_kpts.shape, np.nan)

                # Update current human measurement vector
                human_meas = skeleton_kpts.flatten()

            row_dict.update({'timestamp': timestamp})
            row_dict.update({'human_meas': human_meas.flatten()})

            rows_list.append(row_dict)

    subject_id = bag_file.split('.')[0].split('simple_')[-1]
    bag_data[subject_id] = rows_list

print("Preprocess the rosbag data...")



import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

measurement_data = {}
for subject, bag in bag_data.items():
    data = pd.DataFrame(bag, columns=['timestamp', 'human_meas'])

    # split columns into separate columns
    for c in data.columns.values:
        data = pd.concat([data, data.pop(c).apply(pd.Series).add_prefix(c+"_")], axis=1)

    # change column names
    data.columns = ['timestamp'] + human_meas_names

    # Make time index relative to the start of the recording
    # data['timestamp'] = data['timestamp'] - data['timestamp'][0]

    # Convert the 'timestamp' column to a TimeDeltaIndex
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

    # Increase the timestamp by 2 hours to match the system time
    data['timestamp'] = data['timestamp'] + pd.Timedelta(hours=2)

    # Resample the DataFrame to a known frequency
    # resampled_data = data.resample(freq_str, on='timestamp').mean()
    #resampled_data = data.resample(freq_str, on='timestamp').first()
    measurement_data[subject] = data

for key, value in measurement_data.items():
    print(f'{key}: \t\t{value}')



print("Load GUI data...")


gui_dir = os.path.join(package_path, 'logs', 'gui_data')

# Get all the file names in the directory
gui_files = os.listdir(gui_dir)

gui_data = {}
for gui_file in gui_files:
    # Check if the file is a text file
    if gui_file.endswith('.txt'): # they are txt files, but structured as csv
        # Construct the file path
        file_path = os.path.join(gui_dir, gui_file)
        
        # Read the file as a dataframe
        df = pd.read_csv(file_path)
        
        # Add the dataframe to the dictionary using a portion of the file name as the key
        key = gui_file.split('.')[0].split('_')[-2:]
        key = '_'.join(key)
        gui_data[key] = df




VELOCITIES = ['SLOW', 'MEDIUM', 'FAST']
TASK_NAMES = ['PICK-&-PLACE', 'WALKING', 'PASSING-BY']

trigger_data = {}
for subject, gui in gui_data.items():
    for velocity in VELOCITIES:
        for task in TASK_NAMES:
            trigger_data[(subject, velocity, task)] = gui.loc[(gui['Velocity'] == velocity) & (gui['Task_name'] == task)]

            # only keep the first and last timestamp of each trigger_data
            first_last_timestamps = trigger_data[(subject, velocity, task)].iloc[[0, -1]]['Timestamp'].values
            trigger_data[(subject, velocity, task)] = first_last_timestamps

for key, value in trigger_data.items():
    print(f'{key}: \t\t{value}')


print("Define the filtering and prediction functions...")

def get_near_psd(P, max_iter=10):

    eps = 1e-3  # Small positive jitter for regularization
    increment_factor = 10  # Factor to increase eps if needed
        
    def is_symmetric(A):
        return np.allclose(A, A.T)
                    
    def is_positive_definite(A):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
        
    for _ in range(max_iter):
        if is_symmetric(P) and is_positive_definite(P):
            return P  # The matrix is suitable for Cholesky
    
        # Make P symmetric
        P = (P + P.T) / 2
    
        # Set negative eigenvalues to zero
        eigval, eigvec = np.linalg.eig(P)
        eigval[eigval < 0] = 0
        # add a jitter for strictly positive
        eigval += eps
    
        # Reconstruct the matrix
        P = eigvec.dot(np.diag(eigval)).dot(eigvec.T)

        # Force P to be real
        P = np.real(P)
    
        # Check if P is now positive definite
        if is_positive_definite(P):
            return P
    
        # Increase regularization factor for the next iteration
        eps *= increment_factor
    
    raise ValueError("Unable to convert the matrix to positive definite within max iterations.")




from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.kalman import IMMEstimator
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
import copy, time
from tqdm import tqdm

# Parameters
dt = 0.1
predict_k_steps = True
pred_steps = 5          # [paper: n] number of prediction steps
n_kpts = 18
n_var_per_dof = 3       # position, velocity, acceleration
n_dim_per_kpt = 3       # x, y, z
var_r = 0.0025          # [paper: r_y] Hip: 3-sigma (99.5%) = 0.15 m ==> sigma 0.05 m ==> var = (0.05)^2 m^2
var_q = 0.1             # [paper: q_a] a_dot = u (u = 0 is very uncertain ==> add variance here)
var_P_pos = var_r       # [paper: p_y] Set equal to the measurement noise since the state is initialized with the measurement
var_P_vel = 0.02844     # [paper: p_v] Hip: no keypoint moves faster than 1.6 m/s ==> 3-sigma (99.5%) = 1.6 m/s ==> var = (1.6/3)^2 m^2/s^2
var_P_acc = 1.1111      # [paper: p_a] Hip: no keypoint accelerates faster than 10 m/s^2 ==> 3-sigma (99.5%) = 10 m/s^2 ==> var = (10/3)^2 m^2/s^4
max_time_no_meas = pd.Timedelta(seconds=1.0)

# Transition matrix for IMM
NUM_FILTERS_IN_BANK = 3
M = np.array([[0.55, 0.15, 0.30],
              [0.15, 0.75, 0.10],
              [0.60, 0.30, 0.10]])
mu = np.array([0.55, 0.40, 0.05])

# Dimensions
dim_x = n_var_per_dof * n_dim_per_kpt * n_kpts # 3D (position, velocity, acceleration) for each keypoint
dim_z = n_dim_per_kpt * n_kpts # 3D position for each keypoint

# Initial covariance matrix for all keypoints
def initialize_P(n_dim_per_kpt, n_kpts, var_P_pos, var_P_vel, var_P_acc):
    if n_var_per_dof == 1:
        init_P = np.eye(n_dim_per_kpt * n_kpts) * var_P_pos
    elif n_var_per_dof == 2:
        init_P = np.diag([var_P_pos, var_P_vel]) # initial state covariance for the single keypoint
        init_P = block_diag(*[init_P for _ in range(n_dim_per_kpt * n_kpts)])
    elif n_var_per_dof == 3:
        init_P = np.diag([var_P_pos, var_P_vel, var_P_acc]) # initial state covariance for the single keypoint
        init_P = block_diag(*[init_P for _ in range(n_dim_per_kpt * n_kpts)])
    else:
        raise ValueError('Invalid n_var_per_dof')
    return init_P
    
init_P = initialize_P(n_dim_per_kpt, n_kpts, var_P_pos, var_P_vel, var_P_acc)

# Position indices
p_idx = np.arange(0, dim_x, n_var_per_dof)

# Column names
state_names = ['kp{}_{}'.format(i, suffix)
               for i in range(n_kpts)
               for suffix in ['x', 'xd', 'xdd', 'y', 'yd', 'ydd', 'z', 'zd', 'zdd']]
measurement_names = ['kp{}_{}'.format(i, suffix)
                     for i in range(n_kpts)
                     for suffix in ['x', 'y', 'z']]
filtered_column_names = ['{}_kp{}_{}'.format(filt_type, i, suffix)
                         for filt_type in ['ca', 'cv', 'imm']
                         for i in range(n_kpts)
                         for suffix in ['x', 'xd', 'xdd', 'y', 'yd', 'ydd', 'z', 'zd', 'zdd']]
filtered_pred_column_names = ['{}_kp{}_{}'.format(filt_type, i, suffix)
                              for filt_type in ['ca', 'imm']
                              for i in range(n_kpts)
                              for suffix in ['x', 'xd', 'xdd', 'y', 'yd', 'ydd', 'z', 'zd', 'zdd']]
col_names_imm = ['imm_pos', 'imm_vel', 'imm_acc']
col_names_prob_imm = ['prob_ca', 'prob_ca_no', 'prob_cv']

# measurement function: only the position is measured
def hx(x):
    return x[p_idx]

sigmas = MerweScaledSigmaPoints(n=dim_x, alpha=.1, beta=2., kappa=1.)

# CONSTANT ACCELERATION UKF
F_block_ca = np.array([[1, dt, 0.5*dt**2],
                       [0, 1, dt],
                       [0, 0, 1]])
F_ca = block_diag(*[F_block_ca for _ in range(n_dim_per_kpt * n_kpts)])

# state transition function: const acceleration
def fx_ca(x, dt):
    # return np.dot(F_ca, x)
    return F_ca @ x

# CONSTANT VELOCITY UKF
F_block_cv = np.array([[1, dt, 0],
                       [0, 1, 0],
                       [0, 0, 0]])
F_cv = block_diag(*[F_block_cv for _ in range(n_dim_per_kpt * n_kpts)])

# state transition function: const velocity
def fx_cv(x, dt):
    # return np.dot(F_cv, x)
    return F_cv @ x

import torch
def custom_inv(a):
    t = torch.from_numpy(a)
    t = t.cuda()
    t_inv = torch.inverse(t)
    return t_inv.cpu().numpy()


def run_filtering_loop(subject_ids, velocities, task_names, pred_horizons, init_P, var_r, var_q):
    # CONSTANT ACCELERATION UKF
    ca_ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=dt, hx=hx, fx=fx_ca, points=sigmas)
    ca_ukf.x = np.nan * np.ones(dim_x)
    ca_ukf.P = init_P
    ca_ukf.R = np.eye(dim_z)* var_r
    ca_ukf.Q = Q_discrete_white_noise(dim=n_var_per_dof, dt=dt, var=var_q, block_size=n_dim_per_kpt * n_kpts)
    ca_ukf.inv = custom_inv

    # CONSTANT ACCELERATION UKF WITH NO PROCESS ERROR
    ca_no_ukf = copy.deepcopy(ca_ukf)
    ca_no_ukf.Q = np.zeros((dim_x, dim_x))

    # CONSTANT VELOCITY UKF
    cv_ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=dt, hx=hx, fx=fx_cv, points=sigmas)
    cv_ukf.x = np.nan * np.ones(dim_x)
    cv_ukf.P = init_P
    cv_ukf.R = np.eye(dim_z)* var_r
    cv_ukf.Q = Q_discrete_white_noise(dim=n_var_per_dof, dt=dt, var=var_q, block_size=n_dim_per_kpt * n_kpts)
    ca_ukf.inv = custom_inv

    # IMM ESTIMATOR
    filters = [copy.deepcopy(ca_ukf), ca_no_ukf, copy.deepcopy(cv_ukf)]

    bank = IMMEstimator(filters, mu, M)

    # K-STEP AHEAD PREDICTION FILTERS (declare k dictionaries to store the time series of predicted states and covariances)
    ca_ukf_pred = copy.deepcopy(ca_ukf)
    uxs_ca_pred = {}
    uxs_ca_pred_cov = {}
    bank_pred = copy.deepcopy(bank)
    uxs_bank_pred = {}
    uxs_bank_pred_cov = {}
    probs_bank_pred = {}

    # Create dictionary to store results
    measurement_split = {}   # dictionary of DataFrames with the measurements split by task
    filtering_results = {}   # dictionary of dictionaries with the filtering results
    prediction_results = {}  # dictionary of dictionaries with the k-step ahead prediction results

    for k in pred_horizons:
        for subject_id in subject_ids:
            for velocity in velocities:
                for task in task_names:
                    print(f'Processing {subject_id} - {velocity} - {task} for {k} steps ahead...')

                    # Reinitialize the lists to store the filtering results
                    uxs_ca, uxs_cv, uxs_bank, probs_bank = [], [], [], []
                    uxs_ca_cov, uxs_bank_cov = [], []

                    # Get the trigger timestamps for the current task
                    trigger_timestamps = trigger_data[(subject_id, velocity, task)]

                    # Get only the measurements whose timestamps are within the trigger timestamps
                    start_trigger = pd.to_datetime(trigger_timestamps[0])
                    end_trigger = pd.to_datetime(trigger_timestamps[1])

                    print("Selecting measurements from: ", start_trigger, "to", end_trigger)

                    zs = measurement_data[subject_id].loc[(measurement_data[subject_id]['timestamp'] >= start_trigger) &
                                                        (measurement_data[subject_id]['timestamp'] <= end_trigger)]
                    #zs.set_index('timestamp', inplace=True)

                    # Resample the measurements to a known frequency and subtract initial time
                    # zs = zs.resample(freq_str).mean()
                    #zs.index = zs.index - zs.index[0]
                    zs["timestamp"] = zs["timestamp"] - zs["timestamp"].iloc[0]
                    
                    measurement_split[(k, subject_id, velocity, task)] = zs

                    # Define times
                    t = zs["timestamp"].iloc[0]
                    t_end = zs["timestamp"].iloc[-1]
                    t_incr = pd.Timedelta(seconds=dt)

                    print("Start time:", t, "End time:", t_end)

                    # Initialization flag
                    time_no_meas = pd.Timedelta(seconds=0)
                    ufk_initialized = False
                    filt_timestamps = []
                    elapsed_time = 0.0

                    # Main loop
                    total_iterations = int((t_end - t) / t_incr) + 1
                    pbar = tqdm(total=total_iterations)

                    # Create dictionaries to store the k-step ahead prediction results
                    if predict_k_steps:
                        for i in range(k):
                            uxs_ca_pred[i] = []
                            uxs_bank_pred[i] = []
                            probs_bank_pred[i] = []
                            uxs_ca_pred_cov[i] = []
                            uxs_bank_pred_cov[i] = []

                    while t <= t_end:
                        tic = time.time()
                        filt_timestamps.append(t)
                        k_step_pred_executed = False

                        # Get the measurements in the current time window
                        tmp_db =zs.loc[(zs["timestamp"]>=t) & (zs["timestamp"]<=t+t_incr)]
                        measure_received = False
                        if (tmp_db.shape[0] > 0):
                            z = np.double(np.array(tmp_db.iloc[-1][1:])) # Select the last measurement in the time window
                            measure_received = not np.isnan(z).any() # Consider the measurement only if it is not NaN
                            
                        if measure_received and not ufk_initialized:
                            # print('timestamp:', t, 'measure:', z, 'initializing filters')
                            # initial state: [pos, vel, acc] = [current measured position, 0.0, 0.0]
                            ca_ukf.x = np.zeros(dim_x)
                            ca_ukf.x[p_idx] = z
                            cv_ukf.x = np.zeros(dim_x)
                            cv_ukf.x[p_idx] = z
                            for f in bank.filters:
                                f.x = np.zeros(dim_x)
                                f.x[p_idx] = z
                            ufk_initialized = True

                        else:
                            if not measure_received and ufk_initialized:
                                time_no_meas += t_incr
                                # print('timestamp:', t, 'no measure received for', time_no_meas, 'seconds')

                            if time_no_meas >= max_time_no_meas:
                                ufk_initialized = False
                            
                                # Reset filter states
                                ca_ukf.x = np.nan * np.ones(dim_x)
                                cv_ukf.x = np.nan * np.ones(dim_x)
                                bank.x = np.nan * np.ones(dim_x)
                                if predict_k_steps:
                                    ca_ukf_pred.x = np.nan * np.ones(dim_x)
                                    bank_pred.x = np.nan * np.ones(dim_x)

                                # Reset filter covariances
                                ca_ukf.P = init_P
                                cv_ukf.P = init_P
                                bank.P = init_P
                                if predict_k_steps:
                                    ca_ukf_pred.P = init_P
                                    bank_pred.P = init_P
                                
                            if ufk_initialized:
                                try:
                                    # make sure covariance matrices are positive semidefinite
                                    ca_ukf.P = get_near_psd(ca_ukf.P)
                                    cv_ukf.P = get_near_psd(cv_ukf.P)
                                    for f in bank.filters:
                                        f.P = get_near_psd(f.P)
                                    
                                    ca_ukf.predict()
                                    cv_ukf.predict()
                                    bank.predict()

                                    if measure_received:
                                        time_no_meas = pd.Timedelta(seconds=0)
                                        ca_ukf.update(z)
                                        cv_ukf.update(z)
                                        bank.update(z)

                                    if predict_k_steps:
                                        # Predict k steps ahead starting from the current state and covariance
                                        ca_ukf_pred.x = ca_ukf.x.copy()
                                        ca_ukf_pred.P = ca_ukf.P.copy()
                                        bank_pred.x = bank.x.copy()
                                        for f_pred, f in zip(bank_pred.filters, bank.filters):
                                            f_pred.x = f.x.copy()
                                            f_pred.P = f.P.copy()
                                            
                                        for i in range(k):
                                            # make sure covariance matrices are positive semidefinite
                                            ca_ukf_pred.P = get_near_psd(ca_ukf_pred.P)
                                            for f in bank_pred.filters:
                                                f.P = get_near_psd(f.P)

                                            ca_ukf_pred.predict()
                                            bank_pred.predict()

                                            uxs_ca_pred[i].append(ca_ukf_pred.x.copy())
                                            uxs_bank_pred[i].append(bank_pred.x.copy())
                                            probs_bank_pred[i].append(bank_pred.mu.copy())
                                            uxs_ca_pred_cov[i].append(ca_ukf_pred.P.copy().flatten())
                                            uxs_bank_pred_cov[i].append(bank_pred.P.copy().flatten())

                                        k_step_pred_executed = True

                                except np.linalg.LinAlgError as e:
                                    print(f"LinAlgError: {e}")

                                    # Reset filters
                                    ufk_initialized = False
                                        
                                    # Reset filter states
                                    ca_ukf.x = np.nan * np.ones(dim_x)
                                    cv_ukf.x = np.nan * np.ones(dim_x)
                                    bank.x = np.nan * np.ones(dim_x)
                                    if predict_k_steps:
                                        ca_ukf_pred.x = np.nan * np.ones(dim_x)
                                        bank_pred.x = np.nan * np.ones(dim_x)
                                        bank_pred.mu = np.nan * np.ones(NUM_FILTERS_IN_BANK) # IMM probabilities (3 filters)

                                    # Reset filter covariances
                                    ca_ukf.P = init_P
                                    cv_ukf.P = init_P
                                    bank.P = init_P
                                    if predict_k_steps:
                                        ca_ukf_pred.P = init_P
                                        bank_pred.P = init_P
                                
                        uxs_ca.append(ca_ukf.x.copy())
                        uxs_cv.append(cv_ukf.x.copy())
                        uxs_bank.append(bank.x.copy())
                        probs_bank.append(bank.mu.copy())
                        uxs_ca_cov.append(ca_ukf.P.copy().flatten())
                        uxs_bank_cov.append(bank.P.copy().flatten())

                        if not k_step_pred_executed:
                            for i in range(k):
                                uxs_ca_pred[i].append(ca_ukf.x.copy())
                                uxs_bank_pred[i].append(bank.x.copy())
                                probs_bank_pred[i].append(bank.mu.copy())
                                uxs_ca_pred_cov[i].append(ca_ukf.P.copy().flatten())
                                uxs_bank_pred_cov[i].append(bank.P.copy().flatten())

                        t += t_incr
                        toc = time.time()
                        elapsed_time += (toc - tic)
                        
                        pbar.update()

                    pbar.close()
                    print("Mean loop frequency: {:.2f} Hz".format(1.0 / (elapsed_time / len(filt_timestamps))))

                    # Create DataFrames with the filtered data
                    uxs_ca = np.array(uxs_ca)
                    uxs_cv = np.array(uxs_cv)
                    uxs_bank = np.array(uxs_bank)
                    uxs = np.concatenate((uxs_ca, uxs_cv, uxs_bank), axis=1)
                    probs_bank = np.array(probs_bank)
                    uxs_ca_cov = np.array(uxs_ca_cov)
                    uxs_bank_cov = np.array(uxs_bank_cov)
                    uxs_cov = np.concatenate((uxs_ca_cov, uxs_bank_cov), axis=1)

                    filtered_data = pd.DataFrame(uxs, index=filt_timestamps, columns=filtered_column_names)
                    imm_probs = pd.DataFrame(probs_bank, index=filt_timestamps, columns=col_names_prob_imm)
                    filtered_data_cov = pd.DataFrame(uxs_cov, index=filt_timestamps) # the elements of the flattened covariance matrices are stored in separate anonymous columns

                    if predict_k_steps:
                        kstep_pred_data = {}
                        kstep_pred_imm_probs = {}
                        kstep_pred_cov = {}

                        for i in range(k):
                            uxs_pred = np.concatenate((np.array(uxs_ca_pred[i]), np.array(uxs_bank_pred[i])), axis=1)
                            uxs_pred_cov = np.concatenate((np.array(uxs_ca_pred_cov[i]), np.array(uxs_bank_pred_cov[i])), axis=1)

                            kstep_pred_data[i] = pd.DataFrame(uxs_pred, index=filt_timestamps, columns=filtered_pred_column_names)
                            kstep_pred_imm_probs[i] = pd.DataFrame(np.array(probs_bank_pred[i]), index=filt_timestamps, columns=col_names_prob_imm)
                            kstep_pred_cov[i] = pd.DataFrame(uxs_pred_cov, index=filt_timestamps) # the elements of the flattened covariance matrices are stored in separate anonymous columns

                            # Shift the i-step ahead prediction data by i steps
                            kstep_pred_data[i] = kstep_pred_data[i].shift(+i)
                            kstep_pred_imm_probs[i] = kstep_pred_imm_probs[i].shift(+i)
                            kstep_pred_cov[i] = kstep_pred_cov[i].shift(+i)  

                    # Store filtering results
                    filtering_results[(k, subject_id, velocity, task)] = {
                        'filtered_data': filtered_data,
                        'imm_probs': imm_probs,
                        'filtered_data_cov': filtered_data_cov
                    }

                    # Store k-step prediction results
                    if predict_k_steps: 
                        prediction_results[(k, subject_id, velocity, task)] = {
                            'kstep_pred_data': kstep_pred_data,
                            'kstep_pred_imm_probs': kstep_pred_imm_probs,
                            'kstep_pred_cov': kstep_pred_cov
                        }

                    print(f"Processed {subject_id} - {velocity} - {task} for {k} steps ahead.")

    return measurement_split, filtering_results, prediction_results




# Subject IDS
subject_ids = gui_data.keys()
print("Subjects: ", subject_ids)

# Split subjects into train and test
train_subjects = ['sub_9', 'sub_4', 'sub_11', 'sub_7', 'sub_8', 'sub_10', 'sub_6']
test_subjects = ['sub_13', 'sub_12', 'sub_3']

# Define which keypoints to consider for each task
keypoints = {'PICK-&-PLACE': [4, 7], # [2, 3, 4, 5, 6, 7],
             'WALKING': [0, 1], # [0, 1, 2, 5, 8, 11],
             'PASSING-BY': [0, 1]} # [0, 1, 2, 5, 8, 11]}

dimensions_per_keypoint = {0: ['y'],
                           1: ['y'],
                           4: ['x', 'y', 'z'],
                           7: ['x', 'y', 'z']}

# Parameters
dt = 0.1
predict_k_steps = True
n_kpts = 18
n_var_per_dof = 3       # position, velocity, acceleration
n_dim_per_kpt = 3       # x, y, z
max_time_no_meas = pd.Timedelta(seconds=1.0)
NUM_FILTERS_IN_BANK = 3

# Initial values for the parameters
var_r = 0.0025          # [paper: r_y] Hip: 3-sigma (99.5%) = 0.15 m ==> sigma 0.05 m ==> var = (0.05)^2 m^2
var_q = 0.03            # [paper: q_a] a_dot = u (u = 0 is very uncertain ==> add variance here)
var_P_pos = var_r       # [paper: p_y] Set equal to the measurement noise since the state is initialized with the measurement
var_P_vel = 0.02844     # [paper: p_v] Hip: no keypoint moves faster than 1.6 m/s ==> 3-sigma (99.5%) = 1.6 m/s ==> var = (1.6/3)^2 m^2/s^2
var_P_acc = 1.1111      # [paper: p_a] Hip: no keypoint accelerates faster than 10 m/s^2 ==> 3-sigma (99.5%) = 10 m/s^2 ==> var = (10/3)^2 m^2/s^4

M = np.array([[0.55, 0.15, 0.30], # transition matrix for the IMM estimator
              [0.15, 0.75, 0.10],
              [0.60, 0.30, 0.10]])

mu = np.array([0.55, 0.40, 0.05]) # initial mode probabilities for the IMM estimator

# Tuning parameters
iter_P = 1
iter_q = 10
decrement_factor_q = 0.75
decrement_factor_P = 0.5

PRED_HORIZONS = [1, 3, 5]



# Compute the average error between the filtered states and the k-step ahead predictions
def compute_mean_error(filt, kpred, ca_states, imm_states):
    ca_error = np.mean(filt[ca_states] - kpred[ca_states])
    imm_error = np.mean(filt[imm_states] - kpred[imm_states])

    return ca_error, imm_error

# Compute the root mean squared error between the filtered states and the k-step ahead predictions
def compute_rmse_error(filt, kpred, ca_states, imm_states):
    ca_error = np.sqrt(np.mean((filt[ca_states] - kpred[ca_states])**2))
    imm_error = np.sqrt(np.mean((filt[imm_states] - kpred[imm_states])**2))

    return ca_error, imm_error
    
# Compute the average standard deviation of the error between the filtered states and the k-step ahead predictions 
def compute_std_error(filt, kpred, ca_states, imm_states):
    ca_error = filt[ca_states] - kpred[ca_states]
    ca_error_mean = np.mean(ca_error)
    ca_error_std = np.std(ca_error - ca_error_mean)

    imm_error = filt[imm_states] - kpred[imm_states]
    imm_error_mean = np.mean(imm_error)
    imm_error_std = np.std(imm_error - imm_error_mean)
    
    # Average value for all states
    ca_error_std = np.mean(ca_error_std)
    imm_error_std = np.mean(imm_error_std)

    return ca_error_std, imm_error_std

# Compute the percentage of k-step ahead samples that fall within the band current filtered state +- 1*std
def compute_avg_perc(filt, kpred, kpred_var, ca_states, imm_states, ca_variance_idxs, imm_variance_idxs):
    ca_filtered_state = filt[ca_states]
    
    ca_std = kpred_var[ca_variance_idxs].apply(np.sqrt)
    ca_std.columns = ca_states
    ca_pred_lcl = kpred[ca_states] - 1 * ca_std
    ca_pred_ucl = kpred[ca_states] + 1 * ca_std

    ca_in_CI_band = (ca_filtered_state >= ca_pred_lcl) & (ca_filtered_state <= ca_pred_ucl)
    ca_perc = 100 * np.sum(ca_in_CI_band) / len(filt.dropna())

    imm_filtered_state = filt[imm_states]

    imm_std = kpred_var[imm_variance_idxs].apply(np.sqrt)
    imm_std.columns = imm_states
    imm_pred_lcl = kpred[imm_states] - 1 * imm_std
    imm_pred_ucl = kpred[imm_states] + 1 * imm_std

    imm_in_CI_band = (imm_filtered_state >= imm_pred_lcl) & (imm_filtered_state <= imm_pred_ucl)
    imm_perc = 100 * np.sum(imm_in_CI_band) / len(filt.dropna())
    
    # Average value for all states
    ca_perc = np.mean(ca_perc)
    imm_perc = np.mean(imm_perc)

    return ca_perc, imm_perc




def evaluate_metrics(subjects, velocities, tasks, keypoints, dimensions_per_keypoint,
                 init_P, var_r, var_q, decrement_factor_q, n_var_per_dof, n_dim_per_kpt, dim_x, n_kpts, k,
                 filtering_results, prediction_results):
    
    avg_errors = {}
    avg_rmse = {}
    avg_std = {}
    avg_perc = {}
        
    # Compute the average error between the filtered states and the k-step ahead predictions
    avg_errors[k] = {'PICK-&-PLACE': {'CA': 0, 'IMM': 0},
                'WALKING': {'CA': 0, 'IMM': 0},
                'PASSING-BY': {'CA': 0, 'IMM': 0}}

    # Compute the average RMSE between the filtered states and the k-step ahead predictions
    avg_rmse[k] = copy.deepcopy(avg_errors[k])

    # Compute the average standard deviation of the filtered states and the k-step ahead predictions
    avg_std[k] = copy.deepcopy(avg_errors[k])

    # Compute the percentage of k-step ahead samples that fall within the band current filtered state +- 1*std
    avg_perc = copy.deepcopy(avg_errors)

    for subject_id in subjects:
        for velocity in velocities:
            for task in tasks:
                filt = filtering_results[(k, subject_id, velocity, task)]['filtered_data']
                kpred = prediction_results[(k, subject_id, velocity, task)]['kstep_pred_data'][k-1]
                kpred_var = prediction_results[(k, subject_id, velocity, task)]['kstep_pred_cov'][k-1]
                
                ca_states = []
                ca_variance_idxs = []
                imm_states = []
                imm_variance_idxs = []
                for kpt in keypoints[task]:
                    for dim in ['x', 'y', 'z']:
                        ca_states.append('ca_kp{}_{}'.format(kpt, dim))
                        imm_states.append('imm_kp{}_{}'.format(kpt, dim))

                        state_idx = ['x', 'xd', 'xdd', 'y', 'yd', 'ydd', 'z', 'zd', 'zdd'].index(dim) + n_var_per_dof * n_dim_per_kpt * kpt
                        ca_variance_idx = dim_x * state_idx + state_idx
                        imm_variance_idx = dim_x * state_idx + state_idx + dim_x * dim_x
                        ca_variance_idxs.append(ca_variance_idx)
                        imm_variance_idxs.append(imm_variance_idx)
                
                ca_error, imm_error = compute_mean_error(filt, kpred, ca_states, imm_states)
                ca_rmse, imm_rmse = compute_rmse_error(filt, kpred, ca_states, imm_states)
                ca_std, imm_std = compute_std_error(filt, kpred, ca_states, imm_states)
                ca_perc, imm_perc = compute_avg_perc(filt, kpred, kpred_var,
                                                    ca_states, imm_states, ca_variance_idxs, imm_variance_idxs)
                
                avg_errors[k][task]['CA'] += ca_error
                avg_errors[k][task]['IMM'] += imm_error
                avg_rmse[k][task]['CA'] += ca_rmse
                avg_rmse[k][task]['IMM'] += imm_rmse
                avg_std[k][task]['CA'] += ca_std
                avg_std[k][task]['IMM'] += imm_std
                avg_perc[k][task]['CA'] += ca_perc
                avg_perc[k][task]['IMM'] += imm_perc

    # Compute the average values of these aggregated metrics
    num_sums = len(subjects) * len(velocities)
    for task in tasks:
        avg_errors[k][task]['CA'] /= num_sums
        avg_errors[k][task]['IMM'] /= num_sums
        avg_rmse[k][task]['CA'] /= num_sums
        avg_rmse[k][task]['IMM'] /= num_sums
        avg_std[k][task]['CA'] /= num_sums
        avg_std[k][task]['IMM'] /= num_sums
        avg_perc[k][task]['CA'] /= num_sums
        avg_perc[k][task]['IMM'] /= num_sums

        # Average over all selected keypoints
        avg_errors[k][task]['CA'] = np.mean(avg_errors[k][task]['CA'])
        avg_errors[k][task]['IMM'] = np.mean(avg_errors[k][task]['IMM'])
        avg_rmse[k][task]['CA'] = np.mean(avg_rmse[k][task]['CA'])
        avg_rmse[k][task]['IMM'] = np.mean(avg_rmse[k][task]['IMM'])
        avg_std[k][task]['CA'] = np.mean(avg_std[k][task]['CA'])
        avg_std[k][task]['IMM'] = np.mean(avg_std[k][task]['IMM'])
        avg_perc[k][task]['CA'] = np.mean(avg_perc[k][task]['CA'])
        avg_perc[k][task]['IMM'] = np.mean(avg_perc[k][task]['IMM'])

    # Display the results aggregating results for all keypoints
    print(f"Results for k={k}")
    print("Average error (CA): {:.6f}, {:.6f}, {:.6f}".format(
        avg_errors[k]['PICK-&-PLACE']['CA'], avg_errors[k]['WALKING']['CA'], avg_errors[k]['PASSING-BY']['CA']))
    print("Average error (IMM): {:.6f}, {:.6f}, {:.6f}".format(
        avg_errors[k]['PICK-&-PLACE']['IMM'], avg_errors[k]['WALKING']['IMM'], avg_errors[k]['PASSING-BY']['IMM']))
    print("Average RMSE (CA): {:.6f}, {:.6f}, {:.6f}".format(
        avg_rmse[k]['PICK-&-PLACE']['CA'], avg_rmse[k]['WALKING']['CA'], avg_rmse[k]['PASSING-BY']['CA']))
    print("Average RMSE (IMM): {:.6f}, {:.6f}, {:.6f}".format(
        avg_rmse[k]['PICK-&-PLACE']['IMM'], avg_rmse[k]['WALKING']['IMM'], avg_rmse[k]['PASSING-BY']['IMM']))
    print("Average std (CA): {:.4f}, {:.4f}, {:.4f}".format(
        avg_std[k]['PICK-&-PLACE']['CA'], avg_std[k]['WALKING']['CA'], avg_std[k]['PASSING-BY']['CA']))
    print("Average std (IMM): {:.4f}, {:.4f}, {:.4f}".format(
        avg_std[k]['PICK-&-PLACE']['IMM'], avg_std[k]['WALKING']['IMM'], avg_std[k]['PASSING-BY']['IMM']))
    print("Average percentage (CA): {:.4f}, {:.4f}, {:.4f}".format(
        avg_perc[k]['PICK-&-PLACE']['CA'], avg_perc[k]['WALKING']['CA'], avg_perc[k]['PASSING-BY']['CA']))
    print("Average percentage (IMM): {:.4f}, {:.4f}, {:.4f}".format(
        avg_perc[k]['PICK-&-PLACE']['IMM'], avg_perc[k]['WALKING']['IMM'], avg_perc[k]['PASSING-BY']['IMM']))
    print("===============================================\n\n")

    # Export results to a CSV file
    results = {'CA_error': [avg_errors[k]['PICK-&-PLACE']['CA'], avg_errors[k]['WALKING']['CA'], avg_errors[k]['PASSING-BY']['CA']],
                'IMM_error': [avg_errors[k]['PICK-&-PLACE']['IMM'], avg_errors[k]['WALKING']['IMM'], avg_errors[k]['PASSING-BY']['IMM']],
                'CA_RMSE': [avg_rmse[k]['PICK-&-PLACE']['CA'], avg_rmse[k]['WALKING']['CA'], avg_rmse[k]['PASSING-BY']['CA']],
                'IMM_RMSE': [avg_rmse[k]['PICK-&-PLACE']['IMM'], avg_rmse[k]['WALKING']['IMM'], avg_rmse[k]['PASSING-BY']['IMM']],
                'CA_std': [avg_std[k]['PICK-&-PLACE']['CA'], avg_std[k]['WALKING']['CA'], avg_std[k]['PASSING-BY']['CA']],
                'IMM_std': [avg_std[k]['PICK-&-PLACE']['IMM'], avg_std[k]['WALKING']['IMM'], avg_std[k]['PASSING-BY']['IMM']],
                'CA_perc': [avg_perc[k]['PICK-&-PLACE']['CA'], avg_perc[k]['WALKING']['CA'], avg_perc[k]['PASSING-BY']['CA']],
                'IMM_perc': [avg_perc[k]['PICK-&-PLACE']['IMM'], avg_perc[k]['WALKING']['IMM'], avg_perc[k]['PASSING-BY']['IMM']]}
    
    results_df = pd.DataFrame(results, index=['PICK-&-PLACE', 'WALKING', 'PASSING-BY'])
    results_df.to_csv(f'results_{k}.csv')




tic = time.time()
train_subjects = train_subjects
velocities = VELOCITIES
tasks = TASK_NAMES
pred_horizons = PRED_HORIZONS

keypoints = {'PICK-&-PLACE': [4, 7], # [2, 3, 4, 5, 6, 7],
             'WALKING': [0, 1], # [0, 1, 2, 5, 8, 11],
             'PASSING-BY': [0, 1]} # [0, 1, 2, 5, 8, 11]}

dimensions_per_keypoint = {0: ['y'],
                           1: ['y'],
                           4: ['x', 'y', 'z'],
                           7: ['x', 'y', 'z']}

# Initial values for the parameters
var_r = 0.0025          # [paper: r_y] Hip: 3-sigma (99.5%) = 0.15 m ==> sigma 0.05 m ==> var = (0.05)^2 m^2
var_q = 0.01            # [paper: q_a] a_dot = u (u = 0 is very uncertain ==> add variance here)
var_P_pos = var_r       # [paper: p_y] Set equal to the measurement noise since the state is initialized with the measurement
var_P_vel = 0.02844     # [paper: p_v] Hip: no keypoint moves faster than 1.6 m/s ==> 3-sigma (99.5%) = 1.6 m/s ==> var = (1.6/3)^2 m^2/s^2
var_P_acc = 1.1111      # [paper: p_a] Hip: no keypoint accelerates faster than 10 m/s^2 ==> 3-sigma (99.5%) = 10 m/s^2 ==> var = (10/3)^2 m^2/s^4
init_P = initialize_P(n_dim_per_kpt, n_kpts, var_P_pos, var_P_vel, var_P_acc)

# print("===============================================")
# print("Evaluation of the metrics for the test subjects...")

# tic = time.time()
# _, test1_filtering_results, test1_prediction_results = run_filtering_loop(test_subjects, velocities, tasks, [1],
#                                                                     init_P, var_r, var_q)

# evaluate_metrics(test_subjects, velocities, tasks, keypoints, dimensions_per_keypoint,
#                  init_P, var_r, var_q, decrement_factor_q, n_var_per_dof, n_dim_per_kpt, dim_x, n_kpts, 1, test1_filtering_results, test1_prediction_results)

# _, test3_filtering_results, test3_prediction_results = run_filtering_loop(test_subjects, velocities, tasks, [3],
#                                                                     init_P, var_r, var_q)

# evaluate_metrics(test_subjects, velocities, tasks, keypoints, dimensions_per_keypoint,
#                  init_P, var_r, var_q, decrement_factor_q, n_var_per_dof, n_dim_per_kpt, dim_x, n_kpts, 3, test3_filtering_results, test3_prediction_results)

# _, test5_filtering_results, test5_prediction_results = run_filtering_loop(test_subjects, velocities, tasks, [5],
#                                                                     init_P, var_r, var_q, )

# evaluate_metrics(test_subjects, velocities, tasks, keypoints, dimensions_per_keypoint,
#                  init_P, var_r, var_q, decrement_factor_q, n_var_per_dof, n_dim_per_kpt, dim_x, n_kpts, 5, test5_filtering_results, test5_prediction_results)

# toc = time.time()
# minutes, seconds = divmod(toc - tic, 60)
# print(f"[VALIDATION] Metrics evaluation took {minutes:.0f} minutes and {seconds:.2f} seconds.")

print("===============================================")
print("Evaluation of the metrics for the train subjects...")

_, train1_filtering_results, train1_prediction_results = run_filtering_loop(train_subjects, velocities, tasks, [1],
                                                                    init_P, var_r, var_q)

evaluate_metrics(train_subjects, velocities, tasks, keypoints, dimensions_per_keypoint,
                init_P, var_r, var_q, decrement_factor_q, n_var_per_dof, n_dim_per_kpt, dim_x, n_kpts, 1, train1_filtering_results, train1_prediction_results)

# _, train3_filtering_results, train3_prediction_results = run_filtering_loop(train_subjects, velocities, tasks, [3],
#                                                                     init_P, var_r, var_q)

# evaluate_metrics(train_subjects, velocities, tasks, keypoints, dimensions_per_keypoint,
#                  init_P, var_r, var_q, decrement_factor_q, n_var_per_dof, n_dim_per_kpt, dim_x, n_kpts, 3, train3_filtering_results, train3_prediction_results)

# _, train5_filtering_results, train5_prediction_results = run_filtering_loop(train_subjects, velocities, tasks, [5],
#                                                                     init_P, var_r, var_q)

# evaluate_metrics(train_subjects, velocities, tasks, keypoints, dimensions_per_keypoint,
#                  init_P, var_r, var_q, decrement_factor_q, n_var_per_dof, n_dim_per_kpt, dim_x, n_kpts, 5, train5_filtering_results, train5_prediction_results)

# toc = time.time()
# minutes, seconds = divmod(toc - tic, 60)
# print(f"[IDENTIFICATION] Metrics evaluation took {minutes:.0f} minutes and {seconds:.2f} seconds.")


