import numpy as np
import pandas as pd
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.kalman import IMMEstimator
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
import copy, torch, tqdm, time

pd.options.mode.chained_assignment = None  # default='warn'

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


# Initial covariance matrix for all keypoints
def initialize_P(n_var_per_dof, n_dim_per_kpt, n_kpts, var_P_pos, var_P_vel, var_P_acc):
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


def initialize_filters(dim_x, dim_z, p_idx, dt,
                       init_P, var_r, var_q,
                       n_var_per_dof, n_dim_per_kpt, n_kpts,
                       custom_inv, mu, M):
    
    # measurement function: only the position is measured
    def hx(x: np.ndarray) -> np.ndarray:
        return x[p_idx]

    # Sigma points for the UKF
    sigmas = MerweScaledSigmaPoints(n=dim_x, alpha=.1, beta=2., kappa=1.)

    # CONSTANT ACCELERATION UKF
    F_block_ca = np.array([[1, dt, 0.5*dt**2],
                           [0, 1, dt],
                           [0, 0, 1]])
    F_ca = block_diag(*[F_block_ca for _ in range(n_dim_per_kpt * n_kpts)])

    def fx_ca(x: np.ndarray, dt: float) -> np.ndarray:
        return F_ca @ x

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
    F_block_cv = np.array([[1, dt, 0],
                           [0, 1, 0],
                           [0, 0, 0]])
    F_cv = block_diag(*[F_block_cv for _ in range(n_dim_per_kpt * n_kpts)])

    # state transition function: const velocity
    def fx_cv(x: np.ndarray, dt: float) -> np.ndarray:
        return F_cv @ x

    cv_ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=dt, hx=hx, fx=fx_cv, points=sigmas)
    cv_ukf.x = np.nan * np.ones(dim_x)
    cv_ukf.P = init_P
    cv_ukf.R = np.eye(dim_z)* var_r
    cv_ukf.Q = Q_discrete_white_noise(dim=n_var_per_dof, dt=dt, var=var_q, block_size=n_dim_per_kpt * n_kpts)
    ca_ukf.inv = custom_inv

    # IMM ESTIMATOR
    filters = [copy.deepcopy(ca_ukf), ca_no_ukf, copy.deepcopy(cv_ukf)]

    bank = IMMEstimator(filters, mu, M)

    return (ca_ukf, cv_ukf, bank)


# Define a custom inverse function for PyTorch to use GPU
def custom_inv(a):
    # Detect if GPU is available
    if torch.cuda.is_available():
        t = torch.from_numpy(a)
        t = t.cuda()
        t_inv = torch.inverse(t)
        return t_inv.cpu().numpy()
    else:
        return np.linalg.inv(a)


def run_filtering_loop(trigger_data, measurement_data,
                       subject_ids, velocities, task_names, pred_horizons, predict_k_steps,
                       dim_x, dim_z, p_idx, dt,
                       n_var_per_dof, n_dim_per_kpt, n_kpts,
                       init_P, var_r, var_q,
                       mu, M, num_filters_in_bank,
                       custom_inv, max_time_no_meas,
                       filtered_column_names, filtered_pred_column_names, col_names_prob_imm):

    # Create dictionary to store results
    measurement_split = {}   # dictionary of DataFrames with the measurements split by task
    filtering_results = {}   # dictionary of dictionaries with the filtering results
    prediction_results = {}  # dictionary of dictionaries with the k-step ahead prediction results

    for k in pred_horizons:
        for subject_id in subject_ids:
            for velocity in velocities:
                for task in task_names:
                    print(f'Processing {subject_id} - {velocity} - {task} for {k} steps ahead...')

                    # Initialize the filters
                    ca_ukf, cv_ukf, bank = initialize_filters(dim_x, dim_z, p_idx, dt,
                                                              init_P, var_r, var_q,
                                                              n_var_per_dof, n_dim_per_kpt, n_kpts,
                                                              custom_inv, mu, M)
                    ca_ukf_pred = copy.deepcopy(ca_ukf)
                    bank_pred = copy.deepcopy(bank)

                    # K-STEP AHEAD PREDICTION FILTERS (declare k dictionaries to store the time series of predicted states and covariances)
                    ca_ukf_pred = copy.deepcopy(ca_ukf)
                    uxs_ca_pred = {}
                    uxs_ca_pred_cov = {}
                    bank_pred = copy.deepcopy(bank)
                    uxs_bank_pred = {}
                    uxs_bank_pred_cov = {}
                    probs_bank_pred = {}

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
                    zs.loc[:, "timestamp"] = zs["timestamp"] - zs["timestamp"].iloc[0]
                    
                    measurement_split[(k, subject_id, velocity, task)] = zs

                    # Define times
                    t = zs["timestamp"].iloc[0]
                    t_end = zs["timestamp"].iloc[-1]
                    t_incr = pd.Timedelta(seconds=dt)

                    print("Start time:", t, "End time:", t_end)

                    # Initialization flag
                    time_no_meas = pd.Timedelta(seconds=0)
                    ukf_initialized = False
                    filt_timestamps = []
                    
                    # Main loop
                    total_iterations = int((t_end - t) / t_incr) + 1
                    pbar = tqdm.tqdm(total=total_iterations)

                    # Create dictionaries to store the k-step ahead prediction results
                    if predict_k_steps:
                        for i in range(k):
                            uxs_ca_pred[i] = []
                            uxs_bank_pred[i] = []
                            probs_bank_pred[i] = []
                            uxs_ca_pred_cov[i] = []
                            uxs_bank_pred_cov[i] = []

                    start_time = time.time()
                    while t <= t_end:
                        filt_timestamps.append(t)
                        k_step_pred_executed = False

                        # Get the measurements in the current time window
                        tmp_db =zs.loc[(zs["timestamp"]>=t) & (zs["timestamp"]<=t+t_incr)]
                        measure_received = False
                        if (tmp_db.shape[0] > 0):
                            z = np.double(np.array(tmp_db.iloc[-1][1:])) # Select the last measurement in the time window
                            measure_received = not np.isnan(z).any() # Consider the measurement only if it is not NaN
                            
                        if measure_received and not ukf_initialized:
                            # print(f'[timestamp: {t.total_seconds():.2f}s] Initializing filters with the first measurement.')
                            # initial state: [pos, vel, acc] = [current measured position, 0.0, 0.0]
                            ca_ukf.x = np.zeros(dim_x)
                            ca_ukf.x[p_idx] = z
                            cv_ukf.x = np.zeros(dim_x)
                            cv_ukf.x[p_idx] = z
                            for f in bank.filters:
                                f.x = np.zeros(dim_x)
                                f.x[p_idx] = z
                            ukf_initialized = True

                        else:
                            if not measure_received and ukf_initialized:
                                time_no_meas += t_incr
                                # print(f'[timestamp: {t.total_seconds():.2f}s] No measure received for {time_no_meas.total_seconds():.2f} seconds.')

                            if time_no_meas >= max_time_no_meas or not ukf_initialized:
                                if ukf_initialized:
                                    # print(f'[timestamp: {t.total_seconds():.2f}s] No-measure-received timeout. Resetting filters.')
                                    ukf_initialized = False
                                # else:
                                    # print(f'[timestamp: {t.total_seconds():.2f}s] No measure received and filters not initialized.')

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
                                
                            if ukf_initialized:
                                try:
                                    # print(f'[timestamp: {t.total_seconds():.2f}s] Filtering and k-step-ahead prediction.')

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
                                    ukf_initialized = False
                                        
                                    # Reset filter states
                                    ca_ukf.x = np.nan * np.ones(dim_x)
                                    cv_ukf.x = np.nan * np.ones(dim_x)
                                    bank.x = np.nan * np.ones(dim_x)
                                    if predict_k_steps:
                                        ca_ukf_pred.x = np.nan * np.ones(dim_x)
                                        bank_pred.x = np.nan * np.ones(dim_x)
                                        bank_pred.mu = np.nan * np.ones(num_filters_in_bank) # IMM probabilities (3 filters)

                                    # Reset filter covariances
                                    ca_ukf.P = init_P
                                    cv_ukf.P = init_P
                                    bank.P = init_P
                                    if predict_k_steps:
                                        ca_ukf_pred.P = init_P
                                        bank_pred.P = init_P

                            else:
                                total_iterations -= 1 # do not count the iteration if the filters are not initialized
                                
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
                        pbar.update()

                    pbar.close()
                    print(f"Average loop frequency: {(total_iterations / (time.time() - start_time)):.2f} Hz")

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

                    print(f"Processed {subject_id} - {velocity} - {task} for {k} steps ahead.\n\n")

    return measurement_split, filtering_results, prediction_results


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