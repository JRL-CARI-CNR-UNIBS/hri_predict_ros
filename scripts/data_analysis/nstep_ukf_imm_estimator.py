import os, copy, torch, tqdm, time, pickle
import numpy as np
import pandas as pd
import human_model_binding as hkm # human kinematic model
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, IMMEstimator, unscented_transform
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
from multipledispatch import dispatch

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
def initialize_P(n_var_per_dof, n_dim_per_kpt, n_kpts, var_P_pos, var_P_vel, var_P_acc,
                 space='cartesian', var_P_param= None, n_param=None, n_joints=None):
    if space == 'cartesian':
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
    
    elif space == 'joint'and var_P_param is not None and n_param is not None and n_joints is not None:

        if n_var_per_dof == 1:
            init_P = np.eye(n_joints) * var_P_pos
        elif n_var_per_dof == 2:
            init_P = np.diag([var_P_pos, var_P_vel]) # initial state covariance for the single joint
            init_P = block_diag(*[init_P for _ in range(n_joints)])
        elif n_var_per_dof == 3:
            init_P = np.diag([var_P_pos, var_P_vel, var_P_acc]) # initial state covariance for the single joint
            init_P = block_diag(*[init_P for _ in range(n_joints)])
        else:
            raise ValueError('Invalid n_var_per_dof')

        param_P = np.eye(n_param) * var_P_param
        init_P = np.block([
            [init_P, np.zeros((init_P.shape[0], n_param))],
            [np.zeros((n_param, init_P.shape[1])), param_P]
        ])

    return init_P


def initialize_filters(dim_x, dim_z, p_idx, dt,
                       init_P, var_r, var_q,
                       n_var_per_dof, n_dim_per_kpt, n_kpts,
                       custom_inv, mu, M,
                       space='cartesian', human_kinematic_model=None,
                       param_idx=None,
                       sindy_model=None):
    
    assert space in ['cartesian', 'joint'], "Invalid space. Choose between 'cartesian' or 'joint'."
    assert (human_kinematic_model is not None) if space == 'joint' else True, "Human kinematic model must be provided for joint space."
    
    n_param = len(param_idx) if param_idx is not None else 0

    # measurement function: only the keypoint position is measured
    if space == 'cartesian':
        @dispatch(np.ndarray)
        def hx(x: np.ndarray): # type: ignore (function overload made possible by multipledispatch)
            return x[p_idx]
    elif space == 'joint':
        @dispatch(np.ndarray, param=np.ndarray)
        def hx(x: np.ndarray, param: np.ndarray):
            kp_in_ext = hkm.Keypoints()
            human_kinematic_model.forward_kinematics(x, param, kp_in_ext) # type: ignore (assertion already ensures that human_kinematic_model is not None)
            return kp_in_ext.get_keypoints()
    else:
        raise ValueError('Invalid space. Choose between "cartesian" or "joint".')

    # Sigma points for the UKF
    sigmas = MerweScaledSigmaPoints(n=dim_x, alpha=.1, beta=2., kappa=1.)

    # CONSTANT ACCELERATION UKF
    F_block_ca = np.array([[1, dt, 0.5*dt**2],
                           [0, 1, dt],
                           [0, 0, 1]])
    F_ca = block_diag(*[F_block_ca for _ in range(len(p_idx))])

    # Append the identity matrix block for the parameters to F_ca
    if space == 'joint' and n_param is not None:
        F_ca = np.block([
            [F_ca, np.zeros((F_ca.shape[0], n_param))],
            [np.zeros((n_param, F_ca.shape[1])), np.eye(n_param)]
        ])
    
    def fx_ca(x: np.ndarray, dt: float) -> np.ndarray:
        return F_ca @ x

    ca_ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=dt, hx=hx, fx=fx_ca, points=sigmas)
    ca_ukf.x = np.nan * np.ones(dim_x)
    ca_ukf.P = init_P
    ca_ukf.R = np.eye(dim_z)* var_r
    ca_ukf.Q = Q_discrete_white_noise(dim=n_var_per_dof, dt=dt, var=var_q, block_size=len(p_idx))
    if space == 'joint' and n_param is not None:
        ca_ukf.Q = np.block([
            [ca_ukf.Q, np.zeros((ca_ukf.Q.shape[0], n_param))],
            [np.zeros((n_param, ca_ukf.Q.shape[1])), np.eye(n_param)*var_q]
        ])
    ca_ukf.inv = custom_inv

    # CONSTANT ACCELERATION UKF WITH NO PROCESS ERROR
    ca_no_ukf = copy.deepcopy(ca_ukf)
    ca_no_ukf.Q = np.zeros((dim_x, dim_x))

    # SINDY MODEL UKF
    if sindy_model is not None:
        assert param_idx is not None, "Parameter indices must be provided for SINDy model."

        WALKING_BODY_JOINTS = [0, 1, 2, 8, 9] + list(range(10, 26)) # up to 25th joint (included)
        WALKING_BODY_PARAMS = [i for i in range(n_param)]
        
        PICKPLACE_RH_BODY_JOINTS = list(range(3, 7)) + list(range(10, 14))
        PICKPLACE_LH_BODY_JOINTS = list(range(3, 7)) + list(range(14, 18))
        PICKPLACE_BODY_PARAMS = list(range(0, 5))

        UPPER_BODY_JOINTS = list(range(3, 8)) + list(range(10, 18)) + list(range(26, 28))
        UPPER_BODY_PARAMS = list(range(0, 5)) + [7]

        def compute_sindy_prediction(x, selected_joints, selected_params, model_name):
            # Define indices for the joint and parameter states
            p_idx_temp = p_idx[selected_joints]
            v_idx_temp = p_idx_temp + 1
            a_idx_temp = v_idx_temp + 1
            param_idx_temp = np.array(param_idx[selected_params])

            # Select the SINDy model
            model = sindy_model[model_name]

            # Predict the next state using the SINDy models
            state = x[a_idx_temp]
            input = x[np.sort(np.concatenate((p_idx_temp, v_idx_temp, param_idx_temp)))]

            # Reshape the state and input for the SINDy model
            state = state.reshape(1, -1)
            input = input.reshape(1, -1)

            # Predict the next state using the SINDy model
            x_dot_pred = model.predict(state, u=input)

            return x_dot_pred
        

        def fx_sindy(x: np.ndarray, dt: float) -> np.ndarray:
            # Chest position and leg joints
            x_dot_pred_chest_pos_legs = \
                compute_sindy_prediction(x,
                                         WALKING_BODY_JOINTS,
                                         WALKING_BODY_PARAMS,
                                         'chest_pos_legs')

            # Chest rotation and right arm joints
            x_dot_pred_chest_rot_right_arm = \
                compute_sindy_prediction(x,
                                         PICKPLACE_RH_BODY_JOINTS,
                                         PICKPLACE_BODY_PARAMS,
                                         'chest_rot_right_arm')

            # Chest rotation and left arm joints
            x_dot_pred_chest_rot_left_arm = \
                compute_sindy_prediction(x,
                                         PICKPLACE_LH_BODY_JOINTS,
                                         PICKPLACE_BODY_PARAMS,
                                         'chest_rot_left_arm')
            
            # Upper body
            x_dot_pred_upper_body = \
                compute_sindy_prediction(x,
                                         UPPER_BODY_JOINTS,
                                         UPPER_BODY_PARAMS,
                                         'upper_body')

            # Merge the partial predicted jerks into a single prediction
            combined_pred = np.zeros_like(x[p_idx])
            count_pred = np.zeros_like(x[p_idx])

            # Add predictions for WALKING_BODY_JOINTS
            combined_pred[WALKING_BODY_JOINTS] += x_dot_pred_chest_pos_legs[0]
            count_pred[WALKING_BODY_JOINTS] += 1

            # Add predictions for PICKPLACE_RH_BODY_JOINTS
            combined_pred[PICKPLACE_RH_BODY_JOINTS] += x_dot_pred_chest_rot_right_arm[0]
            count_pred[PICKPLACE_RH_BODY_JOINTS] += 1

            # Add predictions for PICKPLACE_LH_BODY_JOINTS
            combined_pred[PICKPLACE_LH_BODY_JOINTS] += x_dot_pred_chest_rot_left_arm[0]
            count_pred[PICKPLACE_LH_BODY_JOINTS] += 1

            # Add predictions for UPPER_BODY_JOINTS
            combined_pred[UPPER_BODY_JOINTS] += x_dot_pred_upper_body[0]
            count_pred[UPPER_BODY_JOINTS] += 1

            # Avoid division by zero
            count_pred[count_pred == 0] = 1

            # Calculate the average jerk predictions
            jerk_pred = combined_pred / count_pred

            # Create the predicted state applying triple integration
            x_next_pred = np.zeros_like(x)
            x_next_pred[p_idx] = x[p_idx] + x[p_idx + 1]*dt             # pos_next = pos + vel*dt
            x_next_pred[p_idx + 1] = x[p_idx + 1] + x[p_idx + 2]*dt     # vel_next = vel + acc*dt
            x_next_pred[p_idx + 2] = x[p_idx + 2] + jerk_pred*dt        # acc_next = acc + jerk*dt
        
            return x_next_pred
        
        sindy_ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=dt, hx=hx, fx=fx_sindy, points=sigmas)
        sindy_ukf.x = np.nan * np.ones(dim_x)
        sindy_ukf.P = init_P
        sindy_ukf.R = np.eye(dim_z)* var_r
        sindy_ukf.Q = Q_discrete_white_noise(dim=n_var_per_dof, dt=dt, var=var_q, block_size=len(p_idx))
        if space == 'joint' and n_param is not None:
            sindy_ukf.Q = np.block([
                [sindy_ukf.Q, np.zeros((sindy_ukf.Q.shape[0], n_param))],
                [np.zeros((n_param, sindy_ukf.Q.shape[1])), np.eye(n_param)*var_q]
            ])
        sindy_ukf.inv = custom_inv


    # CONSTANT VELOCITY UKF
    F_block_cv = np.array([[1, dt, 0],
                           [0, 1, 0],
                           [0, 0, 0]])
    F_cv = block_diag(*[F_block_cv for _ in range(len(p_idx))])

    # Append the identity matrix block for the parameters to F_ca
    if space == 'joint' and n_param is not None:
        F_cv = np.block([
            [F_cv, np.zeros((F_cv.shape[0], n_param))],
            [np.zeros((n_param, F_cv.shape[1])), np.eye(n_param)]
        ])

    # state transition function: const velocity
    def fx_cv(x: np.ndarray, dt: float) -> np.ndarray:
        return F_cv @ x

    cv_ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=dt, hx=hx, fx=fx_cv, points=sigmas)
    cv_ukf.x = np.nan * np.ones(dim_x)
    cv_ukf.P = init_P
    cv_ukf.R = np.eye(dim_z)* var_r
    cv_ukf.Q = Q_discrete_white_noise(dim=n_var_per_dof, dt=dt, var=var_q, block_size=len(p_idx))
    if space == 'joint' and n_param is not None:
        cv_ukf.Q = np.block([
            [cv_ukf.Q, np.zeros((cv_ukf.Q.shape[0], n_param))],
            [np.zeros((n_param, cv_ukf.Q.shape[1])), np.eye(n_param)*var_q]
        ])
    cv_ukf.inv = custom_inv

    # IMM ESTIMATOR
    if sindy_model is not None:
        filters = [copy.deepcopy(ca_ukf), sindy_ukf, copy.deepcopy(cv_ukf)]
    else:
        filters = [copy.deepcopy(ca_ukf), ca_no_ukf, copy.deepcopy(cv_ukf)]

    bank = IMMEstimator(filters, mu, M)

    return (ca_ukf, cv_ukf, bank)


def propagate_covariance(dim_x, dim_z, ukf, body_model, p_idx, param_idx):
    if (np.isnan(ukf.x)).any() or (np.isnan(ukf.P)).any():
        return np.nan * np.ones(dim_z), np.nan * np.ones(dim_z * dim_z)
    
    # Sigma points for the UKF
    sigma_fn = MerweScaledSigmaPoints(n=dim_x, alpha=.1, beta=2., kappa=1.)

    # Calculate sigma points for given mean and covariance
    sigmas = sigma_fn.sigma_points(ukf.x, ukf.P)

    sigmas_h = []
    kpts = hkm.Keypoints()
    for s in sigmas:
        body_model.forward_kinematics(s[p_idx], s[param_idx], kpts)
        sigmas_h.append(kpts.get_keypoints())

    sigmas_h = np.atleast_2d(sigmas_h)

    # Mean and covariance of prediction passed through unscented transform
    if isinstance(ukf, IMMEstimator):
        # Select the first filter in the bank (arbitrary choice)
        Wm = ukf.filters[0].Wm
        Wc = ukf.filters[0].Wc
        R = ukf.filters[0].R
    else:
        Wm = ukf.Wm
        Wc = ukf.Wc
        R = ukf.R

    # kpts_mean, kpts_cov = unscented_transform(sigmas_h, Wm, Wc, R)
    kpts_mean = np.mean(sigmas_h, axis=0)
    kpts_cov = np.cov(sigmas_h.T)
    
    return kpts_mean, kpts_cov


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


# Compute the average error between the filtered states and the k-step ahead predictions
def compute_mean_error(filt, kpred, ca_states, imm_states):
    with np.errstate(invalid='ignore'):
        ca_diff = (filt[ca_states] - kpred[ca_states]).dropna()
        ca_error = np.mean(ca_diff.values, axis=0)

        imm_diff = (filt[imm_states] - kpred[imm_states]).dropna()
        imm_error = np.mean(imm_diff.values, axis=0)

        # Average value for all states
        ca_error = np.mean(ca_error)
        imm_error = np.mean(imm_error)

        if np.isnan(ca_error):
            # raise ValueError('CA: Mean Error is NaN')
            print('CA: Mean Error is NaN')
        if np.isnan(imm_error):
            # raise ValueError('IMM: Mean Error is NaN')
            print('IMM: Mean Error is NaN')

    return ca_error, imm_error


# Compute the root mean squared error between the filtered states and the k-step ahead predictions
def compute_rmse_error(filt, kpred, ca_states, imm_states):
    with np.errstate(invalid='ignore'):
        ca_squared_diff = ((filt[ca_states] - kpred[ca_states])**2).dropna()
        ca_error = np.sqrt(np.mean(ca_squared_diff.values, axis=0))

        imm_squared_diff = ((filt[imm_states] - kpred[imm_states])**2).dropna()
        imm_error = np.sqrt(np.mean(imm_squared_diff.values, axis=0))

        # Average value for all states
        ca_error = np.mean(ca_error)
        imm_error = np.mean(imm_error)
    
        if np.isnan(ca_error):
            # raise ValueError('CA: RMSE is NaN')
            print('CA: RMSE is NaN')
        if np.isnan(imm_error):
            # raise ValueError('IMM: RMSE is NaN')
            print('IMM: RMSE is NaN')

    return ca_error, imm_error
    

# Compute the average standard deviation of the error between the filtered states and the k-step ahead predictions 
def compute_std_error(filt, kpred, ca_states, imm_states):
    with np.errstate(invalid='ignore'):
        ca_error = (filt[ca_states] - kpred[ca_states]).dropna()
        ca_error_std = np.std(ca_error.values, axis=0)

        imm_error = (filt[imm_states] - kpred[imm_states]).dropna()
        imm_error_std = np.std(imm_error.values, axis=0)
        
        # Average value for all states
        ca_error_std = np.mean(ca_error_std)
        imm_error_std = np.mean(imm_error_std)

        if np.isnan(ca_error_std):
            # raise ValueError('CA: Error Std.Dev. is NaN')
            print('CA: Error Std.Dev. is NaN')
        if np.isnan(imm_error_std):
            # raise ValueError('IMM: Error Std.Dev. is NaN')
            print('IMM: Error Std.Dev. is NaN')

    return ca_error_std, imm_error_std


# Compute the percentage of k-step ahead samples that fall within the band current filtered state +- 1*std
def compute_avg_perc(filt, kpred, kpred_var, ca_states, imm_states, ca_variance_idxs, imm_variance_idxs):
    with np.errstate(invalid='ignore'):
        # CA states
        ca_filtered_state = filt[ca_states]
        
        ca_std = kpred_var[ca_variance_idxs].apply(np.sqrt)
        ca_std.columns = ca_states
        ca_pred_lcl = kpred[ca_states] - 1 * ca_std
        ca_pred_ucl = kpred[ca_states] + 1 * ca_std

        ca_in_CI_band = (ca_filtered_state >= ca_pred_lcl) & (ca_filtered_state <= ca_pred_ucl)
        ca_perc = 100 * np.sum(ca_in_CI_band) / len(filt)

        # IMM states
        imm_filtered_state = filt[imm_states]
        
        imm_std = kpred_var[imm_variance_idxs].apply(np.sqrt)
        imm_std.columns = imm_states
        imm_pred_lcl = kpred[imm_states] - 1 * imm_std
        imm_pred_ucl = kpred[imm_states] + 1 * imm_std

        imm_in_CI_band = (imm_filtered_state >= imm_pred_lcl) & (imm_filtered_state <= imm_pred_ucl)
        imm_perc = 100 * np.sum(imm_in_CI_band) / len(filt)
        
        # Average value for all states
        ca_perc = np.mean(ca_perc)
        imm_perc = np.mean(imm_perc)

        if np.isnan(ca_perc):
            # raise ValueError('CA: Percentage is NaN')
            print('CA: Percentage is NaN')
        if np.isnan(imm_perc):
            # raise ValueError('IMM: Percentage is NaN')
            print('IMM: Percentage is NaN')

    return ca_perc, imm_perc


# Evaluate the metrics for the prediction results
def evaluate_metrics(subjects, velocities, tasks, instructions,
                     keypoints, dim_name_per_kpt,
                     n_var_per_dof, n_dim_per_kpt, dim_x, k,
                     filtering_results, prediction_results, results_dir,
                     space_compute='cartesian', space_eval='cartesian',
                     conf_names=[]):
    
    results = {}

    avg_errors = {}
    avg_rmse = {}
    avg_std = {}
    avg_perc = {}
        
    # Compute the average error between the filtered states and the k-step ahead predictions
    avg_errors[k] = {'PICK-&-PLACE': {'CA': 0.0, 'IMM': 0.0},
                     'WALKING':      {'CA': 0.0, 'IMM': 0.0},
                     'PASSING-BY':   {'CA': 0.0, 'IMM': 0.0}}

    # Compute the average RMSE between the filtered states and the k-step ahead predictions
    avg_rmse[k] = copy.deepcopy(avg_errors[k])

    # Compute the average standard deviation of the filtered states and the k-step ahead predictions
    avg_std[k] = copy.deepcopy(avg_errors[k])

    # Compute the percentage of k-step ahead samples that fall within the band current filtered state +- 1*std
    avg_perc = copy.deepcopy(avg_errors)

    num_sums = {'PICK-&-PLACE': 0,
                'WALKING':      0,
                'PASSING-BY':   0}
    for subject_id in subjects:
        for velocity in velocities:
            for task in tasks:
                for instruction in instructions[task]:
                    # Get the filtered data and the k-step ahead prediction data
                    filt = filtering_results[(k, subject_id, velocity, task, instruction)]['filtered_data']
                    kpred = prediction_results[(k, subject_id, velocity, task, instruction)]['kstep_pred_data'][k-1]
                    kpred_var = prediction_results[(k, subject_id, velocity, task, instruction)]['kstep_pred_cov'][k-1]

                    # Identify rows that contain all NaN values and discard them
                    nan_rows = filt[filt.isnull().all(axis=1)]
                    filt = filt.dropna(how='all')

                    # Keep track of the discarded rows
                    discarded_rows = nan_rows.index.tolist()

                    # Remove rows at indexes discarded_rows
                    kpred = kpred.drop(discarded_rows)
                    kpred_var = kpred_var.drop(discarded_rows)
                    

                    ca_states = []
                    ca_variance_idxs = []
                    imm_states = []
                    imm_variance_idxs = []

                    if space_eval == 'cartesian':
                        for kpt in keypoints[task]:
                            for dim in dim_name_per_kpt[kpt]:    
                                ca_states.append('ca_kp{}_{}'.format(kpt, dim))
                                imm_states.append('imm_kp{}_{}'.format(kpt, dim))
                                state_idx = ['x', 'xd', 'xdd', 'y', 'yd', 'ydd', 'z', 'zd', 'zdd'].index(dim) + n_var_per_dof * n_dim_per_kpt * kpt
                                
                                ca_variance_idx = dim_x * state_idx + state_idx
                                imm_variance_idx = dim_x * state_idx + state_idx + dim_x * dim_x
                                
                                ca_variance_idxs.append(ca_variance_idx)
                                imm_variance_idxs.append(imm_variance_idx)

                    elif space_eval == 'joint':
                        visited_confs = []
                        for conf in conf_names:
                            if 'ca' in conf:
                                ca_states.append(conf)
                                substring = conf.split('ca_', 1)[1]
                            elif 'imm' in conf:
                                imm_states.append(conf)
                                substring = conf.split('imm_', 1)[1]
                            else:
                                continue

                            if substring not in visited_confs:
                                state_idx = conf_names.index(conf)
                                
                                ca_variance_idx = dim_x * state_idx + state_idx
                                imm_variance_idx = dim_x * state_idx + state_idx + dim_x * dim_x

                                ca_variance_idxs.append(ca_variance_idx)
                                imm_variance_idxs.append(imm_variance_idx)

                            visited_confs.append(substring)  

                    # Only compute metrics if both filtered and predicted data are available
                    # i.e., filt-kpred do not produce all NaN rows
                    if not (filt-kpred).isnull().all(bool_only=True).all():
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

                        num_sums[task] += 1

    # Compute the average values of these aggregated metrics
    for task in tasks:
        print(f"Number of sums for {task}: {num_sums[task]}")
        if num_sums[task] == 0:
            avg_errors[k][task]['CA'] = np.nan
            avg_errors[k][task]['IMM'] = np.nan
            avg_rmse[k][task]['CA'] = np.nan
            avg_rmse[k][task]['IMM'] = np.nan
            avg_std[k][task]['CA'] = np.nan
            avg_std[k][task]['IMM'] = np.nan
            avg_perc[k][task]['CA'] = np.nan
            avg_perc[k][task]['IMM'] = np.nan

        else:
            avg_errors[k][task]['CA'] /= num_sums[task]
            avg_errors[k][task]['IMM'] /= num_sums[task]
            avg_rmse[k][task]['CA'] /= num_sums[task]
            avg_rmse[k][task]['IMM'] /= num_sums[task]
            avg_std[k][task]['CA'] /= num_sums[task]
            avg_std[k][task]['IMM'] /= num_sums[task]
            avg_perc[k][task]['CA'] /= num_sums[task]
            avg_perc[k][task]['IMM'] /= num_sums[task]


    # Display the results aggregating results for all keypoints
    print(f"Results for k = {k}")
    print("================================================================")
    print("Average error (CA):\t {:.4f},\t {:.4f},\t {:.4f}".format(
        avg_errors[k]['PICK-&-PLACE']['CA'], avg_errors[k]['WALKING']['CA'], avg_errors[k]['PASSING-BY']['CA']))
    print("Average error (IMM):\t {:.4f},\t {:.4f},\t {:.4f}".format(
        avg_errors[k]['PICK-&-PLACE']['IMM'], avg_errors[k]['WALKING']['IMM'], avg_errors[k]['PASSING-BY']['IMM']))
    print("Average RMSE (CA):\t {:.4f},\t {:.4f},\t {:.4f}".format(
        avg_rmse[k]['PICK-&-PLACE']['CA'], avg_rmse[k]['WALKING']['CA'], avg_rmse[k]['PASSING-BY']['CA']))
    print("Average RMSE (IMM):\t {:.4f},\t {:.4f},\t {:.4f}".format(
        avg_rmse[k]['PICK-&-PLACE']['IMM'], avg_rmse[k]['WALKING']['IMM'], avg_rmse[k]['PASSING-BY']['IMM']))
    print("Average std (CA):\t {:.4f},\t {:.4f},\t {:.4f}".format(
        avg_std[k]['PICK-&-PLACE']['CA'], avg_std[k]['WALKING']['CA'], avg_std[k]['PASSING-BY']['CA']))
    print("Average std (IMM):\t {:.4f},\t {:.4f},\t {:.4f}".format(
        avg_std[k]['PICK-&-PLACE']['IMM'], avg_std[k]['WALKING']['IMM'], avg_std[k]['PASSING-BY']['IMM']))
    print("Average % in CI (CA):\t {:.4f},\t {:.4f},\t {:.4f}".format(
        avg_perc[k]['PICK-&-PLACE']['CA'], avg_perc[k]['WALKING']['CA'], avg_perc[k]['PASSING-BY']['CA']))
    print("Average % in CI (IMM):\t {:.4f},\t {:.4f},\t {:.4f}".format(
        avg_perc[k]['PICK-&-PLACE']['IMM'], avg_perc[k]['WALKING']['IMM'], avg_perc[k]['PASSING-BY']['IMM']))
    print("================================================================\n\n")


    # Export results to a CSV file
    results = {'Metric': ['CA_error', 'IMM_error', 'CA_RMSE', 'IMM_RMSE', 'CA_std', 'IMM_std', 'CA_perc', 'IMM_perc'],
               'PICK-&-PLACE': [avg_errors[k]['PICK-&-PLACE']['CA'], avg_errors[k]['PICK-&-PLACE']['IMM'],
                                avg_rmse[k]['PICK-&-PLACE']['CA'], avg_rmse[k]['PICK-&-PLACE']['IMM'],
                                avg_std[k]['PICK-&-PLACE']['CA'], avg_std[k]['PICK-&-PLACE']['IMM'],
                                avg_perc[k]['PICK-&-PLACE']['CA'], avg_perc[k]['PICK-&-PLACE']['IMM']
                ],
               'WALKING': [avg_errors[k]['WALKING']['CA'], avg_errors[k]['WALKING']['IMM'],
                           avg_rmse[k]['WALKING']['CA'], avg_rmse[k]['WALKING']['IMM'],
                           avg_std[k]['WALKING']['CA'], avg_std[k]['WALKING']['IMM'],
                           avg_perc[k]['WALKING']['CA'], avg_perc[k]['WALKING']['IMM']
                ],
               'PASSING-BY': [avg_errors[k]['PASSING-BY']['CA'], avg_errors[k]['PASSING-BY']['IMM'],
                              avg_rmse[k]['PASSING-BY']['CA'], avg_rmse[k]['PASSING-BY']['IMM'],
                              avg_std[k]['PASSING-BY']['CA'], avg_std[k]['PASSING-BY']['IMM'],
                              avg_perc[k]['PASSING-BY']['CA'], avg_perc[k]['PASSING-BY']['IMM']
                ]
    }
    df_results = pd.DataFrame(results)
    df_results.set_index('Metric', inplace=True)      
    df_results.to_csv(os.path.join(results_dir,f'results_{k}_steps_{space_compute}Compute_{space_eval}Eval.csv'))
    

def run_filtering_loop(X_train_list, time_train_list, train_traj_idx,
                       pred_horizons, predict_k_steps,
                       dim_x, dim_z, p_idx, dt,
                       n_var_per_dof, n_dim_per_kpt, n_kpts,
                       init_P, var_r, var_q,
                       mu, M, num_filters_in_bank,
                       custom_inv, max_time_no_meas,
                       prob_imm_col_names,
                       output_col_names,
                       results_dir,
                       filename,
                       space='cartesian',
                       param_idx=np.array([]),
                       sindy_model=None):
    
    assert space in ['cartesian', 'joint'], "Invalid space. Choose between 'cartesian' and 'joint'."
    
    # Get the column names for the filtered and predicted states
    filt_col_names = output_col_names['filtered_column_names']
    filt_pred_col_names = output_col_names['filtered_pred_column_names']
    if space == 'joint':
        filt_joint_col_names = output_col_names['filtered_joint_column_names']
        filt_pred_joint_col_names = output_col_names['filtered_pred_joint_column_names']
        filt_param_col_names = output_col_names['filtered_param_names']

    # Loop over the prediction horizons
    measurements_saved = False
    for hor_idx, k in enumerate(pred_horizons):
        # Create dictionary to store results
        measurement_split = {}   # dictionary of DataFrames with the measurements split by task
        filtering_results = {}   # dictionary of dictionaries with the filtering results
        prediction_results = {}  # dictionary of dictionaries with the k-step ahead prediction results

        # Loop over the training trajectories
        for traj_idx in range(len(X_train_list)):
            traj_info = train_traj_idx[traj_idx]
            subject_id, velocity, task, instruction = traj_info['subject'], traj_info['velocity'], traj_info['task'], traj_info['instruction']
            print(f'\n[ITER {hor_idx*len(X_train_list) + traj_idx + 1} / {len(X_train_list)*len(pred_horizons)}]: processing {traj_info} for {k} steps ahead...')

            # Create a HumanBodyModel object if the space is 'joint'
            if space == 'joint':
                body_model = hkm.Human28DOF()

                # Initialize joint limits
                qbounds = [hkm.JointLimits(-np.pi, np.pi)]*28

                # Set the shoulder rot y joint limits
                qbounds[12] = hkm.JointLimits(-np.pi/2, np.pi/2) # right shoulder
                qbounds[16] = hkm.JointLimits(-np.pi/2, np.pi/2) # left shoulder
                qbounds[20] = hkm.JointLimits(-np.pi/2, np.pi/2) # right hip
                qbounds[24] = hkm.JointLimits(-np.pi/2, np.pi/2) # left hip

                # Initialize Keypoints object
                kp_in_ext = hkm.Keypoints()

            elif space == 'cartesian':
                body_model = None

            # Initialize the filters
            ca_ukf, cv_ukf, bank = initialize_filters(dim_x, dim_z, p_idx, dt,
                                                      init_P, var_r, var_q,
                                                      n_var_per_dof, n_dim_per_kpt, n_kpts,
                                                      custom_inv, mu, M,
                                                      space=space, human_kinematic_model=body_model,
                                                      param_idx=param_idx,
                                                      sindy_model=sindy_model)
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

            if space == 'joint':
                uxs_ca_kpt_pred = {}
                uxs_bank_kpt_pred = {}
                uxs_ca_kpt_pred_cov = {}
                uxs_bank_kpt_pred_cov = {}

            # Reinitialize the lists to store the filtering results
            uxs_ca, uxs_cv, uxs_bank, probs_bank = [], [], [], []
            uxs_ca_cov, uxs_bank_cov = [], []

            if space == 'joint':
                uxs_ca_kpt, uxs_cv_kpt, uxs_bank_kpt = [], [], []
                uxs_ca_kpt_cov, uxs_bank_kpt_cov = [], []


            # Get the measurements
            meas = X_train_list[traj_idx]
            time_info = time_train_list[traj_idx]

            zs = pd.DataFrame([time_info, meas], index=['timestamp', 'meas']).T
            zs['timestamp'] = pd.to_timedelta(zs['timestamp'], unit='s')

            if not measurements_saved:
                measurement_split[(subject_id, velocity, task, instruction)] = zs

            # If zs only contains NaN values, skip the current iteration
            nan_idxs = zs['meas'].apply(lambda x: all(pd.isna(x)))
            if nan_idxs.all():
                print(f"Skipping {subject_id} - {velocity} - {task} for {k} steps ahead.")
                continue

            # Define times
            t = zs["timestamp"].iloc[0]
            t_end = zs["timestamp"].iloc[-1]
            t_incr = pd.Timedelta(seconds=dt)

            print("Start time:", t, " [s] | End time:", t_end, " [s]")

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

                    if space == 'joint':
                        uxs_ca_kpt_pred[i] = []
                        uxs_bank_kpt_pred[i] = []
                        uxs_ca_kpt_pred_cov[i] = []
                        uxs_bank_kpt_pred_cov[i] = []

            start_time = time.time()
            while t <= t_end:
                filt_timestamps.append(t)
                k_step_pred_executed = False

                # Get the measurements in the current time window
                tmp_db =zs.loc[(zs["timestamp"]>=t) & (zs["timestamp"]<=t+t_incr)]
                measure_received = False
                if (tmp_db.shape[0] > 0):
                    z = tmp_db.iloc[-1]['meas'] # Select the last measurement in the time window
                    measure_received = not np.isnan(z).any() # Consider the measurement only if it is not NaN
                    
                if space == 'joint':
                    if measure_received:
                        # Update the Keypoints object with the current joint angles
                        kpts = {
                            "head":           z[0:3],
                            "left_shoulder":  z[3:6],
                            "left_elbow":     z[6:9],
                            "left_wrist":     z[9:12],
                            "left_hip":       z[12:15],
                            "left_knee":      z[15:18],
                            "left_ankle":     z[18:21],
                            "right_shoulder": z[21:24],
                            "right_elbow":    z[24:27],
                            "right_wrist":    z[27:30],
                            "right_hip":      z[30:33],
                            "right_knee":     z[33:36],
                            "right_ankle":    z[36:39]
                        }

                        kp_in_ext.set_keypoints(kpts)
                        
                        # Update the HumanBodyModel with the current joint angles
                        q = np.zeros(28)
                        param = np.zeros(8)
                        if body_model is not None:
                            q, param = body_model.inverse_kinematics(kp_in_ext, qbounds, q, param) 

                if measure_received and not ukf_initialized:
                    # print(f'[timestamp: {t.total_seconds():.2f}s] Initializing filters with the first measurement.')
                    # initial state: [pos, vel, acc] = [current measured position, 0.0, 0.0]
                    ca_ukf.x = np.zeros(dim_x)
                    ca_ukf.x[p_idx] = (z if space == 'cartesian' else q)
                    if space == 'joint':
                        ca_ukf.x[param_idx] = param 
                    cv_ukf.x = np.zeros(dim_x)
                    cv_ukf.x[p_idx] = (z if space == 'cartesian' else q)
                    if space == 'joint':
                        cv_ukf.x[param_idx] = param 
                    for f in bank.filters:
                        f.x = np.zeros(dim_x)
                        f.x[p_idx] = (z if space == 'cartesian' else q)
                        if space == 'joint':
                            f.x[param_idx] = param 
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
                                if space == 'cartesian':
                                    ca_ukf.update(z)
                                    cv_ukf.update(z)
                                    bank.update(z)
                                elif space == 'joint':
                                    ca_ukf.update(z, R=None, UT=None, hx=None, param=param)
                                    cv_ukf.update(z, R=None, UT=None, hx=None, param=param)
                                    bank.update(z, param=param)
                                else:
                                    raise ValueError("Invalid space. Choose between 'cartesian' and 'joint'.")

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

                                    if space == 'joint':
                                        # Propagate the covariance matrices from joint space to Cartesian space
                                        ca_ukf_pred_kpt, ca_ukf_pred_kpt_cov = \
                                            propagate_covariance(dim_x, dim_z, ca_ukf, body_model, p_idx, param_idx)
                                        bank_pred_kpt, bank_pred_kpt_cov = \
                                            propagate_covariance(dim_x, dim_z, bank, body_model, p_idx, param_idx)
                                        
                                        uxs_ca_kpt_pred[i].append(ca_ukf_pred_kpt)
                                        uxs_bank_kpt_pred[i].append(bank_pred_kpt)
                                        uxs_ca_kpt_pred_cov[i].append(ca_ukf_pred_kpt_cov.flatten())
                                        uxs_bank_kpt_pred_cov[i].append(bank_pred_kpt_cov.flatten())

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
                        

                # Store the filtered states and covariances
                uxs_ca.append(ca_ukf.x.copy())
                uxs_cv.append(cv_ukf.x.copy())
                uxs_bank.append(bank.x.copy())
                probs_bank.append(bank.mu.copy())
                uxs_ca_cov.append(ca_ukf.P.copy().flatten())
                uxs_bank_cov.append(bank.P.copy().flatten())

                if space == 'joint':
                    # Propagate the covariance matrices from joint space to Cartesian space
                    ca_ukf_kpt, ca_ukf_kpt_cov = propagate_covariance(dim_x, dim_z, ca_ukf, body_model, p_idx, param_idx)
                    ca_ukf_kpt_cov = ca_ukf_kpt_cov.flatten()
                    cv_ukf_kpt, _ = propagate_covariance(dim_x, dim_z, cv_ukf, body_model, p_idx, param_idx)
                    bank_kpt, bank_kpt_cov = propagate_covariance(dim_x, dim_z, bank, body_model, p_idx, param_idx)
                    bank_kpt_cov = bank_kpt_cov.flatten()

                    # Store the filtered states and covariances in cartesian space
                    uxs_ca_kpt.append(ca_ukf_kpt) # type: ignore
                    uxs_cv_kpt.append(cv_ukf_kpt) # type: ignore
                    uxs_bank_kpt.append(bank_kpt) # type: ignore
                    uxs_ca_kpt_cov.append(ca_ukf_kpt_cov) # type: ignore
                    uxs_bank_kpt_cov.append(bank_kpt_cov) # type: ignore


                if not k_step_pred_executed:
                    for i in range(k):
                        uxs_ca_pred[i].append(ca_ukf.x.copy())
                        uxs_bank_pred[i].append(bank.x.copy())
                        probs_bank_pred[i].append(bank.mu.copy())
                        uxs_ca_pred_cov[i].append(ca_ukf.P.copy().flatten())
                        uxs_bank_pred_cov[i].append(bank.P.copy().flatten())

                        if space == 'joint':
                            uxs_ca_kpt_pred[i].append(ca_ukf_kpt)
                            uxs_bank_kpt_pred[i].append(bank_kpt)
                            uxs_ca_kpt_pred_cov[i].append(ca_ukf_kpt_cov)
                            uxs_bank_kpt_pred_cov[i].append(bank_kpt_cov)

                t += t_incr                        
                pbar.update()

            pbar.close()
            print(f"Average loop frequency: {(total_iterations / (time.time() - start_time)):.2f} Hz")


            # Convert the lists to numpy arrays
            uxs_ca = np.array(uxs_ca)
            uxs_cv = np.array(uxs_cv)
            uxs_bank = np.array(uxs_bank)
            probs_bank = np.array(probs_bank)
            uxs_ca_cov = np.array(uxs_ca_cov)
            uxs_bank_cov = np.array(uxs_bank_cov)

            if space == 'joint':
                uxs_ca_kpt = np.array(uxs_ca_kpt)
                uxs_cv_kpt = np.array(uxs_cv_kpt)
                uxs_bank_kpt = np.array(uxs_bank_kpt)
                uxs_ca_kpt_cov = np.array(uxs_ca_kpt_cov)
                uxs_bank_kpt_cov = np.array(uxs_bank_kpt_cov)


            # Concatenate the filtered states and covariances
            if space == 'cartesian':
                uxs = np.concatenate((uxs_ca, uxs_cv, uxs_bank), axis=1)
                uxs_cov = np.concatenate((uxs_ca_cov, uxs_bank_cov), axis=1)
            elif space == 'joint':
                uxs = np.concatenate((uxs_ca, uxs_cv, uxs_bank, uxs_ca_kpt, uxs_cv_kpt, uxs_bank_kpt), axis=1)
                uxs_cov = np.concatenate((uxs_ca_cov, uxs_bank_cov, uxs_ca_kpt_cov, uxs_bank_kpt_cov), axis=1)
            else:
                raise ValueError("Invalid space. Choose between 'cartesian' and 'joint'.")
            

            # Create DataFrames with the filtered data
            if space == 'cartesian':
                filtered_data = pd.DataFrame(uxs, index=filt_timestamps, columns=filt_col_names)
            elif space == 'joint':
                # Split filt_joint_col_names into three portions
                ca_cols = [col for col in filt_joint_col_names if col.startswith('ca_')]
                cv_cols = [col for col in filt_joint_col_names if col.startswith('cv_')]
                imm_cols = [col for col in filt_joint_col_names if col.startswith('imm_')]

                ca_kpt_cols = [col for col in filt_col_names if col.startswith('ca_') and not col.endswith('d')]
                cv_kpt_cols = [col for col in filt_col_names if col.startswith('cv_') and not col.endswith('d')]
                imm_kpt_cols = [col for col in filt_col_names if col.startswith('imm_') and not col.endswith('d')]

                # Combine the portions with filt_param_col_names in between
                combined_cols = ca_cols + filt_param_col_names + cv_cols + filt_param_col_names + imm_cols + filt_param_col_names \
                                + ca_kpt_cols + cv_kpt_cols + imm_kpt_cols
                filtered_data = pd.DataFrame(uxs, index=filt_timestamps, columns=combined_cols)
            else:
                raise ValueError("Invalid space. Choose between 'cartesian' and 'joint'.")

            imm_probs = pd.DataFrame(probs_bank, index=filt_timestamps, columns=prob_imm_col_names)
            filtered_data_cov = pd.DataFrame(uxs_cov, index=filt_timestamps) # the elements of the flattened covariance matrices are stored in separate anonymous columns


            # Create DataFrames with the k-step ahead prediction data if predict_k_steps is True
            if predict_k_steps:
                kstep_pred_data = {}
                kstep_pred_imm_probs = {}
                kstep_pred_cov = {}

                for i in range(k):
                    if space == 'cartesian':
                        uxs_pred = np.concatenate((np.array(uxs_ca_pred[i]), np.array(uxs_bank_pred[i])), axis=1)
                        uxs_pred_cov = np.concatenate((np.array(uxs_ca_pred_cov[i]), np.array(uxs_bank_pred_cov[i])), axis=1)
                        kstep_pred_data[i] = pd.DataFrame(uxs_pred, index=filt_timestamps, columns=filt_pred_col_names)

                    elif space == 'joint':
                        uxs_pred = np.concatenate((np.array(uxs_ca_pred[i]), np.array(uxs_bank_pred[i]), 
                                                   np.array(uxs_ca_kpt_pred[i]), np.array(uxs_bank_kpt_pred[i])), axis=1)
                        uxs_pred_cov = np.concatenate((np.array(uxs_ca_pred_cov[i]), np.array(uxs_bank_pred_cov[i]),
                                                       np.array(uxs_ca_kpt_pred_cov[i]), np.array(uxs_bank_kpt_pred_cov[i])), axis=1)

                        # Split filt_joint_col_names into three portions
                        ca_cols = [col for col in filt_pred_joint_col_names if col.startswith('ca_')]
                        imm_cols = [col for col in filt_pred_joint_col_names if col.startswith('imm_')]

                        ca_kpt_cols = [col for col in filt_col_names if col.startswith('ca_') and not col.endswith('d')]
                        imm_kpt_cols = [col for col in filt_col_names if col.startswith('imm_') and not col.endswith('d')]

                        # Combine the portions with filt_param_col_names in between
                        combined_cols = ca_cols + filt_param_col_names + imm_cols + filt_param_col_names \
                                        + ca_kpt_cols + imm_kpt_cols
                        kstep_pred_data[i] = pd.DataFrame(uxs_pred, index=filt_timestamps, columns=combined_cols)
                    else:
                        raise ValueError("Invalid space. Choose between 'cartesian' and 'joint'.")
            
                    kstep_pred_imm_probs[i] = pd.DataFrame(np.array(probs_bank_pred[i]), index=filt_timestamps, columns=prob_imm_col_names)
                    kstep_pred_cov[i] = pd.DataFrame(uxs_pred_cov, index=filt_timestamps) # the elements of the flattened covariance matrices are stored in separate anonymous columns

                    # Shift the i-step ahead prediction data by i steps
                    kstep_pred_data[i] = kstep_pred_data[i].shift(+i+1)
                    kstep_pred_imm_probs[i] = kstep_pred_imm_probs[i].shift(+i+1)
                    kstep_pred_cov[i] = kstep_pred_cov[i].shift(+i+1)  


            # Store filtering results
            filtering_results[(k, subject_id, velocity, task, instruction)] = {
                'filtered_data': filtered_data,
                'imm_probs': imm_probs,
                'filtered_data_cov': filtered_data_cov
            }

            # Store k-step prediction results
            if predict_k_steps: 
                prediction_results[(k, subject_id, velocity, task, instruction)] = {
                    'kstep_pred_data': kstep_pred_data,
                    'kstep_pred_imm_probs': kstep_pred_imm_probs,
                    'kstep_pred_cov': kstep_pred_cov
                }

        # Save the results for the current prediction horizon and free up memory
        print(f"Saving results for {k} steps ahead to a compressed archive and freeing up memory...")
        tic = time.time()
        if not measurements_saved:
            with open(os.path.join(results_dir,
                               '_'.join([filename, 'measurements', space, '.pkl'])), 'wb') as f:
                pickle.dump(measurement_split, f, protocol=pickle.HIGHEST_PROTOCOL)
            measurements_saved = True
            del measurement_split
        with open(os.path.join(results_dir,
                               '_'.join([filename, 'filtering_results', str(k), 'steps', space, '.pkl'])), 'wb') as f:
            pickle.dump(filtering_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(results_dir,
                               '_'.join([filename, 'prediction_results', str(k), 'steps', space, '.pkl'])), 'wb') as f:
            pickle.dump(prediction_results, f,protocol=pickle.HIGHEST_PROTOCOL)

        del filtering_results, prediction_results
        toc = time.time()
        print(f"Results for {k} steps ahead saved and memory freed up. Time elapsed: {toc-tic:.2f} seconds.")


### ===== OLD VERSION OF run_filtering_loop ===== ###
# def run_filtering_loop(trigger_data, measurement_data,
#                        subject_ids, velocities, task_names, pred_horizons, predict_k_steps,
#                        dim_x, dim_z, p_idx, dt,
#                        n_var_per_dof, n_dim_per_kpt, n_kpts,
#                        init_P, var_r, var_q,
#                        mu, M, num_filters_in_bank,
#                        custom_inv, max_time_no_meas,
#                        filtered_column_names, filtered_pred_column_names, col_names_prob_imm):

#     # Create dictionary to store results
#     measurement_split = {}   # dictionary of DataFrames with the measurements split by task
#     filtering_results = {}   # dictionary of dictionaries with the filtering results
#     prediction_results = {}  # dictionary of dictionaries with the k-step ahead prediction results

#     for k in pred_horizons:
#         for subject_id in subject_ids:
#             for velocity in velocities:
#                 for task in task_names:
#                     print(f'Processing {subject_id} - {velocity} - {task} for {k} steps ahead...')

#                     # Initialize the filters
#                     ca_ukf, cv_ukf, bank = initialize_filters(dim_x, dim_z, p_idx, dt,
#                                                               init_P, var_r, var_q,
#                                                               n_var_per_dof, n_dim_per_kpt, n_kpts,
#                                                               custom_inv, mu, M)
#                     ca_ukf_pred = copy.deepcopy(ca_ukf)
#                     bank_pred = copy.deepcopy(bank)

#                     # K-STEP AHEAD PREDICTION FILTERS (declare k dictionaries to store the time series of predicted states and covariances)
#                     ca_ukf_pred = copy.deepcopy(ca_ukf)
#                     uxs_ca_pred = {}
#                     uxs_ca_pred_cov = {}
#                     bank_pred = copy.deepcopy(bank)
#                     uxs_bank_pred = {}
#                     uxs_bank_pred_cov = {}
#                     probs_bank_pred = {}

#                     # Reinitialize the lists to -> np.ndarray:
#                     print("Selecting measurements from: ", start_trigger, "to", end_trigger)

#                     zs = measurement_data[subject_id].loc[(measurement_data[subject_id]['timestamp'] >= start_trigger) &
#                                                         (measurement_data[subject_id]['timestamp'] <= end_trigger)]
#                     #zs.set_index('timestamp', inplace=True)

#                     # Resample the measurements to a known frequency and subtract initial time
#                     # zs = zs.resample(freq_str).mean()
#                     #zs.index = zs.index - zs.index[0]
#                     zs.loc[:, "timestamp"] = zs["timestamp"] - zs["timestamp"].iloc[0]
                    
#                     measurement_split[(k, subject_id, velocity, task)] = zs

#                     # Define times
#                     t = zs["timestamp"].iloc[0]
#                     t_end = zs["timestamp"].iloc[-1]
#                     t_incr = pd.Timedelta(seconds=dt)

#                     print("Start time:", t, "End time:", t_end)

#                     # Initialization flag
#                     time_no_meas = pd.Timedelta(seconds=0)
#                     ukf_initialized = False
#                     filt_timestamps = []
                    
#                     # Main loop
#                     total_iterations = int((t_end - t) / t_incr) + 1
#                     pbar = tqdm.tqdm(total=total_iterations)

#                     # Create dictionaries to store the k-step ahead prediction results
#                     if predict_k_steps:
#                         for i in range(k):
#                             uxs_ca_pred[i] = []
#                             uxs_bank_pred[i] = []
#                             probs_bank_pred[i] = []
#                             uxs_ca_pred_cov[i] = []
#                             uxs_bank_pred_cov[i] = []

#                     start_time = time.time()
#                     while t <= t_end:
#                         filt_timestamps.append(t)
#                         k_step_pred_executed = False

#                         # Get the measurements in the current time window
#                         tmp_db =zs.loc[(zs["timestamp"]>=t) & (zs["timestamp"]<=t+t_incr)]
#                         measure_received = False
#                         if (tmp_db.shape[0] > 0):
#                             z = np.double(np.array(tmp_db.iloc[-1][1:])) # Select the last measurement in the time window
#                             measure_received = not np.isnan(z).any() # Consider the measurement only if it is not NaN
                            
#                         if measure_received and not ukf_initialized:
#                             # print(f'[timestamp: {t.total_seconds():.2f}s] Initializing filters with the first measurement.')
#                             # initial state: [pos, vel, acc] = [current measured position, 0.0, 0.0]
#                             ca_ukf.x = np.zeros(dim_x)
#                             ca_ukf.x[p_idx] = z
#                             cv_ukf.x = np.zeros(dim_x)
#                             cv_ukf.x[p_idx] = z
#                             for f in bank.filters:
#                                 f.x = np.zeros(dim_x)
#                                 f.x[p_idx] = z
#                             ukf_initialized = True

#                         else:
#                             if not measure_received and ukf_initialized:
#                                 time_no_meas += t_incr
#                                 # print(f'[timestamp: {t.total_seconds():.2f}s] No measure received for {time_no_meas.total_seconds():.2f} seconds.')

#                             if time_no_meas >= max_time_no_meas or not ukf_initialized:
#                                 if ukf_initialized:
#                                     # print(f'[timestamp: {t.total_seconds():.2f}s] No-measure-received timeout. Resetting filters.')
#                                     ukf_initialized = False
#                                 # else:
#                                     # print(f'[timestamp: {t.total_seconds():.2f}s] No measure received and filters not initialized.')

#                                 # Reset filter states
#                                 ca_ukf.x = np.nan * np.ones(dim_x)
#                                 cv_ukf.x = np.nan * np.ones(dim_x)
#                                 bank.x = np.nan * np.ones(dim_x)
#                                 if predict_k_steps:
#                                     ca_ukf_pred.x = np.nan * np.ones(dim_x)
#                                     bank_pred.x = np.nan * np.ones(dim_x)

#                                 # Reset filter covariances
#                                 ca_ukf.P = init_P
#                                 cv_ukf.P = init_P
#                                 bank.P = init_P
#                                 if predict_k_steps:
#                                     ca_ukf_pred.P = init_P
#                                     bank_pred.P = init_P
                                
#                             if ukf_initialized:
#                                 try:
#                                     # print(f'[timestamp: {t.total_seconds():.2f}s] Filtering and k-step-ahead prediction.')

#                                     # make sure covariance matrices are positive semidefinite
#                                     ca_ukf.P = get_near_psd(ca_ukf.P)
#                                     cv_ukf.P = get_near_psd(cv_ukf.P)
#                                     for f in bank.filters:
#                                         f.P = get_near_psd(f.P)
                                    
#                                     ca_ukf.predict()
#                                     cv_ukf.predict()
#                                     bank.predict()

#                                     if measure_received:
#                                         time_no_meas = pd.Timedelta(seconds=0)
#                                         ca_ukf.update(z)
#                                         cv_ukf.update(z)
#                                         bank.update(z)

#                                     if predict_k_steps:
#                                         # Predict k steps ahead starting from the current state and covariance
#                                         ca_ukf_pred.x = ca_ukf.x.copy()
#                                         ca_ukf_pred.P = ca_ukf.P.copy()
#                                         bank_pred.x = bank.x.copy()
#                                         for f_pred, f in zip(bank_pred.filters, bank.filters):
#                                             f_pred.x = f.x.copy()
#                                             f_pred.P = f.P.copy()
                                            
#                                         for i in range(k):
#                                             # make sure covariance matrices are positive semidefinite
#                                             ca_ukf_pred.P = get_near_psd(ca_ukf_pred.P)
#                                             for f in bank_pred.filters:
#                                                 f.P = get_near_psd(f.P)

#                                             ca_ukf_pred.predict()
#                                             bank_pred.predict()

#                                             uxs_ca_pred[i].append(ca_ukf_pred.x.copy())
#                                             uxs_bank_pred[i].append(bank_pred.x.copy())
#                                             probs_bank_pred[i].append(bank_pred.mu.copy())
#                                             uxs_ca_pred_cov[i].append(ca_ukf_pred.P.copy().flatten())
#                                             uxs_bank_pred_cov[i].append(bank_pred.P.copy().flatten())

#                                         k_step_pred_executed = True

#                                 except np.linalg.LinAlgError as e:
#                                     print(f"LinAlgError: {e}")

#                                     # Reset filters
#                                     ukf_initialized = False
                                        
#                                     # Reset filter states
#                                     ca_ukf.x = np.nan * np.ones(dim_x)
#                                     cv_ukf.x = np.nan * np.ones(dim_x)
#                                     bank.x = np.nan * np.ones(dim_x)
#                                     if predict_k_steps:
#                                         ca_ukf_pred.x = np.nan * np.ones(dim_x)
#                                         bank_pred.x = np.nan * np.ones(dim_x)
#                                         bank_pred.mu = np.nan * np.ones(num_filters_in_bank) # IMM probabilities (3 filters)

#                                     # Reset filter covariances
#                                     ca_ukf.P = init_P
#                                     cv_ukf.P = init_P
#                                     bank.P = init_P
#                                     if predict_k_steps:
#                                         ca_ukf_pred.P = init_P
#                                         bank_pred.P = init_P

#                             else:
#                                 total_iterations -= 1 # do not count the iteration if the filters are not initialized
                                
#                         uxs_ca.append(ca_ukf.x.copy())
#                         uxs_cv.append(cv_ukf.x.copy())
#                         uxs_bank.append(bank.x.copy())
#                         probs_bank.append(bank.mu.copy())
#                         uxs_ca_cov.append(ca_ukf.P.copy().flatten())
#                         uxs_bank_cov.append(bank.P.copy().flatten())

#                         if not k_step_pred_executed:
#                             for i in range(k):
#                                 uxs_ca_pred[i].append(ca_ukf.x.copy())
#                                 uxs_bank_pred[i].append(bank.x.copy())
#                                 probs_bank_pred[i].append(bank.mu.copy())
#                                 uxs_ca_pred_cov[i].append(ca_ukf.P.copy().flatten())
#                                 uxs_bank_pred_cov[i].append(bank.P.copy().flatten())

#                         t += t_incr                        
#                         pbar.update()

#                     pbar.close()
#                     print(f"Average loop frequency: {(total_iterations / (time.time() - start_time)):.2f} Hz")

#                     # Create DataFrames with the filtered data
#                     uxs_ca = np.array(uxs_ca)
#                     uxs_cv = np.array(uxs_cv)
#                     uxs_bank = np.array(uxs_bank)
#                     uxs = np.concatenate((uxs_ca, uxs_cv, uxs_bank), axis=1)
#                     probs_bank = np.array(probs_bank)
#                     uxs_ca_cov = np.array(uxs_ca_cov)
#                     uxs_bank_cov = np.array(uxs_bank_cov)
#                     uxs_cov = np.concatenate((uxs_ca_cov, uxs_bank_cov), axis=1)

#                     filtered_data = pd.DataFrame(uxs, index=filt_timestamps, columns=filtered_column_names)
#                     imm_probs = pd.DataFrame(probs_bank, index=filt_timestamps, columns=col_names_prob_imm)
#                     filtered_data_cov = pd.DataFrame(uxs_cov, index=filt_timestamps) # the elements of the flattened covariance matrices are stored in separate anonymous columns

#                     if predict_k_steps:
#                         kstep_pred_data = {}
#                         kstep_pred_imm_probs = {}
#                         kstep_pred_cov = {}

#                         for i in range(k):
#                             uxs_pred = np.concatenate((np.array(uxs_ca_pred[i]), np.array(uxs_bank_pred[i])), axis=1)
#                             uxs_pred_cov = np.concatenate((np.array(uxs_ca_pred_cov[i]), np.array(uxs_bank_pred_cov[i])), axis=1)

#                             kstep_pred_data[i] = pd.DataFrame(uxs_pred, index=filt_timestamps, columns=filtered_pred_column_names)
#                             kstep_pred_imm_probs[i] = pd.DataFrame(np.array(probs_bank_pred[i]), index=filt_timestamps, columns=col_names_prob_imm)
#                             kstep_pred_cov[i] = pd.DataFrame(uxs_pred_cov, index=filt_timestamps) # the elements of the flattened covariance matrices are stored in separate anonymous columns

#                             # Shift the i-step ahead prediction data by i steps
#                             kstep_pred_data[i] = kstep_pred_data[i].shift(+i)
#                             kstep_pred_imm_probs[i] = kstep_pred_imm_probs[i].shift(+i)
#                             kstep_pred_cov[i] = kstep_pred_cov[i].shift(+i)  

#                     # Store filtering results
#                     filtering_results[(k, subject_id, velocity, task)] = {
#                         'filtered_data': filtered_data,
#                         'imm_probs': imm_probs,
#                         'filtered_data_cov': filtered_data_cov
#                     }

#                     # Store k-step prediction results
#                     if predict_k_steps: 
#                         prediction_results[(k, subject_id, velocity, task)] = {
#                             'kstep_pred_data': kstep_pred_data,
#                             'kstep_pred_imm_probs': kstep_pred_imm_probs,
#                             'kstep_pred_cov': kstep_pred_cov
#                         }

#                     print(f"Processed {subject_id} - {velocity} - {task} for {k} steps ahead.\n\n")

#     return measurement_split, filtering_results, prediction_results
### =================================== ###