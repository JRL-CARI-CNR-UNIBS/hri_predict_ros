import os, rospkg, pickle, copy, time
import numpy as np
import pandas as pd
import nstep_ukf_imm_estimator as ukf_predictor

pd.options.mode.chained_assignment = None  # default='warn'

# ====================================================================================================
print("\n1 / 5. Load preprocessed data...")

# Get the path to the package this script is in
rospack = rospkg.RosPack()
package_path = rospack.get_path('hri_predict_ros')

# Load preprocessed data from pickle files
preprocessed_dir = os.path.join(package_path, 'logs', 'preprocessed')

with open(os.path.join(preprocessed_dir, 'bag_data.pkl'), 'rb') as f:
    bag_data = pickle.load(f)

with open(os.path.join(preprocessed_dir, 'measurement_data.pkl'), 'rb') as f:
    measurement_data = pickle.load(f)

with open(os.path.join(preprocessed_dir, 'trigger_data.pkl'), 'rb') as f:
    trigger_data = pickle.load(f)


# ====================================================================================================
print("\n2 / 5. Define a function to tune parameters...")

def parameter_tuning(trigger_data, measurement_data, training_subjects,
                     velocities, tasks, keypoints, dim_name_per_kpt, pred_horizons,
                     init_P, var_r, var_q, decrement_factor_q, decrement_factor_P, n_iter_P, n_iter_q,
                     n_var_per_dof, n_dim_per_kpt, dim_x, n_kpts,
                     var_P_pos, var_P_vel, var_P_acc):
    
    var_P_acc_init = var_P_acc
    
    for i in range(n_iter_q):
        for j in range(n_iter_P):
            print(f"Iteration {i+1}/{n_iter_q} for var_q and {j+1}/{n_iter_P} for var_P")
            print(f"Current values: var_q = {var_q}, var_P_acc = {var_P_acc}")

            init_P = ukf_predictor.initialize_P(n_var_per_dof, n_dim_per_kpt, n_kpts, var_P_pos, var_P_vel, var_P_acc)

            _, filtering_results, prediction_results = ukf_predictor.run_filtering_loop(
                trigger_data, measurement_data,
                training_subjects, velocities, tasks, pred_horizons, predict_k_steps,
                dim_x, dim_z, p_idx, dt,
                n_var_per_dof, n_dim_per_kpt, n_kpts,
                init_P, var_r, var_q,
                mu, M, num_filters_in_bank,
                ukf_predictor.custom_inv, max_time_no_meas,
                filtered_column_names, filtered_pred_column_names, col_names_prob_imm
            )

            # Compute the average error between the filtered states and the k-step ahead predictions
            avg_errors = {'PICK-&-PLACE': {'CA': 0.0, 'IMM': 0.0},
                          'WALKING':      {'CA': 0.0, 'IMM': 0.0},
                          'PASSING-BY':   {'CA': 0.0, 'IMM': 0.0}}

            # Compute the average RMSE between the filtered states and the k-step ahead predictions
            avg_rmse = copy.deepcopy(avg_errors)
            
            # Compute the average standard deviation of the filtered states and the k-step ahead predictions
            avg_std = copy.deepcopy(avg_errors)
            
            # Compute the percentage of k-step ahead samples that fall within the band current filtered state +- 1*std
            avg_perc = copy.deepcopy(avg_errors)
            
            for k in pred_horizons:
                for subject_id in training_subjects:
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
                                for dim in dim_name_per_kpt[kpt]:
                                    ca_states.append('ca_kp{}_{}'.format(kpt, dim))
                                    imm_states.append('imm_kp{}_{}'.format(kpt, dim))

                                    state_idx = ['x', 'xd', 'xdd', 'y', 'yd', 'ydd', 'z', 'zd', 'zdd'].index(dim) + n_var_per_dof * n_dim_per_kpt * kpt
                                    ca_variance_idx = dim_x * state_idx + state_idx
                                    imm_variance_idx = dim_x * state_idx + state_idx + dim_x * dim_x
                                    ca_variance_idxs.append(ca_variance_idx)
                                    imm_variance_idxs.append(imm_variance_idx)
                            
                            ca_error, imm_error = ukf_predictor.compute_mean_error(filt, kpred, ca_states, imm_states)
                            ca_rmse, imm_rmse = ukf_predictor.compute_rmse_error(filt, kpred, ca_states, imm_states)
                            ca_std, imm_std = ukf_predictor.compute_std_error(filt, kpred, ca_states, imm_states)
                            ca_perc, imm_perc = ukf_predictor.compute_avg_perc(filt, kpred, kpred_var,
                                                                               ca_states, imm_states, ca_variance_idxs, imm_variance_idxs)
                           
                            avg_errors[task]['CA']  += ca_error
                            avg_errors[task]['IMM'] += imm_error
                            avg_rmse[task]['CA']    += ca_rmse
                            avg_rmse[task]['IMM']   += imm_rmse
                            avg_std[task]['CA']     += float(ca_std)
                            avg_std[task]['IMM']    += float(imm_std)
                            avg_perc[task]['CA']    += ca_perc
                            avg_perc[task]['IMM']   += imm_perc

            # Compute the average values of these aggregated metrics
            num_sums = len(pred_horizons) * len(training_subjects) * len(velocities)
            for task in tasks:
                avg_errors[task]['CA'] /= num_sums
                avg_errors[task]['IMM'] /= num_sums
                avg_rmse[task]['CA'] /= num_sums
                avg_rmse[task]['IMM'] /= num_sums
                avg_std[task]['CA'] /= num_sums
                avg_std[task]['IMM'] /= num_sums
                avg_perc[task]['CA'] /= num_sums
                avg_perc[task]['IMM'] /= num_sums

                # Average over all selected keypoints
                avg_errors[task]['CA']  = float(np.mean(avg_errors[task]['CA']))
                avg_errors[task]['IMM'] = float(np.mean(avg_errors[task]['IMM']))
                avg_rmse[task]['CA']    = float(np.mean(avg_rmse[task]['CA']))
                avg_rmse[task]['IMM']   = float(np.mean(avg_rmse[task]['IMM']))
                avg_std[task]['CA']     = float(np.mean(avg_std[task]['CA']))
                avg_std[task]['IMM']    = float(np.mean(avg_std[task]['IMM']))
                avg_perc[task]['CA']    = float(np.mean(avg_perc[task]['CA']))
                avg_perc[task]['IMM']   = float(np.mean(avg_perc[task]['IMM']))

            # Display the results aggregating results for all keypoints
            print("Average error (CA): {:.6f}, {:.6f}, {:.6f}".format(
                avg_errors['PICK-&-PLACE']['CA'], avg_errors['WALKING']['CA'], avg_errors['PASSING-BY']['CA']))
            print("Average error (IMM): {:.6f}, {:.6f}, {:.6f}".format(
                avg_errors['PICK-&-PLACE']['IMM'], avg_errors['WALKING']['IMM'], avg_errors['PASSING-BY']['IMM']))
            print("Average RMSE (CA): {:.6f}, {:.6f}, {:.6f}".format(
                avg_rmse['PICK-&-PLACE']['CA'], avg_rmse['WALKING']['CA'], avg_rmse['PASSING-BY']['CA']))
            print("Average RMSE (IMM): {:.6f}, {:.6f}, {:.6f}".format(
                avg_rmse['PICK-&-PLACE']['IMM'], avg_rmse['WALKING']['IMM'], avg_rmse['PASSING-BY']['IMM']))
            print("Average std (CA): {:.4f}, {:.4f}, {:.4f}".format(
                avg_std['PICK-&-PLACE']['CA'], avg_std['WALKING']['CA'], avg_std['PASSING-BY']['CA']))
            print("Average std (IMM): {:.4f}, {:.4f}, {:.4f}".format(
                avg_std['PICK-&-PLACE']['IMM'], avg_std['WALKING']['IMM'], avg_std['PASSING-BY']['IMM']))
            print("Average percentage (CA): {:.4f}, {:.4f}, {:.4f}".format(
                avg_perc['PICK-&-PLACE']['CA'], avg_perc['WALKING']['CA'], avg_perc['PASSING-BY']['CA']))
            print("Average percentage (IMM): {:.4f}, {:.4f}, {:.4f}".format(
                avg_perc['PICK-&-PLACE']['IMM'], avg_perc['WALKING']['IMM'], avg_perc['PASSING-BY']['IMM']))
            print("===============================================\n\n")

            # Update the init_P parameter
            var_P_acc *= decrement_factor_P

        # Update the var_q parameter
        var_q *= decrement_factor_q

        # Reset the var_P_acc parameter to its initial value
        var_P_acc = var_P_acc_init


# ====================================================================================================
print("\n3 / 5. Define parameters and constants...")

# Define velocity and task names
VELOCITIES = ['SLOW', 'MEDIUM', 'FAST']
TASK_NAMES = ['PICK-&-PLACE', 'WALKING', 'PASSING-BY']

# Define which keypoints to consider for each task
KEYPOINTS = {'PICK-&-PLACE': [4, 7], # [2, 3, 4, 5, 6, 7],
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

# Define possible prediction horizons
PRED_HORIZONS = [1, 3, 5]

# Subject IDS
subject_ids = trigger_data.keys()
print("Subjects: ", subject_ids)

# Split subjects into train and test
train_subjects = ['sub_9', 'sub_4', 'sub_11', 'sub_7', 'sub_8', 'sub_10', 'sub_6']
test_subjects = ['sub_13', 'sub_12', 'sub_3']

# Filter parameters
dt = 0.1
predict_k_steps = True
n_kpts = 18
n_var_per_dof = 3       # position, velocity, acceleration
n_dim_per_kpt = 3       # x, y, z
dim_x = n_var_per_dof * n_dim_per_kpt * n_kpts
dim_z = n_dim_per_kpt * n_kpts
max_time_no_meas = pd.Timedelta(seconds=1.0)

# Initial uncertainty parameters for the filters
var_r = 0.0025          # [paper: r_y] Hip: 3-sigma (99.5%) = 0.15 m ==> sigma 0.05 m ==> var = (0.05)^2 m^2
var_q = 0.20            # [paper: q_a] a_dot = u (u = 0 is very uncertain ==> add variance here)
var_P_pos = var_r       # [paper: p_y] Set equal to the measurement noise since the state is initialized with the measurement
var_P_vel = 0.02844     # [paper: p_v] Hip: no keypoint moves faster than 1.6 m/s ==> 3-sigma (99.5%) = 1.6 m/s ==> var = (1.6/3)^2 m^2/s^2
var_P_acc = 1.1111      # [paper: p_a] Hip: no keypoint accelerates faster than 10 m/s^2 ==> 3-sigma (99.5%) = 10 m/s^2 ==> var = (10/3)^2 m^2/s^4
init_P = ukf_predictor.initialize_P(n_var_per_dof, n_dim_per_kpt, n_kpts, var_P_pos, var_P_vel, var_P_acc)

# Parameters for the IMM estimator
num_filters_in_bank = 3
M = np.array([[0.55, 0.15, 0.30], # transition matrix for the IMM estimator
              [0.15, 0.75, 0.10],
              [0.60, 0.30, 0.10]])

mu = np.array([0.55, 0.40, 0.05]) # initial mode probabilities for the IMM estimator

# Tuning parameters
iter_P = 5
iter_q = 10
decrement_factor_q = 0.75
decrement_factor_P = 0.5

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

# Define directory to store csv results
results_dir = os.path.join(package_path, 'logs', 'results')


# ====================================================================================================
print("\n4 / 5. Tune the *var_q* parameter. Consider the worst-case scenario, namely:\n \
       \t- FAST velocity\n \
       \t- LARGEST prediction horizon for the IDENTIFICATION subjects...")

tic = time.time()

velocities = ['FAST']       # worst-case scenario
pred_horizons = [5]         # worst-case scenario
tasks = TASK_NAMES          # consider all tasks to get a general idea of the performance
subjects = train_subjects   # consider only the IDENTIFICATION subjects

parameter_tuning(trigger_data, measurement_data, subjects,
                 velocities, tasks, KEYPOINTS, DIMENSIONS_PER_KEYPOINT, pred_horizons,
                 init_P, var_r, var_q, decrement_factor_q, decrement_factor_P, iter_P, iter_q,
                 n_var_per_dof, n_dim_per_kpt, dim_x, n_kpts,
                 var_P_pos, var_P_vel, var_P_acc)
toc = time.time()

minutes, seconds = divmod(toc - tic, 60)
print(f"Parameter tuning took {minutes:.0f} minutes and {seconds:.2f} seconds.")