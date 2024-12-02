# Tuning parameters
iter_P = 1
iter_q = 10
decrement_factor_q = 0.75
decrement_factor_P = 0.5


def parameter_tuning(training_subjects, velocities, tasks, pred_horizons, keypoints, dim_per_kpt,
                     init_P, var_r, var_q, decrement_factor_q, decrement_factor_P, n_iter_P, n_iter_q,
                     n_var_per_dof, n_dim_per_kpt, dim_x, n_kpts,
                     var_P_pos, var_P_vel, var_P_acc):
    
    var_P_acc_init = var_P_acc
    
    for i in range(n_iter_q):
        for j in range(n_iter_P):
            print(f"Iteration {i+1}/{n_iter_q} for var_q and {j+1}/{n_iter_P} for var_P")
            print(f"Current values: var_q = {var_q}, var_P_acc = {var_P_acc}")

            init_P = initialize_P(n_dim_per_kpt, n_kpts, var_P_pos, var_P_vel, var_P_acc)

            _, filtering_results, prediction_results = run_filtering_loop(training_subjects, velocities, tasks, pred_horizons,
                                                                        init_P, var_r, var_q)

            # Compute the average error between the filtered states and the k-step ahead predictions
            avg_errors = {'PICK-&-PLACE': {'CA': 0, 'IMM': 0},
                        'WALKING': {'CA': 0, 'IMM': 0},
                        'PASSING-BY': {'CA': 0, 'IMM': 0}}

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
                                for dim in dim_per_kpt[kpt]: # ['x', 'y', 'z']:
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
                            
                            avg_errors[task]['CA'] += ca_error
                            avg_errors[task]['IMM'] += imm_error
                            avg_rmse[task]['CA'] += ca_rmse
                            avg_rmse[task]['IMM'] += imm_rmse
                            avg_std[task]['CA'] += ca_std
                            avg_std[task]['IMM'] += imm_std
                            avg_perc[task]['CA'] += ca_perc
                            avg_perc[task]['IMM'] += imm_perc

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
                avg_errors[task]['CA'] = np.mean(avg_errors[task]['CA'])
                avg_errors[task]['IMM'] = np.mean(avg_errors[task]['IMM'])
                avg_rmse[task]['CA'] = np.mean(avg_rmse[task]['CA'])
                avg_rmse[task]['IMM'] = np.mean(avg_rmse[task]['IMM'])
                avg_std[task]['CA'] = np.mean(avg_std[task]['CA'])
                avg_std[task]['IMM'] = np.mean(avg_std[task]['IMM'])
                avg_perc[task]['CA'] = np.mean(avg_perc[task]['CA'])
                avg_perc[task]['IMM'] = np.mean(avg_perc[task]['IMM'])

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


# Tune the *var_q* parameter. Consider the worst-case scenario, namely:
# - FAST velocity
# - LARGEST prediction horizon

tic = time.time()
train_subjects = train_subjects
velocities = ['FAST'] # worst-case scenario
tasks = TASK_NAMES
pred_horizons = [5] # worst-case scenario
parameter_tuning(train_subjects, ['FAST'], tasks, pred_horizons, keypoints, dimensions_per_keypoint,
                init_P, var_r, var_q, decrement_factor_q, decrement_factor_P, iter_P, iter_q,
                n_var_per_dof, n_dim_per_kpt, dim_x, n_kpts,
                var_P_pos, var_P_vel, var_P_acc)
toc = time.time()

minutes, seconds = divmod(toc - tic, 60)
print(f"Parameter tuning took {minutes:.0f} minutes and {seconds:.2f} seconds.")