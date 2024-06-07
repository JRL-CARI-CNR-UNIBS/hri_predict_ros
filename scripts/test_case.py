import unittest
import numpy as np
import pandas as pd

# Assuming the function compute_avg_perc is already imported
# from the module where it is defined

# Compute the percentage of k-step ahead samples that fall within the band current filtered state +- 1*std
def compute_avg_perc(filt, kpred, kpred_var, ca_states, imm_states, ca_variance_idxs, imm_variance_idxs):
    ca_filtered_state = filt[ca_states]

    print("ca_variance_idxs: ", ca_variance_idxs)
    
    ca_std = kpred_var[ca_variance_idxs].apply(np.sqrt)
    ca_std.columns = ca_states

    print("ca_std: ", ca_std)

    ca_pred_lcl = kpred[ca_states] - 1 * ca_std
    ca_pred_ucl = kpred[ca_states] + 1 * ca_std

    print("ca_pred_lcl: ", ca_pred_lcl)
    print("ca_pred_ucl: ", ca_pred_ucl)

    ca_in_CI_band = (ca_filtered_state >= ca_pred_lcl) & (ca_filtered_state <= ca_pred_ucl)
    
    print("ca_in_CI_band: ", ca_in_CI_band)
    
    ca_perc = 100 * np.sum(ca_in_CI_band) / len(filt)

    print("ca_perc: ", ca_perc)

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

    return ca_perc, imm_perc


def get_var_idx(n_dim_per_kpt, n_var_per_dof, dim_x, keypoints, task):
    ca_variance_idxs = []
    for kpt in keypoints[task]:
        for dim in ['x', 'y', 'z']:
            state_idx = ['x', 'xd', 'xdd', 'y', 'yd', 'ydd', 'z', 'zd', 'zdd'].index(dim) + n_var_per_dof * n_dim_per_kpt * kpt
            variance_idx = dim_x * state_idx + state_idx
            ca_variance_idxs.append(variance_idx)

    return ca_variance_idxs


class TestComputeAvgPerc(unittest.TestCase):
    def test_compute_avg_perc(self):
        # Sample input data
        filt = pd.DataFrame({
            'ca_state1': [1, 2, 3, 4, 5],
            'ca_state2': [2, 3, 4, 5, 6],
            'imm_state1': [1, 2, 1, 2, 1],
            'imm_state2': [2, 3, 2, 3, 2]
        })
        
        kpred = pd.DataFrame({
            'ca_state1': [1.5, 2.5, 3.5, 4.5, 5.5],
            'ca_state2': [2.5, 3.5, 4.5, 5.5, 6.5],
            'imm_state1': [1.5, 2.5, 1.5, 2.5, 1.5],
            'imm_state2': [2.5, 3.5, 2.5, 3.5, 2.5]
        })
        
        kpred_var = pd.DataFrame({
            'ca_var1': [0.25, 0.25, 0.25, 0.25, 0.25],
            'ca_var2': [0.25, 0.25, 0.25, 0.25, 0.25],
            'imm_var1': [0.25, 0.25, 0.25, 0.25, 0.25],
            'imm_var2': [0.25, 0.25, 0.25, 0.25, 0.25]
        })

        ca_states = ['ca_state1', 'ca_state2']
        imm_states = ['imm_state1', 'imm_state2']
        ca_variance_idxs = []
        imm_variance_idxs = ['imm_var1', 'imm_var2']
        
        # Expected output
        expected_ca_perc = 100.0
        expected_imm_perc = 100.0

        # Running the function
        ca_perc, imm_perc = compute_avg_perc(filt, kpred, kpred_var, ca_states, imm_states, ca_variance_idxs, imm_variance_idxs)

        # Asserting the output
        self.assertAlmostEqual(ca_perc, expected_ca_perc, places=1)
        self.assertAlmostEqual(imm_perc, expected_imm_perc, places=1)


    def test_get_var_idx(self):
        # Sample input data
        n_dim_per_kpt = 3  # For example, 3 dimensions per keypoint (x, y, z)
        n_var_per_dof = 3  # For example, 3 variances per degree of freedom (dof)
                
        keypoints = {
            'task1': [0, 1, 2],  # Keypoints for task1
            'task2': [3, 4]      # Keypoints for task2
        }
        
        task = 'task1'  # Task we are calculating indices for
        dim_x = len(keypoints[task]) * n_dim_per_kpt * n_var_per_dof  # Total number of dimensions (x, y, z) for the task (e.g. 27)

        # Expected output for task1
        expected_ca_variance_idxs = [
            0,  # For kpt=0, dim='x'
            3*(dim_x+1), # For kpt=0, dim='y'
            6*(dim_x+1), # For kpt=0, dim='z'
            9*(dim_x+1),  # For kpt=1, dim='x'
            12*(dim_x+1), # For kpt=1, dim='y'
            15*(dim_x+1), # For kpt=1, dim='z'
            18*(dim_x+1),  # For kpt=2, dim='x'
            21*(dim_x+1), # For kpt=2, dim='y'
            24*(dim_x+1)  # For kpt=2, dim='z'
        ]

        # Running the function
        ca_variance_idxs = get_var_idx(n_dim_per_kpt, n_var_per_dof, dim_x, keypoints, task)

        # Asserting the output
        self.assertEqual(ca_variance_idxs, expected_ca_variance_idxs)

if __name__ == '__main__':
    unittest.main()
