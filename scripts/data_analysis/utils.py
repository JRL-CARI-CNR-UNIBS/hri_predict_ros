import numpy as np
import plotly.express as px
import os
import pandas as pd

DEFAULT_PLOTLY_COLORS=['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                       'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                       'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                       'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                       'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

# Add transparency to the colors
alpha = 0.2
DEFAULT_PLOTLY_COLORS_ALPHA = [color.replace('rgb', 'rgba').replace(')', f', {alpha})') for color in DEFAULT_PLOTLY_COLORS] # used for 5-step ahead prediction CI

alpha = 0.1
DEFAULT_PLOTLY_COLORS_ALPHA_HIGH = [color.replace('rgb', 'rgba').replace(')', f', {alpha})') for color in DEFAULT_PLOTLY_COLORS] # used for covariance cones


# Select a portion of the dataset based on the subjects,
# instructions,velocities and the configuration names
def select_trajectory_dataset(df,
                              subjects,
                              instructions,
                              velocities,
                              task_names,
                              column_names) -> tuple:
    X_list = []
    time_list = []
    traj_idx = []

    counter = 0
    for sub in subjects:
        for task in task_names:
            for instruction in instructions[task]:
                for velocity in velocities:
                    selection_filter = (df['Subject'] == sub) & \
                                       (df['Instruction_id'] == instruction) & \
                                       (df['Velocity'] == velocity) & \
                                       (df['Task_name'] == task)
                    
                    X = df[selection_filter][column_names].values
                    X_list.append(X)

                    time = df[selection_filter]['Time'].values
                    time_list.append(time)

                    iter_data = dict(counter=counter,
                                    subject=sub,
                                    instruction=instruction,
                                    velocity=velocity,
                                    task=task)
                    traj_idx.append(iter_data)

                    counter += 1

                    print(f'Trajectory {counter} (# samples: {X.shape[0]}):\t',
                        f'Subject {sub}, Instruction {instruction}, Velocity {velocity}, Task {task}')
                
    return X_list, time_list, traj_idx


def plot_time_series(subject, velocity, task, kpt, dim, description,
                     dim_type, k, n_var_per_dof, n_dim_per_kpt,
                     dim_x, dt, plot_dir,
                     measurements, filtering_results, prediction_results,
                     predict_k_steps=False):
    
    state = 'kp{}_{}'.format(kpt, dim)
    state_idx = ['x', 'xd', 'xdd', 'y', 'yd', 'ydd', 'z', 'zd', 'zdd'].index(dim) + n_var_per_dof * n_dim_per_kpt * kpt
    variance_idx = dim_x * state_idx + state_idx

    if dim_type == 'pos':
        meas = measurements[(k, subject, velocity, task)].copy()
        meas_seconds = (meas["timestamp"] - meas["timestamp"].iloc[0]).dt.total_seconds()

    filt = filtering_results[(k, subject, velocity, task)]['filtered_data']
    filt_seconds = (filt.index - filt.index[0]).total_seconds()

    # Display the k-step ahead prediction (only last step)
    if predict_k_steps:
        kpred = prediction_results[(k, subject, velocity, task)]['kstep_pred_data'][k-1]
        kpred_seconds = (kpred.index - kpred.index[0]).total_seconds()
        kpred_variance = prediction_results[(k, subject, velocity, task)]['kstep_pred_cov'][k-1]

        # CA K-step ahead prediction
        std = kpred_variance.iloc[:, variance_idx].apply(np.sqrt)

        # Create pandas Series for the upper and lower confidence limits (1-sigma)
        kpred['_'.join(['ca', state, 'ucl'])] = kpred['_'.join(['ca', state])] + 1 * std
        kpred['_'.join(['ca', state, 'lcl'])] = kpred['_'.join(['ca', state])] - 1 * std

        # IMM K-step ahead prediction
        # (the index must be increased by the dimension of the flattened covariance matrix according to the way covariance matrices are stored)
        std = kpred_variance.iloc[:, variance_idx + dim_x*dim_x].apply(np.sqrt)

        # Create pandas Series for the upper and lower confidence limits (1-sigma)
        kpred['_'.join(['imm', state, 'ucl'])] = kpred['_'.join(['imm', state])] + 1 * std
        kpred['_'.join(['imm', state, 'lcl'])] = kpred['_'.join(['imm', state])] - 1 * std

    fig = px.line()
    if dim_type == 'pos':
        fig.add_scatter(x=meas_seconds,
                        y=meas['_'.join(('human',state))],
                        mode='lines+markers',
                        name='Measurements',
                        line=dict(color=DEFAULT_PLOTLY_COLORS[0])
        )
    fig.add_scatter(x=filt_seconds,
                    y=filt['_'.join(('ca',state))],
                    mode='lines+markers',
                    name='UKF CA',
                    line=dict(color=DEFAULT_PLOTLY_COLORS[1])
    )
    fig.add_scatter(x=filt_seconds,
                    y=filt['_'.join(('cv',state))],
                    mode='lines+markers',
                    name='UKF CV',
                    line=dict(color=DEFAULT_PLOTLY_COLORS[5])
    )
    fig.add_scatter(x=filt_seconds,
                    y=filt['_'.join(('imm',state))],
                    mode='lines+markers',
                    name='UKF IMM',
                    line=dict(color=DEFAULT_PLOTLY_COLORS[3])
    )
    if predict_k_steps:
        fig.add_scatter(x=kpred_seconds,
                        y=kpred['_'.join(('ca',state))],
                        mode='lines+markers',
                        name=f'CA {k}-step ahead prediction',
                        line=dict(color=DEFAULT_PLOTLY_COLORS[4])
        )
        fig.add_scatter(x=kpred_seconds,
                        y=kpred['_'.join(('imm',state))],
                        mode='lines+markers',
                        name=f'IMM {k}-step ahead prediction',
                        line=dict(color=DEFAULT_PLOTLY_COLORS[2])
        )
        fig.add_scatter(x=kpred_seconds,
                        y=kpred['_'.join(['ca', state, 'ucl'])],
                        mode='lines',
                        name=f'CA {k}-step ahead prediction (UCL)',
                        marker=dict(color=DEFAULT_PLOTLY_COLORS_ALPHA[4]),
                        line=dict(width=0),
                        fillcolor=DEFAULT_PLOTLY_COLORS_ALPHA[4],
        )
        fig.add_scatter(x=kpred_seconds,
                        y=kpred['_'.join(['ca', state, 'lcl'])],
                        mode='lines',
                        name=f'CA {k}-step ahead prediction (LCL)',
                        marker=dict(color=DEFAULT_PLOTLY_COLORS_ALPHA[4]),
                        line=dict(width=0),
                        fillcolor=DEFAULT_PLOTLY_COLORS_ALPHA[4],
                        fill='tonexty'
        )
        fig.add_scatter(x=kpred_seconds,
                        y=kpred['_'.join(['imm', state, 'ucl'])],
                        mode='lines',
                        name=f'IMM {k}-step ahead prediction (UCL)',
                        marker=dict(color=DEFAULT_PLOTLY_COLORS_ALPHA[2]),
                        line=dict(width=0),
                        fillcolor=DEFAULT_PLOTLY_COLORS_ALPHA[2]
        )
        fig.add_scatter(x=kpred_seconds,
                        y=kpred['_'.join(['imm', state, 'lcl'])],
                        mode='lines',
                        name=f'IMM {k}-step ahead prediction (LCL)',
                        marker=dict(color=DEFAULT_PLOTLY_COLORS_ALPHA[2]),
                        line=dict(width=0),
                        fillcolor=DEFAULT_PLOTLY_COLORS_ALPHA[2],
                        fill='tonexty'
        )

    fig.update_traces(marker=dict(size=2), line=dict(width=1))

    if dim_type == 'pos':
        fig.update_layout(title=description+" position"+f" [{subject}, {velocity}, {task}] "+f" (k={k})",
                          xaxis_title='Time (s)',
                          yaxis_title='Position (m)',
                          hovermode="x")
    elif dim_type == 'vel':
        fig.update_layout(title=description+" velocity"+f" [{subject}, {velocity}, {task}] "+f" (k={k})",
                          xaxis_title='Time (s)',
                          yaxis_title='Velocity (m/s)',
                          hovermode="x")
    elif dim_type == 'acc':
        fig.update_layout(title=description+" acceleration"+f" [{subject}, {velocity}, {task}] "+f" (dt={dt}, k={k})",
                          xaxis_title='Time (s)',
                          yaxis_title='Acceleration (m/s^2)',
                          hovermode="x")
    else:
        raise ValueError("Invalid dimension type. Use 'pos', 'vel', or 'acc'.")

    fig.show()

    # Save the plot to the plots folder in html format
    plot_name = '_'.join([subject, velocity, task, state, "dt", str(dt), "k", str(k), str(dim_type), "pred", str(predict_k_steps)])
    fig.write_html(os.path.join(plot_dir, plot_name+'.html')) # interactive plot
    fig.write_image(os.path.join(plot_dir, plot_name+ '.pdf')) # static plot


# Define state indices for the CA and IMM filters
def compute_state_indices(space, keypoints, task, dim_name_per_kpt,
                          n_var_per_dof, n_dim_per_kpt, dim_x, conf_names):
    ca_states, imm_states, ca_variance_idxs, imm_variance_idxs = [], [], [], []
    if space == 'cartesian':
        for kpt in keypoints[task]:
            for dim in dim_name_per_kpt[kpt]:    
                ca_states.append('ca_kp{}_{}'.format(kpt, dim))
                imm_states.append('imm_kp{}_{}'.format(kpt, dim))
                state_idx = ['x', 'xd', 'xdd', 'y', 'yd', 'ydd', 'z', 'zd', 'zdd'].index(dim) + n_var_per_dof * n_dim_per_kpt * kpt
                
                ca_variance_idx = dim_x * state_idx + state_idx
                imm_variance_idx = dim_x * state_idx + state_idx + dim_x * dim_x
                
                ca_variance_idxs.append(ca_variance_idx)
                imm_variance_idxs.append(imm_variance_idx)

    elif space == 'joint':
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

    else:
        raise ValueError('Invalid space. Choose between "cartesian" or "joint".')
    
    return ca_states, imm_states, ca_variance_idxs, imm_variance_idxs


def plot_covariance_cone(measurements, filtering_results, prediction_results,
                         subject, velocity, task, instruction, kpt_num, kpt_idx, dim, dim_type,
                         dim_x, n_var_per_dof, n_dim_per_kpt, dt, predict_k_steps,
                         k, selected_timestamps, selected_range, filter_type, y_axes_lim, plot_dir,
                         num_sigmas=1):
                    
    # Define the state index and the variance index for the selected keypoint and dimension
    state = 'kp{}_{}'.format(kpt_num, dim)
    state_idx = ['x', 'xd', 'xdd', 'y', 'yd', 'ydd', 'z', 'zd', 'zdd'].index(dim) + n_var_per_dof * n_dim_per_kpt * kpt_idx
    ca_variance_idx = dim_x * state_idx + state_idx
    imm_variance_idx = ca_variance_idx + dim_x * dim_x

    # Select the measurements and the filtering results for the selected subject, velocity, task, and instruction
    meas = measurements[(subject, velocity, task, instruction)]
    filt = filtering_results[(k, subject, velocity, task, instruction)]['filtered_data']
    filt_cov = filtering_results[(k, subject, velocity, task, instruction)]['filtered_data_cov']

    # Select time range between selected_range[0] and selected_range[1]
    meas_cut = meas.loc[selected_range[0]:selected_range[1]]
    meas_seconds = meas_cut.index.total_seconds()
    filt_cut = filt.loc[selected_range[0]:selected_range[1]]
    filt_seconds = filt_cut.index.total_seconds()
    
    # Create an empty Plotly line plot
    fig = px.line()

    # Plot mesurements
    fig.add_scatter(x=meas_seconds,
                    y=meas_cut[state],
                    mode='markers',
                    name='Measurements',
                    line=dict(color=DEFAULT_PLOTLY_COLORS[0]),
                    marker=dict(size=2)
    )
    # Plot filtered data
    fig.add_scatter(x=filt_seconds,
                    y=filt_cut['_'.join((filter_type.lower(), state))],
                    mode='lines',
                    name=' '.join(('UKF', filter_type)),
                    line=dict(color=DEFAULT_PLOTLY_COLORS[1],
                                width=1)
    )
    
    # Display the k-step ahead prediction (only last step)
    next_dataframes = {}

    # Only plot the legend once for the next 5 predictions at time t
    ca_cone_legend_not_plotted = True
    imm_cone_legend_not_plotted = True
    
    for t in selected_timestamps:
        # Initialize lists to store the next k-step ahead states and confidence limits (1-sigma)
        std = np.sqrt(filt_cov.loc[t].iloc[ca_variance_idx])
        next_k_states_ca = [filt.loc[t, '_'.join(['ca', state])]]
        next_k_lcls_ca = [filt.loc[t, '_'.join(['ca', state])] - num_sigmas * std]
        next_k_ucls_ca = [filt.loc[t, '_'.join(['ca', state])] + num_sigmas * std]
        next_k_states_imm = [filt.loc[t, '_'.join(['imm', state])]]
        next_k_lcls_imm = [filt.loc[t, '_'.join(['imm', state])] - num_sigmas * std]
        next_k_ucls_imm = [filt.loc[t, '_'.join(['imm', state])] + num_sigmas * std]
        times = [t]

        for step in range(k):
            # shift the timestamp by step*dt (had been shifted before for visualization purposes)
            t_shifted = t - pd.Timedelta(seconds=(step+1)*dt) 

            kpred = prediction_results[(k, subject, velocity, task, instruction)]['kstep_pred_data'][step]
            kpred_variance = prediction_results[(k, subject, velocity, task, instruction)]['kstep_pred_cov'][step]

            # CA K-step ahead prediction
            std = kpred_variance.iloc[:, ca_variance_idx].apply(np.sqrt)

            # Create pandas Series for the upper and lower confidence limits (1-sigma)
            kpred['_'.join(['ca', state, 'ucl'])] = kpred['_'.join(['ca', state])] + num_sigmas * std
            kpred['_'.join(['ca', state, 'lcl'])] = kpred['_'.join(['ca', state])] - num_sigmas * std

            # IMM K-step ahead prediction
            # (the index must be increased by the dimension of the flattened covariance matrix according to the way covariance matrices are stored)
            std = kpred_variance.iloc[:, imm_variance_idx].apply(np.sqrt)

            # Create pandas Series for the upper and lower confidence limits (1-sigma)
            kpred['_'.join(['imm', state, 'ucl'])] = kpred['_'.join(['imm', state])] + num_sigmas * std
            kpred['_'.join(['imm', state, 'lcl'])] = kpred['_'.join(['imm', state])] - num_sigmas * std

            t_shifted = t + pd.Timedelta(seconds=(step+1)*dt)

            # Store the next k-step ahead states and confidence limits
            next_k_states_ca.append(kpred.loc[t_shifted, '_'.join(['ca', state])])
            next_k_lcls_ca.append(kpred.loc[t_shifted, '_'.join(['ca', state, 'lcl'])])
            next_k_ucls_ca.append(kpred.loc[t_shifted, '_'.join(['ca', state, 'ucl'])])
            next_k_states_imm.append(kpred.loc[t_shifted, '_'.join(['imm', state])])
            next_k_lcls_imm.append(kpred.loc[t_shifted, '_'.join(['imm', state, 'lcl'])])
            next_k_ucls_imm.append(kpred.loc[t_shifted, '_'.join(['imm', state, 'ucl'])])
            
            # create array of timestamps from to t + k*dt
            times.append(t + pd.Timedelta(seconds=(step+1)*dt))

        # Transform the lists into Pandas DataFrames
        next_k_states_ca = pd.DataFrame(next_k_states_ca, index=times, columns=['CA'])
        next_k_lcls_ca = pd.DataFrame(next_k_lcls_ca, index=times, columns=['CA_LCL'])
        next_k_ucls_ca = pd.DataFrame(next_k_ucls_ca, index=times, columns=['CA_UCL'])
        next_k_states_imm = pd.DataFrame(next_k_states_imm, index=times, columns=['IMM'])
        next_k_lcls_imm = pd.DataFrame(next_k_lcls_imm, index=times, columns=['IMM_LCL'])
        next_k_ucls_imm = pd.DataFrame(next_k_ucls_imm, index=times, columns=['IMM_UCL'])

        next_dataframes[t] = {
            'next_k_states_ca': next_k_states_ca,
            'next_k_lcls_ca': next_k_lcls_ca,
            'next_k_ucls_ca': next_k_ucls_ca,
            'next_k_states_imm': next_k_states_imm,
            'next_k_lcls_imm': next_k_lcls_imm,
            'next_k_ucls_imm': next_k_ucls_imm
        }

        next_pred_ca = next_dataframes[t]['next_k_states_ca']['CA']
        next_pred_ca_lcl = next_dataframes[t]['next_k_lcls_ca']['CA_LCL']
        next_pred_ca_ucl = next_dataframes[t]['next_k_ucls_ca']['CA_UCL']
        next_pred_imm = next_dataframes[t]['next_k_states_imm']['IMM']
        next_pred_imm_lcl = next_dataframes[t]['next_k_lcls_imm']['IMM_LCL']
        next_pred_imm_ucl = next_dataframes[t]['next_k_ucls_imm']['IMM_UCL']

        next_pred_seconds = next_dataframes[t]['next_k_states_ca'].index.total_seconds()

        # Select time range before plotting between selected_range[0] and selected_range[1]
        kpred_cut = kpred.loc[selected_range[0]:selected_range[1]]
        kpred_seconds = filt_seconds
    
        # Plot next k-step ahead predictions and the covariance cones
        if filter_type == 'CA':
            fig.add_scatter(x=next_pred_seconds,
                            y=next_pred_ca,
                            mode='lines',
                            name=f'Prediction at current time',
                            line=dict(color=DEFAULT_PLOTLY_COLORS[2],
                                      width=2),
                            showlegend=ca_cone_legend_not_plotted
            )
            ca_cone_legend_not_plotted = False

            fig.add_scatter(x=next_pred_seconds,
                        y=next_pred_ca_lcl,
                        mode='lines',
                        name=f'CA next {k}-step at time {t} (LCL)',
                        marker=dict(color=DEFAULT_PLOTLY_COLORS_ALPHA_HIGH[2]),
                        line=dict(width=0),
                        fillcolor=DEFAULT_PLOTLY_COLORS_ALPHA_HIGH[2],
                        showlegend=False
            )
            fig.add_scatter(x=next_pred_seconds,
                            y=next_pred_ca_ucl,
                            mode='lines',
                            name=f'CA next {k}-step at time {t} (UCL)',
                            marker=dict(color=DEFAULT_PLOTLY_COLORS_ALPHA_HIGH[2]),
                            line=dict(width=0),
                            fillcolor=DEFAULT_PLOTLY_COLORS_ALPHA_HIGH[2],
                            fill='tonexty',
                            showlegend=False
            )
        elif filter_type == 'IMM':
            fig.add_scatter(x=next_pred_seconds,
                            y=next_pred_imm,
                            mode='lines',
                            name=f'Prediction at current time',
                            line=dict(color=DEFAULT_PLOTLY_COLORS[2],
                                      width=2),
                            showlegend=imm_cone_legend_not_plotted
            )
            imm_cone_legend_not_plotted = False
            
            fig.add_scatter(x=next_pred_seconds,
                            y=next_pred_imm_lcl,
                            mode='lines',
                            name=f'IMM next {k}-step at time {t} (LCL)',
                            marker=dict(color=DEFAULT_PLOTLY_COLORS_ALPHA_HIGH[2]),
                            line=dict(width=0),
                            fillcolor=DEFAULT_PLOTLY_COLORS_ALPHA_HIGH[2],
                            showlegend=False
            )
            fig.add_scatter(x=next_pred_seconds,
                            y=next_pred_imm_ucl,
                            mode='lines',
                            name=f'IMM next {k}-step at time {t} (UCL)',
                            marker=dict(color=DEFAULT_PLOTLY_COLORS_ALPHA_HIGH[2]),
                            line=dict(width=0),
                            fillcolor=DEFAULT_PLOTLY_COLORS_ALPHA_HIGH[2],
                            fill='tonexty',
                            showlegend=False
            )
        else:
            raise ValueError("Invalid filter type. Use 'CA' or 'IMM'.")

    # Plot k-step ahead predictions with their confidence intervals
    if predict_k_steps:
        fig.add_scatter(x=kpred_seconds,
                        y=kpred_cut['_'.join((filter_type.lower(), state))],
                        mode='lines',
                        name=f'{k}-step ahead prediction',
                        line=dict(color=DEFAULT_PLOTLY_COLORS[4],
                                    width=1)
        )
        fig.add_scatter(x=kpred_seconds,
                        y=kpred_cut['_'.join([filter_type.lower(), state, 'ucl'])],
                        mode='lines',
                        name=f'{k}-step ahead prediction (UCL)',
                        marker=dict(color=DEFAULT_PLOTLY_COLORS_ALPHA[4]),
                        line=dict(width=0),
                        fillcolor=DEFAULT_PLOTLY_COLORS_ALPHA[4],
                        showlegend=False
        )
        fig.add_scatter(x=kpred_seconds,
                        y=kpred_cut['_'.join([filter_type.lower(), state, 'lcl'])],
                        mode='lines',
                        name=f'{k}-step ahead prediction (LCL)',
                        marker=dict(color=DEFAULT_PLOTLY_COLORS_ALPHA[4]),
                        line=dict(width=0),
                        fillcolor=DEFAULT_PLOTLY_COLORS_ALPHA[4],
                        fill='tonexty',
                        showlegend=False
        )

        # # Add two vertical lines, one at the first element of selected_timestamps and another 0.5 seconds after
        # # (the line must touch the top and the bottom of the plot)
        # fig.add_vrect(x0=selected_timestamps[0].total_seconds(),
        #               x1=(selected_timestamps[0] + pd.Timedelta(seconds=0.5)).total_seconds(),
        #               fillcolor="LightSalmon",
        #               opacity=0.5,
        #               layer="below",
        #               line_width=0
        # )
    
    # Fix task name
    if task == 'PICK-&-PLACE':
        task = 'REACH-TO-GRASP'

    # Customize the title of the plot
    fig.update_layout(title=dict(text=f"Filter: {filter_type}  |  Velocity: {velocity}  |  Task: {task}",
                                 font=dict(size=25),
                                 x=0.5,
                                 y=0.94),
                      xaxis_title=dict(text='Time (s)',
                                       font=dict(size=20)),
                      yaxis_title=dict(text='Position (m)',
                                       font=dict(size=20))
    )

    # Update the legend layout
    fig.update_layout(legend=dict(orientation="v",
                                  yanchor="bottom",
                                  y=0.020,
                                  xanchor="right",
                                  x=0.992,
                                  font=dict(size=16))
    )

    # Update the font type
    fig.update_layout(font_family='Open-Sherif',
                      font_color='black')

    # Update the x-axis layout to match the time range selected by the user        
    fig.update_xaxes(range=[selected_range[0].total_seconds(), selected_range[1].total_seconds()])

    # Update the y-axis to the given limits to improve visualization
    # fig.update_yaxes(range=y_axes_lim,
    #                  title_standoff = 20)

    fig.show()

    # Save the plot to the plots folder in html format
    plot_name = '_'.join([subject, velocity, task, state, "dt", str(dt),
                          "k", str(k), str(dim_type), "pred", str(predict_k_steps),
                          "cone", str(filter_type), "sigmas", str(num_sigmas)])
    fig.write_html(os.path.join(plot_dir, plot_name+ '.html')) # interactive plot
    fig.write_image(os.path.join(plot_dir, plot_name+ '.pdf')) # static plot


class IncrementalCovariance:
    def __init__(self, n_features):
        self.n_features = n_features
        self.sumx = np.zeros(n_features)
        self.sumx2 = np.zeros((n_features, n_features))
        self.n_samples = 0

    def update(self, x):
        self.n_samples += 1
        # If x has 1 dimension
        if x.ndim == 1:
            self.sumx += x
            self.sumx2 += np.outer(x, x)
        # If x has 2 dimensions
        elif x.ndim == 2:
            for i in range(x.shape[0]):
                assert x[i].shape[0] == self.n_features, "The number of features in x does not match the expected number of features."
                self.sumx += x[i]
                self.sumx2 += np.outer(x[i], x[i])
        else:
            raise ValueError("x must have 1 or 2 dimensions.")

    @property
    def mean(self):
        return self.sumx / self.n_samples

    @property
    def cov(self):
        mean = self.mean
        return (self.sumx2 - np.outer(mean, mean) * self.n_samples) / (self.n_samples - 1)