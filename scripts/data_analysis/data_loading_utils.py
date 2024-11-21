# Check if any of the upper_body_frames strings
# are substrings of the names in conf_names_*
def is_in_upper_body(name, list_of_names):
    return any(item in name for item in list_of_names)


# Filter the names for the upper body
def filter_upper_body_joints(names, list_of_names):
    return [name for name in names if is_in_upper_body(name, list_of_names)]


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