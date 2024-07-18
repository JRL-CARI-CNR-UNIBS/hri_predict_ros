import os, rospkg, rosbag, pickle
import numpy as np
import pandas as pd
from tf import transformations

pd.options.mode.chained_assignment = None  # default='warn'

# ====================================================================================================
print("\n1 / 5. Define parameters and load data...")

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
rotation_matrix_world_camera = transformations.quaternion_matrix(quaternion_world_camera)

# Create a translation matrix
translation_matrix_world_camera = transformations.translation_matrix(translation_world_camera)

# Combine the rotation and translation to get the transformation matrix from the world frame to the camera frame
cam_to_world_matrix = transformations.concatenate_matrices(
    translation_matrix_world_camera,
    rotation_matrix_world_camera
)

human_meas_names = ['human_kp{}_{}'.format(i, suffix)
                    for i in range(n_kpts)
                    for suffix in ['x', 'y', 'z']]

# ====================================================================================================
print("\n2 / 5. Load rosbag data...")

# Define the path to the bag directory
bag_dir = os.path.join(package_path, 'logs', 'bag')

bag_files = os.listdir(bag_dir)
bag_files = [os.path.join(bag_dir, bag_file) for bag_file in bag_files]

bag_data = {}
for bag_file in bag_files:
    with rosbag.Bag(bag_file, 'r') as bag:
        rows_list = []
        for topic, msg, t in bag.read_messages(topics=topic_names): # type: ignore
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

# ====================================================================================================
print("\n3 / 5. Preprocess measurement data...")

measurement_data = {}
for subject, bag in bag_data.items():
    data = pd.DataFrame(bag, columns=['timestamp', 'human_meas'])

    # split columns into separate columns
    for c in data.columns.values:
        data = pd.concat([data, data.pop(c).apply(pd.Series).add_prefix(c+"_")], axis=1)

    # change column names
    data.columns = ['timestamp'] + human_meas_names

    # Convert the 'timestamp' column to a TimeDeltaIndex
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

    # Increase the timestamp by 2 hours to match the system time
    data['timestamp'] = data['timestamp'] + pd.Timedelta(hours=2)

    # Resample the DataFrame to a known frequency
    measurement_data[subject] = data

# ====================================================================================================
print("\n4 / 5. Load GUI data and extract trigger times...")

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

# Extract trigger times from the GUI data
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

# ====================================================================================================
print("\n5 / 5. Save measurement and trigger data to file...")

preprocessed_dir = os.path.join(package_path, 'logs', 'preprocessed')

if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)

with open(os.path.join(preprocessed_dir, 'bag_data.pkl'), 'wb') as f:
    pickle.dump(bag_data, f)

with open(os.path.join(preprocessed_dir, 'measurement_data.pkl'), 'wb') as f:
    pickle.dump(measurement_data, f)

with open(os.path.join(preprocessed_dir, 'trigger_data.pkl'), 'wb') as f:
    pickle.dump(trigger_data, f)