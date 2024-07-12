import os, rospkg, pickle

# ====================================================================================================
print("1 / 5. Load preprocessed data...")

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

