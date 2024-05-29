import subprocess
import time
import rospkg
import os
import rosbag
import sys
import time

RATE = 10
MAX_FOLDER_SIZE = 10e9 # Bytes

# Create a RosPack object
rospack = rospkg.RosPack()

# Get the path to the package this script is in
package_path = rospack.get_path('hri_predict_ros')

# Paths to the ROS node launch files and bag files
ros_node_launch_file = os.path.join(package_path, 'launch','predictor_bringup.launch')

npz_dir = os.path.join(package_path, 'logs', 'npz')
bag_dir = os.path.join(package_path, 'logs', 'bag')
bag_files = [os.path.join(bag_dir, filename) for filename in os.listdir(bag_dir)]

def get_bag_length(bag_file):
    with rosbag.Bag(bag_file, 'r') as bag:
        return bag.get_end_time() - bag.get_start_time()
    
def get_folder_size(folder):
    total = 0
    for path, dirs, files in os.walk(folder):
        for f in files:
            fp = os.path.join(path, f)
            total += os.path.getsize(fp)
    return total


if __name__ == '__main__':
    for bag_file in bag_files:
        folder_size = get_folder_size(npz_dir)
        if folder_size > MAX_FOLDER_SIZE:
            print(f"\n\nFolder size is greater than {(MAX_FOLDER_SIZE/1e3):.0f} kB. Stopping script.\n")
            sys.exit(1)
    
        print(f"Processing bag file: {bag_file}")
        # Get the length of the bag file
        bag_length = get_bag_length(bag_file)
        print(f"File length: {bag_length/60:.2f} minutes")

        # Get the subject number from the bag file name
        sub_number = bag_file.split('_')[-1].split('.')[0]

        # Launch the ROS node
        command = f"roslaunch {ros_node_launch_file} offline:=true sub_number:={sub_number}"
        subprocess.run(["gnome-terminal", "--", "bash", "-c", command])

        # Play the bag file
        command = f"rosbag play {bag_file} -r {RATE} --clock"
        subprocess.run(["gnome-terminal", "--", "bash", "-c", command])

        # Wait for the bag file to finish playing
        wait_time = (bag_length+60)*1/RATE
        print(f"Waiting for {wait_time/60:.2f} minutes...")
        
        start_time = time.time()
        while True:
            time.sleep(60*1/RATE)  # Check every minute*1/RATE
            folder_size = get_folder_size(npz_dir) # in bytes
            print(f"Folder size: {folder_size/1e6:.2f} MB")

            if time.time() - start_time >= wait_time:
                break
        
        # Wait for a short period to ensure the process is terminated properly
        print("Waiting for 10 seconds...")
        time.sleep(10)

    # Kill any hanging prediction_node.py instances
    subprocess.run(["pkill", "-f", "prediction_node.py"])

    print("Finished processing all bag files.")

