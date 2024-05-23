import tkinter as tk
from tkinter import ttk
import time
import os
import rospkg
import logging
import argparse


# Task instructions
INSTRUCTIONS = {'PICK-&-PLACE': {0: 'Place BOTH HANDS in home position',
                                 1: 'Reach object 1 with RIGHT HAND',
                                 2: 'Place BOTH HANDS in home position',
                                 3: 'Reach object 2 with LEFT HAND',
                                 4: 'Place BOTH HANDS in home position',
                                 5: 'Reach object 3 with ANY HAND',
                                 6: 'Place BOTH HANDS in home position',
                                 7: 'Reach robot end-effector with BOTH HANDS',
                                 8: 'Place BOTH HANDS in home position'},

                'WALKING'     : {0: 'Stand still in position A',
                                 1: 'Walk to position B parallel to the cell',
                                 2: 'Rotate 180 degrees and stand still',
                                 3: 'Walk to position A parallel to the cell'},

                'PASSING-BY'  : {0: 'Stand still in position C',
                                 1: 'Walk to position D parallel to the cell',
                                 2: 'Walk to position C parallel to the cell',}}

# Execution times in seconds for each task and velocity
VELOCITIES = {'SLOW':   {'PICK-&-PLACE': 5.0, 'WALKING': 4.0, 'PASSING-BY': 10.0}, 
              'MEDIUM': {'PICK-&-PLACE': 3.0, 'WALKING': 3.0, 'PASSING-BY':  7.0},
              'FAST':   {'PICK-&-PLACE': 1.0, 'WALKING': 2.0, 'PASSING-BY':  5.0}}

# Total number of subtasks (each task at each velocity)
N_SUBTASKS = sum(len(task_dict) for task_dict in VELOCITIES.values())

# Define temporal constants
DELAY_PAUSE = 5.0
DELAY_INITIAL = 10.0
DELAY_TASK = 2.0


# Create an argument parser
parser = argparse.ArgumentParser(description="Specify the subject number.")
# Add an argument for the subject number
parser.add_argument("subject", type=int, help="The number of the subject executing the task.")
# Parse the arguments
args = parser.parse_args()

# Create a RosPack object
rospack = rospkg.RosPack()
# Get the path to the package this script is in
package_path = rospack.get_path('hri_predict_ros')
# Define the path to the logs directory
log_dir = os.path.join(package_path, 'logs/gui_data')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Specify the path to the log file
log_file = os.path.join(log_dir,f'gui_log_sub_{args.subject}.txt')

# Open the log file in write mode to overwrite it
with open(log_file, 'w'):
    pass

# Configure the logger
logging.basicConfig(filename=log_file,
                    level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Log the header
logging.info(f"Task_name, Velocity, Instruction_id, Instruction")


def interactive_gui(root, progress_bar, duration, delay=0.0):
    time.sleep(delay)  # Delay before starting the progress bar
    duration = int(duration*100) # 0.01 seconds per iteration

    # Update the progress bar in a loop
    for i in range(duration):
        time.sleep(0.01)  # Sleep for 0.01 second
        progress_bar['value'] = (i+1)/duration*100  # Update progress
        root.update()  # Refresh the window


def main():
    # Create the root window
    root = tk.Tk()
    root.title("HRI Prediction Test")
    root.configure(bg='lightgrey')

    # Center the window
    window_width = 1280
    window_height = 720
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

    # Create label for global status
    status_var = tk.StringVar()
    status_label = tk.Label(root, textvariable=status_var,
                            bg='lightgrey', fg='blue', font=('Helvetica', 60, 'bold'))
    status_label.place(relx=0.5, rely=0.1, anchor='n')

    # Create label for task name
    task_name_var = tk.StringVar()
    task_label = tk.Label(root, textvariable=task_name_var,
                          bg='lightgrey', fg='blue', font=('Helvetica', 40, 'bold'))
    task_label.place(relx=0.5, rely=0.2, anchor='n')

    # Create label for instruction
    instruction_var = tk.StringVar()
    instruction_label = tk.Label(root, textvariable=instruction_var,
                                 bg='lightgrey', fg='green', font=('Helvetica', 40))
    instruction_label.place(relx=0.5, rely=0.5, anchor='n')

    # Create a progress bar
    progress_bar = ttk.Progressbar(root, length=1000, mode='indeterminate')
    progress_bar.place(relx=0.5, rely=0.8, anchor='center')

    # Configure the style of the progress bar
    style = ttk.Style()
    style.configure("TProgressbar", thickness=200, background='green', foreground='green', troughcolor ='grey')

    # Start the timer
    start_time = time.time()

    # Loop over all tasks and execution velocities
    i = 1
    for velocity_id, velocity_values in VELOCITIES.items():
        for task_name, duration in velocity_values.items():

            # Interleave a pause (5.0 seconds) between tasks
            status_var.set(f"Task {i} / {N_SUBTASKS}")
            task_name_var.set("PAUSE")
            
            # Tell the operator to move to next position and get ready
            if task_name == 'PICK-&-PLACE':
                instruction_var.set(f"Move in front of HOME position and get ready for {task_name} task")
            elif task_name == 'WALKING':
                instruction_var.set(f"Move to position A and get ready for {task_name} task")
            elif task_name == 'PASSING-BY':
                instruction_var.set(f"Move to position C and get ready for {task_name} task")
            else:
                instruction_var.set(f"Get ready for {task_name} task")

            # Update the window
            root.update()

            # Display progress bar for the pause
            if i == 1:
                # Initial delay to set up the environment
                interactive_gui(root, progress_bar, DELAY_PAUSE, DELAY_INITIAL)
            else:
                interactive_gui(root, progress_bar, DELAY_PAUSE)
            
            # Loop through all the task instructions
            for instruction_id, instruction in INSTRUCTIONS[task_name].items():
                # Display the task name and instruction
                task_name_var.set(f"{task_name} at {velocity_id} speed")
                instruction_var.set(f"{instruction}")

                # Reset progress bar and update the window
                progress_bar['value'] = 0
                root.update()

                # Display progress bar for the task
                interactive_gui(root, progress_bar, duration, DELAY_TASK)

                # Log the timestamp, task name, velocity, and instruction
                logging.info(f"{task_name}, {velocity_id}, {instruction_id}, {instruction}")

            i += 1

    # Delete progress bar and labels
    progress_bar.destroy()
    status_label.destroy()
    task_label.destroy()
    instruction_label.destroy()

    # Create a label for the completion message
    completion_message_var = tk.StringVar()
    completion_message_var.set("Test completed")
    completion_message_label = tk.Label(root, textvariable=completion_message_var,
                                        bg='lightgrey', fg='green', font=('Helvetica', 100, 'bold'))
    completion_message_label.place(relx=0.5, rely=0.5, anchor='center')

    # End the timer
    end_time = time.time()

    # Calculate the total time
    total_time = end_time - start_time
    print(f"Total time: {int(total_time // 60)}m {int(total_time % 60):.2f}s")
    print(f"Logs written to file: {os.path.join(log_dir,'execution_log.txt')}")

    root.mainloop()


if __name__ == '__main__':
    main()