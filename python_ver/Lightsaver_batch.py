import tkinter as tk
from tkinter import simpledialog, filedialog, ttk
from fnmatch import fnmatch
from natsort import natsorted
import os, time, cv2

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def clean_img_names(img_paths):

    cleaned_file_names = [os.path.basename(path) for path in img_paths]
    common_prefix = os.path.commonprefix(cleaned_file_names)
    cleaned_file_names = [file_name[len(common_prefix):] for file_name in cleaned_file_names]

    common_suffix = os.path.commonprefix([w[::-1] for w in cleaned_file_names])[::-1]
    cleaned_file_names = [file_name[:-len(common_suffix)] for file_name in cleaned_file_names]

    counter = 0
    for i,this_name in enumerate(cleaned_file_names):
        if this_name in img_paths[i]:
            counter += 1
    assert counter == len(img_paths)

    return cleaned_file_names

def get_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Make the window always on top
    root.attributes('-topmost', True)

    # get the os a current path for data selections
    detected_opterating_system = os.name
    current_path = os.getcwd()

    # find where the lightsaver script is being ran from for data selection
    idx_of_lightsvr = current_path.rfind('LightSaver')

    # if in the defualt folders then do nothing
    if idx_of_lightsvr != -1:
        initial_dir = os.path.join(current_path[:idx_of_lightsvr+11],'data')
    else: # if not the defualt to 'documents' folder 
        # Determine the default documents folder based on the operating system
        if detected_opterating_system == 'posix':  # Linux or macOS
            initial_dir = os.path.expanduser("~/Documents")
        elif detected_opterating_system == 'nt':   # Windows
            initial_dir = os.path.join(os.path.expanduser("~"), "Documents")
        else:
            initial_dir = os.getcwd()  # Fallback to current directory
    
    # make the output path for all the images
    output_path = os.path.join(current_path[:idx_of_lightsvr+11],'exported_images')

    # use the gui to make the data selection
    directory = filedialog.askdirectory(initialdir=initial_dir)

    return directory, output_path

def get_user_inputs():
    root = tk.Tk()
        # Make the window always on top
    root.attributes('-topmost', True)
    root.title("User Inputs for Lightsaver")

    defaults = ['5', '0', '0', '', '1', '1', '1', '0']  # Default values for each input field

    fields = ['Number of worms to detect:',
              'Show output images - yes(1) - no(0):',
              'Use large blob fix - yes (1) - no(0):',
              'Output name - leave blank for defaults - or enter name for exported_images sub-folder:',
              'Remove 001,002, etc, from tif names - yes(1) - no(0) Will overwrite data files:',
              'Export processed images - yes (1) - no (0):',
              'Automatic data analysis and export - yes (1) - no (0):',
              'Does the experiment folder have condition names in it? (ex: 01-1-11_N2_vs_SKN-1) - yes (1) - no (0):']

    entries = []
    for i, field in enumerate(fields):
        row = tk.Frame(root)
        label = tk.Label(row, width=75, text=field, anchor='w')
        entry = tk.Entry(row)
        entry.insert(0, defaults[i])  # Insert default value
        row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        label.pack(side=tk.LEFT)
        entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entries.append(entry)

    def get_input():
        inputs = [e.get() for e in entries]
        root.destroy()  # Close the window
        return inputs

    button = tk.Button(root, text='Submit', command=lambda: root.quit())
    button.pack(side=tk.BOTTOM, padx=5, pady=5)

    root.mainloop()

    return get_input()

def create_progress_window():
    # Create Tkinter window
    root_progressbar = tk.Tk()
    root_progressbar.attributes('-topmost', True)
    root_progressbar.title("Progress Bar")

    # Create progress bar
    progress_bar = ttk.Progressbar(root_progressbar, orient='horizontal', length=1000, mode='determinate')
    progress_bar.pack(pady=10)

    # Create label for displaying text over the progress bar
    label = tk.Label(root_progressbar, text="")
    label.pack()

    return root_progressbar, progress_bar, label

def update_progress_bar(progress_bar, label, current_iteration, total, text=""):
    progress_bar['value'] = (current_iteration / total) * 100
    # print("Progress: {:.1f}%".format(current_iteration / total * 100), end='\r')

    # Update label text
    text = text[:50]  # Limit text length to 50 characters
    text = text.ljust(50)  # Pad text with spaces to ensure consistent display width
    label.config(text=text)

    progress_bar.update()  # Update the progress bar

    time.sleep(0.1)  # Introduce a delay of 0.1 seconds

# Get user inputs
inputs = get_user_inputs()

# This is for the exported images
# Faster is with the jpg format -> 0 but less quality on the images
high_quality_output = 1
if high_quality_output:
    output_img_format = '.png'
else:
    output_img_format = '.jpg'

# Get current path
curr_path = os.getcwd()

# get the selected directory and the first part of the output path for the exported images
selected_directory, output_path = get_directory()
print("Selected directory:", selected_directory)

# get the final save name of the experiment and create the output directory for it
final_save_name = os.path.split(selected_directory)[-1]
output_path = os.path.join(output_path,final_save_name)
os.makedirs(output_path,exist_ok=True)

# step through all the files in the selected data folder to get a list of the tif images paths
pattern = "*.tif"
img_paths = []

# find all the image files and then natsort them
for path, subdirs, files in os.walk(selected_directory):
    for name in files:
        if fnmatch(name, pattern):
            img_paths.append(os.path.join(path, name))
img_paths = natsorted(img_paths)
print('There are',img_paths.__len__(),'files with the *.tif ests.\n')

# check to make sure that there are acutally images in the directory
if img_paths.__len__() == 0:
    exit("No TIF images found")

# this is for the export names, just cleans them up and removes ANYTHING that is the same across all names
img_names = clean_img_names(img_paths)

# Set total iterations and set up the progress bar
root_progressbar, progress_bar, label = create_progress_window()
# update_progress_bar(progress_bar, label, 0, len(img_paths), text= "starting")

for i in range(len(img_paths)):
    print(i,img_names[i])
    update_progress_bar(progress_bar, label, i, len(img_paths), text= "Processing --- " + img_names[i] + '\t' + str(i+1) + '/' + str(len(img_paths)))
    # Update progress bar
    time.sleep(1)
    
# Print completion message after the loop
print("\nWork completed.")

# Close Tkinter window after the loop completes
root_progressbar.destroy()

print('eof')