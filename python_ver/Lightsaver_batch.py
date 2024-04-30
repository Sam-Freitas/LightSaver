import tkinter as tk
from tkinter import simpledialog, filedialog
import os

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Make the window always on top
    root.attributes('-topmost', True)

    detected_opterating_system = os.name

    # Determine the default documents folder based on the operating system
    if detected_opterating_system == 'posix':  # Linux or macOS
        initial_dir = os.path.expanduser("~/Documents")
    elif detected_opterating_system == 'nt':   # Windows
        initial_dir = os.path.join(os.path.expanduser("~"), "Documents")
    else:
        initial_dir = os.getcwd()  # Fallback to current directory

    directory = filedialog.askdirectory(initialdir=initial_dir)
    return directory

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

# Get user inputs
inputs = get_user_inputs()

# Print out the inputs
print("User inputs:")
for input_ in inputs:
    print(input_)

# This is for the exported images
# Faster is with the jpg format -> 0 but less quality on the images
high_quality_output = 1
if high_quality_output:
    output_img_format = '.png'
else:
    output_img_format = '.jpg'

# Get current path
curr_path = os.getcwd()

selected_directory = get_directory()
print("Selected directory:", selected_directory)
