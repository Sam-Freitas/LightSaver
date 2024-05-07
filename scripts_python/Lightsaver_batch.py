import tkinter as tk
import numpy as np
from tkinter import simpledialog, filedialog, ttk
from fnmatch import fnmatch
from natsort import natsorted
import os, time, cv2, re, pathlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects
from skimage.measure import label

import matplotlib # for some reason using the 'agg' 
matplotlib.use('TkAgg')#)'qtagg')

# https://github.com/AndersDHenriksen/SanityChecker/blob/master/AllChecks.py
def bwareafilt(mask, n=1, area_range=(0, np.inf)):
    """Extract objects from binary image by size """
    # For openCV > 3.0 this can be changed to: areas_num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    labels = label(mask.astype('uint8'), background=0)
    area_idx = np.arange(1, np.max(labels) + 1)
    areas = np.array([np.sum(labels == i) for i in area_idx])
    inside_range_idx = np.logical_and(areas >= area_range[0], areas <= area_range[1])
    area_idx = area_idx[inside_range_idx]
    areas = areas[inside_range_idx]
    keep_idx = area_idx[np.argsort(areas)[::-1][0:n]]
    kept_areas = areas[np.argsort(areas)[::-1][0:n]]
    if np.size(kept_areas) == 0:
        kept_areas = np.array([0])
    if n == 1:
        kept_areas = kept_areas[0]
    kept_mask = np.isin(labels, keep_idx)
    return kept_mask, kept_areas

def matlab_bwareaopen(bw_img, min_size = 64, connectivity = 4):
    return remove_small_objects(bw_img, min_size=min_size, connectivity=connectivity)

# https://stackoverflow.com/questions/58417310/matlabs-imgaussfilt-equivalent-in-python
def matlab_gaussian_filter(image, sigma):
    return gaussian_filter(
        image.astype(float),
        sigma=sigma,
        radius=np.ceil(2 * sigma).astype(int),
    )

def load_fluor_image(img_paths,i, fill_value = 0):
    # this function is an attempt to load in the images as a grayscale image 
    # given list of inputs 'img_paths' and index 'i' it loads and transforms specified image

    # slight difference between matlab and python scripts
    # the uint8 number class wraps around in python eg 1-5 = 252
    # whereas in matlab it hard cuts off 1-5 = 0

    this_img_name = os.path.split(img_paths[i])[-1]

    # try to read in the image and report if there is an error 
    try:
        this_img = cv2.imread(img_paths[i]).astype(np.float64)
    except:
        print(['ERROR: reading image - ', this_img_name])
        print(['Image will be treated as corrupted and skipped'])
        try:
            this_img = np.zeros_like(this_img)
        except:
            this_img = np.zeros(1024,1024)

    # if for some reason that the LUA was also saved at the 4th zpos 
    # this assumes that images will get loaded at x,y,c
    if this_img.shape[-1] > 3:
        this_img = this_img[:,:,0:3]
    
    # check for rgb
    if len(this_img.shape) > 2:

        R,G,B = this_img[:,:,0],this_img[:,:,1],this_img[:,:,2]

        # find the most dominant color
        color_choice = np.argmax([np.sum(R),np.sum(G),np.sum(B)])

        if   color_choice == 0: # red fluorescence
            data = R - G - B
        elif color_choice == 1: # green fluorescence
            data = G - B - R
        elif color_choice == 2: # blue fluorescence 
            data = B - R - G
        else: # otherwise 
            data = np.mean(this_img,axis=-1)

        data = (data*(data>0)).astype(np.uint8)

    # get rid of scale bar
    if np.sum(data) < 1000:
        data = this_img[:,:,color_choice]
        # replace the scale bar with the median of the data (should be the background)
        if fill_value == 0:
            data[-100:-1,0:256] = 0 #np.median(data)
        elif 'str' in type(fill_value):
            data[-100:-1,0:256] = np.median(data)
        else:
            data[-100:-1,0:256] = fill_value #np.median(data)

    return data,this_img

def clean_img_names(img_paths):

    cleaned_file_names = [os.path.basename(path) for path in img_paths]

    for i, this_name in enumerate(cleaned_file_names):
        cleaned_file_names[i] = re.sub(r'00[0-9]', '', this_name)

    # remove common prefix
    common_prefix = os.path.commonprefix(cleaned_file_names)
    cleaned_file_names = [file_name[len(common_prefix):] for file_name in cleaned_file_names]
    # remove common suffix
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

    # Make the window starts on top
    root.attributes('-topmost', True)
    root.update()
    root.attributes('-topmost', False)

    # get the os a current path for data selections
    detected_opterating_system = os.name
    current_path = os.getcwd()
    current_path = str(pathlib.Path(__file__).parent.resolve())
    print("CURRENT PATH",current_path)

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
    # Make the window starts on top
    root.attributes('-topmost', True)
    root.update()
    root.attributes('-topmost', False)
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

def convert_user_inputs(inputs):

    number_worms_to_detect = int(inputs[0])
    show_output_images = int(inputs[1])
    use_large_blob_fix = int(inputs[2])
    output_name = inputs[3]
    rename_tifs_choice = int(inputs[4])
    export_processed_images = int(inputs[5])
    data_analysis_and_export_bool = int(inputs[6])
    experimental_name_has_conditions_in_it = int(inputs[7])

    return number_worms_to_detect,show_output_images,use_large_blob_fix,output_name,rename_tifs_choice,export_processed_images,data_analysis_and_export_bool,experimental_name_has_conditions_in_it

def create_progress_window():
    # Create Tkinter window
    root_progressbar = tk.Tk()
    # make sure the window starts on top
    root_progressbar.attributes('-topmost', True)
    root_progressbar.update()
    root_progressbar.attributes('-topmost', False)

    root_progressbar.title("Progress Bar")

    # Create progress bar
    progress_bar = ttk.Progressbar(root_progressbar, orient='horizontal', length=1000, mode='determinate')
    progress_bar.pack(pady=10)

    # Create label for displaying text over the progress bar
    progress_bar_label = tk.Label(root_progressbar, text="", anchor="w")
    progress_bar_label.pack()

    return root_progressbar, progress_bar, progress_bar_label

def update_progress_bar(progress_bar, label, current_iteration, total, text=""):
    progress_bar['value'] = (current_iteration / total) * 100
    # print("Progress: {:.1f}%".format(current_iteration / total * 100), end='\r')

    # Update label text
    text = text[:100]  # Limit text length to 50 characters
    text = text.ljust(50)  # Pad text with spaces to ensure consistent display width
    label.config(text=text)

    progress_bar.update()  # Update the progress bar

def display_images(images, blocking = False, fig = None):
    if fig == None:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')  # Assuming images are grayscale
        ax.axis('off')
    plt.tight_layout()
    plt.show(block = blocking)

if __name__ ==  "__main__":
    # Get user inputs
    inputs = get_user_inputs()
    [number_worms_to_detect,
     show_output_images,
     use_large_blob_fix,
     output_name,
     rename_tifs_choice,
     export_processed_images,
     data_analysis_and_export_bool,
     experimental_name_has_conditions_in_it] = convert_user_inputs(inputs)

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
    # find all the image files and then natsort them
    img_paths = [os.path.join(path, name) for path, subdirs, files in os.walk(selected_directory) for name in files if fnmatch(name, pattern)]
    img_paths = natsorted(img_paths)
    print('There are',img_paths.__len__(),'files with the *.tif extension.\n')

    # check to make sure that there are acutally images in the directory
    if img_paths.__len__() == 0:
        exit("No TIF images found")

    # this is for the export names, just cleans them up and removes ANYTHING that is the same across all names
    # it also removes any usages of the 001,002,003...009 from the names (from leica weird export stuff)
    img_names = clean_img_names(img_paths)
    # Set total iterations and set up the progress bar
    root_progressbar, progress_bar, progress_bar_label = create_progress_window()


    # this is the main loop 
    for i in range(len(img_paths)):
        print(i,img_names[i])
        # Update progress bar
        update_progress_bar(progress_bar, progress_bar_label, i, len(img_paths), text= '\t\t' + "Processing --- " + img_names[i] + '\t\t' + str(i+1) + '/' + str(len(img_paths)))

        data, this_img = load_fluor_image(img_paths,i)

        data_norm = data.astype(np.float64)/255

        # this algorithm is supposed to be an exact match to the matlab (base) version
        for j in range(1,7):
            # create first threshold
            this_thresh = np.mean(data_norm) + (np.std(data_norm)*(1/5)*(j-1))
            # create a mask
            this_mask = matlab_gaussian_filter(data_norm,2) > this_thresh
            # remove any small blobs from the mask
            this_mask = matlab_bwareaopen(this_mask,min_size = 3000, connectivity = 4)
            # label the mask
            this_label = label(this_mask)

            # if there are N blobs in the image 
            if np.max(this_label) == number_worms_to_detect:
                # step one iterathion further
                this_thresh2 = np.mean(data_norm) + (np.std(data_norm)*(1/5)*(j))
                this_mask2 = matlab_gaussian_filter(data_norm,2)>this_thresh2
                this_mask2 = matlab_bwareaopen(this_mask2,min_size = 3000, connectivity = 4)
                this_label2 = label(this_mask2)

                # if there are still N blobs then keep this mask
                if np.max(this_label2) == number_worms_to_detect:
                    this_mask = this_mask2
                    this_label = this_label2
                    break
                # break out of the loop if N blobs are detected
                break
        
        # if there are many blobks still detected only take the N largest
        if np.max(this_label) > number_worms_to_detect:
            print('Warning: MORE than ', (number_worms_to_detect),' worms detected - ', img_names[i])
            print('Using only the ',(number_worms_to_detect),' largest blobs')

            this_mask, returned_areas = bwareafilt(this_mask,n = number_worms_to_detect)

        if np.max(this_label) < number_worms_to_detect:
            print('Warning: LESS than ', (number_worms_to_detect),' worms detected - ', img_names[i])
            print('Using only the ',np.max(this_label),' largest blobs')

            this_mask, returned_areas = bwareafilt(this_mask,n = number_worms_to_detect)

        display_images(np.asarray([data,this_mask,this_label,data*this_mask]),blocking = True)

        time.sleep(0.1)
        
    # Print completion message after the loop
    print("\nWork completed.")

    # Close Tkinter window after the loop completes
    root_progressbar.destroy()

    print('eof')