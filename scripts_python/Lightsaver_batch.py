import tkinter as tk
import numpy as np, pandas as pd
from tkinter import simpledialog, filedialog, ttk
from fnmatch import fnmatch
from natsort import natsorted
import os, time, cv2, re, pathlib, glob, shutil, subprocess
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.morphology import remove_small_objects, binary_closing, disk
from skimage.measure import label, regionprops

import matplotlib # for some reason using the 'agg' 
matplotlib.use('TkAgg')#)'qtagg')

def open_file_explorer(img_export_path):
    if not os.path.exists(img_export_path):
        raise FileNotFoundError(f"The path {img_export_path} does not exist.")
    
    if os.name == 'nt':  # For Windows
        os.startfile(img_export_path)
    elif os.name == 'posix':
        if 'darwin' in os.uname().sysname.lower():  # For macOS
            subprocess.run(['open', img_export_path])
        else:  # For Linux
            subprocess.run(['xdg-open', img_export_path])
    else:
        raise OSError("Unsupported operating system")

def label_to_rgb(label_img):

    label_to_color = {
        0: [0,0,0],
        1: [0,128,255],
        2: [0,255,255],
        3: [128,255,128],
        4: [255,255,0],
        5: [255,128,0]
    }

    out = np.zeros(shape = (label_img.shape[0],label_img.shape[1],3)).astype(np.uint8)

    # https://stackoverflow.com/questions/68192717/how-to-efficiently-convert-image-labels-to-rgb-values
    for gray, rgb in label_to_color.items():
        out[label_img == gray, :] = list(reversed(rgb)) # needs to be reversed for cv2 output idk

    return out

def normalize_image(image, return_uint8=True):
    # Convert int32 to float32
    image_float = image.astype(np.float32)
    # Normalize the image
    image_normalized = (image_float - np.min(image_float)) / (1+(np.max(image_float) - np.min(image_float)))
    if return_uint8:
        # Scale to 0-255
        image_scaled = (image_normalized * 255).astype(np.uint8)
        return image_scaled
    else:
        return image_normalized

def resize_and_construct_grid(images, title=''):
    # Ensure all images are color images
    resized_images = []
    for img in images:
        if 'int' in str(img.dtype): # this should only be for label
            img = normalize_image(img)
        if img.dtype == bool:
            img = (img * 255).astype(np.uint8)  # Convert boolean to uint8
        if len(img.shape) == 2:  # If the image is grayscale
            img = cv2.cvtColor(normalize_image(img), cv2.COLOR_GRAY2RGB)  # Convert to color image
        elif len(img.shape) == 3 and img.shape[2] == 1:  # If the image has only one channel
            img = cv2.cvtColor(normalize_image(img), cv2.COLOR_GRAY2RGB)  # Convert to color image
        resized_images.append(img)
    
    # Determine the maximum dimensions among all images
    max_height = max(img.shape[0] for img in resized_images)
    max_width = max(img.shape[1] for img in resized_images)
    
    # Resize images to the maximum dimensions
    resized_images = [cv2.resize(img, (max_width, max_height)) for img in resized_images]
    
    # Construct the grid
    grid = np.zeros((max_height, 3*max_width, 3), dtype=np.uint8)
    for i, img in enumerate(resized_images):
        col_start = i * max_width
        col_end = (i + 1) * max_width
        grid[:, col_start:col_end] = img
    
    # Superimpose title on top middle
    font_scale = 2  # Larger font scale
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
    text_x = (grid.shape[1] - title_size[0]) // 2
    text_y = title_size[1] + 20
    cv2.putText(grid, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3)
    
    return grid

def imclearborder(binary_image):

    # Label connected components
    labeled_image, num_labels = label(binary_image, connectivity=2, return_num=True)
    
    # Get the shape of the image
    rows, cols = binary_image.shape
    
    # Iterate through each label
    for region in regionprops(labeled_image):
        min_row, min_col, max_row, max_col = region.bbox
        
        # Check if the region touches the border
        if min_row == 0 or min_col == 0 or max_row == rows or max_col == cols:
            # Remove the region touching the border
            for coordinates in region.coords:
                labeled_image[coordinates[0], coordinates[1]] = 0
    
    # Convert the labeled image back to binary
    cleared_image = (labeled_image > 0)
    
    return cleared_image
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
        data[980:data.shape[0],0:215] = 0 #np.median(data)
    elif 'str' in str(type(fill_value)):
        data[980:data.shape[0],0:215] = np.median(data)
    else:
        data[980:data.shape[0],0:215] = fill_value #np.median(data)

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

    for i,this_name in enumerate(cleaned_file_names):
        if this_name[0] == '_':
            cleaned_file_names[i] = this_name[1:]

    counter = 0
    non_counter = 0
    for i,this_name in enumerate(cleaned_file_names):
        if this_name in img_paths[i]:
            counter += 1
        else:
            non_counter += 1
    
    if counter == len(img_paths):
        print('Correct amount of img paths and img names detected')
    else:
        print('WARNING:: ')
        print('WARNING:: ')
        print('Naming scheme is not consistent and MOST LIKELY needs to be redone -- EXPORT MIGHT BE WRONG')
        print('Please remove extraneous information from the image names')

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

def display_images(images, blocking = False, fig = None, axes = None, title = None):
    if fig == None and axes == None:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    else:
        fig.clf()
    if title is not None:
        plt.suptitle(title)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')  # Assuming images are grayscale
        ax.axis('off')
    plt.tight_layout()
    if 'bool' in str(type(blocking)):
        plt.show(block = blocking)
    elif 'float' in str(type(blocking)) or 'int' in str(type(blocking)):
        plt.draw()
        plt.pause(blocking)

def display_images2(images, blocking=False, fig=None, axes=None, title=None):
    if fig is None and axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    else:
        for ax in axes.flat:
            ax.clear()  # Clear the axes
    if title is not None:
        fig.suptitle(title)  # Set title for the figure
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')  # Assuming images are grayscale
        ax.axis('off')
    fig.tight_layout()  # Adjust layout
    if blocking is not None:
        if isinstance(blocking, bool):
            plt.show(block=blocking)
        elif isinstance(blocking, (float, int)):
            plt.draw()
            plt.pause(blocking)

def segment_blobs_from_image(data_norm,this_img,min_worm_size = 3000, i = 0):
    
    # this algorithm is supposed to be an exact match to the matlab (base) version
    for j in range(1,7):
        check_flag = True
        # create first threshold
        this_thresh = np.mean(data_norm) + (np.std(data_norm)*(1/5)*(j-1))
        # create a mask
        this_mask = matlab_gaussian_filter(data_norm,2) > this_thresh
        # remove any small blobs from the mask
        this_mask = matlab_bwareaopen(this_mask,min_size = min_worm_size, connectivity = 4)
        # # remove any blobs that touch the sides
        # this_mask = imclearborder(this_mask)
        # label the mask
        this_label, n_labels = label(this_mask,connectivity=2,return_num=True)

        # if there are N blobs in the image 
        if np.max(this_label) == number_worms_to_detect:

            label_sizes = []
            for k in range( int(np.max(this_label))):
                label_sizes.append(np.sum(this_label==(k+1)))

            if np.sum((np.asarray(label_sizes)/(this_img.shape[0]*this_img.shape[1]))>0.18):
                print('LARGE MASK DETECTED ATTEMPT FIX')
                j = j+1
                check_flag = False
            
            if check_flag:
                # step one iterathion further
                this_thresh2 = np.mean(data_norm) + (np.std(data_norm)*(1/5)*(j))
                this_mask2 = matlab_gaussian_filter(data_norm,2)>this_thresh2
                this_mask2 = matlab_bwareaopen(this_mask2,min_size = 3000, connectivity = 4)
                this_label2, n_labels2 = label(this_mask2,connectivity=2,return_num=True)

                # if there are still N blobs then keep this mask
                if np.max(this_label2) == number_worms_to_detect:
                    this_mask = this_mask2
                    this_label = this_label2
                    break
                # break out of the loop if N blobs are detected
                break
    
    return this_mask, this_label

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

    # set the minimum worm size, might be updated if no masks can be found
    min_worm_size = 3000

    # Get current path
    curr_path = os.getcwd()

    # get the selected directory and the first part of the output path for the exported images
    selected_directory, output_path = get_directory()
    print("Selected directory:", selected_directory)

    # get the final save name of the experiment and create the output directory for it
    final_save_name = os.path.split(selected_directory)[-1]
    output_path = os.path.join(output_path,final_save_name)
    os.makedirs(output_path,exist_ok=True)
    shutil.rmtree(output_path) ############################################ THIS IS TESTING ONLY IT DELETES THE EXPORTED IMGAGES 
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

    if show_output_images:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    image_integral_intensities = np.zeros(shape = (len(img_paths),number_worms_to_detect))
    image_integral_area = np.zeros(shape = (len(img_paths),number_worms_to_detect))

    if show_output_images:
        blocking = 0.1
    else:
        if export_processed_images:
            blocking = None
        else:
            blocking = False

    open_file_explorer(output_path)

    # this is the main loop 
    for i in range(len(img_paths)):
        print(i,img_names[i])
        # Update progress bar
        update_progress_bar(progress_bar, progress_bar_label, i, len(img_paths), text= '\t\t' + "Processing --- " + img_names[i] + '\t\t' + str(i+1) + '/' + str(len(img_paths)))

        data, this_img = load_fluor_image(img_paths,i,fill_value='median')#fill_value='median)

        data_norm = data.astype(np.float64)/255

        # run the segmentation 
        this_mask, this_label = segment_blobs_from_image(data_norm,this_img,min_worm_size = min_worm_size, i = i)

        # if there are no blobs detetced. redo the last threshold and take the N largest
        if np.max(this_label) == 0 or np.max(this_label)==1:
            print('Warning: NO WORMS FOUND USING STANDARD METHODS ON - ', img_names[i])
            print('Reducing minimum blob size from', min_worm_size, ' to ', str(1000))
            this_mask, this_label = segment_blobs_from_image(data_norm,this_img,min_worm_size = 1000)

        # if there are many blobks still detected only take the N largest
        if np.max(this_label) > number_worms_to_detect:
            print('Warning: MORE than ', (number_worms_to_detect),' worms detected - ', img_names[i])
            print('Using only the ',(number_worms_to_detect),' largest blobs')

            this_mask, returned_areas = bwareafilt(this_mask,n = number_worms_to_detect)

        if np.max(this_label) < number_worms_to_detect:
            print('Warning: LESS than ', (number_worms_to_detect),' worms detected - ', img_names[i])
            print('Using only the ',np.max(this_label),' largest blobs')

            this_mask, returned_areas = bwareafilt(this_mask,n = number_worms_to_detect)

        # thicken the masks (different than matlabs interp)
        new_mask = (matlab_gaussian_filter(this_mask,0.25))>0
        # close small edges and zones
        new_mask = binary_closing(new_mask,footprint = disk(5))
        # fill the holes 
        new_mask = binary_fill_holes(new_mask)
        # re-label the masks
        labeled_masks = np.rot90(label(np.rot90(new_mask,-1), connectivity=2),1)

        # mask the inital data without the normalization step
        # attempts to get rid of the background signals
        masked_data = new_mask*(data.astype(np.float64))

        for j in range(int(np.max(labeled_masks))):
            this_labeled_mask = masked_data*(labeled_masks == (j+1))
            image_integral_intensities[i,j] = np.sum(this_labeled_mask)
            image_integral_area[i,j] = np.sum(this_labeled_mask>0)

        # fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        if show_output_images:
            display_images2(np.asarray([data,new_mask,labeled_masks,masked_data]),blocking = blocking, fig=fig,axes=axes, title = img_names[i])
        if export_processed_images:
            labeled_masks_rgb = label_to_rgb(labeled_masks)
            out = resize_and_construct_grid([this_img,labeled_masks_rgb,masked_data], title=img_names[i])
            cv2.imwrite(os.path.join(output_path,str(i+1) + '_' + img_names[i] + output_img_format),out)

        # time.sleep(0.1)
        
    # Print completion message after the loop
    print("\nWork completed.")

    df = pd.DataFrame(columns = ['Image names','Worm 1 (blue) integrated Intensity',
        'Worm 2 (teal) integrated Intensity','Worm 3 (green) integrated Intensity',
        'Worm 4 (yellow/red) integrated Intensity','Worm 5 (orange) integrated Intensity',
        'Worm 1 (blue) integrated Area','Worm 2 (teal) integrated Area',
        'Worm 3 (green) integrated Area','Worm 4 (yellow/red) integrated Area',
        'Worm 5 (orange) integrated Area'])
    df['Image names'] = img_names
    df.iloc[:,1:6] = image_integral_intensities
    df.iloc[:,6:] = image_integral_area
    df.to_csv(os.path.join(selected_directory,'data_python.csv'), index = False)

    # Close Tkinter window after the loop completes
    root_progressbar.destroy()

    print('eof')