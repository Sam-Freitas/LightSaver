import os, glob, tqdm, cv2
import numpy as np
from natsort import natsorted
from pathlib import Path
import matplotlib.pyplot as plt

def del_dir_contents(path_to_dir):
    files = glob.glob(os.path.join(path_to_dir,'*'))
    for f in files:
        os.remove(f)

dataset_dir = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\data\Terasaki Validation SU10\SU10_EV"
filetype = ".tif"

dataset_dir2 = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\data\Terasaki Validation SU10\SU10_RNAi"
filetype = ".tif"

output_dataset = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\data\SICKO_testing_dataset"
output_filetype = '.tif'
output_dataset2 = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\data\SICKO_testing_dataset_png"
output_filetype2 = '.png'
# all_images = natsorted(glob.glob(os.path.join(dataset_dir,'*/**' + filetype), recursive= True))
days_directories = natsorted(glob.glob(os.path.join(dataset_dir,'*/'))) + natsorted(glob.glob(os.path.join(dataset_dir2,'*/')))

img_max = 0

def norm(img):

    img_min = np.min(img)
    img = img-img_min
 
    img_max = np.max(img)

    if img_max > 0:
        return img/img_max
    else:
        return img
    
def norm_5_to_95(img):

    img_mean = np.mean(img)
    img_std = np.std(img)

    img2 = img-(img_mean + (2*img_std))

    img_mean2 = np.mean(img2)
    img_std2 = np.std(img2)

    img3 = img2/(np.abs(img_mean2) + np.abs(2*img_std2))
    img3 = (np.clip(img3,-1,1)+1)/2

    return img3

os.makedirs(output_dataset,exist_ok=True)
del_dir_contents(output_dataset)
os.makedirs(output_dataset2,exist_ok=True)
del_dir_contents(output_dataset2)

# write all the images to the output folder 
for i in tqdm.tqdm(range(len(days_directories))):

    each_day = days_directories[i]
    print(each_day)

    base_name = Path(each_day).parts[-1]

    all_images_in_this_day = natsorted(glob.glob(os.path.join(each_day,'*' + filetype), recursive= True))
    specified_images_in_this_day = all_images_in_this_day[0::3]

    for j,each_img in enumerate(specified_images_in_this_day):
        img = cv2.imread(each_img,-1)

        img = img.astype(np.float32)#/img_max # read into 16 bit, change to float, divide by max value
        img2 = norm_5_to_95(img)
        img2 = (img2*255).astype(np.uint8)

        this_img_name = base_name + '_' + str(j) + '_' + 'ch00' + output_filetype
        this_img_name2 = base_name + '_' + str(j) + '_' + 'ch00' + output_filetype2
        cv2.imwrite(os.path.join(output_dataset,this_img_name), img2)
        cv2.imwrite(os.path.join(output_dataset2,this_img_name2), img2)

print('EOF')