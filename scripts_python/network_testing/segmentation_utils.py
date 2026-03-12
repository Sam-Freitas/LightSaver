from torchvision.io.image import read_file, read_image, write_jpeg, write_png
from torchvision.models import segmentation 
from torchvision import transforms #, ops

import torch
from torch import nn
from torch.utils.data import Dataset

# from sklearn.model_selection import train_test_split
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import os, datetime, cv2, glob, tqdm, time
from natsort import natsorted
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

from kornia.contrib import connected_components
from skimage.measure import label, regionprops
from functools import reduce
import ntpath

from skimage.morphology import disk, ball
from skimage.filters import rank

def flatten_list(lst):
    return reduce(lambda x,y: x+y, lst)

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def every_nth(lst, nth):
    """Returns a new list containing every nth element of the original list.
    Args:
        lst: The original list.
        nth: The interval between selected elements
    Returns:
        A new list containing every nth element.
    """
    if not isinstance(lst, list):
        raise TypeError("Input must be a list")
    if not isinstance(nth, int):
        raise TypeError("nth must be an integer")
    if nth <= 0:
        raise ValueError("nth must be greater than 0")
    return lst[nth - 1::nth]

def norm(a, b_max = None,b_min = False, convert_to_float64 = True):
    if convert_to_float64: 
        if 'torch' in str(type(a)):
            b = a.to(torch.float64)
        else:
            b = a.astype(np.float64)
    else:
        b = a
    if b_min:
        b = b - b.min()
    if b_max == None:
        b_max = b.max()
    b = (b/b_max)
    return b

def filter_raw_img(img,kernel):
        # Calculate the average of the images in the chunk.
    # Add extra dimensions to the averaged image for further operations.
    a1 = img[None, ::]
    # Perform morphological operations (opening and closing) on the image.
    a4 = (2 * a1) - (kornia.morphology.opening(a1, kernel) + kornia.morphology.closing(a1, kernel))
    # Clip values in `a4` to ensure they are not below the mean value.
    a5 = torch.clip(a4, min=a4.mean()).squeeze()
    # Normalize the clipped result.
    # a5 = norm(a5, b_min=True).squeeze()

    return a5

def flatten(xss):
    return [x for xs in xss for x in xs]

def group_tensors_by_x_delta(tensors, delta_threshold):
    # First, group tensors by x-coordinate delta
    sorted_tensors = sorted(tensors, key=lambda t: t[0].item())
    groups = []
    current_group = [sorted_tensors[0]]

    for i in range(1, len(sorted_tensors)):
        x_delta = sorted_tensors[i][0].item() - current_group[-1][0].item()
        if x_delta <= delta_threshold:
            current_group.append(sorted_tensors[i])
        else:
            groups.append(current_group)
            current_group = [sorted_tensors[i]]

    if current_group:
        groups.append(current_group)

    return groups

def group_and_sort_by_y(tensors, delta_threshold):
    # Group tensors first
    groups = group_tensors_by_x_delta(tensors, delta_threshold)
    
    # Then sort each group by y-coordinate in ascending order
    sorted_groups = [sorted(group, key=lambda t: t[1].item()) for group in groups]
    return sorted_groups

def rescale_centers(sorted_centers,binary_image):

    out = sorted_centers.copy()

    binary_image = binary_image.squeeze()
    if binary_image.dim() == 2:  # If grayscale, add a channel dimension
        binary_image = binary_image.unsqueeze(0)  # Shape becomes (1, H, W)
    
    _, height, width = binary_image.shape

    for i in range(len(out)):
        out[i] = torch.tensor([out[i][0]/height,out[i][1]/width,out[i][2]])

    return out

def segment_squares_from_image(image, points, sq_percentage=5):
    """
    Segments squares centered at specified points from the input image using PyTorch.

    Args:
    - image (torch.Tensor): Input image tensor of shape (C, H, W) for RGB or (H, W) for grayscale.
    - points (list of tuples): List of (x, y) coordinates for the center of each square.
    - sq_percentage (float): Percentage of the image width or height to define square size.

    Returns:
    - List of torch.Tensor: Cropped square patches centered at each specified point.
    """
    image = image.squeeze()
    if image.dim() == 2:  # If grayscale, add a channel dimension
        image = image.unsqueeze(0)  # Shape becomes (1, H, W)
    
    n_dim_flag = False
    try:
        _, height, width = image.shape
    except:
        n_dim_flag = True
        height,width = image.shape[-2:]
    square_size = int(min(width, height) * (sq_percentage / 100))
    half_size = square_size // 2

    cropped_squares = []
    for (x, y, idx) in points:
        # Calculate bounds and ensure they stay within image dimensions

        if (x < 1) or (y < 1): # if given scaled coordinates then reconvert
            x = int(x*height)
            y = int(y*width) 

        left = max(0, int(x) - half_size)
        right = min(width, int(x) + half_size)
        top = max(0, int(y) - half_size)
        bottom = min(height, int(y) + half_size)
        
        # Crop the square patch from the image
        if n_dim_flag:
            square_patch = image[:, :, left:right,top:bottom]
        else:
            square_patch = image[:, left:right,top:bottom]
        cropped_squares.append(square_patch)

    return cropped_squares

def sort_and_fix_labeled_images(binary_image, labeled_binary_image, device = 'cpu'):

    # get image
    # label image
    # isolate blobs and get centers
    unique_labels = torch.unique(labeled_binary_image)
    centers = torch.zeros(size = (len(unique_labels)-1,3))
    for i, this_label in enumerate(unique_labels[1:]):  # Skip background label 0
        blob_mask = (labeled_binary_image == this_label)
        indices = blob_mask.nonzero()
        this_center = torch.mean(indices.to(torch.float32),axis = 0)
        centers[i,:] = (torch.cat([this_center.cpu(),torch.tensor([i])]))

    sorted_centers = group_and_sort_by_y(centers, delta_threshold=25)
    sorted_centers = flatten(sorted_centers)
    sorted_rounded_centers = [torch.round(this_center).to(torch.int64) for this_center in sorted_centers]

    temp_labeled_binary_image = torch.zeros_like(labeled_binary_image)
    for i, this_label in enumerate(unique_labels[1:]):
        blob_mask = (labeled_binary_image == this_label)
        indices = blob_mask.nonzero().cpu()
        for this_center in sorted_rounded_centers:
            if torch.abs(torch.sum(indices-this_center[0:2],axis = 1)).min() == 0:
                temp_labeled_binary_image[blob_mask] = i+1
                break
    labeled_binary_image = temp_labeled_binary_image

    return labeled_binary_image, sorted_centers

def segment_out_wells_from_inital_img(binary_image, intial_image, labeled_binary_image, sq_percentage = 25, return_img_squares = True):

    # get image
    # label image
    # isolate blobs and get centers
    unique_labels = torch.unique(labeled_binary_image)
    centers = torch.zeros(size = (len(unique_labels)-1,3))
    for i, this_label in enumerate(unique_labels[1:]):  # Skip background label 0
        blob_mask = (labeled_binary_image == this_label)
        indices = blob_mask.nonzero()
        this_center = torch.mean(indices.to(torch.float32),axis = 0)
        centers[i,:] = (torch.cat([this_center.cpu(),torch.tensor([i])]))

    sorted_centers = group_and_sort_by_y(centers, delta_threshold=25)
    sorted_centers = flatten(sorted_centers)
    scaled_sorted_centers = rescale_centers(sorted_centers,binary_image)

    for i, this_center in enumerate(scaled_sorted_centers):
        scaled_sorted_centers[i][-1] = i

    if return_img_squares:
        image_squares = segment_squares_from_image(intial_image,scaled_sorted_centers,sq_percentage = sq_percentage)

        return image_squares,sorted_centers,scaled_sorted_centers
    else:
        return None,sorted_centers,scaled_sorted_centers

def filter_blobs(binary_image, border_percent, min_size=500, device='cpu'):
    """
    Filters blobs in a binary image by removing small blobs and those near the borders.

    Parameters:
    - binary_image (torch.Tensor): 2D binary image tensor (HxW) with values 0 and 1.
    - border_percent (float): Percentage of the image width/height defining the border region.
    - min_size (int): Minimum number of pixels for a component to be retained.
    - device (str): Device to perform the operation on ('cpu' or 'cuda').

    Returns:
    - torch.Tensor: Filtered binary image with small and border blobs removed.
    """
    # Ensure binary_image is a 2D binary tensor on the specified device
    binary_image = binary_image.to(device).float().squeeze()
    if binary_image.dim() != 2:
        raise ValueError("binary_image must be a 2D binary tensor.")

    # Step 1: Label connected components
    labeled_image = connected_components(binary_image.unsqueeze(0).unsqueeze(0), num_iterations=100).squeeze()
    labeled_image = labeled_image.to(torch.int64)

    # Step 2: Remove small blobs
    component_sizes = torch.bincount(labeled_image.flatten())
    size_mask = component_sizes >= min_size
    size_mask[0] = 0  # Exclude the background

    # Filter out small blobs
    filtered_image = size_mask[labeled_image].float()

    # Step 3: Remove border blobs
    height, width = filtered_image.shape
    border_size_x = int(width * border_percent / 100)
    border_size_y = int(height * border_percent / 100)

    # Create a border mask
    border_mask = torch.ones_like(filtered_image, dtype=torch.bool, device=device)
    border_mask[border_size_y:height - border_size_y, border_size_x:width - border_size_x] = False

    # Remove blobs that touch the border
    unique_labels = torch.unique(labeled_image)
    for this_label in unique_labels[1:]:  # Skip background label 0
        blob_mask = (labeled_image == this_label)
        if torch.any(border_mask & blob_mask):
            filtered_image[blob_mask] = 0

    labeled_image = connected_components(filtered_image.unsqueeze(0).unsqueeze(0), num_iterations=100).squeeze()
    labeled_image = labeled_image.to(torch.int64)

    # labeled_image_temp = torch.zeros_like(labeled_image)
    # for i,this_label in enumerate(torch.unique(labeled_image)):
    #     this_label_img = (labeled_image==this_label)
    #     labeled_image_temp[this_label_img] == i

    return filtered_image.to(device), labeled_image.to(device)

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, target)
        smooth = 1e-5
        # inputs = torch.sigmoid(inputs)
        num = target.size(0)
        inputs = inputs.view(num, -1)
        target = target.view(num, -1)
        intersection = (inputs * target)
        dice = (2. * intersection.sum(1) + smooth) / (inputs.sum(1) + target.sum(1) + smooth)
        dice = dice.sum() / num 
        dice_loss = 1 - dice

        BCEDiceLoss_num = 0.5 * (bce + dice_loss)

        return BCEDiceLoss_num

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.nn.functional.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        num = targets.size(0)
        inputs = inputs.view(num, -1)
        targets = targets.view(num, -1)
        intersection = (inputs * targets)
        dice = (2. * intersection.sum(1) + smooth) / (inputs.sum(1) + targets.sum(1) + smooth)
        dice = dice.sum() / num 
        
        return 1 - dice

class BCEDiceLoss_blobPunish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target):

        num_iterations = 100

        labels = connected_components(1*(inputs > 0.5).float(),num_iterations=num_iterations)
        target_number = connected_components(1*(target > 0.5).float(),num_iterations=num_iterations)

        num_label_blobs = torch.numel(torch.unique(labels))-torch.tensor(1)
        num_target_blobs = torch.numel(torch.unique(target_number))

        blob_number_penalty = torch.sqrt(num_label_blobs/num_target_blobs) # sqare root of the number of blobs/batch size

        if torch.isinf(blob_number_penalty) or torch.isnan(blob_number_penalty) or torch.isneginf(blob_number_penalty):
            blob_number_penalty = inputs.shape[0]
        if blob_number_penalty < 1:
            blob_number_penalty = 1 
        if blob_number_penalty > inputs.shape[0]:
            blob_number_penalty = inputs.shape[0]

        bce = nn.functional.binary_cross_entropy_with_logits(inputs, target)
        smooth = 1e-5
        # inputs = torch.sigmoid(inputs)
        num = target.size(0)
        inputs = inputs.view(num, -1)
        target = target.view(num, -1)
        intersection = (inputs * target)
        dice = (2. * intersection.sum(1) + smooth) / (inputs.sum(1) + target.sum(1) + smooth)
        dice = dice.sum() / num 
        dice_loss = 1 - dice

        BCEDiceLoss_num = 0.5 * (bce + dice_loss)

        return BCEDiceLoss_num * blob_number_penalty

class BinaryDiceLoss(nn.Module):
    # """Dice loss of binary class
    # Args:
    #     smooth: A float number to smooth loss, and avoid NaN error, default: 1
    #     p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
    #     predict: A tensor of shape [N, *]
    #     target: A tensor of shape same with predict
    #     reduction: Reduction method to apply, return mean over batch if 'mean',
    #         return sum if 'sum', return a tensor of shape [N,] if 'none'
    # Returns:
    #     Loss tensor according to arg reduction
    # Raise:
    #     Exception if unexpected reduction
    # """
    def __init__(self, smooth=1, p=2, reduction='mean' ):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

def del_dir_contents(path_to_dir):
    files = glob.glob(os.path.join(path_to_dir,'*'))
    for f in files:
        try:
            os.remove(f)
        except:
            print('Cant delete:', f)

def read_img_custom(path):
    if 'Tensor' in str(type(path)):
        return path
    else:
        if path[-3:] == 'png' or path[-3:] =='jpg':
            img = read_image(path).float()/255
        else:
            img = Image.open(path)
            img = np.asarray(img)
            img = np.atleast_3d(img)
            img = np.moveaxis(img,2,0)
            img_possible_max = np.iinfo(img.dtype).max
            img = torch.tensor(img).float()/img_possible_max
        return img

def find_files(folder_path, file_extension='.png', filter='', filter2 = None, exclude_filter = None):
    """
    Recursively finds and returns all files with the specified extension in the given folder,
    only if the filter string is contained within the file path.

    Args:
        folder_path (str): The path to the folder to search.
        file_extension (str): The file extension to search for (default is '.png').
        filter (str): The filter string that must be in the file path to be included in the result (default is '').

    Returns:
        list: A list of file paths that match the specified extension and contain the filter string.
    """
    found_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(file_extension.lower()) and filter in os.path.join(root, file):
                found_files.append(os.path.join(root, file))

    # secondary optional filter
    if filter2:
        found_files2 = []
        for file in found_files:
            if filter2 in file:
                found_files2.append(file)
        found_files =  found_files2
    
    if exclude_filter:
        non_excluded_files = []
        for file in found_files:
            if exclude_filter not in file:
                non_excluded_files.append(file)
        found_files = non_excluded_files

    return found_files

def read_all_images(paths, transforms = None, number_to_stop_at = None, silent = False):
    out = []
    for i,this_path in enumerate(tqdm.tqdm(paths, disable=silent)):
        temp_img = read_img_custom(this_path)
        if transforms is not None:
            try:
                temp_img = transforms()(temp_img)
            except:
                temp_img = transforms(temp_img)
        out.append(temp_img)
        if number_to_stop_at is not None:
            if i >= number_to_stop_at:
                break
    return out

def normalize_img(input_img):
    input_img = input_img - input_img.min()
    input_img = input_img/input_img.max()
    return input_img

def s(img, title = None, block = True, norm = True):
    # matplotlib.use('TkAgg')
    if 'torch' in str(img.dtype):

        if 'bool' in str(img.dtype):
            img = img*1

        img = img.squeeze()
        if len(img.shape) > 2: # check RGB
            if np.argmin(torch.tensor(img.shape)) == 0: # check if CHW 
                img = img.permute((1, 2, 0)) # change to HWC
        if norm:
            img = normalize_img(img)*255
            img = img.to('cpu').to(torch.uint8)
        else:
            img = img.cpu()
    else:
        img_shape = img.shape
        if len(img_shape) > 2:
            if np.argmin(img_shape) == 0:
                img = np.moveaxis(img,0,-1)
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    if block:
        plt.show(block = True)

def impair(A, B):

    if 'torch' not in str(type(A)):
        A = torch.tensor(A)
    if 'torch' not in str(type(B)):
        B = torch.tensor(B)
    
    if A.device != B.device:
        B = B.to(A.device)

    if A.dtype != B.dtype:
        B = B.to(A.dtype)

    return torch.stack([B, A, B])#, dim=2)

def impair_green_mask_outline(A,B):
    # this assumes that A is a mask outline

    if 'torch' not in str(type(A)):
        A = torch.tensor(A)
    if 'torch' not in str(type(B)):
        B = torch.tensor(B)
    
    if A.device != B.device:
        B = B.to(A.device)

    if A.dtype != B.dtype:
        B = B.to(A.dtype)

    out = torch.stack([B,B,B])
    out[1][A>0] = 1

    return out

def s_grid_combined(images, border=5):
    """
    Combines images into a single array with a white border between each.

    Args:
    - images (list): List of 6 or 9 images in torch.Tensor format.
    - border (int): Width of the border between images.

    Returns:
    - torch.Tensor: Combined image with borders, ready for display.
    """
    num_images = len(images)
    if num_images not in [6, 9]:
        raise ValueError("s_grid_combined requires exactly 6 or 9 images.")

    if num_images == 9:
        grid_size=(3, 3)
    else:
        grid_size=(3, 2)

    rows, cols = grid_size

    # Normalize images and ensure they are all the same size
    img_shape = images[0].shape
    height, width = img_shape[-2], img_shape[-1]  # Assuming (C, H, W) or (H, W) format
    normalized_images = []

    img_max = 0
    for img in images:
        if torch.max(img) > img_max:
            img_max = torch.max(img)
    norm_multiplier = (1/img_max)*255

    for img in images:
        if img.dim() == 2:  # Convert grayscale (H, W) to (H, W, 1)
            img = img.unsqueeze(-1)
        elif img.dim() == 3 and img.shape[0] == 1:  # Convert grayscale (1, H, W) to (H, W, 1)
            img = img.squeeze(0).unsqueeze(-1)
        elif img.dim() == 3 and img.shape[0] == 3:  # RGB (C, H, W) to (H, W, C)
            img = img.permute(1, 2, 0)
        # img = normalize_img(img) * 255
        img = img*norm_multiplier
        img = img.to(torch.uint8)
        
        # If grayscale, expand to 3 channels for consistency
        if img.shape[-1] == 1:
            img = img.expand(-1, -1, 3)
        
        normalized_images.append(img)

    # Define the combined image shape with borders
    combined_height = rows * height + (rows - 1) * border
    combined_width = cols * width + (cols - 1) * border
    combined_image = torch.full((combined_height, combined_width, 3), 255, dtype=torch.uint8)

    # Place each image in the combined array
    for idx, img in enumerate(normalized_images):
        row, col = divmod(idx, cols)
        top = row * (height + border)
        left = col * (width + border)
        combined_image[top:top + height, left:left + width] = img

    return combined_image

def mask_blur_function(mask):

    temp_mask = transforms.GaussianBlur(kernel_size = (35,35), sigma = (9))(mask)
    out = torch.clip(temp_mask+mask, min = 0, max = mask.max())

    return out

class Normalize(nn.Module):
    def __init__(self):
        """
        Initializes the Normalize module to normalize a tensor of any shape
        between 0 and 1 based on 3 standard deviations of the input data.
        """
        super(Normalize, self).__init__()

    def forward(self, tensor, std_bound = 3):
        """
        Normalizes a tensor of any shape between 0 and 1 using 3 standard deviations
        of the data's mean, then clips the values to the [0, 1] range.

        Args:
            tensor (torch.Tensor): An n-dimensional tensor to be normalized.

        Returns:
            torch.Tensor: The normalized and clipped tensor.
        """
        # Compute the mean and standard deviation of the input tensor
        mean = tensor.mean()
        std = tensor.std()

        # Define the lower and upper bounds based on 3 standard deviations
        lower_bound = mean - std_bound * std
        upper_bound = mean + std_bound * std

        if lower_bound < 0:
            lower_bound = 0

        # Normalize the tensor between 0 and 1
        normalized_tensor = (tensor - lower_bound) / (upper_bound - lower_bound)

        # Clip values to be between 0 and 1
        normalized_tensor = torch.clamp(normalized_tensor, min=0.0, max=1.0)

        return normalized_tensor

class Normalize_indiv_worm(nn.Module):
    def __init__(self):
        """
        Initializes the Normalize module to normalize a tensor of any shape
        between 0 and 1 based on 3 standard deviations of the input data.
        """
        super(Normalize_indiv_worm, self).__init__()

    def forward(self, tensor):
        """
        Normalizes a tensor of any shape between 0 and 1 
        of the data's mean, then clips the values to the [0, 1] range.
        Args:
            tensor (torch.Tensor): An n-dimensional tensor to be normalized.
        Returns:
            torch.Tensor: The normalized and clipped tensor.
        """

        # Define the lower and upper bounds based on 3 standard deviations
        lower_bound = torch.min(tensor)
        upper_bound = torch.max(tensor)

        if lower_bound < 0:
            lower_bound = 0

        # Normalize the tensor between 0 and 1
        normalized_tensor = (tensor - lower_bound) / (upper_bound - lower_bound)

        # Clip values to be between 0 and 1
        normalized_tensor = torch.clamp(normalized_tensor, min=0.0, max=1.0)

        return normalized_tensor

class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths = None, transforms = None, resize = None, device = torch.device('cpu'), return_intial_img_aswell = False, blur_masks = False, return_path_aswell = False):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.blur_masks = blur_masks
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms
        self.resize = resize
        self.device = device
        self.return_intial_img_aswell = return_intial_img_aswell
        self.return_path_aswell = return_path_aswell
    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)
    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.imagePaths[idx]
        # load the image from disk, swap its channels from BGR to RGB,
        # and read the associated mask from disk in grayscale mode
        image = read_img_custom(imagePath) # cv2.imread(imagePath)
        if self.return_intial_img_aswell:
            image_init = image.clone().detach()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.maskPaths is not None:
            mask = read_img_custom(self.maskPaths[idx])#cv2.imread(maskPath, 0)
        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            # image = self.transforms(image)
            # if self.maskPaths is not None:
            #     mask = self.resize(mask)
            if self.maskPaths is not None:
                temp = self.transforms(image = image.squeeze().numpy(), mask = mask.squeeze().numpy())
                image = temp['image']
                mask = temp['mask'].unsqueeze(0)
            else:
                temp = self.transforms(image = image.squeeze().numpy())
                image = temp['image']

        if self.resize is not None:
            try:
                image = self.resize()(image.clone().detach().unsqueeze(0))
            except:
                image = self.resize(image.clone().detach().unsqueeze(0))
            if self.maskPaths is not None:
                try:
                    mask = self.resize()(mask)
                except:
                    mask = self.resize(mask)

        # return a tuple of the image and its mask
        if self.maskPaths is not None:
            image, mask = image.to(self.device), mask.to(self.device)
            if self.blur_masks:
                if self.return_path_aswell:
                    return (image, mask_blur_function(mask), imagePath)
                else:
                    return (image, mask_blur_function(mask))
            else:
                if self.return_path_aswell:
                    return (image, mask, imagePath)
                else:
                    return (image, mask)
        else:
            image = image.to(self.device)
            if self.return_intial_img_aswell:
                image_init = image_init.to(self.device)
                if self.return_path_aswell:
                    return (image, image_init, imagePath)
                else:
                    return (image, image_init)
            else:
                if self.return_path_aswell:
                    return (image, imagePath)
                else:
                    return (image)

class PreprocessNormalize(nn.Module):
    def __init__(self):
        """
        Initializes the Normalize module to normalize a tensor of any shape
        between 0 and 1 based on 3 standard deviations of the input data.
        """
        super(PreprocessNormalize, self).__init__()

    def forward(self, tensor, std_bound = 6):
        """
        Normalizes a tensor of any shape between 0 and 1 using 3 standard deviations
        of the data's mean, then clips the values to the [0, 1] range.

        Args:
            tensor (torch.Tensor): An n-dimensional tensor to be normalized.

        Returns:
            torch.Tensor: The normalized and clipped tensor.
        """
        # Compute the mean and standard deviation of the input tensor
        mean = tensor.mean()
        std = tensor.std()

        # Define the lower and upper bounds based on 3 standard deviations
        lower_bound = mean - std_bound * std
        upper_bound = mean + std_bound * std

        if lower_bound < 0:
            lower_bound = 0

        # Normalize the tensor between 0 and 1
        normalized_tensor = (tensor - lower_bound) / (upper_bound - lower_bound)

        # Clip values to be between 0 and 1
        normalized_tensor = torch.clamp(normalized_tensor, min=0.0, max=1.0)

        return normalized_tensor

def preprocess(img_size=224):
    preprocess_func = nn.Sequential( 
        transforms.Resize([img_size,img_size],antialias=True),
        Normalize(),
        # transforms.Resize([520,520],antialias=True),
        # transforms.Resize([384,384],antialias=True),
        transforms.Grayscale()
    )
    return preprocess_func

def preprocess_indiv_worm(img_size=224):
    preprocess_func = nn.Sequential( 
        transforms.Resize([img_size,img_size],antialias=True),
        Normalize_indiv_worm(),
        # transforms.Resize([520,520],antialias=True),
        # transforms.Resize([384,384],antialias=True),
        transforms.Grayscale()
    )
    return preprocess_func

def preprocess_labels(img_size=224):
    preprocess_labels_func = nn.Sequential( 
        transforms.Resize([img_size,img_size],antialias=True),
        # Normalize(),
        # transforms.Resize([520,520],antialias=True),
        # transforms.Resize([384,384],antialias=True),
        transforms.Grayscale()
    )
    return preprocess_labels_func

def detect_hough_circles(input_img, img_size = 128):

    img = input_img>torch.mean(input_img)
    img = np.atleast_3d((img.cpu().squeeze().numpy()*255).astype(np.uint8))
    a_img_c = np.concatenate((img,img,img), axis = -1)
    detected_circles = cv2.HoughCircles(np.squeeze(img),  
                       cv2.HOUGH_GRADIENT, 1, 20, 
                    param1 = 60, # these two parameters param1 > 2*param2 (??)
                    param2 = 10, # lower is more false readings
                    minRadius = 20, maxRadius = 75) 

    if detected_circles is not None:
        detected_circles = np.squeeze(np.uint16(np.around(detected_circles)))
        if detected_circles.shape.__len__() > 1:
            detected_circles = detected_circles[0,:]
        a, b, r = detected_circles[0], detected_circles[1], detected_circles[2] 
        circle_mask = torch.tensor(cv2.circle(np.zeros((img_size,img_size)), (a, b), r, 1, -1)) 
    else:
        circle_mask = torch.zeros((img_size,img_size))

    return circle_mask

def bwareaopen(binary_image, min_size, device = 'cpu'):
    """
    Removes all connected components in a binary image that have fewer than `min_size` pixels.

    Args:
        binary_image (torch.Tensor): A 2D binary image tensor (HxW) with values 0 and 1.
        min_size (int): The minimum number of pixels for a component to be retained.

    Returns:
        torch.Tensor: The binary image with small components removed.
    """
    # Ensure binary_image is a single-channel 2D binary tensor
    if len(binary_image.shape) != 2 or binary_image.dtype != torch.uint8:
        raise ValueError("binary_image must be a 2D binary tensor of type torch.uint8. or IMAGE IS WRONG SHAPE")

    # Reshape to (1, 1, H, W) for kornia connected_components function
    binary_image = binary_image.unsqueeze(0).unsqueeze(0).float()  # Shape: (1, 1, H, W)

    # Label connected components using Kornia's connected_components
    labels = connected_components(binary_image, num_iterations=1000)
    labels = labels.squeeze(0).squeeze(0)  # Remove unnecessary dimensions to get back to (H, W)

    # Convert labels to CPU and int64 for bincount
    labels = labels.to(torch.int64).cpu()

    # Compute the area (number of pixels) for each component
    component_sizes = torch.bincount(labels.flatten())

    # Create a mask to keep only components with size >= min_size
    size_mask = component_sizes >= min_size
    size_mask[0] = 0  # Exclude the background component

    # Move size_mask to the same device as labels and filter the labels
    size_mask = size_mask.to(labels.device)
    filtered_image = size_mask[labels].float()

    filtered_image = filtered_image.to(device)

    return filtered_image

def remove_border_blobs(binary_image, border_percent):
    """
    Remove binary blobs that are within a specified percentage of the borders.

    Parameters:
    - binary_image (np.ndarray): 2D binary image array with blobs (1s) and background (0s).
    - border_percent (float): Percentage of the image width/height defining the border region.

    Returns:
    - np.ndarray: A binary image with border blobs removed.
    """
    # Calculate the border size in pixels
    binary_image = binary_image.squeeze().cpu()
    height, width = binary_image.shape
    border_size_x = int(width * border_percent / 100)
    border_size_y = int(height * border_percent / 100)

    # Label the connected components in the binary image
    labeled_image, num_labels = label(binary_image, connectivity=1, return_num=True)

    # Create a mask to mark components touching the border
    mask = np.ones_like(binary_image, dtype=bool)
    mask[border_size_y:height - border_size_y, border_size_x:width - border_size_x] = False

    # Identify and remove border blobs
    for region in regionprops(labeled_image):
        if np.any(mask[region.coords[:, 0], region.coords[:, 1]]):
            # Remove blobs touching the border by setting their pixels to 0
            binary_image[region.coords[:, 0], region.coords[:, 1]] = 0

    return binary_image

def find_and_mark_centroids(binary_image):
    """
    Finds centroids of each binary blob in a binary image tensor and marks each centroid with a star.
    
    Parameters:
    - binary_image (torch.Tensor): A binary image tensor (1s and 0s) on a CUDA device with dtype torch.float32.
    
    Returns:
    - torch.Tensor: An output tensor on CUDA with only the centroids marked.
    """
    # Ensure binary image is on CUDA and float32
    assert binary_image.is_cuda and binary_image.dtype == torch.float32, \
        "Input image must be a CUDA float32 tensor."

    # Label connected components in the binary image
    labels = connected_components(binary_image, num_iterations=100)

    # Initialize an empty image to place the centroids (stars)
    star_image = torch.zeros_like(binary_image)
    delta = 5

    # For each label, calculate the centroid and mark it with a star
    for this_label in torch.unique(labels)[1:]:
        # Find the pixels belonging to the current label
        blob_mask = (labels == this_label)
        if blob_mask.sum() == 0:
            continue

        # Calculate the centroid coordinates (mean of x and y positions)
        y_coords, x_coords = torch.where(blob_mask.squeeze())
        centroid_x = int(torch.mean(x_coords.to(torch.float32)).item())
        centroid_y = int(torch.mean(y_coords.to(torch.float32)).item())

        star_image[0,centroid_y-delta:centroid_y+delta,centroid_x-delta:centroid_x+delta] = 1

    return star_image

def write_outputs_to_images(inputs, outputs, output_path, i = 0, j = '', binary_threshold = 0.5, export_inputs_and_outputs = True):
    if 'Dict' in str(type(outputs)):
        outputs = outputs['out']

    if j != '':
        j = '_' + str(j)

    if inputs.shape != outputs.shape:
        outputs = transforms.Resize([inputs.squeeze().shape[0],inputs.squeeze().shape[1]], antialias=True)(outputs.squeeze().unsqueeze(0).unsqueeze(0)).squeeze()
    
    scaled_toutputs = 1*(normalize_img(outputs)).squeeze().unsqueeze(0)
    BW_toutputs = 1*(outputs>binary_threshold).squeeze().unsqueeze(0)

    Rscaled_toutputs = scaled_toutputs.clone()
    Rscaled_toutputs[BW_toutputs>0.5] = 1
    scaled_toutputs[BW_toutputs>0.5] = 0

    if export_inputs_and_outputs:

        BW_toutputs = torch.concatenate((Rscaled_toutputs, scaled_toutputs, scaled_toutputs ), dim = 0)

        this_tin_rgb = torch.concatenate((inputs,inputs,inputs),0).squeeze()
        norm_output = outputs/outputs.max()
        this_tin_rgb[0] = this_tin_rgb[0] + (norm_output)/2
        this_tin_rgb = torch.clip(this_tin_rgb,min=0.,max=1.)

        out1 = (this_tin_rgb*255).to(torch.uint8).cpu().squeeze()
        out2 = (BW_toutputs*255).to(torch.uint8).cpu().squeeze()
        out = torch.concatenate((out1,out2), dim = -1)

        try:
            write_jpeg(out,os.path.join(output_path,str(i) + str(j) + '.jpg'),100)
        except:
            time.sleep(0.1)
            os.makedirs(output_path,exist_ok=True)
            write_jpeg(out,os.path.join(output_path,str(i) +  str(j) + '.jpg'),100)
    else:
        out = (torch.concatenate((BW_toutputs, BW_toutputs, BW_toutputs ), dim = 0)*255).to(torch.uint8).cpu()
        write_png(out,os.path.join(output_path,str(i) + str(j) + '.png'))

def write_filtered_outputs_to_images(image_squares,output_path, i = 0, j = ''):

    out = s_grid_combined(image_squares)
    out = torch.permute(out,(2,0,1)).cpu()

    try:
        write_jpeg(out,os.path.join(output_path,str(i) + str(j) + '.jpg'),100)
    except:
        time.sleep(0.1)
        write_jpeg(out,os.path.join(output_path,str(i) +  str(j) + '.jpg'),100)

def get_model(model_size = 50, device = torch.device('cpu'), freeze_layers = None, weights = True):

    from torch.nn import Parameter, Conv2d, Sigmoid
    from torch import mean as torch_mean

    if model_size == 50:
        if weights:
            model = segmentation.fcn_resnet50(weights = segmentation.FCN_ResNet50_Weights.DEFAULT)
        else:
            model = segmentation.fcn_resnet50(weights = None, aux_loss = False)
        # model = segmentation.deeplabv3_resnet50(weights = segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
    else:
        if weights:
            model = segmentation.fcn_resnet101(weights = segmentation.FCN_ResNet101_Weights.DEFAULT)
        else:
            model = segmentation.fcn_resnet101(weights = None, aux_loss = False)
        # model = segmentation.deeplabv3_resnet101(weights = segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)

    if freeze_layers is not None:
        if freeze_layers == True:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True

    model.backbone.conv1 = Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
    model.backbone.conv1.requires_grad = True
    model.classifier._modules['4'] = Conv2d(512,1,kernel_size = (1,1),stride = (1,1), bias=True)
    model.classifier._modules['4'].requires_grad = True
    model.classifier._modules['5'] = Sigmoid()

    if weights:
        model.aux_classifier._modules['4'] = Conv2d(256,1,kernel_size = (1,1),stride = (1,1), bias=True)
        model.aux_classifier._modules['4'].requires_grad = True
        model.aux_classifier._modules['5'] = Sigmoid()

        input_weights = Parameter(torch_mean(model.backbone.conv1.weight,1).unsqueeze(1))
        model.backbone.conv1.weight = input_weights

    model = model.to(device)

    return model

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            print('Early Stop counter: 0')
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            print('Early Stop counter: ' + str(self.counter))
            if self.counter >= self.patience:
                return True
        return False

def train_one_epoch(model, epoch, training_loader, optimizer, loss_fn, tb_writer = None):
    running_loss_total = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    stream = tqdm.tqdm(training_loader)
    for i, (inputs, labels) in enumerate(stream, start=1):

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)#['out']

        if 'Dict' in str(type(outputs)):
            outputs = outputs['out']

        if 'tuple' in str(type(outputs)):
            classification_head = outputs[1]
            outputs = outputs[0]

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss_total += loss.item()
        last_loss = running_loss_total / i # loss per batch

        stream.set_description("Epoch: {epoch}. Running LOSS: {metric_monitor}".format(epoch=epoch, metric_monitor=last_loss))

    return last_loss

def training_loop(model, EPOCHS, loss_fn, optimizer, training_loader, validation_loader, testing_loader, output_path = './output/', 
                  weights_outputs_path = os.getcwd(),best_vloss = 1000000, timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                  epoch_number = 0, testing_binary_threshold = 0.5,
                  save_weights = True, save_weights_suffix = 'training', 
                  use_early_stopping = False, early_stop_patience = 10, crop_into_circle = True,
                  test_before_training = True, graph_output_path = ''):

    losses = []

    early_stopper = EarlyStopper(patience=early_stop_patience, min_delta=0)
        
    for epoch in range(epoch_number,EPOCHS):
        torch.cuda.empty_cache()  # Clear unused memory cache

        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model, epoch_number, training_loader, optimizer, loss_fn)#, writer)

        running_vloss_total = 0.0
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            torch.cuda.empty_cache()  # Clear unused memory cache
            stream = tqdm.tqdm(validation_loader)
            # for i, vdata in enumerate(tqdm.tqdm(validation_loader)):
            for i, (vinputs, vlabels) in enumerate(stream, start=1):
                voutputs = model(vinputs)#['out']
                if 'Dict' in str(type(voutputs)):
                    voutputs = voutputs['out']
                if 'tuple' in str(type(voutputs)):
                    vclassification_head = voutputs[1]
                    voutputs = voutputs[0]
                vloss = loss_fn(voutputs, vlabels)
                running_vloss_total += vloss 
                running_vloss = running_vloss_total / i
                stream.set_description("Epoch: {epoch}. Running VAL_LOSS: {metric_monitor}".format(epoch=epoch, metric_monitor=running_vloss))

        avg_vloss = running_vloss #/ (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        losses.append(np.asarray([avg_loss, avg_vloss.cpu().numpy()]))
        loss_epoch_graph_barrier = 0
        running_graph_length = 10000
        if (len(losses) > loss_epoch_graph_barrier) and (len(losses) <= running_graph_length) :
            l1 = plt.plot(np.arange(len(losses)),np.asarray(losses)[:,0],'r')
            l2 = plt.plot(np.arange(len(losses)),np.asarray(losses)[:,1],'b')
            p1 = plt.plot(np.argmin(np.asarray(losses)[:,1]),np.asarray(losses)[:,1][np.argmin(np.asarray(losses)[:,1])], 'go')#, markersize = 15)
            plt.legend(['loss','val_loss','best_val_loss'])
            plt.ylim(bottom = np.min(losses), top = (np.max(losses[loss_epoch_graph_barrier])))
            plt.title('min val loss: ' + str(round(np.min(np.asarray(losses)[:,1]),5)) + ' --- epoch: ' + str(np.argmin(np.asarray(losses)[:,1])))
            plt.savefig(os.path.join(graph_output_path,'output_losses_' + os.path.split(output_path)[-1] + '.png'))
            plt.close('all')
        elif (len(losses) > running_graph_length) and ((len(losses) - np.argmin(np.asarray(losses)[:,1])) <= running_graph_length):
            l1 = plt.plot(np.arange(start=len(losses)-running_graph_length-1, stop = len(losses)), np.asarray(losses)[-running_graph_length-1:,0],'r')
            l2 = plt.plot(np.arange(start=len(losses)-running_graph_length-1, stop = len(losses)),np.asarray(losses)[-running_graph_length-1:,1],'b')
            p1 = plt.plot(np.argmin(np.asarray(losses)[:,1]),np.asarray(losses)[:,1][np.argmin(np.asarray(losses)[:,1])], 'go')#, markersize = 15)
            plt.legend(['loss','val_loss','best_val_loss'])
            plt.ylim(bottom = np.min(losses), top = (np.max(np.asarray(losses)[-running_graph_length-1:,0])))
            plt.title('min val loss: ' + str(round(np.min(np.asarray(losses)[:,1]),5)) + ' --- epoch: ' + str(np.argmin(np.asarray(losses)[:,1])))
            plt.savefig(os.path.join(graph_output_path,'output_losses_' + os.path.split(output_path)[-1] + '.png'))
            plt.close('all')
        else:
            this_running_graph_length = len(losses) - np.argmin(np.asarray(losses)[:,1])
            l1 = plt.plot(np.arange(start=len(losses)-this_running_graph_length-1, stop = len(losses)), np.asarray(losses)[-this_running_graph_length-1:,0],'r')
            l2 = plt.plot(np.arange(start=len(losses)-this_running_graph_length-1, stop = len(losses)),np.asarray(losses)[-this_running_graph_length-1:,1],'b')
            p1 = plt.plot(np.argmin(np.asarray(losses)[:,1]),np.asarray(losses)[:,1][np.argmin(np.asarray(losses)[:,1])], 'go')#, markersize = 15)
            plt.legend(['loss','val_loss','best_val_loss'])
            plt.ylim(bottom = np.min(losses), top = (np.max(losses[-this_running_graph_length:])))
            plt.title('min val loss: ' + str(round(np.min(np.asarray(losses)[:,1]),5)) + ' --- epoch: ' + str(np.argmin(np.asarray(losses)[:,1])))
            plt.savefig(os.path.join(graph_output_path,'output_losses_' + os.path.split(output_path)[-1] + '.png'))
            plt.close('all')
        # Track best performance, and save the model's state
        csv_output = np.round(np.asarray(losses),15)
        csv_output2 = np.zeros(shape=(csv_output.shape[0],csv_output.shape[1]+1))
        csv_output2[:,:2] = csv_output
        csv_output2[np.argmin(csv_output[:,1]),2] = 1
        np.savetxt(os.path.join(graph_output_path,"output_losses_" + os.path.split(output_path)[-1] + ".csv"),csv_output2,delimiter = ',',header = 'loss,val_loss', fmt = '%.15f')

        if avg_vloss < best_vloss:

            del_dir_contents(output_path)
            with torch.no_grad():
                for i, tdata in enumerate(tqdm.tqdm(testing_loader)):
                    tinputs = tdata

                    toutputs = model(tinputs)#['out']

                    if 'tuple' in str(type(toutputs)):
                        tclassification_head = toutputs[1]
                        toutputs = toutputs[0]

                    if toutputs.shape[0] == 1:
                        write_outputs_to_images(tinputs, toutputs, output_path, i = i, binary_threshold = testing_binary_threshold)
                    else:
                        testing_counter = 0
                        for testing_counter in range(toutputs.shape[0]):
                            this_in,this_out = tinputs[testing_counter],toutputs[testing_counter]
                            write_outputs_to_images(this_in, this_out, output_path, i = i,j=testing_counter, binary_threshold = testing_binary_threshold)

            best_vloss = avg_vloss
            if save_weights == True:
                model_path = 'model_{}_{}'.format(timestamp, epoch_number) + '.pt'
                print('improved val_loss')
                torch.save(model.state_dict(), os.path.join(weights_outputs_path,model_path))
            if save_weights == False:
                print('improved val_loss')
            if save_weights == 'last':
                model_path = 'model_{}_{}'.format(timestamp, save_weights_suffix) + '.pt'
                print('improved val_loss')
                torch.save(model.state_dict(), os.path.join(weights_outputs_path,model_path))
        else:
            print('Did NOT improve - best val loss:', best_vloss)

        if use_early_stopping:
            if early_stopper.early_stop(avg_vloss):             
                break

        epoch_number += 1

    return model_path, losses