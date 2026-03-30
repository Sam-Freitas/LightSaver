from torchvision.io.image import read_file, read_image, write_jpeg
# from torchvision.models import segmentation 
# from torchvision import transforms
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
# from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import os, datetime, cv2, glob, tqdm, time, pathlib
from torch.utils.data import Dataset
from natsort import natsorted
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.exposure import rescale_intensity
from segmentation_utils import *

# from network.CMUNeXt import CMUNeXt, CMUNeXt_relu# cmunext, cmunext_s, cmunext_l

plt.ioff()
# check cuda or mps
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    if torch.backends.mps.is_available():
        device = torch.device('mps')

def load_data(img_size, number_to_stop_at = 1000000000000, file_extension = '*.png'):

    print('---loading in images')
    all_imgs = read_all_images(natsorted(glob.glob(os.path.join(imgs_path,file_extension))), transforms = preprocess_indiv_worm(img_size), number_to_stop_at = int(number_to_stop_at))
    print('---loading in labels')
    all_masks= read_all_images(natsorted(glob.glob(os.path.join(masks_path,file_extension))), transforms = preprocess_labels(img_size), number_to_stop_at = int(number_to_stop_at))
    # all_test_imgs = read_all_images(natsorted(glob.glob(os.path.join(testing_path,file_extension))), transforms = preprocess(img_size), number_to_stop_at = int(number_to_stop_at))
    print('---loading in tests')
    all_test_imgs = read_all_images(every_nth(natsorted(glob.glob(os.path.join(testing_path,file_extension))),nth=2), transforms = preprocess_indiv_worm(img_size), number_to_stop_at = int(1000))

    return all_imgs, all_masks, all_test_imgs

# set up all the pathings for graphs, trained weights, and intermediate outputs
graph_output_path = os.path.dirname(os.path.abspath(__file__))

weights_outputs_path = os.path.join(graph_output_path,'trained_weights_mask_fixer')
os.makedirs(weights_outputs_path,exist_ok=True)

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_path =  os.path.join(graph_output_path,'output_training_model_mask_fixer')
os.makedirs(output_path,exist_ok=True)

img_size = 128

# Custom Salt-and-Pepper Noise Transform
class SaltAndPepperNoise(A.ImageOnlyTransform):
    def __init__(self, amount=0.01, salt_vs_pepper=0.5, p=0.3):
        super(SaltAndPepperNoise, self).__init__(p=p)
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper

    def apply(self, img, **params):
        if np.random.rand() < self.p:
            img = np.array(img)
            h, w = img.shape[:2]

            # Salt (white) noise
            num_salt = int(self.amount * h * w * self.salt_vs_pepper)
            coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
            img[coords[0], coords[1]] = 1

            # Pepper (black) noise
            num_pepper = int(self.amount * h * w * (1.0 - self.salt_vs_pepper))
            coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
            img[coords[0], coords[1]] = 0

        return img

class RandomBlobNoise(A.ImageOnlyTransform):
    def __init__(self, blob_size = [150,210], p = 0.3):
        super(RandomBlobNoise, self).__init__(p=p)
        self.blob_size = blob_size

    def apply(self, img, **params):
        if np.random.rand() < self.p:
            img = np.array(img)
            h, w = img.shape[:2]

            this_blob = np.random.randint(self.blob_size[0],self.blob_size[1])

            rng = np.random.default_rng()
            noise = rng.integers(0, 255, (h,w), np.uint8, True)
            # blur the noise image to control the size
            blur = cv2.GaussianBlur(noise, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
            # stretch the blurred image to full dynamic range
            stretch = rescale_intensity(blur, in_range='image', out_range=(0,255)).astype(np.uint8)
            # threshold stretched image to control the size
            thresh = cv2.threshold(stretch, this_blob, 255, cv2.THRESH_BINARY)[1]
            # apply morphology open and close to smooth out and make 3 channels
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
  
            # re-blur the mask to it fits image
            blured_mask = cv2.GaussianBlur(mask, (0,0), sigmaX=0.5, sigmaY=0.5, borderType = cv2.BORDER_DEFAULT)
            # convert and scale back to input
            blurred_mask = ((blured_mask.astype(img.dtype))/255)*np.max(img)

            # add mask to input
            img = np.clip((img+blurred_mask),a_max=np.max(img), a_min = 0)
            pass

        return img

# Define the augmentation pipeline
training_transforms = A.Compose([

    # apply noise to the image
    A.ShotNoise(scale_range=(0.1, 0.33), p=0.25),

    # apply invert
    A.InvertImg(p=0.25),

    # Grid-based operations
    A.RandomGridShuffle(grid=(2,2), p=0.25),
    
    # # Dropout transformations
    A.GridDropout(ratio=0.2, unit_size_range=(10,25), random_offset=True, p=0.25),
    
    # # Crop transformations
    A.RandomResizedCrop(size=(img_size,img_size), scale=(0.5, 1.0), ratio=(0.75, 1.33), p=0.25),
    # A.CenterCrop(height=96, width=96, p=0.25),
    # # Rotation and flips
    A.D4(p=0.75),
    A.Rotate(limit=(-15,15), border_mode = cv2.BORDER_CONSTANT, p=0.25),
    
    # random blob noise
    RandomBlobNoise(blob_size=[129,210],p=1),

    # apply noise to the image
    A.GaussNoise(std_range=(0.2, 0.44),mean_range=(0.0,0.0), p=1),

    # Tensor conversion
    ToTensorV2()
])

# Define the augmentation pipeline
validation_transforms = A.Compose([
        # apply noise to the image
    A.GaussNoise(std_range=(0.2, 0.44),mean_range=(0.0,0.0), p=1),
    # Tensor conversion
    ToTensorV2()
])

load_weights = False #False
batch_size = int((384-64)/5) #int(384) #124 #4 # int((384-64))
early_stop_patience = 10000
training_epochs = 100
use_h5 = True

imgs_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\exported_images\data\labels"
masks_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\exported_images\data\labels"
testing_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\exported_images\data\labels"

inital_weights_path = r"C:\Users\LabPC2\Documents\GitHub\WP_fluorescent_processor\trained_weights_indiv_worm\model_128x128_add_training_noise.pt"

####### this is now full sending with all the data
########read in all the images and then use the "preprocess" to resize and convert them to grayscale (grayscale is just to make sure theyre single dim)
base_h5_path = r'C:\Users\LabPC2\Documents\GitHub\LightSaver\exported_images\data\h5'
os.makedirs(base_h5_path,exist_ok=True)

print('Initializing model')
aux_params=dict(
    pooling='max',             # one of 'avg', 'max'
    dropout=0.3,               # dropout ratio, default is None
    activation='sigmoid',             # activation function, default is None
    classes=1,                 # define number of output labels
)
model = smp.MAnet(encoder_name= 'resnet152',#'resnext101_32x8d',#'timm-res2net50_48w_2s',#'timm-res2net50_26w_4s', resnet34, resnet152
    encoder_depth=5, #3, #5
    encoder_weights= 'imagenet' , #
    decoder_use_batchnorm=True, 
    decoder_channels= (256, 128, 64, 32, 16),#(64, 32, 16),#(256, 128, 64, 32, 16),(256, 128, 64, 32, 16),
    # decoder_pab_channels=64, 
    in_channels=1, 
    classes=1, 
    activation='sigmoid', aux_params=aux_params
).to(device)

loss_fn = BCEDiceLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

if load_weights:
    print('Loading weights:', os.path.normpath(inital_weights_path))
    model.load_state_dict(torch.load(inital_weights_path
        ,weights_only=True)) #### uncomment this to use a previously trained weights 

print('Loading in data:')
if not use_h5:
    print('Loading in images')
    all_imgs, all_masks, all_test_imgs = load_data(img_size,file_extension='*.png',number_to_stop_at = 100000000)

    X_train, X_val, y_train, y_val = train_test_split(all_imgs, all_masks, test_size=0.33, random_state=42)

    print('saving images to h5 format')
    torch.save(X_train,os.path.join(base_h5_path,"X_train.h5"))
    torch.save(X_val,os.path.join(base_h5_path,"X_val.h5"))
    torch.save(y_train,os.path.join(base_h5_path,"y_train.h5"))
    torch.save(y_val,os.path.join(base_h5_path,"y_val.h5"))
    torch.save(all_test_imgs,os.path.join(base_h5_path,"all_test_imgs.h5"))
else:
    print('loading in h5 data')
    X_train = torch.load(os.path.join(base_h5_path,"X_train.h5"),weights_only=False)
    X_val = torch.load(os.path.join(base_h5_path,"X_val.h5"),weights_only=False)
    y_train = torch.load(os.path.join(base_h5_path,"y_train.h5"),weights_only=False)
    y_val = torch.load(os.path.join(base_h5_path,"y_val.h5"),weights_only=False)
    all_test_imgs = torch.load(os.path.join(base_h5_path,"all_test_imgs.h5"),weights_only=False)
    print('finished loading h5 data')

training_dataset = SegmentationDataset(X_train,y_train, device = device, transforms=training_transforms, blur_masks=False)
validation_dataset = SegmentationDataset(X_val,y_val, device = device, transforms=training_transforms, blur_masks=False) ##################whyyyyyyyyyyyyyyyy
testing_dataset = SegmentationDataset(all_test_imgs, None, device = device, transforms=validation_transforms)

training_loader = torch.utils.data.DataLoader(training_dataset, batch_size = batch_size, shuffle = True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = batch_size, shuffle = True)
testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size = batch_size, shuffle = False)

model_path, training_losses = training_loop(model, EPOCHS = training_epochs, loss_fn = loss_fn, optimizer = optimizer, 
                training_loader = training_loader, validation_loader = validation_loader, testing_loader = testing_loader, 
                output_path = output_path, weights_outputs_path = weights_outputs_path, best_vloss = 100, 
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                save_weights = 'last', save_weights_suffix = 'training', 
                test_before_training = True,
                graph_output_path = graph_output_path, use_early_stopping = True,
                early_stop_patience= early_stop_patience)

# ##########
# ####### add random cropping, sunflare, snow, and salt and pepper
# training_transforms = A.Compose([
#     # A.augmentations.crops.transforms.CropAndPad(pad_cval=0,pad_cval_mask=0,keep_size=True,percent=[-0.15, 0.15],p = 0.25), # pad with zeros
#     # A.augmentations.crops.transforms.CropAndPad(pad_mode=2,keep_size=True,percent=[-0.15, 0.15], p = 0.25), # pad with reflect 
#     A.RandomGridShuffle(grid = (4,4), p = 0.25),
#     # A.RandomResizedCrop(size=(img_size,img_size),scale=(0.2,1),p=0.33),
#     A.D4(p=0.75),
#     A.RandomBrightnessContrast(p=0.2),
#     A.RandomGamma(p=0.25),
#     A.RandomToneCurve(p=0.25),
#     A.Rotate(p=0.2),
#     A.CoarseDropout(num_holes_range=(1,6000), hole_height_range=(2,25), hole_width_range=(2,25), p = 0.25),
#     ToTensorV2()
# ])