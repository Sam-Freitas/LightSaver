from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import torch
import os, glob, tqdm, time
from torch.utils.data import Dataset
from natsort import natsorted
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_utils import *

## this script attempts to take in fluorescent images of individual fluorescent wells and export worm segments
## the script uses a segmentation network to isolate the individual worms 
## then exports 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    if torch.backends.mps.is_available():
        device = torch.device('mps')

testing_transforms = A.Compose([
            # apply noise to the image
    # A.GaussNoise(std_range=(0.2, 0.44),mean_range=(0.0,0.0), p=1),
    ToTensorV2()
])

testing_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\data\training data"
# testing_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\data\Leica Images Kayla Miller"
# testing_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\data\Leica Images Skye Rounsville"
# testing_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\data\Leica Image Aadith Mosur"
# testing_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\data\Leica Images Raul Castro"
# testing_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\data\Leica Images Robert Railey"
# testing_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\data\Leica Images Brad Hull"
testing_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\data"
image_paths = find_files(testing_path,file_extension='.tif')

# specify outputs
output_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\scripts_python\network_testing\outputs"
output_path = os.path.join(output_path,os.path.split(testing_path)[-1])
os.makedirs(output_path,exist_ok=True)
del_dir_contents(output_path)

batch_size = 1
testing_binary_threshold = 0.51
img_size = 128
# model = get_this_model()

aux_params=dict(
    pooling='max',             # one of 'avg', 'max'
    dropout=0.0,               # dropout ratio, default is None
    activation='sigmoid',             # activation function, default is None
    classes=1,                 # define number of output labels
)
model = smp.MAnet(encoder_name= 'resnet152',#'timm-res2net50_48w_2s',#'timm-res2net50_26w_4s', 
    encoder_depth=5, #3, #5
    encoder_weights= 'imagenet' , #
    decoder_use_batchnorm=True, 
    decoder_channels= (256, 128, 64, 32, 16),#(64, 32, 16),#(256, 128, 64, 32, 16),(256, 128, 64, 32, 16),
    # decoder_pab_channels=64, 
    in_channels=1, 
    classes=1, 
    activation='sigmoid', aux_params=aux_params
).to(device)


model_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\scripts_python\network_testing\trained_weights_indiv_worm_imgsz128_p00483\model_20260312_153227_training.pt"

print(model_path)
# load previously trained weights for the model and set it as evaluation mode 
model.load_state_dict(
    torch.load(model_path
        , weights_only = True)) #### uncomment this to use a previously trained weights 
model.eval()

all_test_imgs = read_all_images(
    image_paths, 
    transforms = preprocess_indiv_worm(img_size), 
    number_to_stop_at = 1000000000000)

testing_dataset = SegmentationDataset(all_test_imgs, None, device = device, transforms=testing_transforms,
                                        return_intial_img_aswell=True,return_path_aswell=True)
testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size = batch_size, shuffle = False)

# run through all and dump to jpg
with torch.no_grad():
    # for i, (tinputs, raw_img) in enumerate(tqdm.tqdm(testing_loader)): # this is for retunring the raw image of the input without processing -- doesnt work with mixed datasets of different sizes 
    for i, tinputs in enumerate(tqdm.tqdm(testing_loader)):

        if tinputs[0].shape[0] == 1:
            if 'list' in str(type(tinputs)): # this works for single batch
                if len(tinputs)==3:
                    img_path = tinputs[2][0]
                    img_path = image_paths[i]
                    img_name = os.path.split(img_path)[-1][0:-4]
                else:
                    img_path = None
                raw_img = tinputs[1]
                tinputs = tinputs[0]
        else:
            temp = tinputs
            tinputs = tinputs[0]

        if batch_size == 1:
            tinputs = (tinputs.squeeze()).unsqueeze(0).unsqueeze(0)
        else:
            tinputs = tinputs.squeeze().unsqueeze(1)

        # run the images through the model        
        toutputs = model(tinputs)#['out']

        classification_head = toutputs[1]
        toutputs = toutputs[0]

        # dump the outputs to a side by side jpg of the input and output masks
        if tinputs.shape[0] > 1:
            for j, (each_input, each_output) in enumerate(zip(tinputs,toutputs)):
                write_outputs_to_images(each_input, 1*each_output, output_path, i = i, j =  '_' + str(j) ,binary_threshold = testing_binary_threshold)
        else:                                                                                   
            write_outputs_to_images(tinputs, 1*toutputs, output_path, i = img_name , binary_threshold = testing_binary_threshold, export_inputs_and_outputs=True)
