from segmentation_utils import *
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

## this script attempts to take in fluorescent images of individual fluorescent wells and export worm segments
## the script uses a segmentation network to isolate the individual worms 
## then exports 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    if torch.backends.mps.is_available():
        device = torch.device('mps')

testing_transforms = A.Compose([
    ToTensorV2()
])

image_type = '.tif'

# testing_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\data\training data"
testing_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\data\Leica Images Kayla Miller"
# testing_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\data\Leica Images Skye Rounsville"
testing_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\data\Leica Image Aadith Mosur"
testing_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\data\Leica Images Raul Castro"
# testing_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\data\Leica Images Robert Railey"
# testing_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\data\Leica Images Brad Hull"
# testing_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\data"
print('Recursively finding all' ,image_type ,'image(s) in path:')
print(testing_path,'\n')
image_paths = find_files(testing_path,file_extension=image_type)

# specify outputs
output_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\scripts_python\network_testing\outputs"
output_path = os.path.join(output_path,os.path.split(testing_path)[-1])
os.makedirs(output_path,exist_ok=True)
del_dir_contents(output_path)

print('Exporting images to:')
print(output_path,'\n')

batch_size = 1
testing_binary_threshold = 0.51
img_size = 224
lightsaver_output = True

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

# model_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\scripts_python\network_testing\trained_weights_indiv_worm_imgsz224_p00483\model_20260316_145746_training.pt"
model_path = r"C:\Users\LabPC2\Documents\GitHub\LightSaver\scripts_python\network_testing\trained_weights_indiv_worm_imgsz128_p00483\model_20260319_140747_training.pt"

print('Loading in model weights from:')
print(model_path,'\n')
# load previously trained weights for the model and set it as evaluation mode 
model.load_state_dict(
    torch.load(model_path
        , weights_only = True)) #### uncomment this to use a previously trained weights 
model.eval()

testing_dataset = SegmentationDataset(image_paths, None, device = device, transforms=None,
                                        return_intial_img_aswell=True,return_path_aswell=True,resize=preprocess_indiv_worm(img_size))
testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size = batch_size, shuffle = False)

# run through all and dump to jpg
print('Processing images')
with torch.no_grad(): #@torch.inference_mode()
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

        # make sure that the image size and dimensions are correct
        # (X,1,img_size,img_size)
        if batch_size == 1:
            tinputs = (tinputs.squeeze()).unsqueeze(0).unsqueeze(0)
        else:
            tinputs = tinputs.squeeze().unsqueeze(1)

        # run the images through the model        
        toutputs = model(tinputs)#['out']

        classification_head = toutputs[1]
        toutputs = toutputs[0]

        if lightsaver_output:

            labeled_image = label(toutputs)
            normalized_masked_img = np.zeros_like(raw_img)
            
            export_images = [raw_img,tinputs,normalized_masked_img]
            grid = resize_and_construct_grid(images=export_images,title=img_name)
            

        else:
            # dump the outputs to a side by side jpg of the input and output masks
            if tinputs.shape[0] > 1:
                for j, (each_input, each_output) in enumerate(zip(tinputs,toutputs)):
                    write_outputs_to_images(each_input, 1*each_output, output_path, i = i, j =  '_' + str(j) ,binary_threshold = testing_binary_threshold)
            else:                                                                                   
                write_outputs_to_images(tinputs, 1*toutputs, output_path, i = img_name , binary_threshold = testing_binary_threshold, export_inputs_and_outputs=True)
