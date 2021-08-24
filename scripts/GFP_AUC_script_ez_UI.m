close all
clear all
warning('off', 'MATLAB:MKDIR:DirectoryExists');
% img_dir_path = "C:\Users\Lab PC\Documents\GFP_AUC\data\Raul_data\2021-02-16\Exported\";

% User input choice
prompt = {'Enter number of worms to detect: ', ...
    'Show output images - yes(1) - no(0)',...
    'Use large blob fix - yes (1) - no(0)',...
    'Output name - leave blank for defaults - or enter name for exported_images sub-folder'};
dlgtitle = 'User Inputs for Lightsaver';
dims = [1 100];
definput = {'5','0','0',''};
answer = inputdlg(prompt,dlgtitle,dims,definput);

if isempty(answer)
    error('Please select user inputs')
end

number_worms_to_detect = str2double(answer{1});
show_output_images = str2double(answer{2});
use_large_blob_fix = str2double(answer{3});
output_name = answer{4};

% clean up variables
clearvars dims definput dlgtitle prompt answer

% get current path
curr_path = pwd;

data_path = fullfile(erase(curr_path,'scripts'),'data');

img_dir_path = uigetdir(data_path,'Please select the folder containing the *.tiff files');

[~,final_save_name,~] = fileparts(img_dir_path);

output_path = fullfile(erase(erase(pwd,'GFP_AUC_script.m'),'scripts'),'exported_images');
mkdir(output_path);

if isempty(output_name)
    output_name = final_save_name;
end

output_path = fullfile(output_path,output_name);
mkdir(output_path);

img_paths = dir(fullfile(img_dir_path, '*.tif'));
[~,sort_idx,~] = natsort({img_paths.name});

img_paths = img_paths(sort_idx);

image_integral_intensities = zeros(length(img_paths),number_worms_to_detect);
image_integral_area = zeros(length(img_paths),number_worms_to_detect);

se = strel('disk',5);

for i = 1:length(img_paths)
    
    [data,this_img] = load_fluor_image(img_dir_path,img_paths,i);
    
    % this assumes that all the data is in the uint8 format
    data_norm = double(data)/255;
    
    % setp through consequitive iterations of a threshold based off the
    % mean and std of the image intensities
    for j = 1:6
        % create a threshold
        this_thresh = mean2(data_norm)+(std2(data_norm)*(1/5)*(j-1));
        % create a mask
        this_mask = imgaussfilt(data_norm,2)>this_thresh;
        % remove any small blobs from the mask
        this_mask = bwareaopen(this_mask,3000);
        % label the mask
        this_label = bwlabel(this_mask);
        
        % if there are 5 blobs in the mask
        if max(this_label(:)) == number_worms_to_detect
            % step one iteration further
            this_thresh2 = mean2(data_norm)+(std2(data_norm)*(1/5)*(j));
            this_mask2 = imgaussfilt(data_norm,2)>this_thresh2;
            this_mask2 = bwareaopen(this_mask2,3000);
            this_label2 = bwlabel(this_mask2);
            
            % if there are still 5 blobs then keep this mask
            if max(this_label2(:)) == number_worms_to_detect
                this_mask = this_mask2;
                this_label = this_label2;
                
                % break out of the loop
                break
            end
            % break out of the loop if there are 5 blobs
            break
        end
        
    end
    
    % if there are many blobs still detected only take the 5 largest
    if max(this_label(:))>number_worms_to_detect
        disp(['Warning: more than ' num2str(number_worms_to_detect) ' worms detected - ' img_paths(i).name])
        disp(['Using only the ' num2str(number_worms_to_detect) ' largest blobs'])
        
        this_mask = bwareafilt(this_label>0,number_worms_to_detect);
        
    end
    
    % if this is chosen then a semi-smart filter will try to fix the large
    % blobs that contain multiple worms
    if use_large_blob_fix
        
        [this_label,this_mask,large_blob_is_fixed] = large_blob_fix(this_label,...
            this_mask,data_norm,img_paths,number_worms_to_detect,i);
        
    end
    
    
    % thicken all the masks
    new_mask = bwmorph(this_mask,'Thicken',1);
    % close small edges and zones
    new_mask = imclose(new_mask,strel('disk',5));
    % fill the holes
    new_mask = imfill(new_mask,'holes');
    % re-label the masks
    labeled_masks = bwlabel(new_mask);
    
    % mask the inital data without the normalization step
    % gets rid of background signals
    masked_data = new_mask.*double(data);
    
    % integrate the entire signal across the masks and only the masks
    for j = 1:max(labeled_masks(:))
        this_labeled_mask = double(data).*(labeled_masks==j);
        image_integral_intensities(i,j) = sum(sum(this_labeled_mask));
        image_integral_area(i,j) = sum(sum((this_labeled_mask)>0));
    end
    
    % converts the labeled mask to RGB (easier to read)
    rgb_labeled_mask = label2rgb(labeled_masks,'jet','k');
    % get the masked data ready for export
    masked_data_output = masked_data/max(masked_data(:));

    [~,this_img_name,~] = fileparts(img_paths(i).name);
    
    if large_blob_is_fixed
        this_img_name = ['_L_Blob_fix_' this_img_name];
    end
    
    % write the image sequence to the export folder
    imwrite(imtile({this_img,rgb_labeled_mask,masked_data_output},'GridSize',[1,3]),...
        fullfile(output_path,[this_img_name '_img' num2str(i) '.png']))
    
    % show the sequence if necessary
    if show_output_images == 1
        imshow(imtile({this_img,rgb_labeled_mask,masked_data_output},'GridSize',[1,3]),[]);
        title([img_paths(i).name ' -- img ' num2str(i)], 'Interpreter', 'none');
    end
    
    linear_data = nonzeros(masked_data);
    
    [counts,binLoc] = hist(linear_data,255);
    
end

output_csv = cell(1 + length(img_paths),11);

output_header = {'Image names',...
    'Worm 1 (blue) integrated Intensity','Worm 2 (teal) integrated Intensity','Worm 3 (green) integrated Intensity','Worm 4 (yellow/red) integrated Intensity','Worm 5 (orange) integrated Intensity',...
    'Worm 1 (blue) integrated Area','Worm 2 (teal) integrated Area','Worm 3 (green) integrated Area','Worm 4 (yellow/red) integrated Area','Worm 5 (orange) integrated Area'};

output_csv(1,:) = output_header;
output_csv(2:end,2:6) = num2cell(image_integral_intensities);
output_csv(2:end,7:11) = num2cell(image_integral_area);
for i = 1:length(img_paths)
    output_csv{i+1,1} = img_paths(i).name;
end

T = cell2table(output_csv(2:end,:),'VariableNames',output_csv(1,:));
writetable(T,fullfile(char(img_dir_path),'data.csv'))

if isfile(fullfile(data_path,[final_save_name '.csv']))
    writetable(T,fullfile(data_path,[final_save_name,datestr(now, 'dd-mmm-yyyy'),'_.csv']))
else
    writetable(T,fullfile(data_path,[final_save_name '.csv']))
end

disp(' ')
disp('End of scrip')


function [data,this_img] = load_fluor_image(img_dir_path,img_paths,i)
% read the image into ram
try
    this_img = imread(fullfile(img_dir_path,img_paths(i).name));
catch
    disp(['ERROR: reading image - ' img_paths(i).name])
    disp(['Image will be treated as corrupted and skipped']);

    try
        this_img = zeros(size(this_img));
    catch
        this_img = zeros(1024,1024);
    end
end

% if for some reason the luma,blue, or red difference were saved aswell
[~,~,z] = size(this_img);
if z>3
    this_img = this_img(:,:,1:3);
end

% Split channles
[R,G,B] = imsplit(this_img);

% find the dominant color of the fluorescence
[~,color_choice] = max([sum(R(:)),sum(G(:)),sum(B(:))]);

% Remove 1mm bar from our microscope images
switch color_choice
    case 1
        % red fluorescence
        data = R - G - B;
    case 2
        % green fluorescence
        data = G - B - R;
    case 3
        % blue fluorescence
        data = B - R - G;
end
end


function [this_label,this_mask,is_fixed] = large_blob_fix(this_label,this_mask,...
    data_norm,img_paths,number_worms_to_detect,i)

is_fixed = 0;

% get region profile
s = regionprops(this_label,'basic');

% determine if the areas are correct
is_over_large(i) = sum([s.Area]>20000);

% if they are not 
if is_over_large(i)
    disp(['Warning: Large blobs detected in - ' img_paths(i).name])
    disp(['Attempting to fix blobs'])

    % find which blobs are not correct
    is_over_idx = nonzeros(([s.Area]>20000).*(1:length([s.Area])));

    % get the first mask without the improper blobs
    first_mask = zeros(size(this_label));
    for j = 1:max(this_label(:))
        if ~ismember(j,is_over_idx)
            first_mask = first_mask + (this_label==j);
        end
    end

    % iterate
    for j = 1:length(is_over_idx)
        % find the large blob
        temp_mask = (this_label==is_over_idx(j));
        % isolate the data
        temp_data_norm = (temp_mask.*data_norm);
        % create a new mask 
        temp_mask2 = temp_data_norm>mean2(nonzeros(temp_data_norm));
        % add that to the old masks 
        first_mask = first_mask + temp_mask2;

    end
    % isolate the 5 largest blobs
    this_mask = bwareafilt(imfill(first_mask>0,'holes'),...
        number_worms_to_detect);
    % label them
    this_label = bwlabel(this_mask);
    
    is_fixed = 1;

end
        
end
