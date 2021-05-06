close all
clear all
warning('off', 'MATLAB:MKDIR:DirectoryExists');
% img_dir_path = "C:\Users\Lab PC\Documents\GFP_AUC\data\Raul_data\2021-02-16\Exported\";

curr_path = pwd;

data_path = fullfile(erase(curr_path,'scripts'),'data');

img_dir_path = uigetdir(data_path,'Please select the folder containing the *.tiff files');

[~,final_save_name,~] = fileparts(img_dir_path);

show_output_images = 0;

% USER must specify the number of worms that the program SHOULD find
number_worms_to_detect = 5;

output_path = fullfile(erase(erase(pwd,'GFP_AUC_script.m'),'scripts'),'exported_images');
mkdir(output_path);

img_paths = dir(fullfile(img_dir_path, '*.tif'));
[~,sort_idx,~] = natsort({img_paths.name});

img_paths = img_paths(sort_idx);

image_integral_intensities = zeros(length(img_paths),number_worms_to_detect);
image_integral_area = zeros(length(img_paths),number_worms_to_detect);

std_grain = 20;
extra_steps = 10;

for i = 1:length(img_paths)
    
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
    R = this_img(:,:,1); G = this_img(:,:,2); B = this_img(:,:,3);
    
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
    
    % this assumes that all the data is in the uint8 format
    data_norm = double(data)/255;
    
    % setp through consequitive iterations of a threshold based off the
    % mean and std of the image intensities
    
    grain_vec = zeros(1,std_grain+1);
    
    repeat_bool = 1;
    for j = 1:std_grain+extra_steps
        % create a threshold
        this_thresh = mean2(data_norm)+...
            (std2(data_norm)*(1/std_grain)*(j-1));
        % create a mask
        this_mask = imgaussfilt(data_norm,2)>this_thresh;
        % remove any small blobs from the mask
        this_mask = bwareaopen(this_mask,3000);
        
        grain_vec(j) = sum(this_mask(:));
        % label the mask         
        this_label = bwlabel(this_mask);
        
        if repeat_bool
            if max(this_label(:)==number_worms_to_detect)
                first_5_idx = j;
                repeat_bool = 0;
            end
        end
    end
    
    intensity_vec_diff_abs = abs(diff(grain_vec));
    
    good_idx_for_mask = find(intensity_vec_diff_abs<1000,1,'first');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% I HAVE NO IDEA WHAT TO DO
    
    if isempty(good_idx_for_mask)
        [~,good_idx_for_mask] = min(intensity_vec_diff_abs);
    end
    
    if ~isequal(good_idx_for_mask,first_5_idx)
        good_idx_for_mask = first_5_idx+1;
    end
    
    this_thresh = mean2(data_norm)+(std2(data_norm)*(1/std_grain)*(good_idx_for_mask-1));
    this_mask = imgaussfilt(data_norm,2)>this_thresh;
    this_mask = bwareaopen(this_mask,3000);
    this_label = bwlabel(this_mask);
    
    if max(this_label(:))>number_worms_to_detect
        disp(['Warning: more than ' num2str(number_worms_to_detect) ' worms detected - ' img_paths(i).name])
        disp(['Using only the ' num2str(number_worms_to_detect) ' largest blobs'])
        
        this_mask = bwareafilt(this_label>0,number_worms_to_detect);
        
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
        image_integral_intensities(i,j) = sum(sum(this_labeled_mask.*double(data)));
        image_integral_area(i,j) = sum(sum((this_labeled_mask.*double(data))>0));
        
%         if image_integral_area(i,j)>20000
%             image_integral_area(i,j) = 0;
%             image_integral_intensities(i,j) = 0;
%         end
    end
    
    % converts the labeled mask to RGB (easier to read)
    rgb_labeled_mask = label2rgb(labeled_masks,'jet','k');
    % get the masked data ready for export
    masked_data_output = masked_data/max(masked_data(:));
    
    % write the image sequence to the export folder
    imwrite(imtile({this_img,rgb_labeled_mask,masked_data_output},'GridSize',[1,3]),...
        fullfile(output_path,[img_paths(i).name '_img' num2str(i) '.png']))
    
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
