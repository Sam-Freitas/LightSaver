close all
clear all
warning('off', 'MATLAB:MKDIR:DirectoryExists');
% img_dir_path = "C:\Users\Lab PC\Documents\GFP_AUC\data\Raul_data\2021-02-16\Exported\";

curr_path = pwd;

data_path = fullfile(erase(erase(curr_path,'testing'),'scripts'),'data');

img_dir_path = uigetdir(data_path,'Please select the folder containing the *.tiff files');

[~,final_save_name,~] = fileparts(img_dir_path);

show_output_images = 0;

% USER must specify the number of worms that the program SHOULD find
number_worms_to_detect = 5;

% Dual excitation names
fluorescent_names = ["GFP","DAPI"];

output_path = fullfile(erase(erase(pwd,'testing'),'scripts'),'exported_images');
mkdir(output_path);

img_paths = dir(fullfile(img_dir_path, '*.tif'));
[~,sort_idx,~] = natsort({img_paths.name});

img_paths = img_paths(sort_idx);

image_integral_intensities = zeros(length(img_paths),number_worms_to_detect);
image_integral_area = zeros(length(img_paths),number_worms_to_detect);

images_to_skip = [];

for i = 1:length(img_paths)
        
    % load in image pairings
    [images,imgs_already_processed,final_img_name] = ...
        load_images_fluorecscent_imgs(fluorescent_names,img_dir_path,img_paths,i);
        
    if ~ismember(i,images_to_skip)
        
        % covnert them to gray
        gray_images = cell(size(images));
        color_choices = zeros(size(images));
        for j = 1:length(images)
            [gray_images{j},color_choices(j)] = get_dominant_color(images{j});
        end
        
        blue_fluor_img = gray_images{find(color_choices==3,1)};
        green_fluor_img = gray_images{find(color_choices==2,1)};
        
        % normalize them
        gray_images_norm = cell(size(gray_images));
        for j = 1:length(gray_images)
            gray_images_norm{j} = double(gray_images{j})/double(max(nonzeros(gray_images{j})));
        end
        
        combined_image = zeros(size(images{1}));
        for j = 1:length(color_choices)
            combined_image(:,:,color_choices(j)) = gray_images_norm{j};
        end
        
        data_norm = rgb2gray(combined_image);
        
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
        
        if max(this_label(:))>number_worms_to_detect
            disp(['Warning: more than ' num2str(number_worms_to_detect) ' worms detected - ' img_paths(i).name])
            disp(['Using only the ' num2str(number_worms_to_detect) ' largest blobs'])
            
            this_mask = bwareafilt(this_label>0,number_worms_to_detect);
            
        end
                
        % thicken all the masks
        new_mask = bwmorph(this_mask,'thicken',2);
        % close small edges and zones
        new_mask = imclose(new_mask,strel('disk',3));
        % fill the holes
        new_mask = imfill(new_mask,'holes');
        % re-label the masks
        labeled_masks = bwlabel(new_mask);
        
        green_fluor_img = double(green_fluor_img);
        blue_fluor_img = double(blue_fluor_img);
        
        B = blue_fluor_img+1;
        G = green_fluor_img+1;
        
        cmap = colormap('hot');
        
        BG = (B./G).*new_mask;
        
        BG_color = ind2rgb(im2uint8(BG),cmap);
        
        % mask the inital data without the normalization step
        % gets rid of background signals
        
        %%%%%%%%%%%%%%%%%%%% change this to data for each colors
        masked_data = new_mask.*double(data_norm);
        
        % integrate the entire signal across the masks and only the masks
        
        %%%%%%%%%%%%%%%%%%%%%% have to change this for each of the colors 
        
        for j = 1:max(labeled_masks(:))
            this_labeled_mask = double(data_norm).*(labeled_masks==j);
            image_integral_intensities(i,j) = sum(sum(this_labeled_mask.*double(data_norm)));
            image_integral_area(i,j) = sum(sum((this_labeled_mask.*double(data_norm))>0));
        end
        
        % converts the labeled mask to RGB (easier to read)
        rgb_labeled_mask = label2rgb(labeled_masks,'jet','k');
        % get the masked data ready for export
        masked_data_output = masked_data/max(masked_data(:));
        
        % write the image sequence to the export folder
        imwrite(imtile({combined_image,rgb_labeled_mask,masked_data_output,BG_color},'GridSize',[1,4]),...
            fullfile(output_path,[char(final_img_name) '_img' num2str(i) '.png']))
        
        % show the sequence if necessary
        if show_output_images == 1
            imshow(imtile({combined_image,rgb_labeled_mask,masked_data_output,BG_color},'GridSize',[1,4]),[]);
            title([char(final_img_name) ' -- img ' num2str(i)], 'Interpreter', 'none');
        end
        
        linear_data = nonzeros(masked_data);
        
        [counts,binLoc] = hist(linear_data,255);
        
    else
        
        disp('image already processed');
        
    end
    
    images_to_skip = unique([images_to_skip,imgs_already_processed]);

    
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
close all

function [out_img,color_choice] = get_dominant_color(in_img)

% if for some reason the luma,blue, or red difference were saved aswell
[~,~,z] = size(in_img);
if z>3
    in_img = in_img(:,:,1:3);
end

% Split channles
R = in_img(:,:,1); G = in_img(:,:,2); B = in_img(:,:,3);

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

get_rid_of_remainder_simple_mask = bwareaopen(data>0,25,4);

out_img = data.*(uint8(get_rid_of_remainder_simple_mask));

end

function [images,imgs_already_processed,final_img_name] = load_images_fluorecscent_imgs(fluorescent_names,img_dir_path,img_paths,i)

% read the image into ram

first_image_name = img_paths(i).name;

all_img_names = {img_paths.name};

first_image_name2 = first_image_name;
for j = 1:length(fluorescent_names)
    first_image_name2 = erase(first_image_name2,char(fluorescent_names(j)));
end

final_img_name = first_image_name2;

all_img_names2 = all_img_names;
for j = 1:length(all_img_names2)
    for k = 1:length(fluorescent_names)
        all_img_names2{j} = erase(all_img_names2{j},char(fluorescent_names(k)));
    end
end

idx = [];
for j = 1:length(all_img_names2)
    tf = strcmp(first_image_name2,all_img_names2{j});
    
    if tf
        idx = [idx j];
    end
end

images = cell(1,length(idx));
disp('image pairs')
for j = 1:length(idx)
    try
        this_img = imread(fullfile(img_dir_path,img_paths(idx(j)).name));
        disp(img_paths(idx(j)).name)
    catch
        disp(['ERROR: reading image - ' img_paths(i).name])
        disp(['Image will be treated as corrupted and skipped']);
        
        try
            this_img = zeros(size(this_img));
        catch
            this_img = zeros(1024,1024);
        end
    end
    
    images{j} = this_img;
end

imgs_already_processed = idx;

end
