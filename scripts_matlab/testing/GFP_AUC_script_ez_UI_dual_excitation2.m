close all
clear all
warning('off', 'MATLAB:MKDIR:DirectoryExists');
% img_dir_path = "C:\Users\Lab PC\Documents\GFP_AUC\data\Raul_data\2021-02-16\Exported\";

prompt = {'Enter color choice: RG RB GR GB BG BR or RGB',...
    'Enter Red Label: - Leave as is if no selection necessary',...
    'Enter Green Label: - Leave as is if no selection necessary',...
    'Enter Blue Label: - Leave as is if no selection necessary',...
    'Combinatorial (0) or Ratio (1)',...
    'False color scaler',...
    'false color colormap',...
    'Number of worms to select'...
    'Show output images - yes(1) - no(0)'};
dlgtitle = 'User Inputs for Multi-excitation Lightsaver';
dims = [1 100];
definput = {'BG','RFP','GFP','DAPI','1','4','hot','5','0'};
answer = inputdlg(prompt,dlgtitle,dims,definput);

if isempty(answer)
    error('Please select user inputs')
end

color_choice = answer{1};
fluorescent_names = [string(answer{2}),string(answer{3}),string(answer{4})];
combinatorial_or_ratio = str2double(answer{5});
false_color_scaler = str2double(answer{6});
this_cmap_choice = answer{7};
number_worms_to_detect = str2double(answer{8});
show_output_images = str2double(answer{9});

if combinatorial_or_ratio && isequal(color_choice,'RGB')
    error('Cant do ratio and RGB comparison')
end

curr_path = pwd;

data_path = fullfile(erase(erase(curr_path,'testing'),'scripts'),'data');

img_dir_path = uigetdir(data_path,'Please select the folder containing the *.tiff files');

if ~img_dir_path
    error('Please select the folder containing the *.tiff files')
end

[~,final_save_name,~] = fileparts(img_dir_path);

output_path = fullfile(erase(erase(pwd,'testing'),'scripts'),'exported_images');
mkdir(output_path);
mkdir(fullfile(output_path,final_save_name))

output_path = fullfile(output_path,final_save_name);

img_paths = dir(fullfile(img_dir_path, '*.tif'));
[~,sort_idx,~] = natsort({img_paths.name});

cmap = colormap(this_cmap_choice);
cmap(1,:) = [0,0,0];
close all

img_paths = img_paths(sort_idx);

num_pairs = length(img_paths)/length(color_choice);

image_integral_intensities_red = zeros(num_pairs,number_worms_to_detect);
image_integral_intensities_green = zeros(num_pairs,number_worms_to_detect);
image_integral_intensities_blue = zeros(num_pairs,number_worms_to_detect);
image_integral_intensities_comb = zeros(num_pairs,number_worms_to_detect);
image_integral_area = zeros(num_pairs,number_worms_to_detect);

final_img_names = cell(num_pairs,1);

images_to_skip = [];

paired_image_counter = 1;

for i = 1:length(img_paths)
    
    % load in image pairings
    [images,imgs_already_processed,final_img_name] = ...
        load_images_fluorecscent_imgs(fluorescent_names,img_dir_path,img_paths,i);
    
    if ~ismember(i,images_to_skip)
        
        % get the final image name from the pairings
        final_img_names{paired_image_counter} = final_img_name;
        
        % get the RGB and combined (normalized colors) image
        [red_fluor_img,green_fluor_img,blue_fluor_img,combined_image] = ...
            isolate_fluorescence_images(images);
        
        
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
        
        % close small edges and zones
        new_mask = imclose(this_mask,strel('disk',3));
        % fill the holes
        new_mask = imfill(new_mask,'holes');
        % re-label the masks
        labeled_masks = bwlabel(new_mask);
        
        mask3 = double(cat(3,new_mask,new_mask,new_mask));
        combined_image = combined_image.*mask3;
        
        [out_img] = ...
            fluor_data_comparison(red_fluor_img,green_fluor_img,blue_fluor_img,...
            combined_image,new_mask,...
            color_choice,combinatorial_or_ratio);
        
        %         BG_color = ind2rgb(im2uint8(BG_gray/false_color_scaler),cmap);
        out_img_color = ind2rgb(im2uint8(out_img/false_color_scaler),cmap);
        
        %%%%%%%%%%%%%%%%%%%% change this to data for each colors
        masked_data = new_mask.*double(data_norm);
        
        % integrate the entire signal across the masks and only the masks
        for j = 1:max(labeled_masks(:))
            this_labeled_mask = (labeled_masks==j);
            image_integral_intensities_red(paired_image_counter,j) = sum(sum(this_labeled_mask.*double(red_fluor_img)));
            image_integral_intensities_blue(paired_image_counter,j) = sum(sum(this_labeled_mask.*double(blue_fluor_img)));
            image_integral_intensities_green(paired_image_counter,j) = sum(sum(this_labeled_mask.*double(green_fluor_img)));
            image_integral_intensities_comb(paired_image_counter,j) = sum(sum(this_labeled_mask.*double(out_img)));
            image_integral_area(paired_image_counter,j) = sum(sum(this_labeled_mask>0));
        end
        
        % converts the labeled mask to RGB (easier to read)
        rgb_labeled_mask = label2rgb(labeled_masks,'jet','k');
        % get the masked data ready for export
        masked_data_output = masked_data/max(masked_data(:));
        
        % write the image sequence to the export folder
        imwrite(imtile({combined_image,masked_data_output,out_img_color,rgb_labeled_mask},'GridSize',[2,2]),...
            fullfile(output_path,[char(final_img_name) '_img' num2str(paired_image_counter) '.png']))
        
        % show the sequence if necessary
        if show_output_images == 1
            imshow(imtile({combined_image,masked_data_output,out_img_color,rgb_labeled_mask},'GridSize',[2,2]),[]);
            title([char(final_img_name) ' -- img ' num2str(paired_image_counter)], 'Interpreter', 'none');
        end
        
        paired_image_counter = paired_image_counter + 1;
        
    else
        
        disp('image already processed');
        
    end
    
    images_to_skip = unique([images_to_skip,imgs_already_processed]);
    
    
end

output_csv = cell(1 + num_pairs,15);

output_header = {'Image names','color choice','comparison','colormap','colormap scaler',...
    'Worm 1 (blue) integrated Intensity','Worm 2 (teal) integrated Intensity','Worm 3 (green) integrated Intensity','Worm 4 (yellow/red) integrated Intensity','Worm 5 (orange) integrated Intensity',...
    'Worm 1 (blue) integrated Area','Worm 2 (teal) integrated Area','Worm 3 (green) integrated Area','Worm 4 (yellow/red) integrated Area','Worm 5 (orange) integrated Area'};

try
    output_csv(1,:) = output_header;
    output_csv(2:end,6:10) = num2cell(image_integral_intensities_comb);
    output_csv(2:end,11:15) = num2cell(image_integral_area);
    for i = 1:length(final_img_names)
        output_csv{i+1,1} = final_img_names{i};
        output_csv{i+1,2} = color_choice;
        if combinatorial_or_ratio
            output_csv{i+1,3} = 'ratio';
        else
            output_csv{i+1,3} = 'combinatorial';
        end
        output_csv{i+1,4} = this_cmap_choice;
        output_csv{i+1,5} = false_color_scaler;
    end
    
    T = cell2table(output_csv(2:end,:),'VariableNames',output_csv(1,:));
    writetable(T,fullfile(char(img_dir_path),'data.csv'));
    writetable(T,fullfile(output_path,'data.csv'));
    
    if isfile(fullfile(data_path,[final_save_name '.csv']))
        writetable(T,fullfile(data_path,[final_save_name,datestr(now, 'dd-mmm-yyyy'),'_.csv']))
    else
        writetable(T,fullfile(data_path,[final_save_name '.csv']))
    end
    
catch
    error('exporting failed, check user input settings and images -- Make sure if RGB is selected all three images (R,G,B) are present')
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

final_img_name = erase(first_image_name2,'.tif');

final_img_name = strrep(final_img_name,'__','_');

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

function [red_fluor_img,green_fluor_img,blue_fluor_img,combined_image] = ...
    isolate_fluorescence_images(images)


% covnert them to gray and find dominant color
gray_images = cell(size(images));
color_choices = zeros(size(images));
for j = 1:length(images)
    [gray_images{j},color_choices(j)] = get_dominant_color(images{j});
end

% isolate and normalize the different colors
try
    red_fluor_img = gray_images{find(color_choices==1,1)};
    red_fluor_img_norm = double(red_fluor_img)/max(double(red_fluor_img(:)));
catch
    red_fluor_img = zeros(size(gray_images{1}),'uint8');
    red_fluor_img_norm = double(red_fluor_img);
end
try
    green_fluor_img = gray_images{find(color_choices==2,1)};
    green_fluor_img_norm = double(green_fluor_img)/max(double(green_fluor_img(:)));
catch
    green_fluor_img = zeros(size(gray_images{1}),'uint8');
    green_fluor_img_norm = double(green_fluor_img);
end
try
    blue_fluor_img = gray_images{find(color_choices==3,1)};
    blue_fluor_img_norm = double(blue_fluor_img)/max(double(blue_fluor_img(:)));
catch
    blue_fluor_img = zeros(size(gray_images{1}),'uint8');
    blue_fluor_img_norm = double(green_fluor_img);
end


combined_image = cat(3,red_fluor_img_norm,green_fluor_img_norm,blue_fluor_img_norm);


end



function [out_img] = ...
    fluor_data_comparison(red_fluor_img,green_fluor_img,blue_fluor_img,...
    combined_image,new_mask,color_choice,combinatorial_or_ratio)

new_mask = double(new_mask);

red_fluor_img = (double(red_fluor_img).*new_mask) + new_mask;
green_fluor_img = (double(green_fluor_img).*new_mask) + new_mask;
blue_fluor_img = (double(blue_fluor_img).*new_mask) + new_mask;

% combinatorial_or_ratio = 0,1;
% ratio
if combinatorial_or_ratio
    switch color_choice
        case 'RG'
            out_img = red_fluor_img./green_fluor_img;
            out_img(isnan(out_img)) = 0;
        case 'RB'
            out_img = red_fluor_img./blue_fluor_img;
            out_img(isnan(out_img)) = 0;
        case 'GR'
            out_img = green_fluor_img./red_fluor_img;
            out_img(isnan(out_img)) = 0;
        case 'GB'
            out_img = green_fluor_img./blue_fluor_img;
            out_img(isnan(out_img)) = 0;
        case 'BG'
            out_img = blue_fluor_img./green_fluor_img;
            out_img(isnan(out_img)) = 0;
        case 'BR'
            out_img = blue_fluor_img./red_fluor_img;
            out_img(isnan(out_img)) = 0;
        case 'RGB'
            error('Cant do ratio and RGB, can only do combinatorial and RGB')
    end
else
    switch color_choice
        case 'RG'
            out_img = red_fluor_img.*green_fluor_img;
            out_img(isnan(out_img)) = 0;
        case 'RB'
            out_img = red_fluor_img.*blue_fluor_img;
            out_img(isnan(out_img)) = 0;
        case 'GR'
            out_img = green_fluor_img.*red_fluor_img;
            out_img(isnan(out_img)) = 0;
        case 'GB'
            out_img = green_fluor_img.*blue_fluor_img;
            out_img(isnan(out_img)) = 0;
        case 'BG'
            out_img = blue_fluor_img.*green_fluor_img;
            out_img(isnan(out_img)) = 0;
        case 'BR'
            out_img = blue_fluor_img.*red_fluor_img;
            out_img(isnan(out_img)) = 0;
        case 'RGB'
            out_img = red_fluor_img.*blue_fluor_img.*green_fluor_img;
            out_img(isnan(out_img)) = 0;
    end
end




end