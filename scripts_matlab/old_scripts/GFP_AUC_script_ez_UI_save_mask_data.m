close all force hidden
clear all
warning('off', 'MATLAB:MKDIR:DirectoryExists');
% img_dir_path = "C:\Users\Lab PC\Documents\GFP_AUC\data\Raul_data\2021-02-16\Exported\";

curr_path = pwd;

data_path = fullfile(erase(erase(curr_path,'scripts'),'testing'),'data');

% img_dir_path = uigetdir(data_path,'Please select the folder containing the *.tiff files');
% img_dir_path = "C:\Users\LabPC2\Documents\GitHub\LightSaver_testing\data\GFP Marker New set 1 - pngs";
img_dir_path = "C:\Users\LabPC2\Documents\RaulData\Leica Fluorescence Stereoscope";

[~,final_save_name,~] = fileparts(img_dir_path);

show_output_images = 0;

% USER must specify the number of worms that the program SHOULD find
number_worms_to_detect = 5;

% output_path = fullfile(erase(data_path,'data'),'exported_images');
% mkdir(output_path);

[~,message,~] = fileattrib(fullfile(img_dir_path,'*'));
fprintf('\nThere are %i total files & folders in the overarching folder',numel(message));
allExts = cellfun(@(s) s(end-2:end), {message.Name},'uni',0); % Get exts
TIFidx = ismember(allExts,'tif');    % Search ext for "TIF" at the end
img_paths = {message(TIFidx).Name}';  % Use CSVidx to list all paths.
fprintf('\nFound %i image files matching tif\n',length(img_paths));
[~,sort_idx,~] = natsort(img_paths);
img_paths = img_paths(sort_idx);

image_integral_intensities = zeros(length(img_paths),number_worms_to_detect);
image_integral_area = zeros(length(img_paths),number_worms_to_detect);

mkdir('mask_data')
% rmdir('mask_data','s')
mkdir('mask_data')
mkdir(fullfile(pwd,'mask_data','images'))
mkdir(fullfile(pwd,'mask_data','masks'))

use_large_blob_fix = 0;
progress_bar = 0;

% for i = 1:length(img_paths)
for i = 2318:length(img_paths)

    progress_bar = progressbar_function(i,length(img_paths),progress_bar);
    
    % read the image into ram
    try
        this_img = imread(img_paths{i});
    catch
        disp(['ERROR: reading image - ' img_paths{i}])
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
    if z == 1
        temp_img = zeros(size(this_img,1),size(this_img,2),3, 'uint8');
        temp_img(:,:,1) = this_img;
        this_img = temp_img;
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
    
    data(end-100:end,1:256) = median(data(:));

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
        disp(['Warning: more than ' num2str(number_worms_to_detect) ' worms detected - ' img_paths{i}])
        disp(['Using only the ' num2str(number_worms_to_detect) ' largest blobs'])
                
        this_mask = bwareafilt(this_label>0,number_worms_to_detect);
        
    end
    
    if max(this_label(:))<number_worms_to_detect
        disp(['Warning: less than ' num2str(number_worms_to_detect) ' worms detected - ' img_paths{i}])
        disp(['Using only the ' num2str(number_worms_to_detect) ' largest blobs'])
        
        this_mask = bwareafilt(this_label>0,number_worms_to_detect);
        
    end

    large_blob_is_fixed = 0;
    if use_large_blob_fix
        [this_label,this_mask,large_blob_is_fixed] = large_blob_fix(this_label,...
            this_mask,data_norm,img_paths,number_worms_to_detect,i);
    end

    se = strel('disk',5,4);
    new_mask = imerode(this_mask,se);
    new_mask = bwareafilt(new_mask,number_worms_to_detect);
    
    % thicken all the masks
    new_mask = bwmorph(new_mask,'Thicken',3);
    % close small edges and zones
    new_mask = imclose(new_mask,strel('disk',5));
    % fill the holes
    new_mask = imfill(new_mask,'holes');
    % re-label the masks 
    labeled_masks = bwlabel(new_mask);

    if sum(new_mask(:)) > 1000
        rand_worms = randperm(5,3);
        temp_data = data;
        temp_mask = new_mask;
        for j = 1:3
            temp_data(bwmorph((labeled_masks==rand_worms(j)),'thicken',5)) = median(temp_data(:));
            temp_mask = temp_mask-(labeled_masks==rand_worms(j));
        end
        write_different_NN_images(temp_data,fullfile(pwd,'mask_data','images'));
        write_different_NN_masks(temp_mask,fullfile(pwd,'mask_data','masks'));

        write_different_NN_images(data,fullfile(pwd,'mask_data','images'));
        write_different_NN_masks(new_mask,fullfile(pwd,'mask_data','masks'));
    end
%     
%     % mask the inital data without the normalization step
%     % gets rid of background signals 
%     masked_data = new_mask.*double(data); 
%     
%     % integrate the entire signal across the masks and only the masks  
%     for j = 1:max(labeled_masks(:))
%         this_labeled_mask = double(data).*(labeled_masks==j);
%         image_integral_intensities(i,j) = sum(sum(this_labeled_mask.*double(data)));
%         image_integral_area(i,j) = sum(sum((this_labeled_mask.*double(data))>0));
%         
%     end
%     
%     % converts the labeled mask to RGB (easier to read)
%     rgb_labeled_mask = label2rgb(labeled_masks,'jet','k');
%     % get the masked data ready for export
%     masked_data_output = masked_data/max(masked_data(:));
    
%     % write the image sequence to the export folder
%     imwrite(imtile({this_img,rgb_labeled_mask,masked_data_output},'GridSize',[1,3]),...
%         fullfile(output_path,[img_paths(i).name '_img' num2str(i) '.png']))
%     
%     % show the sequence if necessary 
%     if show_output_images == 1
%         imshow(imtile({this_img,rgb_labeled_mask,masked_data_output},'GridSize',[1,3]),[]);
%         title([img_paths(i).name ' -- img ' num2str(i)], 'Interpreter', 'none');
%     end
    
%     linear_data = nonzeros(masked_data);
%         
%     [counts,binLoc] = hist(linear_data,255); 
    
end

% output_csv = cell(1 + length(img_paths),11);
% 
% output_header = {'Image names',...
%     'Worm 1 (blue) integrated Intensity','Worm 2 (teal) integrated Intensity','Worm 3 (green) integrated Intensity','Worm 4 (yellow/red) integrated Intensity','Worm 5 (orange) integrated Intensity',...
%     'Worm 1 (blue) integrated Area','Worm 2 (teal) integrated Area','Worm 3 (green) integrated Area','Worm 4 (yellow/red) integrated Area','Worm 5 (orange) integrated Area'};
% 
% output_csv(1,:) = output_header;
% output_csv(2:end,2:6) = num2cell(image_integral_intensities);
% output_csv(2:end,7:11) = num2cell(image_integral_area);
% for i = 1:length(img_paths)
%     output_csv{i+1,1} = img_paths(i).name;
% end
% 
% T = cell2table(output_csv(2:end,:),'VariableNames',output_csv(1,:));
% writetable(T,fullfile(char(img_dir_path),'data.csv'))
% 
% if isfile(fullfile(data_path,[final_save_name '.csv']))
%     writetable(T,fullfile(data_path,[final_save_name,datestr(now, 'dd-mmm-yyyy'),'_.csv']))
% else
%     writetable(T,fullfile(data_path,[final_save_name '.csv']))
% end

make_dataset_csv()

disp(' ')
disp('End of scrip')




function write_different_NN_images(img,path_to_folder)

img = double(img)/double(max(img(:)));

imgs_dir = natsort({dir(fullfile(path_to_folder,'*.png')).name})';

last_img_name = char(imgs_dir(end));
last_img_name = str2double(last_img_name(1:end-4));

% num_curr_imgs = length(imgs_dir);
num_curr_imgs = last_img_name;

img6 = rot90(img);
img6(img6>mean2(img6)) = mean2(img6);
img6 = imnoise(img6,'gaussian',mean2(img6),var(img6(:)));

imwrite(img,fullfile(path_to_folder,[num2str(num_curr_imgs+1) '.png']));
imwrite(img6,fullfile(path_to_folder,[num2str(num_curr_imgs+2) '.png']));

% img2 = imadjust(img);
% img3 = flip(img,1);
% img4 = imnoise(img,'gaussian');
% img5 = imnoise(img,'salt & pepper');

%%% this creates a ton of data if uncommented
% imwrite(img,fullfile(path_to_folder,[num2str(num_curr_imgs+1) '.png']));
% imwrite(img2,fullfile(path_to_folder,[num2str(num_curr_imgs+2) '.png']));
% imwrite(img3,fullfile(path_to_folder,[num2str(num_curr_imgs+3) '.png']));
% imwrite(img4,fullfile(path_to_folder,[num2str(num_curr_imgs+4) '.png']));
% imwrite(img5,fullfile(path_to_folder,[num2str(num_curr_imgs+5) '.png']));
% imwrite(img6,fullfile(path_to_folder,[num2str(num_curr_imgs+6) '.png']));

end

function write_different_NN_masks(img,path_to_folder)

imgs_dir = natsort({dir(fullfile(path_to_folder,'*.png')).name})';

last_img_name = char(imgs_dir(end));
last_img_name = str2double(last_img_name(1:end-9));

% num_curr_imgs = length(imgs_dir);
num_curr_imgs = last_img_name;

img6 = rot90(img);

imwrite(img,fullfile(path_to_folder,[num2str(num_curr_imgs+1) '_mask.png']));
imwrite(img6,fullfile(path_to_folder,[num2str(num_curr_imgs+2) '_mask.png']));


% img2 = img;
% img3 = flip(img,1);
% img4 = img;
% img5 = img;
% 
% imwrite(img,fullfile(path_to_folder,[num2str(num_curr_imgs+1) '_mask.png']));
% imwrite(img2,fullfile(path_to_folder,[num2str(num_curr_imgs+2) '_mask.png']));
% imwrite(img3,fullfile(path_to_folder,[num2str(num_curr_imgs+3) '_mask.png']));
% imwrite(img4,fullfile(path_to_folder,[num2str(num_curr_imgs+4) '_mask.png']));
% imwrite(img5,fullfile(path_to_folder,[num2str(num_curr_imgs+5) '_mask.png']));
% imwrite(img6,fullfile(path_to_folder,[num2str(num_curr_imgs+6) '_mask.png']));
end


function [this_label,this_mask,is_fixed] = large_blob_fix(this_label,this_mask,...
    data_norm,img_names,number_worms_to_detect,i)

is_fixed = 0;

% get region profile
s = regionprops(this_label,'basic');

% determine if the areas are correct
is_over_large(i) = sum([s.Area]>25000);

% if they are not
if is_over_large(i)
    disp(['Warning: Large blobs detected in - ' img_names{i}])
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

function progress_bar = progressbar_function(i,num_samples,progress_bar)

progress_ratio = i/num_samples;

if isequal(progress_bar,0)
    progress_bar = waitbar(progress_ratio,'Processing data');
    drawnow
else
    progress_bar = waitbar(progress_ratio,progress_bar,['Processing data - '  char(num2str(i))] );
    drawnow
end

end
