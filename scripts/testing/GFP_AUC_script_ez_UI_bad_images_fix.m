close all
clear all
warning('off', 'MATLAB:MKDIR:DirectoryExists');
% img_dir_path = "C:\Users\Lab PC\Documents\GFP_AUC\data\Raul_data\2021-02-16\Exported\";

curr_path = pwd;

data_path = fullfile(erase(erase(curr_path,'scripts'),'testing'),'data');

% img_dir_path = uigetdir2(data_path,'Please select the bad images that need to be manually redone');
img_paths = uigetdir2('Y:\Users\Raul Castro\Microscopes\Leica Fluorescence Stereoscope\2021-06-29\Exported','Please select the bad images that need to be manually redone');

[folder_of_exp,img_names,img_extensions] = fileparts(img_paths);

[~,final_save_name,~] = fileparts(folder_of_exp{1});

% variable to decide to show all the output images as they are processed
% default do not show images - 0
% show output in figures - 1 (slow)
show_output_images = 0;

% User seleted number of worms
number_worms_to_detect = 5;

% If there are large blobs that need fixing (still a beta test)
% default - 0
% use blob fix - 1
use_large_blob_fix = 0;

output_path = fullfile(erase(data_path,'data'),'exported_images');
mkdir(output_path);

[~,sort_idx,~] = natsort(img_paths);

img_paths = img_paths(sort_idx);

image_integral_intensities = zeros(length(img_paths),number_worms_to_detect);
image_integral_area = zeros(length(img_paths),number_worms_to_detect);

se = strel('disk',5);

for i = 1:length(img_paths)
    
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
    
    this_mask = zeros(size(data));
        
    redo = 'Yes';
    while isequal(redo,'Yes')
        
        data2 = imadjust(data);
        
        make_another_roi = 'Yes';
        while isequal(make_another_roi,'Yes')
            
            imshow(data2,[]);
            ROI = images.roi.AssistedFreehand;
            draw(ROI)
            
            bw_ROI = createMask(ROI);
            
            this_mask = this_mask + bw_ROI;
            this_mask = this_mask>0;
            
            data2(bw_ROI) = max(data2(:));
            
            imshow(data2,[])
            
            make_another_roi = questdlg({'Make another ROI?',...
                'No will assume correct, You CAN overlay ROIS to fix them'},'ROI?','Yes','No','Yes');
        end
        
        imshow(this_mask);
        redo = questdlg({'Does this ROI need to be redone? Please double check',...
            'If it does then this script will repeat'},'ROI?','Yes','No','Yes');
    end    
    
    close all
    
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
        fullfile(output_path,[img_names{i} '_img' num2str(i) '.png']))
    
    % show the sequence if necessary
    if show_output_images == 1
        imshow(imtile({this_img,rgb_labeled_mask,masked_data_output},'GridSize',[1,3]),[]);
        title([img_names{i} ' -- img ' num2str(i)], 'Interpreter', 'none');
    end
    
    linear_data = nonzeros(masked_data);
    
    [counts,binLoc] = hist(linear_data,255);
    
end

output_csv = cell(1 + length(img_names),11);

output_header = {'Image names',...
    'Worm 1 (blue) integrated Intensity','Worm 2 (teal) integrated Intensity','Worm 3 (green) integrated Intensity','Worm 4 (yellow/red) integrated Intensity','Worm 5 (orange) integrated Intensity',...
    'Worm 1 (blue) integrated Area','Worm 2 (teal) integrated Area','Worm 3 (green) integrated Area','Worm 4 (yellow/red) integrated Area','Worm 5 (orange) integrated Area'};

output_csv(1,:) = output_header;
output_csv(2:end,2:6) = num2cell(image_integral_intensities);
output_csv(2:end,7:11) = num2cell(image_integral_area);

img_names_string = string(img_names);
img_extensions_string = string(img_extensions);

for i = 1:length(img_names)
    output_csv{i+1,1} = char(img_names_string(i) + img_extensions_string(i));
end

T = cell2table(output_csv(2:end,:),'VariableNames',output_csv(1,:));

input_csv = readtable(fullfile(folder_of_exp{1},'data.csv'),'VariableNamingRule',"preserve");

new_csv = input_csv;

inital_names = string(input_csv.("Image names"));
for i = 1:length(img_names)
    
    this_img_name = img_names_string(i) + img_extensions_string(i);
    
    idx = find(inital_names==this_img_name,1,'first');
    
    if ~isempty(idx)
        new_csv(idx,:) = T(i,:);
    else
        disp(['could not find ' char(this_img_name) ' in data.csv']) 
    end
    
end

writetable(new_csv,fullfile(char(folder_of_exp{1}),'data.csv'))

if isfile(fullfile(data_path,[final_save_name '.csv']))
    writetable(T,fullfile(data_path,[final_save_name,datestr(now, 'dd-mmm-yyyy'),'_.csv']))
else
    writetable(T,fullfile(data_path,[final_save_name '.csv']))
end

disp(' ')
disp('End of scrip')
