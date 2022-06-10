close all force hidden
clear all
warning('off', 'MATLAB:MKDIR:DirectoryExists');
% img_dir_path = "C:\Users\Lab PC\Documents\GFP_AUC\data\Raul_data\2021-02-16\Exported\";

% User input choice
prompt = {'Enter number of worms to detect: ', ...
    'Show output images - yes(1) - no(0)',...
    'Use large blob fix - yes (1) - no(0)',...
    'Output name - leave blank for defaults - or enter name for exported_images sub-folder',...
    'Remove 001,002, etc, from tif names - yes(1) - no(0) Will overwrite data files',...
    'Export processed images - yes (1) - no (0)',...
    'Automaic data analysis and export - yes (1) - no (0)',...
    'Does the experiment folder have condition names in it? (ex: 01-1-11_N2_vs_SKN-1) - yes (1) - no (0)'};
dlgtitle = 'User Inputs for Lightsaver';
dims = [1 100];
definput = {'5','0','0','','1','1','1','0'};
answer = inputdlg(prompt,dlgtitle,dims,definput);

if isempty(answer)
    error('Please select user inputs')
end

number_worms_to_detect = str2double(answer{1});
show_output_images = str2double(answer{2});
use_large_blob_fix = str2double(answer{3});
output_name = answer{4};
rename_tifs_choice = str2double(answer{5});
export_processed_images = str2double(answer{6});
data_analysis_and_export_bool = str2double(answer{7});
experimental_name_has_conditions_in_it = str2double(answer{8});

% this is for the exported images 
% faster is with the jpg format -> 0 but less quality on the images
high_quality_output = 1;
if high_quality_output
    output_img_format = '.png';
else
    output_img_format = '.jpg';
end

% clean up variables
clearvars dims definput dlgtitle prompt answer

% get current path
curr_path = pwd;

data_path = fullfile(erase(erase(curr_path,'multiple_samples'),'scripts'),'data');

img_dir_path = uigetdir(data_path,'Please select the folder containing the *.tiff files');

[~,final_save_name,~] = fileparts(img_dir_path);

output_path = fullfile(erase(erase(curr_path,'multiple_samples'),'scripts'),'exported_images');
mkdir(output_path);

if isempty(output_name)
    output_name = final_save_name;
end

[~,message,~] = fileattrib(fullfile(img_dir_path,'*'));

fprintf('\nThere are %i total files & folders in the overarching folder.\n',numel(message));

allExts = cellfun(@(s) s(end-2:end), {message.Name},'uni',0); % Get exts

TIFidx = ismember(allExts,'tif');    % Search ext for "TIF" at the end
TIF_filepaths = {message(TIFidx).Name}';  % Use CSVidx to list all paths.
fprintf('There are %i files with *.TIF exts.\n',numel(TIF_filepaths));

if isempty(TIF_filepaths)
    error('No images with the .tif file extensions found')
end

output_path = fullfile(output_path,output_name);
mkdir(output_path);

img_paths = TIF_filepaths;
[~,sort_idx,~] = natsort(img_paths);

if rename_tifs_choice
    [img_paths] = rename_tifs(img_paths);
end

img_paths = img_paths(sort_idx);

[~,img_names,~] = fileparts(img_paths);

[img_names] = clean_img_names(img_paths,img_names);

image_integral_intensities = zeros(length(img_paths),number_worms_to_detect);
image_integral_area = zeros(length(img_paths),number_worms_to_detect);

se = strel('disk',5);
progress_bar = 0;

for i = 1:length(img_paths)
    
    progress_bar = progressbar_function(i,length(img_paths),progress_bar);
    
    [data,this_img] = load_fluor_image(img_dir_path,img_paths,i);
    
    % testing to measure scale from exported image with ocr
    %     [out,this_measurement] = ocr_pixel_scale(this_img);
    
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
        disp(['Warning: more than ' num2str(number_worms_to_detect) ' worms detected - ' img_names{i}])
        disp(['Using only the ' num2str(number_worms_to_detect) ' largest blobs'])
        
        E = entropyfilt(data_norm);
        
        this_mask = bwareafilt(this_label>0,number_worms_to_detect);
        
    end
    
    if max(this_label(:))<number_worms_to_detect
        disp(['Warning: more than ' num2str(number_worms_to_detect) ' worms detected - ' img_names{i}])
        disp(['Using only the ' num2str(number_worms_to_detect) ' largest blobs'])
        
        this_mask = bwareafilt(this_label>0,number_worms_to_detect);
        
    end
    
    % if this is chosen then a semi-smart filter will try to fix the large
    % blobs that contain multiple worms
    large_blob_is_fixed = 0;
    if use_large_blob_fix
        
        [this_label,this_mask,large_blob_is_fixed] = large_blob_fix(this_label,...
            this_mask,data_norm,img_names,number_worms_to_detect,i);
        
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
    
    [~,this_img_name,~] = fileparts(img_names{i});
    
    if large_blob_is_fixed
        this_img_name = ['_L_Blob_fix_' this_img_name];
    end
    
    if export_processed_images
        % insert a corresponding text box to each label
        annotated_data_output = annotate_masks(...
            labeled_masks,masked_data_output,image_integral_intensities(i,:));
        
        % write the image sequence to the export folder
        try
            imwrite(imtile({this_img,rgb_labeled_mask,annotated_data_output},'GridSize',[1,3]),...
                fullfile(output_path,[num2str(i) '_' this_img_name output_img_format]))
        catch
            if length(size(this_img)) < 3
                t1 = cat(3,uint8(255*double(this_img)/double(max(this_img(:)))),...
                    uint8(255*double(this_img)/double(max(this_img(:)))),...
                    uint8(255*double(this_img)/double(max(this_img(:)))));
            else
                t1 = uint8(255*double(this_img)/double(max(this_img(:))));
            end
            t2 = rgb_labeled_mask;
            t3 = uint8(255*annotated_data_output);
            temp_img = [t1,t2,t3];
            
            imwrite(temp_img,fullfile(output_path,[num2str(i) '_' this_img_name output_img_format]))
        end
    end
    
    % show the sequence if necessary
    if show_output_images
        imshow(imtile({this_img,rgb_labeled_mask,annotated_data_output},'GridSize',[1,3]),[]);
        title([img_names{i} ' -- img ' num2str(i)], 'Interpreter', 'none');
    end
    
end
close_progressbar(progress_bar)

output_csv = cell(1 + length(img_paths),11);

output_header = {'Image names',...
    'Worm 1 (blue) integrated Intensity','Worm 2 (teal) integrated Intensity',...
    'Worm 3 (green) integrated Intensity',...
    'Worm 4 (yellow/red) integrated Intensity','Worm 5 (orange) integrated Intensity',...
    'Worm 1 (blue) integrated Area','Worm 2 (teal) integrated Area',...
    'Worm 3 (green) integrated Area','Worm 4 (yellow/red) integrated Area',...
    'Worm 5 (orange) integrated Area'};

output_csv(1,:) = output_header;
output_csv(2:end,2:6) = num2cell(image_integral_intensities);
output_csv(2:end,7:11) = num2cell(image_integral_area);
for i = 1:length(img_paths)
    output_csv{i+1,1} = img_names{i};
end

T = cell2table(output_csv(2:end,:),'VariableNames',output_csv(1,:));
writetable(T,fullfile(char(img_dir_path),'data.csv'))

if isfile(fullfile(data_path,[final_save_name '.csv']))
    writetable(T,fullfile(data_path,[final_save_name,datestr(now, 'dd-mmm-yyyy'),'_.csv']))
else
    writetable(T,fullfile(data_path,[final_save_name '.csv']))
end

if data_analysis_and_export_bool
    data_analysis_and_export_function(img_dir_path,experimental_name_has_conditions_in_it)
end

disp(' ')
disp('End of scrip')


function [data,this_img] = load_fluor_image(img_dir_path,img_paths,i)
% read the image into ram

[~,this_img_name,~] = fileparts(img_paths{i});

try
    this_img = imread(img_paths{i});
catch
    disp(['ERROR: reading image - ' this_img_name])
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

if length(size(this_img)) > 2 % if the image is RGB
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
else % else the image is grayscale
    data = this_img;
    % get rid of the scale bar
    %     data(end-100:end,1:256) = 0;
end

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

function [img_names] = clean_img_names(img_paths,img_names)

img_subnames = {};
for i = 1:length(img_names)
    temp = cellstr(strsplit(img_names{i},{' ','_'}));
    img_subnames = [img_subnames,temp];
end
img_subnames = unique(img_subnames);

img_subnames(~isnan(str2double(img_subnames))) = [];

% check to see if the img_subnames contains D(N) data
day_idx_marker = ones(size(img_subnames));
for i = 1:length(img_subnames)
    this_subname = img_subnames{i};
    try
        if isequal(this_subname(1),'D') && isnumeric(str2double(this_subname(2:end))) && ~isnan(str2double(this_subname(2:end)))
            day_idx_marker(i) = 0;
        end
    catch
        % single characters ignored
    end
end
img_subnames = img_subnames(logical(day_idx_marker));

% find if subname is contained in ALL the names
for i = 1:length(img_names)
    for j = 1:length(img_subnames)
        TF(i,j) = contains(img_names{i},img_subnames{j});
    end
end

contained_in_all_names = find(sum(TF,1) == length(img_names));
subnames_in_all_names = img_subnames(contained_in_all_names);

if ~isempty(contained_in_all_names)
    for i = 1:length(contained_in_all_names)
        
        subname_to_erase = img_subnames{contained_in_all_names(i)};
        
        for j = 1:length(img_names)
            img_names{j} = erase(img_names{j},subname_to_erase);
            % remove leading space or underscore
            if isequal(img_names{j}(1),' ') || isequal(img_names{j}(1),'_')
                img_names{j} = img_names{j}(2:end);
            end
            % remove ending space or underscore
            if isequal(img_names{j}(end),' ') || isequal(img_names{j}(end),'_')
                img_names{j} = img_names{j}(1:end-1);
            end
            
        end
        
        
        
        for j = 1:length(img_names)
            
            img_names{j} = replace(img_names{j},'__','_');
            img_names{j} = replace(img_names{j},'  ',' ');
            
            if isequal(img_names{j}(end),' ') || isequal(img_names{j}(end),'_')
                img_names{j} = img_names{j}(1:end-1);
            end
        end
        
    end
end

end


function data_analysis_and_export_function(exp_dir_path,experimental_name_has_conditions_in_it)

% get exp name
[~,experiment_name,~] = fileparts(exp_dir_path);

% this is all to find the data.csv's
[~,message,~] = fileattrib(fullfile(exp_dir_path,'*'));

fprintf('\nThere are %i total files & folders in the overarching folder.\n',numel(message));

[~,fileNames,fileExts] = fileparts({message.Name});

allNamesFull = join(cat(1,fileNames,fileExts),'',1);

CSVidx = ismember(allNamesFull,'data.csv');    % Search ext for "CSV" at the end
CSV_filepaths = {message(CSVidx).Name};  % Use CSVidx to list all paths.

fprintf('There are %i files with *data.csv names.\n',numel(CSV_filepaths));

csv_cells = cell(1,length(CSV_filepaths));

% read in the tables
for i = 1:numel(CSV_filepaths)
    csv_table{i}= readtable(CSV_filepaths{i},'VariableNamingRule','preserve');
    % Your parsing will be different
end

% combine the tables
for i = 1:length(csv_table)
    if isequal(i,1)
        full_table = csv_table{1};
    else
        full_table = [full_table;csv_table{i}];
    end
end

try
    temp_table_array = table2array(full_table(:,2:end));
    %     disp('data not detected on column 2, trying column 6')
catch
    temp_table_array = table2array(full_table(:,6:end));
end

AUC_array = temp_table_array(:,1:5)./temp_table_array(:,6:10);
AUC_array(isnan(AUC_array))=0;

% get the names and split the names into parts
img_names = full_table.("Image names");
img_names_split = cell(1,length(img_names));
img_names_no_day = img_names;
img_names_only_day = img_names;

% get split names
for i = 1:length(img_names)
    % find last D
    D_idx = find(char(img_names{i})=='D',1,'last');
    underscore_idx = find(char(img_names{i})=='_');
    
    D_to_undx = D_idx:(underscore_idx(find(underscore_idx>D_idx,1,'first'))-1);
    % rid of last D
    img_names_no_day{i}(D_idx:end) = [];
    % get only days
    img_names_only_day{i} = img_names{i}(D_to_undx);
    % if there isnt a channel output (ch00) or repeat ( D7_repeat_) then
    % this will be empty
    if isempty(img_names_only_day{i})
        img_names_only_day{i} = img_names{i}(D_idx:end);
    end
    %split the remaining
    img_names_split{i} = strsplit(img_names_no_day{i},{' ','_'});
    % delete empty
    img_names_split{i} = ...
        img_names_split{i}(~cellfun('isempty',img_names_split{i}));
end

% split experiment names
experiment_name_parts = strsplit(experiment_name,{' ','_','-'});

% find if part numerical
only_numerical_name_parts = str2double(experiment_name_parts);
only_numerical_name_parts(isnan(only_numerical_name_parts)) = 0;

only_numerical_name_parts = logical(only_numerical_name_parts);

experiment_name_parts(only_numerical_name_parts) = [];

% get rid of parts that are contained in the experiment

% default is that
if ~experimental_name_has_conditions_in_it
    img_names_split2 = cell(1,length(img_names_split));
    for i = 1:length(img_names_split)
        
        % find parts that are already from the experiment name splits
        % this was changed from true to false
        TF = contains(img_names_split{i},experiment_name_parts,'IgnoreCase',true);
        
        % join the rest
        img_names_split2{i} = char(join(img_names_split{i}(~TF)));
        
    end
else
    img_names_split2 = cell(1,length(img_names_split));
    for i = 1:length(img_names_split)
        % join the rest
        img_names_split2{i} = char(join(img_names_split{i}));
    end
end

% get all the condition names
condition_names = unique(img_names_split2)';
% remove empty condition names (dont know exactly why this can happen)
% condition_names = condition_names(~cellfun('isempty',condition_names));
%
% get all condition subnames
condition_subnames = {};
for i = 1:length(condition_names)
    temp = cellstr(strsplit(condition_names{i}));
    condition_subnames = [condition_subnames,temp];
end
condition_subnames = unique(condition_subnames);

% find if subname is contained in ALL the names
for i = 1:length(condition_names)
    for j = 1:length(condition_subnames)
        TF(i,j) = contains(condition_names{i},condition_subnames{j});
    end
end

contained_in_all_names = find(sum(TF) == length(condition_names));

if ~isempty(contained_in_all_names)
    for i = 1:length(contained_in_all_names)
        for j = 1:length(condition_names)
            condition_names{j} = erase(condition_names{j},...
                condition_subnames{contained_in_all_names(i)});
            condition_names{j} = ...
                remove_space_underscore_first_last(condition_names{j});
        end
        
        for j = 1:length(condition_names)
            condition_names{j} = ...
                remove_space_underscore_first_last(condition_names{j});
        end
        
    end
end

% get all the day names
day_names = natsort(unique(img_names_only_day));

img_names_spaces = cell(size(img_names_no_day));
for i = 1:length(img_names)
    img_names_spaces{i} = strrep(img_names_no_day{i},'_',' ');
    img_names_spaces{i} = remove_space_underscore_first_last(img_names_spaces{i});
end

% find which conditions correspond to what img
condition_idx = zeros(1,length(img_names_spaces))';
for i = 1:length(condition_names)
    
    this_condition_idx = contains(img_names_spaces,condition_names{i},'IgnoreCase',true);
    condition_idx(this_condition_idx) = i;
    
end

% find which day corresponds to what img
day_idx = zeros(1,length(img_names_spaces))';
for i = 1:length(day_names)
    this_day_idx = contains(img_names_only_day,day_names{i},'IgnoreCase',true);
    day_idx(this_day_idx) = i;
end

% indexable list for variables
idx_list = [1:length(day_idx)]';

final_array = cell(length(day_names)+1,length(condition_names)+1);
final_array(2:length(day_names)+1,1) = day_names;
final_array(1,2:length(condition_names)+1) = condition_names;

% combine
for i = 1:length(condition_names)
    
    this_condition_idx = (condition_idx == i);
    
    for j = 1:length(day_names)
        
        this_day_idx = (day_idx == j);
        
        this_combined_idx = nonzeros(this_day_idx.*this_condition_idx.*idx_list);
        
        this_AUC_data = AUC_array(this_combined_idx,:);
        
        this_AUC_data_flat = reshape(this_AUC_data,[1,numel(this_AUC_data)]);
        
        final_array(j+1,i+1) = {this_AUC_data_flat};
        
    end
end

writecell(final_array,fullfile(exp_dir_path,'Analyzed_data.csv'));

disp(['Exported analyzed data to:'])
disp(fullfile(exp_dir_path,'Analyzed_data.csv'));

end

function [img_paths] = rename_tifs(img_paths)

img_paths_new = img_paths;

for i = 1:length(img_paths)
    
    [filepath,name,ext] = fileparts(img_paths_new{i});
    
    new_name = erase(name,{'001',...
        '002','003','004','005','006','007','008','009',});
    
    img_paths_new{i} = fullfile(filepath,[name ext]);
    
    if ~isequal(img_paths{i},img_paths_new{i})
        movefile(img_paths{i},img_paths_new{i});
    end
    
end

img_paths = img_paths_new;

end

function out = remove_space_underscore_first_last(in)

out = in;

if isequal(out(1),{' ','_'})
    out = out(2:end);
end
if isequal(out(end),{' ','_'})
    out = out(1:end-1);
end


end

function progress_bar = progressbar_function(i,num_samples,progress_bar)

progress_ratio = i/num_samples;

if isequal(progress_bar,0)
    progress_bar = waitbar(progress_ratio,'Processing data');
else
    progress_bar = waitbar(progress_ratio,progress_bar,'Processing data');
end

end

function close_progressbar(progress_bar)

close(progress_bar)

end

function annotated_data_output = annotate_masks(labeled_masks,masked_data_output,data)
% add a corresponding colored box that has the integrated intensity for
% each unique label
num_labels = length(nonzeros(unique(labeled_masks)));

cmap = jet(num_labels);

for i = 1:num_labels
    
    this_label = (labeled_masks == i);
    
    s = regionprops(this_label,'Centroid','PixelList');
        
    to_plot(1) = mean(s.PixelList(:,1));
    to_plot(2) = min(s.PixelList(:,2));
    
    centroid = round(s.Centroid);
    masked_data_output = insertText(masked_data_output,to_plot,num2str(data(i)),...
        'BoxColor',cmap(i,:),'AnchorPoint','CenterBottom','FontSize',26);
    
end

annotated_data_output = masked_data_output;

end

function [out,this_measurement] = ocr_pixel_scale(this_img)

this_img = rgb2gray(this_img);

ocr_result = ocr(this_img);

ocr_text = ocr_result.Words{1};

ocr_text_box = ocr_result.WordBoundingBoxes;

this_img(ocr_text_box(2):ocr_text_box(2)+ocr_text_box(4),...
    ocr_text_box(1):ocr_text_box(1)+ocr_text_box(3)) = 0;

this_line = this_img(end-5:end,1:256) == 255;
this_scale = str2double(ocr_text(1:end-2));
this_measurement = ocr_text(end-2:end);

out = (this_scale/sum(this_line(:))) ^ 2;

end