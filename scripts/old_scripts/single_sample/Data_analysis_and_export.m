close all
clear all
warning('off', 'MATLAB:MKDIR:DirectoryExists');
% img_dir_path = "C:\Users\Lab PC\Documents\GFP_AUC\data\Raul_data\2021-02-16\Exported\";

curr_path = pwd;

% get data path
data_path = fullfile(erase(erase(curr_path,'scripts'),'testing'),'data');

% get experiment dir
exp_dir_path = uigetdir(data_path,'Please select the overarching experiment folder');

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
    csv_table{i}= readtable(CSV_filepaths{i},'VariableNamingRule','preserve'); % Your parsing will be different
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
    disp('data not detected on column 2, trying column 6')
catch
    temp_table_array = table2array(full_table(:,6:end));
end

AUC_array = temp_table_array(:,1:5)./temp_table_array(:,6:10);

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
    %split the remaining
    img_names_split{i} = strsplit(img_names_no_day{i},{' ','_'});
    % delete empty 
    img_names_split{i} = img_names_split{i}(~cellfun('isempty',img_names_split{i}));
end

% split experiment names 
experiment_name_parts = strsplit(experiment_name,{' ','_'});

% find if part numerical 
only_numerical_name_parts = str2double(experiment_name_parts);
only_numerical_name_parts(isnan(only_numerical_name_parts)) = 0;

only_numerical_name_parts = logical(only_numerical_name_parts);

experiment_name_parts(only_numerical_name_parts) = [];

% get rid of parts that are contained in the experiment 
img_names_split2 = cell(1,length(img_names_split));
for i = 1:length(img_names_split)
    
    % find parts that are already from the experiment name splits 
    TF = contains(img_names_split{i},experiment_name_parts,'IgnoreCase',true);
    
    % join the rest 
    img_names_split2{i} = char(join(img_names_split{i}(~TF)));
    
end

% get all the condition names
condition_names = unique(img_names_split2)';

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
            condition_names{j} = erase(condition_names{j},condition_subnames{contained_in_all_names(i)});
            if isequal(condition_names{j}(1),' ')
                condition_names{j} = condition_names{j}(2:end);
            end
        end
        
        for j = 1:length(condition_names)
            if isequal(condition_names{j}(1),' ')
                condition_names{j} = condition_names{j}(2:end);
            end
        end
        
    end
end

% get all the day names
day_names = natsort(unique(img_names_only_day));

img_names_spaces = img_names_no_day;
for i = 1:length(img_names)
    img_names_spaces{i} = strrep(img_names_no_day{i},'_',' ');
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

final_array = cell(length(day_names)+1,numel(AUC_array));
final_array(2:length(day_names)+1,1) = day_names;

% combine 
for i = 1:length(condition_idx)
    
    this_condition_idx = (condition_idx == i);
    
    for j = 1:length(day_idx)
                
        this_day_idx = (day_idx == j);
        
        this_combined_idx = nonzeros(this_day_idx.*this_condition_idx.*idx_list);
        
        for k = 1:length(this_combined_idx)
            
            this_AUC_data = num2cell(AUC_array(this_combined_idx(k),:));
            
            this_final_row = final_array(j+1,:);
            
            empty_idx = find(cellfun('isempty', this_final_row),1);
            
            final_array(j+1,empty_idx:empty_idx+length(this_AUC_data)-1) = this_AUC_data;
            
            if isequal(j,1)
                final_array(1,empty_idx:empty_idx+length(this_AUC_data)-1) ...
                    = repmat({condition_names{i}},1,length(this_AUC_data));
            end
            
        end
        
    end
end

writecell(final_array,fullfile(exp_dir_path,'Analyzed_data.csv'));

final_array_names = final_array(1,:);
final_array_names = final_array_names(~cellfun('isempty',final_array_names));
data_subdivisions = cellstr(string(final_array_names));

mkdir(fullfile(exp_dir_path,'output_figures'));
mkdir(fullfile(exp_dir_path,'output_figures','per_day'));
mkdir(fullfile(exp_dir_path,'output_figures','per_condition'));

for i = 1:length(day_names)
    
    out_name = strrep([experiment_name '--' day_names{i}],' ','_');
    
    this_data = final_array(i+1,2:end);
    this_data = cell2mat(this_data(~cellfun('isempty',this_data)));
    
    hFig = figure('units','normalized','outerposition',[0 0 1 1]);
    vs = violinplot(this_data, data_subdivisions);
    ylabel('AUC per condition');
    title(out_name,'Interpreter','none')
    drawnow;
    
    pause(0.1);
    saveas(hFig,fullfile(exp_dir_path,'output_figures','per_day',[char(out_name) '.png'])); 
    
end
close all

day_plot_names = strings();
for i = 1:length(data_subdivisions)
    
    for j = 1:length(day_names)
        day_plot_names(j,i) = join([string(data_subdivisions{i});string(day_names{j})],1);
    end
end

for i = 1:length(condition_names)
    
    out_name = strrep(condition_names{i},' ','_');
    
    this_condition = condition_names{i};
    
    this_condition_idx = find(string(data_subdivisions) == string(this_condition));
    
    this_data = final_array(2:end,this_condition_idx+1);
    this_day_plot_names = day_plot_names(:,this_condition_idx);
    
    this_data_vector = cell2mat(flip(reshape(rot90(flip(this_data)),1,numel(this_data))));
    this_day_plot_names_vector = flip(reshape(rot90(flip(this_day_plot_names)),1,numel(this_day_plot_names)));
    this_data_names = join([repmat(string(condition_names{i}),1,length(day_names));string(day_names)'],' ',1);
    
    this_data_names_cell = cell(1,length(this_data_names));
    for j = 1:length(this_data_names)
        this_data_names_cell{j} = char(this_data_names(j));
    end
    
    this_data_names_cell = natsort(this_data_names_cell);
    
    hFig = figure('units','normalized','outerposition',[0 0 1 1]);
    vs = violinplot(this_data_vector, this_day_plot_names_vector,'GroupOrder',this_data_names_cell);
    ylabel('AUC per condition');
    title(out_name,'Interpreter','none')
    
    pause(0.1);
    saveas(hFig,fullfile(exp_dir_path,'output_figures','per_condition',[char(out_name) '.png'])); 
    
end
close all

disp('All data exported to')
disp(exp_dir_path);

%%%%%%%%%% find a way to combine the day_idx and condition_idx to get rauls
%%%%%%%%%% way of updateing stuff


%               condition1                    conditionN
% D1          AUC1 ............... AUC N 
% D1+1
% D1+2
% 
% 
% 
% 
% 
% 
% 
% 
% 
