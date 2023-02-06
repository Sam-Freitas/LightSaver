function make_dataset_csv()

data_path = fullfile(pwd,'mask_data');

final_save_name = 'dataset';

path_to_imgs = fullfile(pwd,'mask_data','images');
path_to_masks = fullfile(pwd,'mask_data','masks');

masks_dir = dir(fullfile(path_to_masks,'*.jpg'));
imgs_dir = dir(fullfile(path_to_imgs,'*.jpg'));

masks_names = {masks_dir.name};
img_names = {imgs_dir.name};

[~,masks_idx,~] = natsort(masks_names);
[~,imgs_idx,~] = natsort(img_names);

masks_dir = masks_dir(masks_idx);
imgs_dir = imgs_dir(imgs_idx);

csv_header = ["images","masks"];

final_csv = cell(length(imgs_dir),2);

for i = 1:length(imgs_dir)
    
    final_csv(i,1) = {imgs_dir(i).name};
    final_csv(i,2) = {masks_dir(i).name};
    
end

T = cell2table(final_csv,'VariableNames',csv_header);
writetable(T,fullfile(data_path,[final_save_name '.csv']))

disp('Data saved to:')
disp(fullfile(data_path,[final_save_name '.csv']))

end

