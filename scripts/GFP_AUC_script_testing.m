close all
img_dir_path = "C:\Users\Lab PC\Documents\GFP_AUC\data\Raul_data\2021-02-16.1\Exported\";

show_output_images = 0;

output_path = fullfile(erase(erase(pwd,'GFP_AUC_script.m'),'scripts'),'exported_data');
mkdir(output_path);

img_paths = dir(fullfile(img_dir_path, '*.tif'));
[~,sort_idx,~] = natsort({img_paths.name});

img_paths = img_paths(sort_idx);

bad_imgs = [10,23,37,38,39,43,45,47,48,53,67];

for i = 1:length(img_paths)
    
    % read the image into ram
    this_img = imread(fullfile(img_dir_path,img_paths(i).name));
        
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
    
    for j = 1:6
        this_thresh = mean2(data_norm)+(std2(data_norm)*(1/5)*(j-1));
        this_mask = imgaussfilt(data_norm,2)>this_thresh;
        
        this_mask = bwareaopen(this_mask,3000);
        
        this_label = bwlabel(this_mask);
        
        if max(this_label(:)) == 5
            
            this_thresh2 = mean2(data_norm)+(std2(data_norm)*(1/5)*(j));
            this_mask2 = imgaussfilt(data_norm,2)>this_thresh2;
            this_mask2 = bwareaopen(this_mask2,3000);
            this_label2 = bwlabel(this_mask2);
            
            if max(this_label2(:)) == 5
                this_mask = this_mask2;
                this_label = this_label2;
                break
            end
            break
        end
        
    end
    
    new_mask = imfill(this_mask,'holes');
    new_mask = bwmorph(new_mask,'Thicken',1);
    labeled_masks = bwlabel(new_mask);
    
    % mask the inital data without the normalization
    % gets rid of background signals 
    masked_data = new_mask.*double(data); 
    
    % integrate the entire signal across the mask 
    for j = 1:max(labeled_masks(:))
        this_labeled_mask = double(data).*(labeled_masks==j);
        image_integral_intensities(i,j) = sum(sum(this_labeled_mask.*double(data)));
        image_integral_area(i,j) = sum(sum((this_labeled_mask.*double(data))>0));
        
        if image_integral_area(i,j)>20000
            image_integral_area(i,j) = 0;
            image_integral_intensities(i,j) = 0;
        end
    end
    
    imwrite(imtile({this_img,label2rgb(labeled_masks,'jet','k'),masked_data/max(masked_data(:))},'GridSize',[1,3]),...
        fullfile(output_path,[img_paths(i).name '_img' num2str(i) '.png']))
    
    if show_output_images == 1
        imshow(imtile({this_img,label2rgb(labeled_masks,'jet','k'),masked_data/max(masked_data(:))},'GridSize',[1,3]),[]);
        title([img_paths(i).name ' -- img ' num2str(i)], 'Interpreter', 'none');
    end
    
    linear_data = nonzeros(masked_data);
        
    [counts,binLoc] = hist(linear_data,255); 
    
end




