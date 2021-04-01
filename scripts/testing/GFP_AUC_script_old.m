img_dir_path = "C:\Users\Lab PC\Documents\GFP_AUC\data\Raul_data\2021-02-16\Exported\";

show_output_images = 0;

output_path = fullfile(erase(erase(pwd,'GFP_AUC_script.m'),'scripts'),'exported_data');
mkdir(output_path);

img_paths = dir(fullfile(img_dir_path, '*.tif'));
[~,sort_idx,~] = natsort({img_paths.name});

img_paths = img_paths(sort_idx);

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
    
    data_norm = double(data)/double(max(data(:)));
    
    first_thresh = mean2(data_norm)+(std2(data_norm))*(.5);

    % create a mask for the worms 
    new_mask = bwareaopen(imfill(imgaussfilt(data_norm,2)>first_thresh,'holes'),25);
    
    % get rid of any small blobs 
    new_mask = bwareaopen(new_mask,2000);
    
    % label the different mask blobs 
    labeled_masks = bwlabel(new_mask);
    
    % if there are not enough unique blobs then increase the 
    % threshold of the images to 7 
    if max(labeled_masks(:)) ~= 5
        second_thresh = mean2(data_norm)+(std2(data_norm));
        new_mask = bwareaopen(imfill(imgaussfilt(data_norm,2)>second_thresh,'holes'),25);
        new_mask = bwareaopen(new_mask,2000);
        labeled_masks = bwlabel(new_mask);
    end
    
    
    % if there are still not enough blobs then take more drastic measures
    if max(labeled_masks(:)) ~= 5
        se = strel('disk',25);
        filt_data = imsubtract(imadd(data_norm,imtophat(data_norm,se)),imbothat(data_norm,se));
        new_mask = bwareaopen(imfill(filt_data>second_thresh,'holes'),25);
        new_mask = bwareaopen(new_mask,2000);
        new_mask = bwmorph(new_mask,'Thicken',3);
        labeled_masks = bwlabel(new_mask);
        
        labeled_masks2 = zeros(size(new_mask));
        for j = 1:max(labeled_masks(:))
            labeled_masks2 = labeled_masks2 + j*imclose(labeled_masks==j,strel('disk',3));
        end
        new_mask = imfill(labeled_masks2>0,'holes');
        labeled_masks = labeled_masks2;
    end
    

    
    % integrate the entire signal across the mask 
    for j = 1:max(labeled_masks(:))
        this_labeled_mask = double(data).*(labeled_masks==j);
        image_integral_intensities(i,j) = sum(this_labeled_mask(:));
        image_integral_area(i,j) = sum(this_labeled_mask(:)>0);
        
        if image_integral_area(i,j)>20000
            image_integral_area(i,j) = 0;
            image_integral_intensities(i,j) = 0;
        end
    end
    
    % mask the inital data 
    % gets rid of background signals 
    masked_data = new_mask.*double(data); 
    
    labeled_masks2 = zeros(size(new_mask));
    for j = 1:max(labeled_masks(:))
        if image_integral_area(i,j) > 0 
            labeled_masks2 = labeled_masks2 + j*(imfill(labeled_masks==j,'holes'));
        end
    end
    
    labeled_masks = labeled_masks2;
    
    imwrite(imtile({this_img,label2rgb(labeled_masks,'jet','k'),masked_data/max(masked_data(:))},'GridSize',[1,3]),...
        fullfile(output_path,[img_paths(i).name '_img' num2str(i) '.png']))
    
    if show_output_images == 1
        figure;
        imshow(imtile({this_img,label2rgb(labeled_masks,'jet','k'),masked_data/max(masked_data(:))},'GridSize',[1,3]),[]);
        title([img_paths(i).name ' -- img ' num2str(i)], 'Interpreter', 'none');
    end
    
    linear_data = nonzeros(masked_data);
        
    [counts,binLoc] = hist(linear_data,255); 
    
end
