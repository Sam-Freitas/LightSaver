img_dir_path = "C:\Users\Lab PC\Desktop\Raul\12-9-21\Exported";

img_paths = dir(fullfile(img_dir_path, '*.tif'));
[~,sort_idx,~] = natsort({img_paths.name});

img_paths = img_paths(sort_idx);

for i = 1:length(img_paths)
    
    this_img = imread(fullfile(img_dir_path,img_paths(i).name));
    
%     imshow(this_img); title(img_paths(i).name,'Interpreter','none');
    
    % Split channes
    R = this_img(:,:,1); G = this_img(:,:,2); B = this_img(:,:,3);
    
%     imshow(G,[])
    
    % Remove 1mm bar
    data = G - B - R;
%     imshow(data,[]);
    
    mask = data>6; %%%%% please find a better way to do this :)
    
    new_mask = bwareaopen(imfill(imgaussfilt(data,1.5)>6,'holes'),25);
    
    figure;
    imshowpair(label2rgb(bwlabel(bwareaopen(new_mask,2000))),this_img,'montage')
    title(img_paths(i).name);
    
    masked_data = mask.*double(data); % this converts the picture array to mathable stuff 
    
    image_integral_intensities(i) = sum(masked_data(:));
    
    linear_data = nonzeros(masked_data);
        
    [counts,binLoc] = hist(linear_data,255); 
%     stem(binLoc,counts)
    
%     integral = sum(lin_list);
    
end
