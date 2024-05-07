
function out_img = mean2mask(img,amount)


out_img = double(img);
for i = 1:amount
    
    out_mask = out_img>mean2(nonzeros(out_img));
    out_mask = bwareaopen(out_mask,2,4);
    out_mask = imfill(out_mask,'holes');
    out_img = out_img.*out_mask;
    
end


end