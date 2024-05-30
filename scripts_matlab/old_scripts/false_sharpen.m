

function out_img = false_sharpen(img,amount)

out_img = img;

for i = 1:amount
    
    out_img = 2*out_img - imgaussfilt(out_img);
    
end

end