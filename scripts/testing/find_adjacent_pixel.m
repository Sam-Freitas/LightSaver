function [outx,outy] = find_adjacent_pixel(bw,x,y)
try
    idy = y-1:y+1;
    idx = x-1:x+1;
    nhood = bw(idx,idy);
    
    x_hood = [-1,-1,-1;0,0,0;1,1,1];
    y_hood = [-1,0,1;-1,0,1;-1,0,1];
    
    [subx,suby] = find(nhood==1);
    
    x_add = x_hood(subx,suby);
    y_add = y_hood(subx,suby);
    
    outx = x + x_add;
    outy = y + y_add;
catch
    outx = [];
    outy = [];
end
    
end