% fit data try

A = imread('A.png');

lin_data = A;
lin_data = nonzeros(double(lin_data).*(lin_data>mean2(lin_data)));

counts = hist(lin_data,max(lin_data));
counts = medfilt1(counts,3);

x = 1:length(counts);

TF = islocalmin(counts);

local_mins = x(TF);

plot(x,counts,x(TF),counts(TF),'r*')

imshow(double(A).*(A>local_mins(1)),[])