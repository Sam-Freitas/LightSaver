% function to straighten worms
% must give inputs

function output_image = straighten_worms(labels,input_image,output_type_string,interp_seg_num_input,avg_width,img_name)

if isequal(output_type_string,'single')
    output_type = 1;
else
    output_type = 2;
end

interp_seg_num = interp_seg_num_input; % default 10

masks = labels>0;
masks = imclearborder(masks);
masked_image = double(input_image).*masks;

new_labels = bwlabel(masks);

skels = bwskel(masks,'MinBranchLength',50);
skels = bwskel(skels,'MinBranchLength',25);

num_labels = max(new_labels(:));

out = cell(1,num_labels);

for o = 1:num_labels
    this_label = (new_labels == o);
    this_masked_data = (this_label>0).*masked_image;
%     IM = uint8(rescale(this_label.*gr)*255);

    % get single skel line and perimeter
    single_line = skels.*(this_label);

    num_points = sum(nonzeros(single_line>0));

    % find endpoints for finding where to start
    endpts_img = bwmorph(single_line,'endpoints');
    endpts_img = bwareafilt(endpts_img,[1,1]);

    % find starting point by closesnt endpoint to 0,0
    [x,y] = find(endpts_img>0);
    xyz = [x,y, y+x];
    endpts = sortrows(xyz,[3,1,2]);
    endpts = endpts(:,[1,2]);
    starting_point = endpts(1,:);

    % this will be added on to5
    sorted_points = starting_point;
    temp_line = single_line; % slowly gets deleted

    try
        count = 0;
        for i = 1:num_points
            count = count + 1;
            % this finds the next adjacent point and follows it
            if i == 1
                temp_line(starting_point(1),starting_point(2)) = 0;
                [outx,outy] = find_adjacent_pixel(temp_line,starting_point(1),starting_point(2));
            else
                [outx,outy] = find_adjacent_pixel(temp_line,outx,outy);
            end
            % store the sorted point
            sorted_points = [sorted_points;outx,outy];
            temp_line(outx,outy) = 0;
        end

        [sorted_points,num_points,single_line] = extend_skeleton(sorted_points,single_line);

        px = sorted_points(:,1); py = sorted_points(:,2);
        pt = interparc(interp_seg_num,py,px,'spline');
        pt = interparc(num_points,pt(:,1),pt(:,2),'spline');

        [IM2] = straighten(this_masked_data,pt,avg_width,false);

    catch
        disp(['Failure aligning worms on ' img_name])
        IM2 = rand(500,avg_width);
    end

    out{o} = rot90(IM2);

end

switch output_type
    case 1
        output_image = imtile(out,'GridSize',[NaN,1]);
    case 2
        output_image = out;
end

end

function [out,num_points,single_line] = extend_skeleton(sorted_points,single_line)

temp = single_line;
factor_distance = 5;

first10 = sorted_points(1:10,:);
last10 = sorted_points(end-9:end,:);

V = -first10(end,:) +first10(1,:);
pext = first10(1,:) + V*factor_distance;

[x,y]= bresenham(first10(1,1),first10(1,2),pext(1),pext(2));
bad_idx = ((x<1) + (x>2048) + (y<1) + (y>2048))>0;
x = x(~bad_idx); y = y(~bad_idx);
for i = 1:length(x)
    temp(x(i),y(i)) = 1;
end

V = last10(end,:) -last10(1,:);
pext = last10(end,:) + V*factor_distance;

[x2,y2]= bresenham(last10(1,1),last10(1,2),pext(1),pext(2));
bad_idx = ((x2<1) + (x2>2048) + (y2<1) + (y2>2048))>0;
x2 = x2(~bad_idx); y2 = y2(~bad_idx);
for i = 1:length(x2)
    temp(x2(i),y2(i)) = 1;
end

single_line = bwskel(imfill(temp>0,'holes'),'MinBranchLength',25);
single_line([1,end],:) = 0; single_line(:,[1,end]) = 0;
num_points = sum(single_line(:));

% find endpoints for finding where to start
endpts_img = bwmorph(single_line,'endpoints');
endpts_img = bwareafilt(endpts_img,[1,1]);

% find starting point by closesnt endpoint to 0,0
[x,y] = find(endpts_img>0);
xyz = [x,y, y+x];
endpts = sortrows(xyz,[3,1,2]);
endpts = endpts(:,[1,2]);
starting_point = endpts(1,:);
% ending_point = endpts(2,:);

% this will be added on to
out = starting_point;
temp_line = single_line; % slowly gets deleted
% segmented_line = single_line; % will slowly get sigmented
% segment_points = starting_point;

count = 0;
% idx = 1;
for i = 1:num_points
    count = count + 1;
    % this finds the next adjacent point and follows it
    if i == 1
        temp_line(starting_point(1),starting_point(2)) = 0;
        [outx,outy] = find_adjacent_pixel(temp_line,starting_point(1),starting_point(2));
%         segmented_line(starting_point(1),starting_point(2)) = 255;
    else
        [outx,outy] = find_adjacent_pixel(temp_line,outx,outy);
    end
    % store the sorted point
    out = [out;outx,outy];
    % remove it from the line
    temp_line(outx,outy) = 0;
end

end

function [IM2] = straighten(IM,pts,width,d)
%STRAIGHTEN Straightens an image based on the spline interpolation between
%the input points.
% based on the straightenLine function in the Straightener plugin from ImageJ1
% https://github.com/imagej/imagej1/blob/master/ij/plugin/Straightener.java#L101
% https://github.com/imagej/imagej1/blob/master/ij/gui/PolygonRoi.java
debug=d;
if ndims(IM)==3
    for ct = 1:size(IM,3)
        IM2(:,:,ct)=straighten(IM(:,:,ct),pts,width,d);
    end
    return;
end
% must make spline with one pixel segment lengths
% [x,y]=fitSplineForStraightening(pts(:,1),pts(:,2));
% pt = interparc(891,pts(:,1),pts(:,2),'spline');
% x = pt(:,1); y = pt(:,2);

x = pts(:,1); y = pts(:,2);

if debug
    figure(99);clf;
    imagesc(IM);axis image
    hold on;
    plot(x,y,'o-');
    title('spline')
end

for ct = 1:length(x)
    if ct == 1
        dx = x(2)-x(1); %extrapolate first direction
        dy = y(1)-y(2);
    else
        dx = x(ct)-x(ct-1);
        dy = y(ct-1)-y(ct);
    end
    %want to move 1 pixel when we move the pointer dx dy
    L = sqrt(dx^2+dy^2);
    dx = dx/L;
    dy = dy/L;

    xStart = x(ct)-(dy*width)/2; %start half the width away
    yStart = y(ct)-(dx*width)/2;
    if debug
        figure(1);clf;
        imagesc(IM);axis image;hold on;
        plot(x,y,'-');
        plot(x(ct),y(ct),'o-');
        plot([xStart,xStart+dy*width],[yStart,yStart+dx*width],'r--')
    end
    for ct2 = 1:width
        IM2(ct,ct2) = getInterpolatedValue(IM,xStart,yStart);
        xStart=xStart+dy;
        yStart=yStart+dx;
    end
end

end

function I = getInterpolatedValue(IM,y,x)
% interpolation of pixelvalues one-indexed
% IM = [0,1,2;2,3,4;4,5,6];
% getInterpolatedValue(IM,1,1);
% > 0
x = [floor(x),ceil(x),rem(x,1)];
y = [floor(y),ceil(y),rem(y,1)];
try
I(1,1) = IM(x(1),y(1))*(1-x(3))*(1-y(3));
I(2,1) = IM(x(1),y(2))*(1-x(3))*(0+y(3));
I(1,2) = IM(x(2),y(1))*(0+x(3))*(1-y(3));
I(2,2) = IM(x(2),y(2))*(0+x(3))*(0+y(3));
I = sum(I(:));
catch
I = 0;
end
end

function [pt,dudt,fofthandle] = interparc(t,px,py,varargin)
% interparc: interpolate points along a curve in 2 or more dimensions
% usage: pt = interparc(t,px,py)    % a 2-d curve
% usage: pt = interparc(t,px,py,pz) % a 3-d curve
% usage: pt = interparc(t,px,py,pz,pw,...) % a 4-d or higher dimensional curve
% usage: pt = interparc(t,px,py,method) % a 2-d curve, method is specified
% usage: [pt,dudt,fofthandle] = interparc(t,px,py,...) % also returns derivatives, and a function handle
%
% Interpolates new points at any fractional point along
% the curve defined by a list of points in 2 or more
% dimensions. The curve may be defined by any sequence
% of non-replicated points.
%
% arguments: (input)
%  t   - vector of numbers, 0 <= t <= 1, that define
%        the fractional distance along the curve to
%        interpolate the curve at. t = 0 will generate
%        the very first point in the point list, and
%        t = 1 yields the last point in that list.
%        Similarly, t = 0.5 will yield the mid-point
%        on the curve in terms of arc length as the
%        curve is interpolated by a parametric spline.
%
%        If t is a scalar integer, at least 2, then
%        it specifies the number of equally spaced
%        points in arclength to be generated along
%        the curve.
%
%  px, py, pz, ... - vectors of length n, defining
%        points along the curve. n must be at least 2.
%        Exact Replicate points should not be present
%        in the curve, although there is no constraint
%        that the curve has replicate independent
%        variables.
%
%  method - (OPTIONAL) string flag - denotes the method
%        used to compute the points along the curve.
%
%        method may be any of 'linear', 'spline', or 'pchip',
%        or any simple contraction thereof, such as 'lin',
%        'sp', or even 'p'.
%        
%        method == 'linear' --> Uses a linear chordal
%               approximation to interpolate the curve.
%               This method is the most efficient.
%
%        method == 'pchip' --> Uses a parametric pchip
%               approximation for the interpolation
%               in arc length.
%
%        method == 'spline' --> Uses a parametric spline
%               approximation for the interpolation in
%               arc length. Generally for a smooth curve,
%               this method may be most accurate.
%
%        method = 'csape' --> if available, this tool will
%               allow a periodic spline fit for closed curves.
%               ONLY use this method if your points should
%               represent a closed curve.
%               
%               If the last point is NOT the same as the
%               first point on the curve, then the curve
%               will be forced to be periodic by this option.
%               That is, the first point will be replicated
%               onto the end.
%
%               If csape is not present in your matlab release,
%               then an error will result.
%
%        DEFAULT: 'spline'
%
%
% arguments: (output)
%  pt - Interpolated points at the specified fractional
%        distance (in arc length) along the curve.
%
%  dudt - when a second return argument is required,
%       interparc will return the parametric derivatives
%       (dx/dt, dy/dt, dz/dt, ...) as an array.
%
%  fofthandle - a function handle, taking numbers in the interval [0,1]
%       and evaluating the function at those points.
%
%       Extrapolation will not be permitted by this call.
%       Any values of t that lie outside of the interval [0,1]
%       will be clipped to the endpoints of the curve.
%
% Example:
% % Interpolate a set of unequally spaced points around
% % the perimeter of a unit circle, generating equally
% % spaced points around the perimeter.
% theta = sort(rand(15,1))*2*pi;
% theta(end+1) = theta(1);
% px = cos(theta);
% py = sin(theta);
%
% % interpolate using parametric splines
% pt = interparc(100,px,py,'spline');
%
% % Plot the result
% plot(px,py,'r*',pt(:,1),pt(:,2),'b-o')
% axis([-1.1 1.1 -1.1 1.1])
% axis equal
% grid on
% xlabel X
% ylabel Y
% title 'Points in blue are uniform in arclength around the circle'
%
%
% Example:
% % For the previous set of points, generate exactly 6
% % points around the parametric splines, verifying
% % the uniformity of the arc length interpolant.
% pt = interparc(6,px,py,'spline');
%
% % Convert back to polar form. See that the radius
% % is indeed 1, quite accurately.
% [TH,R] = cart2pol(pt(:,1),pt(:,2))
% % TH =
% %       0.86005
% %        2.1141
% %       -2.9117
% %        -1.654
% %      -0.39649
% %       0.86005
% % R =
% %             1
% %        0.9997
% %        0.9998
% %       0.99999
% %        1.0001
% %             1
%
% % Unwrap the polar angles, and difference them.
% diff(unwrap(TH))
% % ans =
% %        1.2541
% %        1.2573
% %        1.2577
% %        1.2575
% %        1.2565
%
% % Six points around the circle should be separated by
% % 2*pi/5 radians, if they were perfectly uniform. The
% % slight differences are due to the imperfect accuracy
% % of the parametric splines.
% 2*pi/5
% % ans =
% %        1.2566
%
%
% See also: arclength, spline, pchip, interp1
%
% Author: John D'Errico
% e-mail: woodchips@rochester.rr.com
% Release: 1.0
% Release date: 3/15/2010

% unpack the arguments and check for errors
if nargin < 3
  error('ARCLENGTH:insufficientarguments', ...
    'at least t, px, and py must be supplied')
end

t = t(:);
if (numel(t) == 1) && (t > 1) && (rem(t,1) == 0)
  % t specifies the number of points to be generated
  % equally spaced in arclength
  t = linspace(0,1,t)';
elseif any(t < 0) || any(t > 1)
  error('ARCLENGTH:impropert', ...
    'All elements of t must be 0 <= t <= 1')
end

% how many points will be interpolated?
nt = numel(t);

% the number of points on the curve itself
px = px(:);
py = py(:);
n = numel(px);

% are px and py both vectors of the same length?
if ~isvector(px) || ~isvector(py) || (length(py) ~= n)
  error('ARCLENGTH:improperpxorpy', ...
    'px and py must be vectors of the same length')
elseif n < 2
  error('ARCLENGTH:improperpxorpy', ...
    'px and py must be vectors of length at least 2')
end

% compose px and py into a single array. this way,
% if more dimensions are provided, the extension
% is trivial.
pxy = [px,py];
ndim = 2;

% the default method is 'linear'
method = 'spline';

% are there any other arguments?
if nargin > 3
  % there are. check the last argument. Is it a string?
  if ischar(varargin{end})
    method = varargin{end};
    varargin(end) = [];
    
    % method may be any of {'linear', 'pchip', 'spline', 'csape'.}
    % any any simple contraction thereof.
    valid = {'linear', 'pchip', 'spline', 'csape'};
    [method,errstr] = validstring(method,valid);
    if ~isempty(errstr)
      error('INTERPARC:incorrectmethod',errstr)
    end
  end
  
  % anything that remains in varargin must add
  % an additional dimension on the curve/polygon
  for i = 1:numel(varargin)
    pz = varargin{i};
    pz = pz(:);
    if numel(pz) ~= n
      error('ARCLENGTH:improperpxorpy', ...
        'pz must be of the same size as px and py')
    end
    pxy = [pxy,pz]; %#ok
  end
  
  % the final number of dimensions provided
  ndim = size(pxy,2);
end

% if csape, then make sure the first point is replicated at the end.
% also test to see if csape is available
if method(1) == 'c'
  if exist('csape','file') == 0
    error('CSAPE was requested, but you lack the necessary toolbox.')
  end
  
  p1 = pxy(1,:);
  pend = pxy(end,:);
  
  % get a tolerance on whether the first point is replicated.
  if norm(p1 - pend) > 10*eps(norm(max(abs(pxy),[],1)))
    % the two end points were not identical, so wrap the curve
    pxy(end+1,:) = p1;
    nt = nt + 1;
  end
end

% preallocate the result, pt
pt = NaN(nt,ndim);

% Compute the chordal (linear) arclength
% of each segment. This will be needed for
% any of the methods.
chordlen = sqrt(sum(diff(pxy,[],1).^2,2));

% Normalize the arclengths to a unit total
chordlen = chordlen/sum(chordlen);

% cumulative arclength
cumarc = [0;cumsum(chordlen)];

% The linear interpolant is trivial. do it as a special case
if method(1) == 'l'
  % The linear method.
  
  % which interval did each point fall in, in
  % terms of t?
  [junk,tbins] = histc(t,cumarc); %#ok
  
  % catch any problems at the ends
  tbins((tbins <= 0) | (t <= 0)) = 1;
  tbins((tbins >= n) | (t >= 1)) = n - 1;
  
  % interpolate
  s = (t - cumarc(tbins))./chordlen(tbins);
  % be nice, and allow the code to work on older releases
  % that don't have bsxfun
  pt = pxy(tbins,:) + (pxy(tbins+1,:) - pxy(tbins,:)).*repmat(s,1,ndim);
  
  % do we need to compute derivatives here?
  if nargout > 1
    dudt = (pxy(tbins+1,:) - pxy(tbins,:))./repmat(chordlen(tbins),1,ndim);
  end
  
  % do we need to create the spline as a piecewise linear function?
  if nargout > 2
    spl = cell(1,ndim);
    for i = 1:ndim
      coefs = [diff(pxy(:,i))./diff(cumarc),pxy(1:(end-1),i)];
      spl{i} = mkpp(cumarc.',coefs);
    end
    
    %create a function handle for evaluation, passing in the splines
    fofthandle = @(t) foft(t,spl);
  end
  
  % we are done at this point
  return
end

% If we drop down to here, we have either a spline
% or csape or pchip interpolant to work with.

% compute parametric splines
spl = cell(1,ndim);
spld = spl;
diffarray = [3 0 0;0 2 0;0 0 1;0 0 0];
for i = 1:ndim
  switch method
    case 'pchip'
      spl{i} = pchip(cumarc,pxy(:,i));
    case 'spline'
      spl{i} = spline(cumarc,pxy(:,i));
      nc = numel(spl{i}.coefs);
      if nc < 4
        % just pretend it has cubic segments
        spl{i}.coefs = [zeros(1,4-nc),spl{i}.coefs];
        spl{i}.order = 4;
      end
    case 'csape'
      % csape was specified, so the curve is presumed closed,
      % therefore periodic
      spl{i} = csape(cumarc,pxy(:,i),'periodic');
      nc = numel(spl{i}.coefs);
      if nc < 4
        % just pretend it has cubic segments
        spl{i}.coefs = [zeros(1,4-nc),spl{i}.coefs];
        spl{i}.order = 4;
      end
  end
  
  % and now differentiate them
  xp = spl{i};
  xp.coefs = xp.coefs*diffarray;
  xp.order = 3;
  spld{i} = xp;
end

% catch the case where there were exactly three points
% in the curve, and spline was used to generate the
% interpolant. In this case, spline creates a curve with
% only one piece, not two.
if (numel(cumarc) == 3) && (method(1) == 's')
  cumarc = spl{1}.breaks;
  n = numel(cumarc);
  chordlen = sum(chordlen);
end

% Generate the total arclength along the curve
% by integrating each segment and summing the
% results. The integration scheme does its job
% using an ode solver.

% polyarray here contains the derivative polynomials
% for each spline in a given segment
polyarray = zeros(ndim,3);
seglen = zeros(n-1,1);

% options for ode45
opts = odeset('reltol',1.e-9);
for i = 1:spl{1}.pieces
  % extract polynomials for the derivatives
  for j = 1:ndim
    polyarray(j,:) = spld{j}.coefs(i,:);
  end
  
  % integrate the arclength for the i'th segment
  % using ode45 for the integral. I could have
  % done this part with quad too, but then it
  % would not have been perfectly (numerically)
  % consistent with the next operation in this tool.
  [tout,yout] = ode45(@(t,y) segkernel(t,y),[0,chordlen(i)],0,opts); %#ok
  seglen(i) = yout(end);
end

% and normalize the segments to have unit total length
totalsplinelength = sum(seglen);
cumseglen = [0;cumsum(seglen)];

% which interval did each point fall into, in
% terms of t, but relative to the cumulative
% arc lengths along the parametric spline?
[junk,tbins] = histc(t*totalsplinelength,cumseglen); %#ok

% catch any problems at the ends
tbins((tbins <= 0) | (t <= 0)) = 1;
tbins((tbins >= n) | (t >= 1)) = n - 1;

% Do the fractional integration within each segment
% for the interpolated points. t is the parameter
% used to define the splines. It is defined in terms
% of a linear chordal arclength. This works nicely when
% a linear piecewise interpolant was used. However,
% what is asked for is an arclength interpolation
% in terms of arclength of the spline itself. Call s
% the arclength traveled along the spline.
s = totalsplinelength*t;

% the ode45 options will now include an events property
% so we can catch zero crossings.
opts = odeset('reltol',1.e-9,'events',@ode_events);

ti = t;
for i = 1:nt
  % si is the piece of arc length that we will look
  % for in this spline segment.
  si = s(i) - cumseglen(tbins(i));
  
  % extract polynomials for the derivatives
  % in the interval the point lies in
  for j = 1:ndim
    polyarray(j,:) = spld{j}.coefs(tbins(i),:);
  end
  
  % we need to integrate in t, until the integral
  % crosses the specified value of si. Because we
  % have defined totalsplinelength, the lengths will
  % be normalized at this point to a unit length.
  %
  % Start the ode solver at -si, so we will just
  % look for an event where y crosses zero.
  [tout,yout,te,ye] = ode45(@(t,y) segkernel(t,y),[0,chordlen(tbins(i))],-si,opts); %#ok
  
  % we only need that point where a zero crossing occurred
  % if no crossing was found, then we can look at each end.
  if ~isempty(te)
    ti(i) = te(1) + cumarc(tbins(i));
  else
    % a crossing must have happened at the very
    % beginning or the end, and the ode solver
    % missed it, not trapping that event.
    if abs(yout(1)) < abs(yout(end))
      % the event must have been at the start.
      ti(i) = tout(1) + cumarc(tbins(i));
    else
      % the event must have been at the end.
      ti(i) = tout(end) + cumarc(tbins(i));
    end
  end
end

% Interpolate the parametric splines at ti to get
% our interpolated value.
for L = 1:ndim
  pt(:,L) = ppval(spl{L},ti);
end

% do we need to compute first derivatives here at each point?
if nargout > 1
  dudt = zeros(nt,ndim);
  for L = 1:ndim
    dudt(:,L) = ppval(spld{L},ti);
  end
end

% create a function handle for evaluation, passing in the splines
if nargout > 2
  fofthandle = @(t) foft(t,spl);
end

% ===============================================
%  nested function for the integration kernel
% ===============================================
  function val = segkernel(t,y) %#ok
    % sqrt((dx/dt)^2 + (dy/dt)^2 + ...)
    val = zeros(size(t));
    for k = 1:ndim
      val = val + polyval(polyarray(k,:),t).^2;
    end
    val = sqrt(val);
    
  end % function segkernel

% ===============================================
%  nested function for ode45 integration events
% ===============================================
  function [value,isterminal,direction] = ode_events(t,y) %#ok
    % ode event trap, looking for zero crossings of y.
    value = y;
    isterminal = ones(size(y));
    direction = ones(size(y));
  end % function ode_events

end % mainline - interparc


% ===============================================
%       end mainline - interparc
% ===============================================
%       begin subfunctions
% ===============================================

% ===============================================
%  subfunction for evaluation at any point externally
% ===============================================
function f_t = foft(t,spl)
% tool allowing the user to evaluate the interpolant at any given point for any values t in [0,1]
pdim = numel(spl);
f_t = zeros(numel(t),pdim);

% convert t to a column vector, clipping it to [0,1] as we do.
t = max(0,min(1,t(:)));

% just loop over the splines in the cell array of splines
for i = 1:pdim
  f_t(:,i) = ppval(spl{i},t);
end
end % function foft


function [str,errorclass] = validstring(arg,valid)
% validstring: compares a string against a set of valid options
% usage: [str,errorclass] = validstring(arg,valid)
%
% If a direct hit, or any unambiguous shortening is found, that
% string is returned. Capitalization is ignored.
%
% arguments: (input)
%  arg - character string, to be tested against a list
%        of valid choices. Capitalization is ignored.
%
%  valid - cellstring array of alternative choices
%
% Arguments: (output)
%  str - string - resulting choice resolved from the
%        list of valid arguments. If no unambiguous
%        choice can be resolved, then str will be empty.
%
%  errorclass - string - A string argument that explains
%        the error. It will be one of the following
%        possibilities:
%
%        ''  --> No error. An unambiguous match for arg
%                was found among the choices.
%
%        'No match found' --> No match was found among 
%                the choices provided in valid.
%
%        'Ambiguous argument' --> At least two ambiguous
%                matches were found among those provided
%                in valid.
%        
%
% Example:
%  valid = {'off' 'on' 'The sky is falling'}
%  
%
% See also: parse_pv_pairs, strmatch, strcmpi
%
% Author: John D'Errico
% e-mail: woodchips@rochester.rr.com
% Release: 1.0
% Release date: 3/25/2010

ind = find(strncmpi(arg,valid,numel(arg)));
if isempty(ind)
  % No hit found
  errorclass = 'No match found';
  str = '';
elseif (length(ind) > 1)
  % Ambiguous arg, hitting more than one of the valid options
  errorclass = 'Ambiguous argument';
  str = '';
  return
else
  errorclass = '';
  str = valid{ind};
end

end % function validstring



% function [xspl,yspl] = fitSplineForStraightening(xVal,yVal)
% %return a spline fit with distances of 1 between the points
% %based on https://github.com/imagej/imagej1/blob/master/ij/gui/PolygonRoi.java#L1006
% 
% %generate intermediate spline with approximately half pixel steps
% pp = spline(xVal,yVal);
% xIspl = interpolate(xVal,yVal,0.5);
% yIspl = ppval(pp,xIspl);
% L=0;%measure spline distance
% %generate spline
% xspl(1) = xVal(1);
% yspl(1) = yVal(1);
% L=0; %keep track of length of spline
% ptw = 1; %points written
% for ct = 2:length(xIspl)
%     dx=xIspl(ct)-xIspl(ct-1);
%     dy=yIspl(ct)-yIspl(ct-1);
%     d = sqrt(dx^2+dy^2);
%     L=L+d;
%     overshoot = L-ptw;
%     if overshoot>0 %we went over the length, must add a point on the spline
%         ptw=ptw+1;
%         frac = overshoot/d; %fractional overshoot for the last step
%         %move back in a straight line towards the previous point
%         xspl(ptw) = xIspl(ct) - frac*dx;
%         yspl(ptw) = yIspl(ct) - frac*dy;
%     end
% end
% end

% function [Xi] = interpolate(xVal,yVal,d)
% %interpolate line with aproximate distance d between points
% Xi = [];
% dists = sqrt(diff(xVal).^2+diff(yVal).^2); %segment distances
% for ct = 1:length(xVal)-1
%     x = linspace(xVal(ct),xVal(ct+1),round(dists(ct)/d)+1);x(end)=[];
%     Xi = [Xi,x];
% end
% end

