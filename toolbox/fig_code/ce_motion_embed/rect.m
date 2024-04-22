function z = rect(bg,fg,point,mode)
% RECT generates samples of a continuous, unity-height rectangle, whose position is
% determind by point and fg (center point/[h,w] || left-up point/[h,w]) 
% on a (given or generated) background rectangle with size N. 
% (Assuming the coordinate origin is at the left-up corner of the rectangle,
%  right=x-positive; down=y-positive; unit = 1 pix)
% 
%   Input:
%   --------
%   - bg: 2-elements int vector, background rectangle size; [hight, width]; Or, 2D matrix, as
%   a background
%	- fg: 2-elements int vector, foreground rectangle size, [hight, width]
%	- mode: input mode, string, 'center'|'corner', default = 'center'
%	- point: 2-elements int vector
%		- mode = 'center': center point / [hight, width]; default
%		- mode = 'corner': left-up point / [hight, width]
%	Default rect position is at the center of the background rectangle
% 
%   Output:
%   --------
%   - z: unity-height rectangle
% 
%   See also:
%   CYL   
% 
%   Info:
%   --------
%   Created:        Zhihong Zhang <z_zhi_hong@163.com>, 2020-06-27
%   Last Modified:  Zhihong Zhang, 2020-06-27       
% 
%   Copyright (c) 2020 Zhihong Zhang.

% input check
if numel(bg)==1
	bg = [bg,bg];		% background size
	z = zeros(bg);	% background
elseif ~isvector(bg)
	z = bg;			% background
	bg = size(z);	% background size
end

if numel(fg)==1
	fg = [fg,fg];
end

if nargin < 4
	mode = 'center';
	if nargin < 3
		point = round(bg/2);
	end
end

% convert 'center-length' mode to 'corner-length' mode for unify calc 
if strcmp(mode, 'center')
	point = round(point - fg/2);
elseif strcmp(mode, 'corner')
	% no operation
else
	error('"mode" param error');
end

x0 = point(1);
x1 = x0 + fg(1);
y0 = point(2);
y1 = y0 + fg(2);

% boundary
x0 = max(1, x0);
x1 = min(x1, bg(1));
y0 = max(1, y0);
y1 = min(y1, bg(2));

z(x0:x1, y0:y1) = 1;
% z(r==R)=0.5; % transition boundary

end