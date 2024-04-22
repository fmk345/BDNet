function z = circ(bg,radius,center)
% CIRC generates a white circle on a black background or given background
% (Assuming the coordinate origin is at the left-up corner of the rectangle,
%  right=x-positive; down=y-positive; unit = 1 pix)
% 
%   Input:
%   --------
%   - bg: 2-elements int vector, background rectangle size; Or, 2D matrix, as
%   a background
%	- center: 2-elements int vector,centre point of the cylinder, default=[0,0]
%	- radius: float scalar, radius of the cylinder
% 
%   Output:
%   --------
%   - z: unity-height cylinder 
% 
%   See also:
%   --------   
%	RECT
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

if nargin < 3
	center = round(bg./2);
end

x0 = center(1);
y0 = center(2);
L1 = bg(1);
L2 = bg(2);

[x,y]=meshgrid(linspace(1, L1, L1), linspace(1, L2, L2)); 
x = x';
y = y';

r = sqrt((x-x0).*(x-x0)+(y-y0).*(y-y0)); % distance map

z(r<=radius) = 1;
% z(r==R)=0.5; % transition boundary

end