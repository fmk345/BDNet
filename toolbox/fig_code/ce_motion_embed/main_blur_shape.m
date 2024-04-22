% -------
% figure: demo -  underlying principle of motion direction embedding using coded exposure phpotography
% ------

clc, clear, close all

% param
ce_code = [1,1,0,1,1];
box_code = [1,1,1,1,1];
code_seg_length = 60;
img_name = 'circ.png';
save_dir = img_name(1:end-4);
mkdir(save_dir)

% 
seg_vec = ones(1, code_seg_length);
kernel_box = kron(box_code, seg_vec); kernel_box = kernel_box/sum(kernel_box(:));
kernel_ce = kron(ce_code, seg_vec); kernel_ce = kernel_ce/sum(kernel_ce(:));
figure
imshow(kernel_ce)

% load patters
x=im2double(imread(img_name));
imshow(x)

% blur and save
x_ce=imfilter(x, kernel_ce,'replicate');
figure, imshow(x_ce), imwrite(x_ce, [save_dir '/x_ce.png'])
figure, plot(x_ce(uint8(size(x_ce,1)/2),:,1),'linewidth',2), axis off, saveas(gcf,  [save_dir '/x_ce_plot.svg'])

x_box=imfilter(x, kernel_box, 'replicate');
figure, imshow(x_box), imwrite(x_box, [save_dir '/x_box.png'])
figure, plot(x_box(uint8(size(x_box,1)/2),:,1),'linewidth',2), axis off, saveas(gcf, [save_dir '/x_box_plot.svg'])