
# --------------------------------------
# generate a video comparer (half-half concatenated) from two image directories
#  - image directories should have the proportionable number of images
# --------------------------------------

import cv2
import imageio
from skimage.transform import resize
import os
from tqdm import tqdm
import numpy as np

# %% Param setting
# path
img_dirs = ["demo/ce", "demo/gt"]
save_path = "./test.avi"  # .gif / .avi / .mp4

# Set the starting and ending image numbers to read
starts = [0, 0]
ends = [3, 24]  # -1 for all
# ends = [-1, -1, -1, -1]  # -1 for all

# Set the frame size and margin
resize_ratio = 1
concat_direction = 'x'  # video concatenation direction, 'x' or 'y'
# color and width of the middle bar
mid_bar_color, mid_bar_width = (255, 0, 0), 2


# Set the frame rate and labels
fps = 10
vid_labels = ['ce', 'gt']
font_face, font_color, font_scale, font_thickness, font_linetype = cv2.FONT_HERSHEY_SIMPLEX, (
    0, 0, 255), 1*resize_ratio, 1, cv2.LINE_AA


# %% Functions
def frame_proc(img_k, dst_size, vid_label_k, frame_idx_k, label_pos='top_right'):
    # process a single frame: resize, add margin, add label
    # resize
    img_k = (resize(img_k, dst_size)*255.0).astype('uint8')

    # add label
    if label_pos == 'bottom_right':
        label_text = f"{vid_label_k}_#{frame_idx_k+1}"
        textsize = cv2.getTextSize(
            label_text, font_face, font_scale, font_thickness)[0]
        label_pos = (img_k.shape[1] - textsize[0]-10,
                     img_k.shape[0] - int(0.5*textsize[1]))
        cv2.putText(img_k, label_text, label_pos, font_face,
                    font_scale, font_color, font_thickness, font_linetype)
    elif label_pos == 'top_right':
        label_text = f"{vid_label_k} #{frame_idx_k+1}"
        textsize = cv2.getTextSize(
            label_text, font_face, font_scale, font_thickness)[0]
        label_pos = (img_k.shape[1] -
                     textsize[0]-10, int(1.3*textsize[1]))
        cv2.putText(img_k, label_text, label_pos, font_face,
                    font_scale, font_color, font_thickness, font_linetype)
    elif label_pos == 'top_left':
        label_text = f"{vid_label_k} #{frame_idx_k+1}"
        textsize = cv2.getTextSize(
            label_text, font_face, font_scale, font_thickness)[0]
        label_pos = (10, int(1.3*textsize[1]))
        cv2.putText(img_k, label_text, label_pos, font_face,
                    font_scale, font_color, font_thickness, font_linetype)
    return img_k


def cat_images(imgs, concat_direction, mid_bar_color=(255, 255, 255), mid_bar_width=10):
    # concatenate two frames according to concat_direction
    if concat_direction == 'x':
        bar_ = np.ones((imgs[0].shape[0], mid_bar_width), dtype='uint8')
        bar = np.stack(
            (bar_*mid_bar_color[0], bar_*mid_bar_color[1], bar_*mid_bar_color[2]), axis=2)
        img_cat = np.concatenate(
            (imgs[0][:, :(imgs[0].shape[1]-mid_bar_width)//2, :], bar, imgs[1][:, (imgs[0].shape[1]+mid_bar_width)//2:, :]), axis=1)
    elif concat_direction == 'y':
        bar_ = np.ones((mid_bar_width, imgs[0].shape[1]), dtype='uint8')
        bar = np.stack(
            (bar_*mid_bar_color[0], bar_*mid_bar_color[1], bar_*mid_bar_color[2]), axis=2)
        img_cat = np.concatenate((imgs[0][:(imgs[0].shape[0]-mid_bar_width)//2, :, :],
                              bar, imgs[1][(imgs[0].shape[0]+mid_bar_width)//2:, :, :]), axis=0)
    else:
        raise ValueError(
            "concat_direction should be 'x' or 'y'.")
    return img_cat


def main():
    # video generating
    # get image paths and info
    assert len(img_dirs) == len(
        vid_labels) == 2, "The number of image directories and labels should equal to 2."
    image_paths = []
    for k, img_dir in enumerate(img_dirs):
        image_paths_k = (sorted([os.path.join(img_dir, f)
                                 for f in os.listdir(img_dir)]))
        ends[k] = len(image_paths_k) if ends[k] == -1 else ends[k]
        image_paths.append(image_paths_k[starts[k]:ends[k]])

    # frame number check
    frame_nums = [len(image_paths_k) for image_paths_k in image_paths]
    max_frame_num = max(frame_nums)
    assert all([max_frame_num % frame_num == 0 for frame_num in frame_nums]
               ), "The number of frames in each video should be the proportionable."
    frame_num_ratio = [max_frame_num//frame_num for frame_num in frame_nums]

    # info
    img = cv2.imread(image_paths[0][0])
    height, width, channels = img.shape

    vid_format = save_path.rsplit('.')[-1]

    dst_size = (int(height*resize_ratio), int(width*resize_ratio))

    if concat_direction == 'x':
        label_pos = ['top_left', 'top_right']
    elif concat_direction == 'y':
        label_pos = ['top_right', 'bottom_right']
    else:
        raise ValueError(
            "concat_direction should be 'x' or 'y'.")

    # video generating
    if vid_format == 'gif':
        # Create a GIF file writer object

        with imageio.get_writer(save_path, mode='I', fps=fps) as writer:
            # Loop through each image, read & proc it, and add to the video file
            for i in tqdm(range(max_frame_num)):
                imgs = []
                for k in range(len(image_paths)):
                    frame_idx_k = i//frame_num_ratio[k]
                    img_k = imageio.v2.imread(image_paths[k][frame_idx_k])
                    img_k = frame_proc(
                        img_k, dst_size, vid_labels[k], frame_idx_k, label_pos[k])
                    imgs.append(img_k)

                # half-half concatenation
                imgs = cat_images(imgs, concat_direction,
                                  mid_bar_color, mid_bar_width)

                writer.append_data(imgs)
    else:
        # Create a video writer object
        if vid_format == 'avi':
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        elif vid_format == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            raise NotImplementedError(
                f'Not supported video saving format: {vid_format}')

        out = cv2.VideoWriter(save_path, fourcc, fps, dst_size[::-1])

        # Loop through each image, read & proc it, and add to the video file
        for i in tqdm(range(max_frame_num)):
            imgs = []
            for k in range(len(image_paths)):
                # proporationally read images and proc
                frame_idx_k = i//frame_num_ratio[k]
                img_k = cv2.imread(image_paths[k][frame_idx_k])
                img_k = frame_proc(
                    img_k, dst_size, vid_labels[k], frame_idx_k, label_pos[k])
                imgs.append(img_k)
            # montage to M*N images
            imgs = cat_images(imgs, concat_direction,
                              mid_bar_color, mid_bar_width)
            out.write(imgs)

        # Release the video object and close all windows
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
