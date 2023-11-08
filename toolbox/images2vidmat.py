# --------------------------------------
# generate a montage video matrix from multiple image directories
#  - image directories should have the proportionable number of images
# --------------------------------------

import cv2
import numpy as np
import imageio
from skimage.transform import resize
import os
from tqdm import tqdm
from einops import rearrange

# %% Param setting
# path
img_dirs = ["outputs/code_dev/BDNeRV_RC/real_test/2023-05-29_15-09-50/outputs/input",
            "outputs/code_dev/BDNeRV_RC/real_test/2023-05-29_15-09-50/outputs/output"]
# "outputs/code_dev/BDNeRV_RC/test/2023-05-01_05-51-33/outputs/target"
save_path = "./test.mp4"  # .gif / .avi / .mp4

# Set the starting and ending image numbers to read
starts = [0, 0]
ends = [-1, -1]  # -1 for all
# ends = [-1, -1, -1, -1]  # -1 for all

# Set the frame size and margin
resize_ratio = 0.5
margin = [30, 5]  # [height, width]
n_row = 1  # number of rows in the montage

# Set the frame rate and labels
fps = 8
show_frame_idx = True
vid_labels = ['ce', 'output'] # 'ce', 'output', 'gt'
font_face, font_color, font_thickness, font_linetype = cv2.FONT_HERSHEY_SIMPLEX, (
    0, 0, 255), 1, cv2.LINE_AA


# %% Functions
def frame_proc(img_k, dst_size, vid_label_k, frame_idx_k):
    # process a single frame: resize, add margin, add label & frame index
    # resize
    img_k = (resize(img_k, dst_size)*255.0).astype('uint8')
    # add margin
    img_k = np.pad(
        img_k, ((margin[0], margin[0]), (margin[1], margin[1]), (0, 0)), 'constant', constant_values=255)
    font_scale = img_k.shape[0]/600  # font size
    # add label
    label_text = f"{vid_label_k}"
    textsize = cv2.getTextSize(
        label_text, font_face, font_scale, font_thickness)[0]
    label_pos = ((img_k.shape[1] - textsize[0]) // 2,
                 img_k.shape[0] - int(0.3*margin[0]))
    cv2.putText(img_k, label_text, label_pos, font_face,
                font_scale, font_color, font_thickness, font_linetype)
    # add frame index
    if show_frame_idx:
        frame_idx_text = f"#{frame_idx_k+1}"
        textsize = cv2.getTextSize(
            frame_idx_text, font_face, font_scale, font_thickness)[0]
        label_pos = (img_k.shape[1] - margin[1] -
                     textsize[0]-10, margin[0] + int(1.3*textsize[1]))
        cv2.putText(img_k, frame_idx_text, label_pos, font_face,
                    font_scale, font_color, font_thickness, font_linetype)
    return img_k


def main():
    # video generating
    assert len(img_dirs) == len(
        vid_labels) , "The number of image directories and labels should be the same."
    # get image paths and info
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
    dst_size = (int(height*resize_ratio), int(width*resize_ratio))

    vid_format = save_path.rsplit('.')[-1]

    # video generating
    if vid_format == 'gif':
        # Create a GIF file writer object
        with imageio.get_writer(save_path, mode='I', fps=fps) as writer:
            # Loop through each image, read & proc it, and add to the video file
            for i in tqdm(range(max_frame_num)):
                imgs = []
                for k in range(len(image_paths)):
                    # proporationally read images and proc
                    frame_idx_k = i//frame_num_ratio[k]
                    img_k = imageio.imread(image_paths[k][frame_idx_k])
                    img_k = frame_proc(
                        img_k, dst_size, vid_labels[k], frame_idx_k)
                    imgs.append(img_k)

                # montage to M*N images
                imgs = rearrange(
                    np.array(imgs), '(n1 n2) h w c->(n1 h) (n2 w) c', n1=n_row)

                writer.append_data(imgs)
    else:
        # Create a video writer object
        vid_size = ((dst_size[1]+margin[1]*2)*len(img_dirs)//n_row,
                    (dst_size[0]+margin[0]*2)*n_row) # (width, height)
        if vid_format == 'avi':
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        elif vid_format == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            raise NotImplementedError(
                f'Not supported video saving format: {vid_format}')

        out = cv2.VideoWriter(save_path, fourcc, fps, vid_size)

        # Loop through each image, read & proc it, and add to the video file
        for i in tqdm(range(max_frame_num)):
            imgs = []
            for k in range(len(image_paths)):
                # proporationally read images and proc
                frame_idx_k = i//frame_num_ratio[k]
                img_k = cv2.imread(image_paths[k][frame_idx_k])
                img_k = frame_proc(
                    img_k, dst_size, vid_labels[k], frame_idx_k)
                imgs.append(img_k)
            # montage to M*N images
            imgs = rearrange(
                np.array(imgs), '(n1 n2) h w c->(n1 h) (n2 w) c', n1=n_row)
            out.write(imgs)

        # Release the video object and close all windows
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
