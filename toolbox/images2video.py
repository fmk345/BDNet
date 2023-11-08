# --------------------------------------
# generate a video from an image directory
# --------------------------------------

import cv2
import imageio
from skimage.transform import resize
import os
from tqdm import tqdm

## params
# path
img_dir = "./outputs/code_dev/BDNeRV_RC/test/2023-04-24_09-56-16/zout"
save_path = "./test_vid.mp4"  # .gif / .avi / .mp4

# Set the starting and ending image numbers to read
start = 0
end = -1  # -1 for all

# Set the frame size
resize_ratio = 1

# Set the frame rate
fps = 10
show_frame_idx = True


## video generating
# video setting
image_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
end = len(image_paths) if end==-1 else end
img = cv2.imread(image_paths[start])
height, width, channels = img.shape
vid_format = save_path.rsplit('.')[-1]
font_face, font_scale, font_color, font_thickness, font_linetype = cv2.FONT_HERSHEY_COMPLEX, height * \
    resize_ratio/600, (255, 255, 255), 1, cv2.LINE_AA

# loop run
if vid_format == 'gif' :
    # Create a GIF file writer object
    dst_size = (int(height*resize_ratio), int(width*resize_ratio))
    with imageio.get_writer(save_path, mode='I', fps=fps) as writer:
        # Loop through each image, read it, and add to the GIF file
        for i in tqdm(range(start, end)):
            img_path = image_paths[i]
            img = imageio.imread(img_path)
            if show_frame_idx:
                label_text = f"#{i+1}"
                textsize = cv2.getTextSize(
                    label_text, font_face, font_scale, font_thickness)[0]
                label_pos = (width - textsize[0]-10, 40)
                cv2.putText(img, label_text, label_pos, font_face,
                            font_scale, font_color, font_thickness, font_linetype)
            img = (resize(img, dst_size)*255.0).astype('uint8')
            writer.append_data(img)
else:
    # Create a video writer object
    dst_size = (int(width*resize_ratio), int(height*resize_ratio))
    if vid_format=='avi':
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    elif vid_format=='mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        raise NotImplementedError(f'Not supported video saving format: {vid_format}')

    out = cv2.VideoWriter(save_path, fourcc, fps, dst_size)

    # Loop through each image, read it, and add to the video file
    for i in tqdm(range(start, end)):
        img_path = image_paths[i]
        img = cv2.imread(img_path)
        if show_frame_idx:
            label_text = f"#{i+1}"
            textsize = cv2.getTextSize(
                label_text, font_face, font_scale, font_thickness)[0]
            label_pos = (width - textsize[0]-10, 40)
            cv2.putText(img, label_text, label_pos, font_face,
                        font_scale, font_color, font_thickness, font_linetype)
        img = cv2.resize(img, dst_size)
        out.write(img)

    # Release the video object and close all windows
    out.release()
    cv2.destroyAllWindows()


