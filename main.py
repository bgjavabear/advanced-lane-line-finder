import cv2
import os
from algorithm.threshold import pipeline
from moviepy.editor import VideoFileClip

image_input_path = 'data/main/test_images/'
image_output_path = 'data/main/output/images/'

test_images = os.listdir(image_input_path)

if not os.path.exists(image_output_path):
    os.mkdir(image_output_path)

for filename in test_images:
    image_path = os.path.join(image_input_path, filename)
    img = cv2.imread(image_path)
    output = pipeline(img)
    cv2.imwrite(os.path.join(image_output_path, filename), output)

video_input_path = 'data/main/test_videos/'
video_output_path = 'data/main/output/videos'
test_videos = os.listdir(video_input_path)

if not os.path.exists(video_output_path):
    os.mkdir(video_output_path)

for test_video_name in test_videos:
    video_path = os.path.join(video_input_path, test_video_name)
    clip = VideoFileClip(video_path)
    out_clip = clip.fl_image(pipeline)
    out_clip.write_videofile(os.path.join(video_output_path, test_video_name), audio=False)
