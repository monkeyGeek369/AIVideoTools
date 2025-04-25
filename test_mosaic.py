from moviepy.editor import VideoFileClip
import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)  # 完全禁用大小限制
from app.services import paddleocr
from app.utils import file_utils
import os


# mac
video0_path = "/Users/monkeygeek/Downloads/compound_video.mp4"
image_frame_path = "/Users/monkeygeek/Downloads/tmp/frame"
mosaic_image_path = "/Users/monkeygeek/Downloads/tmp/mosaic"
tmp_path = "/Users/monkeygeek/Downloads/tmp"

# windows
# video0_path = "F:\download\\test.webm"

def test_video_frame():
    # first frame to image
    clip = VideoFileClip(video0_path)
    index = 1
    for t, frame in clip.iter_frames(with_times=True):
        cv2.imwrite(image_frame_path + f"/{index}.png", frame)
        index += 1

def test_mosaic_telea():
    # get all frames
    files = file_utils.get_file_list(mosaic_image_path, ".png")
    if not files:
        return

    for file in files:
        img = cv2.imread(file.path)
        if img is None:
            print("read image failed")
            return
        
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        coordinates = paddleocr.get_frame_coordinates(file.path,False)
        if not coordinates:
            continue
        for bbox in coordinates:
            (x1, y1), (x2, y2) = bbox
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

        save_path = os.path.join(tmp_path, file.name+"."+file.suffix)
        
        cv2.imwrite(save_path, result)

def test_mosaic_navier():
    files = file_utils.get_file_list(mosaic_image_path, ".png")
    if not files:
        return

    for index,file in enumerate(files):
        img = cv2.imread(file.path)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        # 3. 形态学处理优化掩码
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        if img is None or mask is None:
            return
        
        coordinates = paddleocr.get_frame_coordinates(file.path,False)
        if not coordinates:
            continue
        for bbox in coordinates:
            (x1, y1), (x2, y2) = bbox
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        result = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
        
        save_path = os.path.join(tmp_path, str(index)+"."+file.suffix)
        cv2.imwrite(save_path, result)

if __name__ == "__main__":
    test_video_frame()
    #test_mosaic_telea()
    #test_mosaic_navier()
