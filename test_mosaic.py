from moviepy.editor import VideoFileClip
import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)  # 完全禁用大小限制
from app.services import paddleocr
from app.utils import file_utils


# mac
video0_path = "/Users/monkeygeek/Downloads/国外网友度假时偶遇只搁浅的小魔鬼鱼，好心收养后发现它可爱又会互动.webm"
image_frame_path = "/Users/monkeygeek/Downloads/tmp/frame"

# windows
# video0_path = "F:\download\\test.webm"

def test_video_frame():
    # first frame to image
    clip = VideoFileClip(video0_path)
    for t, frame in clip.iter_frames(with_times=True):
        cv2.imwrite(image_frame_path + f"/{t:.2f}s.png", frame)

def test_mosaic_telea():
    # get all frames
    files = file_utils.get_file_list(image_frame_path, ".png")
    if not files:
        return

    for file in files:
        img = cv2.imread(file.path)
        if img is None:
            print("read image failed")
            return
        
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        coordinates = paddleocr.get_frame_coordinates(file.path,False)
        for bbox in coordinates:
            (x1, y1), (x2, y2) = bbox
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        
        cv2.imwrite(file.path, result)

if __name__ == "__main__":
    test_mosaic_telea()
