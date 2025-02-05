from moviepy.editor import VideoFileClip
import os,pytesseract,re
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def test_video_subtitle_position_recognize():
   # 加载视频文件
   clip = VideoFileClip("F:\download\\test.webm")
   prev_frame = None

   for t, frame in clip.iter_frames(with_times=True):
      if prev_frame is not None:
            # 计算当前帧与前一帧的差异
            diff = np.sum(np.abs(frame - prev_frame))
            if diff > 1000000:
               # 获取指定时间点的帧,保存帧为图片
               frame = clip.get_frame(t)
               frame_path = os.path.join("F:\download\\tmp", f"frame_{t:.2f}s.png")
               img = Image.fromarray(frame)
               img.save(frame_path)
               
               # 识别图片中的文字
               config = '--psm 4 --oem 3'  # 假设一个均匀的文本块
               #config = '--psm 6 --oem 3'  # 假设一个均匀的文本块
               data = pytesseract.image_to_data(frame_path, lang='chi_sim',config=config, output_type=pytesseract.Output.DICT)
               # 遍历识别结果
               for j in range(len(data['text'])):
                  text = data['text'][j]
                  if text.strip() and contains_chinese(text):  # 确保文本不为空
                     x, y, w, h = int(data['left'][j]), int(data['top'][j]), int(data['width'][j]), int(data['height'][j])
                     print(f"Text: '{text}', Position: ({x}, {y}, {w}, {h})")

                     # 标记图片
                     image = Image.open(frame_path)
                     draw = ImageDraw.Draw(image)
                     draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=6)
                     font = ImageFont.truetype("F:\softProject\AIVideoTools\\resource\\fonts\STHeitiMedium.ttc", 60)
                     draw.text((x, y), text, font=font, fill="red")
                     image.save(frame_path)
      prev_frame = frame

   # 关闭视频文件
   clip.close()

if __name__ == '__main__':
   # test video subtitle position recognize
   test_video_subtitle_position_recognize()


