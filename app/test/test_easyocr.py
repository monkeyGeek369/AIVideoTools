from moviepy.editor import VideoFileClip
import os,re,easyocr
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.services.video import video_subtitle_overall_statistics

print(torch.cuda.is_available())
print(torch.__version__)

reader = easyocr.Reader(
    lang_list=['ch_sim', 'en'],  # 语言列表
    gpu=True,  # 是否使用GPU
    model_storage_directory='..\models\easyocr',  # 模型存储目录
    download_enabled=True,  # 是否自动下载模型
    detector=True,  # 是否启用文本检测
    recognizer=True,  # 是否启用文本识别
    verbose=True  # 是否显示详细信息
)


def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def test_video_subtitle_position_recognize():
   # 加载视频文件
   clip = VideoFileClip("F:\download\\test2.mp4")
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
               result = reader.readtext(frame_path,# 读取图片
                                        detail=1,
                                        batch_size=10, # 批处理大小
                                        )
               #print(result)

               # 遍历识别结果
               for item in result:
                    text = item[1]
                    if text.strip() and contains_chinese(text):  # 确保文本不为空
                        top_left = tuple(map(int, item[0][0]))
                        bottom_right = tuple(map(int, item[0][2]))
                        print(f"Text: '{text}', Position: ({top_left}, {bottom_right})")

                        # 标记图片
                        image = Image.open(frame_path)
                        draw = ImageDraw.Draw(image)
                        draw.rectangle([top_left,bottom_right], outline="red", width=6)
                        font = ImageFont.truetype("F:\softProject\AIVideoTools\\resource\\fonts\STHeitiMedium.ttc", 40)
                        draw.text(top_left, text, font=font, fill="red")
                        image.save(frame_path)
        prev_frame = frame

   # 关闭视频文件
   clip.close()

if __name__ == '__main__':
   # test video subtitle position recognize
   #test_video_subtitle_position_recognize()

   # test video_subtitle_overall_statistics
   result = video_subtitle_overall_statistics("F:\download\\test2.mp4", 100, 100)
   print(result)






