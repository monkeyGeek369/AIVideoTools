from moviepy.editor import VideoFileClip
import os
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR, draw_ocr

video_path = "/Users/monkeygeek/Downloads/test.webm"
tmp_path = "/Users/monkeygeek/Downloads/tmp"
font_file_path = "/Users/monkeygeek/Documents/softProject/AIVideoTools/resource/fonts/STHeitiMedium.ttc"

def test_video_subtitle_position_recognize(video_path:str,tmp_path:str,font_file_path:str):
   clip = VideoFileClip(video_path)
   # https://paddlepaddle.github.io/PaddleOCR/latest/model/index.html
   ocr = PaddleOCR(
                use_angle_cls=False, # 关闭方向检测,提升速度
                det_model_dir="./resource/ocr_model/ch_PP-OCRv4_det_train",#区域检测
                rec_model_dir="./resource/ocr_model/ch_PP-OCRv4_rec_train",#方向识别
                cls_model_dir="./resource/ocr_model/ch_ppocr_mobile_v2.0_cls_train",#文本分类
                use_gpu=False, # 关闭GPU
                lang="ch" # 识别中文
                   )

   for t, frame in clip.iter_frames(with_times=True):
        frame = clip.get_frame(t)
        frame_path = os.path.join(tmp_path, f"frame_{t:.2f}s.png")
        img = Image.fromarray(frame)
        img.save(frame_path)
        
        # 识别图片中的文字
        result = ocr.ocr(frame_path,cls=True)
        if result is None or len(result) == 0 or result[0] is None:
            continue

        # 遍历识别结果
        for item in result[0]:
            position = item[0]
            text = item[1][0]
            if text.strip():  # 确保文本不为空
                top_left = tuple(map(int, position[0]))
                bottom_right = tuple(map(int, position[2]))
                print(f"Text: '{text}', Position: ({top_left}, {bottom_right})")

                # 标记图片
                image = Image.open(frame_path)
                draw = ImageDraw.Draw(image)
                draw.rectangle([top_left,bottom_right], outline="red", width=6)
                font = ImageFont.truetype(font_file_path, 40)
                draw.text(top_left, text, font=font, fill="red")
                image.save(frame_path)
   # 关闭视频文件
   clip.close()


if __name__ == '__main__':
   # test video subtitle position recognize
   test_video_subtitle_position_recognize(video_path,
                                          tmp_path,
                                          font_file_path)


