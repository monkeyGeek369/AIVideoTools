
from moviepy.editor import VideoFileClip,AudioFileClip,CompositeAudioClip,TextClip,CompositeVideoClip
import os,cv2,pytesseract

def test_subtitle():
   # load video
   video_clip = VideoFileClip("F:\download\\test2.mp4")

   #colors = TextClip.list('color')
   #print(colors)

   #fonts = TextClip.list('font')
   #print(fonts)

   # subtitle test
   txt_clip = TextClip(
      txt="hello world 234 汉字, this is a subtitle",
      size=(video_clip.size[0], None), # subtitle width and height(auto height)
      fontsize=60,
      #bg_color="blue",
      bg_color="#00FF00",
      #color="white",
      color="#000000",
      #font="Arial", # font name from system and imageMagick
      #font="Microsoft-YaHei-Light-&-Microsoft-YaHei-UI-Light",
      font="./resource/fonts/STHeitiMedium.ttc",
      #stroke_color="black",
      stroke_color="#FFFFFF",
      stroke_width=1.5,
      method="caption", # subtitle while drawn in a picture with fixed size
      kerning=1, # letter spacing
      align="center", # txt align 文本对齐方式
      interline=None,# interline 设置行间距倍数
      transparent=True,
      remove_temp=True
   )
   #txt_clip = txt_clip.set_position('bottom').set_duration(video_clip.duration).set_start(0)
   #txt_clip = txt_clip.set_position((45,150)).set_duration(video_clip.duration).set_start(0)
   #txt_clip = txt_clip.set_position(("center","top")).set_duration(video_clip.duration).set_start(0)
   #txt_clip = txt_clip.set_position((0.4,0.7), relative=True).set_duration(video_clip.duration).set_start(0)
   txt_clip = txt_clip.set_position(lambda t: ('center', 50+t) ).set_duration(video_clip.duration).set_start(0) # x:center y:50px from top

   # combine
   result = CompositeVideoClip([video_clip, txt_clip])
   result.write_videofile("F:\download\output_video.mp4", fps=24)

def test_crop():
   # load video
   video_clip = VideoFileClip("/Users/monkeygeek/Downloads/test.webm")

   width, height = video_clip.size

   # 计算裁剪区域的坐标
   x1 = 0
   y1 = 0
   x2 = width
   y2 = y1 + 500

   # 裁剪视频画面
   cropped_clip = video_clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)

   cropped_clip.write_videofile("/Users/monkeygeek/Downloads/output.mp4", codec="libx264", fps=24)

def test_video_subtitle_position_recognize():
    # 加载视频
    clip = VideoFileClip("/Users/monkeygeek/Downloads/test.webm")
        
    # 获取视频的帧率
    fps = clip.fps
    
    # 遍历视频的每一帧
    for i, frame in enumerate(clip.iter_frames()):
        # 将帧转换为灰度图像
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用 Tesseract 进行文字识别并获取位置信息
        data = pytesseract.image_to_data(gray_frame,lang='chi_sim', output_type=pytesseract.Output.DICT)
        
        # 遍历识别结果
        for j in range(len(data['text'])):
            text = data['text'][j]
            if text.strip():  # 确保文本不为空
               #print(data)
               x, y, w, h = int(data['left'][j]), int(data['top'][j]), int(data['width'][j]), int(data['height'][j])
               print(f"Frame {i}, Time: {i / fps:.2f}s, Text: '{text}', Position: ({x}, {y}, {w}, {h})")
                
               # 将帧数据复制到一个新的可写数组中
               frame_copy = frame.copy()
                
               # 在帧上绘制矩形框
               cv2.rectangle(frame_copy, (x, y), (x + 10, y + 10), (0, 255, 0), 2)

               # 显示处理后的帧
               cv2.imshow('Frame', frame_copy)
        
        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cv2.destroyAllWindows()
    clip.close()

if __name__ == '__main__':
   font_path = "F:\\softProject\\AIVideoTools\\resource\\fonts\\STHeitiMedium.ttc"
   relpath = os.path.relpath(font_path)
   relpath = "./"+relpath.replace(os.sep, '/')
   print(relpath)

   # test subtitle
   #test_subtitle()

   # test crop
   #test_crop()

   # test video subtitle position recognize
   test_video_subtitle_position_recognize()


