
from moviepy.editor import VideoFileClip,AudioFileClip,CompositeAudioClip,TextClip,CompositeVideoClip
import os

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


if __name__ == '__main__':
   font_path = "F:\\softProject\\AIVideoTools\\resource\\fonts\\STHeitiMedium.ttc"
   relpath = os.path.relpath(font_path)
   relpath = "./"+relpath.replace(os.sep, '/')
   print(relpath)

   # test subtitle
   #test_subtitle()

   # test crop
   test_crop()


