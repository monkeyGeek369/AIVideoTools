
from moviepy.editor import VideoFileClip,AudioFileClip,CompositeAudioClip,TextClip,CompositeVideoClip


def test_subtitle():
   # load video
   video_clip = VideoFileClip("/Users/monkeygeek/Downloads/test.mp4")

   #colors = TextClip.list('color')
   #print(colors)

   #fonts = TextClip.list('font')
   #print(fonts)

   # subtitle test
   txt_clip = TextClip(
      txt="hello world , this is a subtitle",
      size=(video_clip.size[0], None), # subtitle width and height(auto height)
      fontsize=60,
      bg_color="blue",
      color="white",
      #font="Arial", # font name from system and imageMagick
      font="/Users/monkeygeek/Documents/softProject/AIVideoTools/resource/fonts/STHeitiMedium.ttc",
      stroke_color="black",
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
   result.write_videofile("/Users/monkeygeek/Downloads/output_video.mp4", fps=24)

if __name__ == '__main__':

   # test subtitle
   test_subtitle()


