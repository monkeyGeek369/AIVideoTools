from moviepy.editor import VideoFileClip,TextClip,CompositeVideoClip
import os,cv2

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

def apply_mosaic(frame, x1, y1, x2, y2, neighbor=10):
    """
    对指定区域添加马赛克效果
    :param frame: 视频帧
    :param x1: 马赛克区域左上角的 x 坐标
    :param y1: 马赛克区域左上角的 y 坐标
    :param x2: 马赛克区域右下角的 x 坐标
    :param y2: 马赛克区域右下角的 y 坐标
    :param neighbor: 马赛克块的大小
    :return: 添加马赛克后的帧
    """
    # 获取指定区域
    roi = frame[y1:y2, x1:x2]
    # 对区域进行马赛克处理
    roi = cv2.resize(roi, (roi.shape[1] // neighbor, roi.shape[0] // neighbor))
    roi = cv2.resize(roi, (roi.shape[1] * neighbor, roi.shape[0] * neighbor))
    # 将处理后的区域放回原图
    copy_frame = frame.copy()
    copy_frame[y1:y2, x1:x2] = roi
    return copy_frame

def add_mosaic_to_video():

    video = VideoFileClip("F:\download\\test2.mp4")
    video_with_mosaic = video.fl_image(lambda frame: apply_mosaic(frame, 100, 100, 400, 400, 50))
    video_with_mosaic.write_videofile("F:\download\\test_mosaic_out.mp4", codec="libx264")

if __name__ == '__main__':
   font_path = "F:\\softProject\\AIVideoTools\\resource\\fonts\\STHeitiMedium.ttc"
   relpath = os.path.relpath(font_path)
   relpath = "./"+relpath.replace(os.sep, '/')
   print(relpath)

   # test subtitle
   #test_subtitle()

   # test crop
   #test_crop()

   # test mosaic
   add_mosaic_to_video()

