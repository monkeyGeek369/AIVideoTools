from moviepy.editor import VideoFileClip,TextClip,CompositeVideoClip
import os,cv2
import numpy as np

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
   特点：实现马赛克模糊化，但无法完全隐藏掉原遮蔽信息，会有明显的透视效果
   可以对遮罩区域进行对比度、亮度等调节
   :param frame: 视频帧
   :param x1: 马赛克区域左上角的 x 坐标
   :param y1: 马赛克区域左上角的 y 坐标
   :param x2: 马赛克区域右下角的 x 坐标
   :param y2: 马赛克区域右下角的 y 坐标
   :param neighbor: 马赛克块的大小
   :return: 添加马赛克后的帧
   """
   
   if neighbor <= 0:
      raise ValueError("Neighbor must be a positive integer.")
   
   # 提取感兴趣区域
   roi = frame[y1:y2, x1:x2]
   
   # 检查 ROI 尺寸是否有效
   if roi.shape[1] == 0 or roi.shape[0] == 0:
      raise ValueError("ROI dimensions must be greater than zero.")
   
   # 计算目标尺寸
   target_width = max(1, roi.shape[1] // neighbor)
   target_height = max(1, roi.shape[0] // neighbor)
   
   # 缩放 ROI
   roi = cv2.resize(roi, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
   
   # 将缩放后的 ROI 放大回原尺寸
   roi = cv2.resize(roi, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

   # 调整亮度
   roi = cv2.convertScaleAbs(roi, alpha=1.5, beta=0)
   
   # 转换到 HSV 色彩空间
   hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
   
   # 调整饱和度
   hsv_roi[:, :, 1] = hsv_roi[:, :, 1] * 0.5
   
   # 转换回 BGR 色彩空间
   roi = cv2.cvtColor(hsv_roi, cv2.COLOR_HSV2BGR)
   
   # 将处理后的 ROI 放回原图
   copy_frame = frame.copy()
   copy_frame[y1:y2, x1:x2] = roi
   return copy_frame


def apply_background_color(frame, x1, y1, x2, y2):
   """
   用背景颜色覆盖指定区域
   :param frame: 图像
   :param x1: 覆盖区域左上角的 x 坐标
   :param y1: 覆盖区域左上角的 y 坐标
   :param x2: 覆盖区域右下角的 x 坐标
   :param y2: 覆盖区域右下角的 y 坐标
   :return: 覆盖后的图像
   """
   copy_frame = frame.copy()

   # get bg color
   roi = frame[y1:y2, x1:x2]
   average_color = np.mean(roi, axis=(0, 1)).astype(int)
   background_color = tuple(average_color)

   # apply bg color
   copy_frame[y1:y2, x1:x2] = background_color

   return copy_frame


def apply_perspective_background_color(frame, x1, y1, x2, y2,extend_factor):
   """
   使用透视变换模拟背景效果覆盖指定区域
   特点：支持背景色计算区域扩展
   具有一定的透视变换效果
   :param frame: 图像
   :param x1: 覆盖区域左上角的 x 坐标
   :param y1: 覆盖区域左上角的 y 坐标
   :param x2: 覆盖区域右下角的 x 坐标
   :param y2: 覆盖区域右下角的 y 坐标
   :return: 覆盖后的图像
   """
   copy_frame = frame.copy()

   # 计算扩展后的区域
   extended_x1 = max(0, x1 - (x2 - x1) * extend_factor)
   extended_y1 = max(0, y1 - (y2 - y1) * extend_factor)
   extended_x2 = min(frame.shape[1], x2 + (x2 - x1) * extend_factor)
   extended_y2 = min(frame.shape[0], y2 + (y2 - y1) * extend_factor)

   extended_roi = frame[int(extended_y1):int(extended_y2), int(extended_x1):int(extended_x2)]
   background_color = np.mean(extended_roi, axis=(0, 1)).astype(int)

   background_color = tuple(background_color)

   # 创建一个与指定区域大小相同的背景颜色块
   background_block = np.full((y2 - y1, x2 - x1, 3), background_color, dtype=np.uint8)

   # 使用透视变换模拟背景效果
   src_pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
   dst_pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
   perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
   warped_block = cv2.warpPerspective(background_block, perspective_matrix, (x2 - x1, y2 - y1))

   # 将透视变换后的颜色块覆盖到原图的指定区域
   copy_frame[y1:y2, x1:x2] = warped_block

   return copy_frame

def add_mosaic_to_video():

    video = VideoFileClip("F:\download\\test2.mp4")
    #video_with_mosaic = video.fl_image(lambda frame: apply_mosaic(frame, 97, 559, 478, 607, 5))
    #video_with_mosaic = video.fl_image(lambda frame: apply_background_color(frame, 97, 559, 478, 607))
    video_with_mosaic = video.fl_image(lambda frame: apply_perspective_background_color(frame, 46, 547, 528, 617,2))
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

