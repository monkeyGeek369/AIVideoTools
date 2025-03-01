from moviepy.editor import VideoFileClip,TextClip,CompositeVideoClip,AudioFileClip,CompositeAudioClip
import os,cv2
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.services.mosaic import apply_perspective_background_color

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

def add_mosaic_to_video():

    video = VideoFileClip("F:\download\\test2.mp4")
    #video_with_mosaic = video.fl_image(lambda frame: apply_mosaic(frame, 97, 559, 478, 607, 5))
    #video_with_mosaic = video.fl_image(lambda frame: apply_background_color(frame, 97, 559, 478, 607))
    video_with_mosaic = video.fl_image(lambda frame: apply_perspective_background_color(frame, 46, 547, 528, 617,2))
    video_with_mosaic.write_videofile("F:\download\\test_mosaic_out.mp4", codec="libx264")

def compress_video():
   video_clip = VideoFileClip("F:\download\\tmp\\edit_video.mp4")
   video_duration = video_clip.duration
   audio_clips = []
   
   # add bg music
   audio_clip = AudioFileClip("F:\download\\tmp\\edit_bg_music.mp3")
   audio_clip = audio_clip.set_duration(video_duration)
   audio_clips.append(audio_clip)

   # add voice
   voice_clip = AudioFileClip("F:\download\\tmp\\edit_audio.mp3")
   audio_clips.append(voice_clip)

   # save
   mix_audio_clip = CompositeAudioClip(audio_clips)
   video_clip = video_clip.set_audio(mix_audio_clip)
   final_clip = video_clip

   # mp4 比原视频略大0.5m可以接受
   temp_audio_path = "temp-audio.aac"
   final_clip.write_videofile(
      "F:/download/tmp/optimized_video.mp4",
      codec='libx264',
      audio_codec='aac',
      fps=video_clip.fps,
      preset='slow',
      threads=os.cpu_count(),
      ffmpeg_params=[
         "-crf", "30",          # 控制视频“质量”，这里的质量主要是指视频的主观视觉质量，即人眼观看视频时的清晰度、细节保留程度以及压缩带来的失真程度
         "-b:v", "2000k", # 设置目标比特率，控制视频每秒数据量，与视频大小有直接关联。
         "-pix_fmt", "yuv420p",#指定像素格式。yuv420p 是一种常见的像素格式，兼容性较好，适用于大多数播放器。
         "-row-mt", "1"#启用行级多线程，允许编码器在单帧内并行处理多行数据，从而提高编码效率。0表示不启用
      ],
      write_logfile=False, #是否写入日志
      remove_temp=True,#是否删除临时文件
      temp_audiofile=temp_audio_path  #指定音频的临时文件路径
   )
   
   # webm格式 与原格式相差无几
   #temp_audio_path = "temp_audio.ogg"
   # final_clip.write_videofile(
   #    "F:/download/tmp/compress_video.webm",
   #    codec="libvpx-vp9",        # VP9 编码器压缩效率更高
   #    audio_codec="vorbis",      # 与视频编码器对应的音频编码器
   #    fps=video_clip.fps,
   #    preset="slow", #预设模式决定了编码器在编码过程中对速度和压缩效率的权衡（slow编码速度较慢，但压缩效率较高，适合需要高质量输出的场景。）
   #    threads=os.cpu_count(),   # 线程数，默认为cpu核心数
   #    ffmpeg_params=[
   #       "-crf", "30",          # 控制视频“质量”，这里的质量主要是指视频的主观视觉质量，即人眼观看视频时的清晰度、细节保留程度以及压缩带来的失真程度
   #       "-b:v", "2000k", # 设置目标比特率，控制视频每秒数据量，与视频大小有直接关联。
   #       "-pix_fmt", "yuv420p",#指定像素格式。yuv420p 是一种常见的像素格式，兼容性较好，适用于大多数播放器。
   #       "-row-mt", "1"#启用行级多线程，允许编码器在单帧内并行处理多行数据，从而提高编码效率。0表示不启用
   #    ],
   #    write_logfile=False, #是否写入日志
   #    remove_temp=True,#是否删除临时文件
   #    temp_audiofile=temp_audio_path  #指定音频的临时文件路径
   # )

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
   #add_mosaic_to_video()

   # test compress video
   compress_video()

