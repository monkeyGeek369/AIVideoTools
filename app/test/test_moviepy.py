
from moviepy.editor import VideoFileClip,AudioFileClip,CompositeAudioClip,TextClip,CompositeVideoClip



if __name__ == '__main__':
   video_clip = VideoFileClip("F:\download\\test.mp4")
   txt_clip = TextClip("Your Subtitle Text", fontsize=50, color='white')
   txt_clip = txt_clip.set_position('bottom').set_duration(video_clip.duration).set_start(0)
   result = CompositeVideoClip([video_clip, txt_clip])
   result.write_videofile("F:\download\output_video.mp4", fps=24)

