from moviepy.editor import VideoFileClip
from moviepy.editor import *

def transfer_origin_video():
    video = VideoFileClip("/Users/monkeygeek/Downloads/222_1741778309.mp4")
    start_time = 12
    end_time = 18
    subclip = video.subclip(start_time, end_time)
    subclip.write_videofile("/Users/monkeygeek/Downloads/baobei.mp4", codec="libx264")

def crossfade():
    clip1 = VideoFileClip("/Users/monkeygeek/Downloads/baobei-0-6.mp4")
    clip2 = VideoFileClip("/Users/monkeygeek/Downloads/baobei-6-12.mp4")

    clip1 = clip1.fx(transfx.crossfadeout, duration=2)
    clip2 = clip2.fx(transfx.crossfadein, duration=2)

    final_clip = concatenate_videoclips([clip1, clip2], method="compose")
    final_clip.write_videofile("/Users/monkeygeek/Downloads/output.mp4")










if __name__ == '__main__':
    # 截取视频
    #transfer_origin_video()

    # 渐变过渡:适合多视频拼接、视频开头、视频结尾
    crossfade()

    # 淡入淡出:开始或者结尾
    # 滑入滑出
    # 旋转
    # 裁剪
    # 局部模糊
    # 尺寸调整
    # 色彩反转
    # 滚动
    # 加速减速:开始或结尾
    # 遮罩:与彩蛋动画结合
    # 页边距
    # 均匀尺寸
    # 超级采样
    # 滚动文字
    # 音频可视化

    pass

