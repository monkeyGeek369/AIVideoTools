import matplotlib
matplotlib.use('Agg')  # 强制使用无头渲染模式
from moviepy.editor import *
from moviepy.video.fx.all import rotate
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.fft import fft
import cv2

# mac
# video1_path = "/Users/monkeygeek/Downloads/baobei-0-6.mp4"
# video2_path = "/Users/monkeygeek/Downloads/baobei-6-12.mp4"
# output_path = "/Users/monkeygeek/Downloads/output.mp4"

# windows
video1_path = "F:\download\\test-0-6.mp4"
video2_path = "F:\download\\test-6-12.mp4"
output_path = "F:\download\\test_out.mp4"

def transfer_origin_video():
    video = VideoFileClip("F:\download\\test.webm")
    start_time = 12
    end_time = 18
    subclip = video.subclip(start_time, end_time)
    subclip.write_videofile(output_path, codec="libx264")

def crossfade():
    clip1 = VideoFileClip(video1_path)
    clip2 = VideoFileClip(video2_path)

    clip1 = clip1.fx(transfx.crossfadeout, duration=2)
    clip2 = clip2.fx(transfx.crossfadein, duration=2)

    final_clip = concatenate_videoclips([clip1, clip2], method="compose")
    final_clip.write_videofile(output_path)

def fade_in_out():
    video = VideoFileClip(video1_path)
    video_with_fade = video.fadein(2).fadeout(2)
    video_with_fade.write_videofile(output_path)

def video_slide():
    transition_duration = 0.5
    clip1 = VideoFileClip(video1_path)
    clip2 = VideoFileClip(video2_path)

    width, height = clip1.size
    def slide_in(t):
        if t < transition_duration:
            x_position = width - (t / transition_duration) * width
        else:
            x_position = 0
        return x_position

    clip2_sliding = clip2.set_position(lambda t: (slide_in(t), "center"))
    clip2_sliding = clip2_sliding.set_start(clip1.duration)
    composite_clip = CompositeVideoClip([clip1, clip2_sliding], size=(width, height))
    composite_clip.write_videofile(output_path, codec='libx264')

def video_rotate():
    transition_duration = 0.5
    clip1 = VideoFileClip(video1_path)
    clip2 = VideoFileClip(video2_path)
    
    def apply_rotate_effect(get_frame, t):
        """ 动态旋转函数（每帧处理） """
        if t < transition_duration:
            progress = t / transition_duration
            angle = 360 * progress  
            
            frame_clip = ImageClip(get_frame(t))
            rotated_clip = rotate(frame_clip, angle, expand=True)
            return rotated_clip.get_frame(0)
        else:
            return get_frame(t)
    
    clip2 = clip2.fl(apply_rotate_effect).set_start(clip1.duration)
    
    final_clip = concatenate_videoclips([
        clip1,
        clip2            
    ], method="compose")
    
    final_clip.write_videofile(output_path, codec="libx264", fps=24, threads=4, preset="fast")

def video_crop():
    video = VideoFileClip(video1_path)
    width, height = video.size

    x1 = width * 0.05  # 左边界，裁剪掉左边的 5%
    y1 = height * 0.05  # 上边界，裁剪掉上边的 5%
    x2 = width * 0.95  # 右边界，裁剪掉右边的 5%
    y2 = height * 0.95  # 下边界，裁剪掉下边的 5%

    cropped_video = video.crop(x1=x1, y1=y1, x2=x2, y2=y2)
    cropped_video.write_videofile(output_path)

def video_zoom():
    # 加载两个视频片段
    clip1 = VideoFileClip(video1_path)  # 第一个视频
    clip2 = VideoFileClip(video2_path)  # 第二个视频

    # 定义变焦效果的开始时间和持续时间
    zoom_start_time = 0  # 变焦开始时间（秒）
    zoom_duration = 0.5    # 变焦持续时间（秒）
    # 定义 resize_frame 函数
    def resize_frame(frame, new_size):
        """
        使用 Pillow 对帧进行缩放。
        :param frame: 原始帧（numpy 数组）
        :param new_size: 新的尺寸 (width, height)
        :return: 缩放后的帧
        """
        # 将 numpy 数组转换为 PIL 图像
        img = Image.fromarray(frame)
        # 缩放图像
        img_resized = img.resize(new_size, Image.LANCZOS)
        # 将 PIL 图像转换回 numpy 数组
        return np.array(img_resized)

    # 定义 crop_frame 函数
    def crop_frame(frame, target_width, target_height):
        """
        裁剪帧以适应目标尺寸。
        :param frame: 缩放后的帧（numpy 数组）
        :param target_width: 目标宽度
        :param target_height: 目标高度
        :return: 裁剪后的帧
        """
        # 获取当前帧的尺寸
        current_height, current_width, _ = frame.shape

        # 计算裁剪的边界
        left = (current_width - target_width) // 2
        top = (current_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height

        # 裁剪帧
        return frame[top:bottom, left:right]

    # 自定义变焦函数
    def zoom_effect(get_frame, t):
        # 获取当前帧
        frame = get_frame(t)
        
        # 计算时间比例
        if zoom_start_time <= t < zoom_start_time + zoom_duration:
            # 时间比例在 [0, 1] 之间变化
            time_ratio = (t - zoom_start_time) / zoom_duration
            scale_factor = 1 + 0.5 * (1 - abs(1 - 2 * time_ratio))  # 先放大到 1.2 倍，再缩小回 1 倍
        else:
            scale_factor = 1  # 不在变焦时间范围内时保持原大小
        
        # 缩放帧
        new_size = (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor))
        scaled_frame = resize_frame(frame, new_size)  # 使用自定义的 resize_frame 函数缩放帧
        
        # 裁剪回原始大小
        cropped_frame = crop_frame(scaled_frame, frame.shape[1], frame.shape[0])  # 使用自定义的 crop_frame 函数裁剪帧
        return cropped_frame

    # 应用变焦效果
    zoomed_clip2 = clip2.fl(zoom_effect)

    # 拼接视频
    final_clip = concatenate_videoclips([clip1, zoomed_clip2])

    # 保存最终结果
    final_clip.write_videofile(output_path)

def video_blur():
    def blur_region(frame, top_ratio, bottom_ratio, sigma=10):
        """
        对图像的上部分和下部分的指定比例区域进行模糊处理。
        :param image: 输入的视频帧图像
        :param top_ratio: 上部分区域的高度比例（0到1之间）
        :param bottom_ratio: 下部分区域的高度比例（0到1之间）
        :param sigma: 高斯模糊的标准差
        :return: 处理后的图像
        """
        image = frame.copy()
        height, width, _ = image.shape
        top_height = int(height * top_ratio)
        bottom_height = int(height * bottom_ratio)

        # 模糊上部分区域
        if top_height > 0:
            image[:top_height, :, :] = gaussian_filter(image[:top_height, :, :], sigma=(sigma, sigma, 0))

        # 模糊下部分区域
        if bottom_height > 0:
            image[-bottom_height:, :, :] = gaussian_filter(image[-bottom_height:, :, :], sigma=(sigma, sigma, 0))

        return image

    # 读取视频
    clip = VideoFileClip(video1_path)

    # 应用模糊效果
    blurred_clip = clip.fl_image(lambda image: blur_region(image, top_ratio=0.2, bottom_ratio=0.2))

    # 输出视频
    blurred_clip.write_videofile(output_path, codec="libx264")

def video_up_factor():
    # 加载视频
    video = VideoFileClip(video1_path)

    # 设置加速倍数（例如加速2倍）
    speed_up_factor = 2

    # 加速视频
    accelerated_video = video.fx(vfx.speedx, speed_up_factor)

    # 输出加速后的视频
    accelerated_video.write_videofile(output_path, codec="libx264")

def video_down_factor():
    video = VideoFileClip(video1_path)

    # 设置减速倍数（例如减速到原来的一半）
    slow_down_factor = 0.5

    # 减速视频
    slowed_video = video.fx(vfx.speedx, slow_down_factor)

    # 输出减速后的视频
    slowed_video.write_videofile(output_path, codec="libx264")

def video_margin():
    # 加载视频文件
    video = VideoFileClip(video1_path)

    # 添加页边距
    # 参数 mar 指定边框宽度，color 指定边框颜色，opacity 指定边框透明度
    new_clip = video.fx(vfx.margin, mar=10, color=(0, 0, 255), opacity=0.5)

    # 输出到新文件
    new_clip.write_videofile(output_path)

def video_supersample():
    # 加载视频文件
    video = VideoFileClip(video1_path)

    # 应用超级采样特效
    # 参数 d=0.1 表示时间范围为 0.1 秒，nframes=5 表示采样 5 帧
    new_clip = video.fx(vfx.supersample, d=0.1, nframes=5)

    # 输出到新文件
    new_clip.write_videofile(output_path)

def audio_visualization_effect(video_path, output_path):
    video = VideoFileClip(video_path)

    with AudioFileClip(video_path) as audio_clip:
        sample_rate = audio_clip.fps
        raw_audio = audio_clip.coreader().iter_frames()
        audio_data = np.vstack([frame for frame in raw_audio])

    audio_data = audio_data[:, 0] if audio_data.ndim == 2 else audio_data.flatten()
    audio_data = audio_data.astype(np.float32)
    
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data /= max_val

    def add_visualization(get_frame, t):
        frame = get_frame(t)
        frame = frame.astype(np.uint8)

        sample_index = int(t * sample_rate)
        start = max(0, sample_index - 100)
        end = min(len(audio_data), sample_index + 100)
        
        segment = audio_data[start:end]
        if len(segment) < 200:
            segment = np.pad(segment, (0, 200 - len(segment)), 'constant')

        # 设置透明背景
        plt.figure(figsize=(8, 2), dpi=100, facecolor='none', edgecolor='none')
        plt.axis('off')
        plt.axhline(y=0, color='white', linewidth=1, alpha=0.3)
        
        num_bars = 40
        sub_grids_per_bar = 15
        bar_width = 1
        sub_height = 0.1
        colors = plt.cm.plasma(np.linspace(0, 1, sub_grids_per_bar))
        
        chunk_size = len(segment) // num_bars
        for i in range(num_bars):
            main_chunk = segment[i*chunk_size : (i+1)*chunk_size]
            amplitude = np.mean(np.abs(main_chunk))
            num_lit = min(int(amplitude * sub_grids_per_bar * 1.2), sub_grids_per_bar)
            
            for k in range(num_lit):
                plt.bar(
                    x=i,
                    height=sub_height,
                    width=bar_width,
                    bottom=k * sub_height,
                    color=colors[k],
                    edgecolor='white',
                    linewidth=0.5,
                    alpha=0.9
                )

        plt.xlim(-1, num_bars)
        plt.ylim(0, sub_grids_per_bar * sub_height + 0.3)
        plt.tight_layout(pad=0)

        # 保留Alpha通道
        canvas = plt.gcf().canvas
        canvas.draw()
        image_data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        image_data = image_data.reshape(canvas.get_width_height()[::-1] + (4,))  # 保持RGBA
        plt.close()

        vis_height = min(200, frame.shape[0] // 4)
        img_pil = Image.fromarray(image_data, 'RGBA').resize((frame.shape[1], vis_height))

        # 使用Alpha通道合成
        frame_pil = Image.fromarray(frame).convert('RGBA')
        frame_pil.paste(img_pil, (0, frame_pil.height - vis_height), img_pil)
        blended = np.array(frame_pil.convert('RGB'))
        
        return blended.astype(np.uint8)

    processed_video = video.fl(add_visualization)
    processed_video.write_videofile(output_path,
                                  codec='libx264',
                                  audio_codec='aac',
                                  threads=4,
                                  logger='bar')

if __name__ == '__main__':
    # 截取视频
    #transfer_origin_video()

    # 渐变过渡:适合多视频拼接、视频开头、视频结尾
    #crossfade()

    # 淡入淡出:适合单视频、视频开头、视频结尾
    #fade_in_out()

    # 滑入滑出:可以针对视频、文字、图片等对象进行滑入滑出，适合多视频拼接、单视频剪辑、特效制作、彩蛋制作等
    #video_slide()

    # 旋转:可以针对视频、文字、图片等对象进行滑入滑出，适合多视频拼接、单视频剪辑、特效制作、彩蛋制作等
    # 缺点是需要逐帧处理，性能不高
    #video_rotate()

    # 裁剪:性能高，可以均匀裁剪
    #video_crop()

    # 变焦特效:适合多视频拼接、单视频剪辑、特效制作等,性能还可以
    #video_zoom()

    # 局部模糊:上下部分做局部模糊,有透明效果，需要逐帧处理，有一定的性能要求
    #video_blur()

    # 加速减速:性能很好，但应用场景有局限性，适合对某些视频题材下做特定的特效处理，或者放在结尾做特效处理
    #video_up_factor()
    #video_down_factor()

    # 页边距:性能较好，适合作为对视频的基础美化特效
    #video_margin()

    # 超级采样:不适合完整视频的处理，只适合对视频中有快速运动的部分做针对性处理。当视频中包含快速切换或快速运动的物体时，画面可能会显得生硬或不自然。超级采样通过计算多个帧的平均值，可以平滑这种快速变化，使运动看起来更加流畅。
    #video_supersample()

    # 音频可视化
    audio_visualization_effect("F:\download\compound_video.mp4", output_path)


    # 尺寸调整
    # 色彩反转
    # 滚动
    # 遮罩:与彩蛋动画结合
    # 均匀尺寸
    # 滚动文字
    
    pass

