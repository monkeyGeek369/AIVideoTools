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
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor,as_completed
from multiprocessing import Pool,shared_memory


# mac
# video1_path = "/Users/monkeygeek/Downloads/baobei-0-6.mp4"
# video2_path = "/Users/monkeygeek/Downloads/baobei-6-12.mp4"
# output_path = "/Users/monkeygeek/Downloads/output.mp4"

# windows
video0_path = "F:\download\\小姐姐惊讶发现生命奇迹，一只小鸡从蛋壳中顽强求生.webm"
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
        # get audio data from frames
        # raw_audio可以理解为一个表格,每一行代表一个采样点，每一列代表一个通道,它只是一个平面数据
        raw_audio = audio_clip.coreader().iter_frames()
        # audio_data表示将每一帧的数据堆叠到一起,组成一个二维数组
        # 它是一个多维数组,表示为(n_samples, n).其中n_samples表示这一帧中所具备的采样点个数,n表示每个采样点的通道数
        # 例如如果是双通道的,那么n=2,每一个采样点就是两个通道的数据,可以通过audio_data[:, 0]或audio_data[:, 1]获得获取单通道的数据
        # 例如audio_data[:, 0]表示提取二维数组所有行:的第一列0,返回一个新的一维数组,这就是单通道
        audio_data = np.vstack([frame for frame in raw_audio])

    # audio_data.ndim == 2表示为多维数组(不一定只是2通道音频)
    # audio_data.flatten()将任何维度的数组转为一维数组
    audio_data = audio_data[:, 0] if audio_data.ndim == 2 else audio_data.flatten()
    audio_data = audio_data.astype(np.float32)
    
    # 归一化
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        # 保证通道中的每个采样数据都在[-1, 1]范围内
        audio_data /= max_val

    def add_visualization(get_frame, t):
        # 帧frame为某一时刻的图像数据,可以表示为一个多维素组(height, width, n),其中height表示图像的高,width表示图像的宽,n表示图像的通道数
        # RGB通道的每个像素点是一个三元组（R, G, B）此时n=3,可以通过frame[:, :, 0]获取红色通道的数据
        # RGBA通道的每个像素点是一个四元组（R, G, B, A）此时n=4,可以通过frame[:, :, 0]获取红色通道的数据
        # 高度和宽度指的是图像在y轴和x轴上的像素个数
        frame = get_frame(t)
        # 将numpy数组转换为uint8(范围0-255)更方便表示像素颜色取值
        frame = frame.astype(np.uint8)

        # 采样率为每秒采样次数,类似于帧率
        # 获取当前时刻的采样点
        sample_index = int(t * sample_rate)
        start = max(0, sample_index - 100)
        end = min(len(audio_data), sample_index + 100)
        
        # 获取当前时刻的音频采样数据
        segment = audio_data[start:end]
        if len(segment) < 200:
            # 如果音频采样数据不足200个采样点，就填充0补充
            # (0, 200 - len(segment))表示前面填充0个采样点，后面填充200 - len(segment)个采样点
            # constant表示使用常数填充
            segment = np.pad(segment, (0, 200 - len(segment)), 'constant')

        # 创建绘图大小:宽8英寸,高2英寸,像素密度100的图像.整体的像素大小是宽800像素,高200像素
        # 背景色为none,边框颜色为none
        plt.figure(figsize=(8, 2), dpi=100, facecolor='none', edgecolor='none')
        # 不显示坐标轴
        plt.axis('off')
        # 水平参考线在y=0处,颜色为白色,宽度为1,透明度为0.3
        plt.axhline(y=0, color='white', linewidth=1, alpha=0.3)
        
        # 条块数量
        num_bars = 40
        # 每一个条块的格子数量
        sub_grids_per_bar = 15
        # 条块宽度:数据单位,1表示一个单位
        bar_width = 1
        # 子格子的高度:数据单位.0.1表示一个单位的10%
        sub_height = 0.1
        # plt.cm.plasma这是渐变颜色,表示从橙色到红色
        # np.linspace(0, 1, sub_grids_per_bar)表示从0到1之间生成sub_grids_per_bar个数
        # 作用是做颜色映射
        colors = plt.cm.plasma(np.linspace(0, 1, sub_grids_per_bar))
        
        # 取整:表示每一个条块占用多少采样
        chunk_size = len(segment) // num_bars
        for i in range(num_bars):
            # 获取当前条块的采样数据
            main_chunk = segment[i*chunk_size : (i+1)*chunk_size]
            # 取绝对值的平均值范围[0-1],因为数据在前面已经归一化
            amplitude = np.mean(np.abs(main_chunk))
            # 计算振幅格子数,放大0.2倍,增加视觉效果
            num_lit = min(int(amplitude * sub_grids_per_bar * 1.2), sub_grids_per_bar)
            
            for k in range(num_lit):
                plt.bar(
                    x=i,# x轴坐标
                    height=sub_height,# 每个格子的高度
                    width=bar_width, # 每个格子的宽度
                    bottom=k * sub_height,# 每个格子的底部位置
                    color=colors[k],# 每个格子的颜色
                    edgecolor='white',# 每个格子的边框颜色
                    linewidth=0.5,# 每个格子的边框宽度
                    alpha=0.9 # 每个格子的透明度
                )

        # 表示x轴的范围,与上文的单位相关
        plt.xlim(-1, num_bars)
        # 表示y轴的范围
        plt.ylim(0, sub_grids_per_bar * sub_height + 0.3)
        # 调整布局，子图与边界之间没有间距
        plt.tight_layout(pad=0)

        # 获取画布
        canvas = plt.gcf().canvas
        # 将图形绘制到画布上
        canvas.draw()
        # 获取画布数据,格式为RGBA,数据类型为uint8
        image_data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        # canvas.get_width_height()：获取画布的宽度和高度（以像素为单位）。返回值是一个元组 (width, height)。
        # [::-1]：将宽度和高度的顺序反转，因为 NumPy 数组的形状是以 (height, width) 的形式表示的。
        # + (4,)：表示每个像素有 4 个通道（RGBA），因此数组的形状为 (height, width, 4)。
        # reshape(...)：将一维的像素数据重新排列为三维数组，形状为 (height, width, 4)，表示 RGBA 格式的图像数据。
        image_data = image_data.reshape(canvas.get_width_height()[::-1] + (4,))
        plt.close()

        # frame.shape表示帧的格式,例如(height, width, channels),因此frame.shape[0]表示帧的高度
        vis_height = min(200, frame.shape[0] // 4)
        # 按照指定宽、高生成图像
        img_pil = Image.fromarray(image_data, 'RGBA').resize((frame.shape[1], vis_height))

        # 将帧转换为RGBA格式
        frame_pil = Image.fromarray(frame).convert('RGBA')

        # im：要粘贴的图像。
        # box：粘贴的位置和大小，是一个元组 (x, y) 或 (x, y, w, h)。
        # mask：可选参数，用于指定蒙版。如果提供，蒙版图像的 Alpha 通道将用于控制粘贴的透明度。
        frame_pil.paste(img_pil, (0, frame_pil.height - vis_height), img_pil)

        # 将帧转换为RGB格式
        blended = np.array(frame_pil.convert('RGB'))
        # 返回混合后的帧
        return blended.astype(np.uint8)

    processed_video = video.fl(add_visualization)
    processed_video.write_videofile(output_path,
                                  codec='libx264',
                                  audio_codec='aac',
                                  threads=4,
                                  logger='bar')

class VideoProcessor:
    def __init__(self, sample_rate,num_bars,sub_grids_per_bar,sub_height,bar_width,colors,vis_height,fps,shm_name, shape, dtype):
        self.sample_rate = sample_rate
        self.num_bars = num_bars
        self.sub_grids_per_bar = sub_grids_per_bar
        self.sub_height=sub_height
        self.bar_width = bar_width
        self.colors = colors
        self.vis_height= vis_height
        self.fps = fps
        self.fig, self.ax = plt.subplots(figsize=(8, 2), dpi=100, facecolor='none', edgecolor='none')

        # 连接共享内存
        self.shm = shared_memory.SharedMemory(name=shm_name)
        self.audio_data = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)

    def __del__(self):
        self.fig.clf()
        plt.close(self.fig)
        self.shm.close()
        
    def add_visualization(self, frame, t):
        self.ax.clear()
    
        # 设置透明背景
        self.fig.patch.set_facecolor('none')
        self.fig.patch.set_edgecolor('none')
        self.ax.patch.set_facecolor('none')
        self.ax.axis('off')
        self.ax.axhline(y=0, color='white', linewidth=1, alpha=0.3)
        self.ax.set_xlim(-1, self.num_bars)
        self.ax.set_ylim(0, self.sub_grids_per_bar * self.sub_height + 0.3)
        self.fig.tight_layout(pad=0)

        sample_index = int(t * self.sample_rate)
        start = max(0, sample_index - 100)
        end = min(len(self.audio_data), sample_index + 100)
        segment = self.audio_data[start:end]
        if len(segment) < 200:
            segment = np.pad(segment, (0, 200 - len(segment)), 'constant')

        chunk_size = len(segment) // self.num_bars
        for i in range(self.num_bars):
            main_chunk = segment[i * chunk_size : (i + 1) * chunk_size]
            amplitude = np.mean(np.abs(main_chunk))
            num_lit = min(int(amplitude * self.sub_grids_per_bar * 1.2), self.sub_grids_per_bar)
            
            for k in range(num_lit):
                self.ax.bar(
                    x=i,
                    height=self.sub_height,
                    width=self.bar_width, 
                    bottom=k * self.sub_height,
                    color=self.colors[k],
                    edgecolor='white',
                    linewidth=0.5,
                    alpha=0.9
                )

        canvas = self.fig.canvas
        canvas.draw()
        image_data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        image_data = image_data.reshape(canvas.get_width_height()[::-1] + (4,))

        # 确保图像宽度与原视频一致
        vis_height = self.vis_height
        vis_width = frame.shape[1]  # 原视频的宽度
        if not hasattr(self, 'vis_image'):
            self.vis_image = np.zeros((vis_height, vis_width, 4), dtype=np.uint8)
        self.vis_image[:, :, :] = 0  # 清空之前的可视化图像

        # 如果生成的图像宽度小于原视频宽度，进行缩放
        if image_data.shape[1] < vis_width:
            image_data = cv2.resize(image_data, (vis_width, image_data.shape[0]), interpolation=cv2.INTER_LINEAR)

        self.vis_image[:image_data.shape[0], :vis_width, :] = image_data[:vis_height, :vis_width, :]

        # 将可视化图像合成到原始帧上，确保透明背景
        if not hasattr(self, 'blended_frame'):
            self.blended_frame = np.zeros_like(frame)
        self.blended_frame[:, :, :] = frame
        alpha_channel = self.vis_image[:, :, 3] / 255.0  # 获取透明度通道
        for c in range(3):  # 对 RGB 通道进行合成
            self.blended_frame[-vis_height:, :, c] = (1 - alpha_channel) * self.blended_frame[-vis_height:, :, c] + alpha_channel * self.vis_image[:vis_height, :, c]

        return self.blended_frame.astype(np.uint8)

    def process_frame(self, t, frame):
        return t,self.add_visualization(frame, t)

def process_frame_wrapper(args):
    t, frame, video_processor = args
    return video_processor.process_frame(t, frame)

def frame_generator(video, fps, video_processor):
    for t, frame in enumerate(video.iter_frames()):
        yield (t / fps, frame, video_processor)

def audio_visualization_effect_v2(video_path, output_path):
    video = VideoFileClip(video_path)
    fps = video.fps

    audio = AudioSegment.from_file(video_path)
    sample_rate = audio.frame_rate
    samples = audio.get_array_of_samples()
    audio_data = np.array(samples)
    audio_data = audio_data[:, 0] if audio_data.ndim == 2 else audio_data.flatten()
    audio_data = audio_data.astype(np.float32)

    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data /= max_val

    num_bars = 40
    sub_grids_per_bar = 15
    bar_width = 1
    sub_height = 0.1
    colors = plt.cm.plasma(np.linspace(0, 1, sub_grids_per_bar))
    vis_height = min(200, video.size[1] // 4)

    # 创建共享内存
    shm = shared_memory.SharedMemory(create=True, size=audio_data.nbytes)
    shared_audio_data = np.ndarray(audio_data.shape, dtype=audio_data.dtype, buffer=shm.buf)
    shared_audio_data[:] = audio_data[:]  # 将数据复制到共享内存

    # muti process
    first_frame = next(video.iter_frames())
    height, width, channels = first_frame.shape
    frame_shape = (height, width, channels)
    num_frames = int(video.duration * fps)
    processed_frames = np.zeros((num_frames, *frame_shape), dtype=np.uint8)
    with Pool(processes=os.cpu_count()) as pool:
        video_processor = VideoProcessor(sample_rate, num_bars, sub_grids_per_bar, sub_height, bar_width, colors, vis_height, fps,
                                     shm.name, audio_data.shape, audio_data.dtype)
        results = pool.imap(process_frame_wrapper, frame_generator(video, fps, video_processor))

        # 将结果存储到 processed_frames 中
        for num, result in enumerate(results):
            processed_frames[int(result[0] * fps)] = result[1]
    
    # video process
    def get_frame(get_frame, t):
        frame_idx = int(t * fps)
        frame = processed_frames[frame_idx]
        if frame is None:
            frame = get_frame(t).astype(np.uint8)
        return frame
    processed_video = video.fl(get_frame)
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
    #audio_visualization_effect(video1_path, output_path)
    audio_visualization_effect_v2(video0_path, output_path)

    # 尺寸调整
    # 色彩反转
    # 滚动
    # 遮罩:与彩蛋动画结合
    # 均匀尺寸
    # 滚动文字
    
    pass

