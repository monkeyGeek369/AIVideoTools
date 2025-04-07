import matplotlib
matplotlib.use('Agg')  # 强制使用无头渲染模式
from moviepy.editor import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from app.utils import file_utils
import gc
from multiprocessing import Pool,shared_memory
from pydub import AudioSegment
import cv2,time


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
        if end == len(self.audio_data) and start >= end:
            segment = self.audio_data[end - 200:end]
        else:
            segment = self.audio_data[start:end]
        if len(segment) < 200:
            end = len(self.audio_data)
            start = max(0, end - 200)
            segment = self.audio_data[start:end]

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

        vis_height = self.vis_height
        vis_width = frame.shape[1]
        if not hasattr(self, 'vis_image'):
            self.vis_image = np.zeros((vis_height, vis_width, 4), dtype=np.uint8)
        self.vis_image[:, :, :] = 0

        if image_data.shape[1] < vis_width:
            image_data = cv2.resize(image_data, (vis_width, image_data.shape[0]), interpolation=cv2.INTER_LINEAR)

        self.vis_image[:image_data.shape[0], :vis_width, :] = image_data[:vis_height, :vis_width, :]

        if not hasattr(self, 'blended_frame'):
            self.blended_frame = np.zeros_like(frame)
        self.blended_frame[:, :, :] = frame
        alpha_channel = self.vis_image[:, :, 3] / 255.0
        for c in range(3):
            self.blended_frame[-vis_height:, :, c] = (1 - alpha_channel) * self.blended_frame[-vis_height:, :, c] + alpha_channel * self.vis_image[:vis_height, :, c]

        return self.blended_frame.astype(np.uint8)

    def process_frame(self, t, frame):
        return t,self.add_visualization(frame, t)

def process_frame_wrapper(args):
    t, frame, video_processor,frame_file_path,index = args
    result = video_processor.process_frame(t, frame)
    file_path = os.path.join(frame_file_path, f"{index}.npy")
    np.save(file_path, result[1])

def audio_visualization_effect(video_clip,task_path):
    start = time.time()

    video = video_clip
    fps = video.fps

    # get bg music
    edit_bg_musics_path = os.path.join(task_path, "edit_bg_musics")
    file_utils.ensure_directory(edit_bg_musics_path)
    edit_bg_musics_file_path = os.path.join(edit_bg_musics_path, "edit_bg_music.mp3")

    audio = AudioSegment.from_file(edit_bg_musics_file_path)
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


    video_processor = VideoProcessor(sample_rate, num_bars, sub_grids_per_bar, sub_height, bar_width, colors, vis_height, fps,
                                    shm.name, audio_data.shape, audio_data.dtype)
    frame_file_path=os.path.join(task_path, "frame")
    def frame_iterator():
        for t, frame in enumerate(video.iter_frames()):
            yield (t / fps, frame, video_processor,frame_file_path,t)

    # muti process
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.imap_unordered(process_frame_wrapper, frame_iterator(), chunksize=1)
        pool.close()
        pool.join()

    # video process
    def get_frame(get_frame, t):
        frame_idx = int(t * fps)
        frame = np.load(os.path.join(frame_file_path, f"{frame_idx}.npy"))
        if frame is None:
            frame = get_frame(t).astype(np.uint8)
        return frame
    
    endtime = time.time()
    print("audio_visualization_effect time:", endtime - start)

    gc.collect()
    return video.fl(get_frame)


