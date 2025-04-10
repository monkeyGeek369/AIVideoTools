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
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

audio_vfx_task_queue = queue.Queue(maxsize=50)


def add_visualization(frame,fig, ax,num_bars,sub_grids_per_bar,sub_height,segment,bar_width,colors,vis_height):
    ax.clear()
    fig.patch.set_facecolor('none')
    fig.patch.set_edgecolor('none')
    ax.patch.set_facecolor('none')
    ax.axis('off')
    ax.axhline(y=0, color='white', linewidth=1, alpha=0.3)
    ax.set_xlim(-1, num_bars)
    ax.set_ylim(0, sub_grids_per_bar * sub_height + 0.3)
    fig.tight_layout(pad=0)


    chunk_size = len(segment) // num_bars
    for i in range(num_bars):
        main_chunk = segment[i * chunk_size : (i + 1) * chunk_size]
        amplitude = np.mean(np.abs(main_chunk))
        num_lit = min(int(amplitude * sub_grids_per_bar * 1.2), sub_grids_per_bar)
        
        for k in range(num_lit):
            ax.bar(
                x=i,
                height=sub_height,
                width=bar_width, 
                bottom=k * sub_height,
                color=colors[k],
                edgecolor='white',
                linewidth=0.5,
                alpha=0.9
            )

    canvas = fig.canvas
    canvas.draw()
    image_data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    image_data = image_data.reshape(canvas.get_width_height()[::-1] + (4,))

    vis_width = frame.shape[1]
    vis_image = np.zeros((vis_height, vis_width, 4), dtype=np.uint8)

    if image_data.shape[1] < vis_width:
        image_data = cv2.resize(image_data, (vis_width, image_data.shape[0]), interpolation=cv2.INTER_LINEAR)

    vis_image[:image_data.shape[0], :vis_width, :] = image_data[:vis_height, :vis_width, :]

    blended_frame = np.zeros_like(frame)
    blended_frame[:, :, :] = frame
    alpha_channel = vis_image[:, :, 3] / 255.0
    for c in range(3):
        blended_frame[-vis_height:, :, c] = (1 - alpha_channel) * blended_frame[-vis_height:, :, c] + alpha_channel * vis_image[:vis_height, :, c]

    return blended_frame.astype(np.uint8)

def audio_data_producer(audio_data,sample_rate,video):
    for index, frame in enumerate(video.iter_frames()):
        time = index/video.fps

        # get audio segment
        sample_index = int(time * sample_rate)
        start = max(0, sample_index - 100)
        end = min(len(audio_data), sample_index + 100)
        if end == len(audio_data) and start >= end:
            segment = audio_data[end - 200:end]
        else:
            segment = audio_data[start:end]
        if len(segment) < 200:
            end = len(audio_data)
            start = max(0, end - 200)
            segment = audio_data[start:end]

        audio_vfx_task_queue.put((index,frame,time,segment))
    audio_vfx_task_queue.put((None, None, None, None))

def audio_data_consumer(num_bars, sub_grids_per_bar, sub_height, bar_width, colors, vis_height,frame_file_path):
    fig, ax = plt.subplots(figsize=(8, 2), dpi=100, facecolor='none', edgecolor='none')

    while True:
        index,frame,time,segment = audio_vfx_task_queue.get()
        if index is None:
            audio_vfx_task_queue.put((None, None, None, None))
            break
        result = add_visualization(frame,fig,ax,num_bars,sub_grids_per_bar,sub_height,segment,bar_width,colors,vis_height)
        file_path = os.path.join(frame_file_path, f"{index}.npy")
        np.save(file_path, result[1])

def audio_visualization_effect(video_clip,task_path):
    start = time.time()
    global audio_vfx_task_queue
    audio_vfx_task_queue = queue.Queue(maxsize=50)
    video = video_clip
    fps = video.fps

    # get bg music
    edit_bg_musics_path = os.path.join(task_path, "edit_bg_musics")
    file_utils.ensure_directory(edit_bg_musics_path)
    edit_bg_musics_file_path = os.path.join(edit_bg_musics_path, "edit_bg_music.mp3")

    # get audio data
    audio = AudioSegment.from_file(edit_bg_musics_file_path)
    sample_rate = audio.frame_rate
    samples = audio.get_array_of_samples()
    audio_data = np.array(samples)
    audio_data = audio_data[:, 0] if audio_data.ndim == 2 else audio_data.flatten()
    audio_data = audio_data.astype(np.float32)

    # get base params
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data /= max_val
    num_bars = 40
    sub_grids_per_bar = 15
    bar_width = 1
    sub_height = 0.1
    colors = plt.cm.plasma(np.linspace(0, 1, sub_grids_per_bar))
    vis_height = min(200, video.size[1] // 4)
    frame_file_path=os.path.join(task_path, "frame")

    # productor
    producer_thread = threading.Thread(target=audio_data_producer, args=(audio_data,sample_rate, video))
    producer_thread.start()

    # 创建线程池
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [executor.submit(audio_data_consumer, num_bars, sub_grids_per_bar, sub_height, bar_width, colors, vis_height,frame_file_path) for _ in range(cpu_count())]
        for future in futures:
            future.result()
    
    # 等待生产者线程结束
    producer_thread.join()
    
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


