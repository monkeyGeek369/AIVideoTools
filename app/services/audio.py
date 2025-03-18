import matplotlib
matplotlib.use('Agg')  # 强制使用无头渲染模式
from moviepy.editor import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from app.utils import file_utils

def audio_visualization_effect(video_clip):
    video = video_clip

    # get bg music
    task_path = st.session_state['task_path']
    edit_bg_musics_path = os.path.join(task_path, "edit_bg_musics")
    file_utils.ensure_directory(edit_bg_musics_path)
    edit_bg_musics_file_path = os.path.join(edit_bg_musics_path, "edit_bg_music.mp3")

    with AudioFileClip(edit_bg_musics_file_path) as audio_clip:
        sample_rate = audio_clip.fps
        raw_audio = audio_clip.coreader().iter_frames()
        audio_data = np.vstack([frame for frame in raw_audio])
        #audio_data = audio_clip.to_soundarray()
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

    plt.figure(figsize=(8, 2), dpi=100, facecolor='none', edgecolor='none')    
    def add_visualization(get_frame, t):
        frame = get_frame(t).astype(np.uint8)
        sample_index = int(t * sample_rate)
        start = max(0, sample_index - 100)
        end = min(len(audio_data), sample_index + 100)
        segment = audio_data[start:end]
        if len(segment) < 200:
            segment = np.pad(segment, (0, 200 - len(segment)), 'constant')

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
        plt.axis('off')
        plt.axhline(y=0, color='white', linewidth=1, alpha=0.3)
        plt.xlim(-1, num_bars)
        plt.ylim(0, sub_grids_per_bar * sub_height + 0.3)
        plt.tight_layout(pad=0)
        canvas = plt.gcf().canvas
        canvas.draw()
        image_data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        image_data = image_data.reshape(canvas.get_width_height()[::-1] + (4,))
        plt.gca().clear()

        img_pil = Image.fromarray(image_data, 'RGBA').resize((frame.shape[1], vis_height))
        frame_pil = Image.fromarray(frame).convert('RGBA')
        frame_pil.paste(img_pil, (0, frame_pil.height - vis_height), img_pil)
        blended = np.array(frame_pil.convert('RGB'))
        return blended.astype(np.uint8)

    return video.fl(add_visualization)


