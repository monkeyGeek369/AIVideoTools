import matplotlib
matplotlib.use('Agg')  # 强制使用无头渲染模式
from moviepy.editor import *
import numpy as np
import matplotlib.pyplot as plt
from app.utils import file_utils,ffmpeg_util,utils
import gc
from pydub import AudioSegment
import cv2,time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import math
from loguru import logger
import streamlit as st

audio_vfx_task_queue = queue.Queue(maxsize=50)


def add_visualization(frame,num_bars,sub_grids_per_bar,sub_height,segment,bar_width,bar_interval,colors,vis_height):
    # 初始化可视化音频图像
    vis_width = frame.shape[1]
    # 四通道的图像，rgba，默认数值都是0
    vis_image = np.zeros((vis_height, vis_width, 4), dtype=np.uint8)
    
    # 计算每个条形的宽度和间距
    bar_total_width = bar_width + bar_interval
    # 计算x轴的起始位置
    start_x = int((vis_width - num_bars * bar_total_width) // 2)

    # 计算每个条形占用的音乐数据尺寸
    chunk_size = len(segment) // num_bars
    # 绘制每个条形
    for i in range(num_bars):
        # 获取当前条形的音乐数据
        main_chunk = segment[i * chunk_size : (i + 1) * chunk_size]
        # 计算音乐数据的平均振幅
        amplitude = np.mean(np.abs(main_chunk))
        # 计算当前条形需要显示的小段数
        num_lit = min(int(amplitude * sub_grids_per_bar * 1.8), sub_grids_per_bar)
        
        # 计算条形的位置
        bar_x = start_x + i * bar_total_width
        
        # 绘制条形图的每个小段
        for k in range(num_lit):
            # 获取当前段的颜色
            color = colors[k]
            
            # 计算当前段的高度和位置（兼容间隔）
            rect_height = sub_height - bar_interval
            rect_y = vis_height - (k + 1) * sub_height + bar_interval
            
            # 绘制矩形
            cv2.rectangle(
                vis_image,
                (int(bar_x), int(rect_y)),  # 矩形的左上角坐标
                (int(bar_x + bar_width), int(rect_y + rect_height)),  # 矩形的右下角坐标
                (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), 255),  # 矩形的颜色和透明度(不透明)
                cv2.FILLED  # 填充方式
            )

    # 从可视化音频特效图像中提取Alpha通道，即透明度通道，并将其转换为浮点数，归一化
    alpha = vis_image[:, :, 3].astype(np.float32) / 255.0
    # 将Alpha通道扩展为与视频帧大小相同的形状，通道数为1，即只有透明度
    alpha = alpha.reshape((alpha.shape[0], alpha.shape[1], 1))

    # 创建一个与视频帧大小相同的透明区域，所有通道值默认填充0
    alpha_full = np.zeros_like(frame, dtype=np.float32)
    # 将Alpha通道填充到透明区域（alpha.shape[0]为高度），即从倒数alpha.shape[0]行开始填充
    alpha_full[-alpha.shape[0]:, :, :] = alpha

    # 创建一个与视频帧大小相同的透明区域
    overlay = np.zeros_like(frame)
    overlay[-vis_height:, :, :] = vis_image[:, :, :3]
    
    # 将视频帧与透明区域进行融合
    # (1 - alpha_full) * frame.astype(np.float32) 表示将视频帧与透明区域进行透明度融合
    # + alpha_full * overlay.astype(np.float32) 表示将透明区域与融合后的视频帧进行融合
    blended_frame = (1 - alpha_full) * frame.astype(np.float32) + alpha_full * overlay.astype(np.float32)
    blended_frame = blended_frame.clip(0, 255).astype(np.uint8)

    return blended_frame

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

def audio_data_consumer(num_bars, sub_grids_per_bar, sub_height, bar_width,bar_interval, colors, vis_height,frame_file_path):
    while True:
        index,frame,time,segment = audio_vfx_task_queue.get()
        if index is None:
            audio_vfx_task_queue.put((None, None, None, None))
            break
        result = add_visualization(frame,num_bars,sub_grids_per_bar,sub_height,segment,bar_width,bar_interval,colors,vis_height)

        # test np to image
        # file_path = os.path.join(frame_file_path, f"{index}.png")
        # rgb_frame = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(file_path, rgb_frame)

        file_path = os.path.join(frame_file_path, f"{index}.npy")
        np.save(file_path, result)

def audio_visualization_effect(video_clip,task_path):
    # init params
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

    # normalization
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data /= max_val
    
    # base params
    num_bars = 40
    sub_grids_per_bar = 30

    # sub height
    vis_height = min(200, int(video.size[1] // 7))
    sub_height = vis_height//sub_grids_per_bar

    # bar width
    bar_origin_width = video.size[0] // num_bars
    bar_interval = math.ceil(bar_origin_width * 0.06)
    bar_width = bar_origin_width - bar_interval

    # bar colors
    colors = plt.cm.coolwarm(np.linspace(0, 1, sub_grids_per_bar))

    # productor
    producer_thread = threading.Thread(target=audio_data_producer, args=(audio_data,sample_rate, video))
    producer_thread.start()

    # consumer
    frame_file_path=os.path.join(task_path, "frame")
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [executor.submit(audio_data_consumer, num_bars, sub_grids_per_bar, sub_height, bar_width,bar_interval, colors, vis_height,frame_file_path) for _ in range(cpu_count())]
        for future in futures:
            future.result()
    
    # producer join
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

def merge_audio_files(out_path: str, audio_files: list, subtitle_list: list,merged_subtitle_path: str):
    """
    merge audio files into a single audio file with subtitles.
    """
    if not ffmpeg_util.check_ffmpeg():
        logger.error("ffmpeg not found, please install ffmpeg.")
        return None
    video_duration = st.session_state.get("video_duration")
    if video_duration is None:
        raise ValueError("video_duration is None, please check the video duration.")
    if audio_files is None or len(audio_files) == 0:
        raise ValueError("audio_files is empty")
    if subtitle_list is None or len(subtitle_list) == 0:
        raise ValueError("subtitle_list is empty")
    if len(audio_files) != len(subtitle_list):
        raise ValueError("audio_files and subtitle_list must have the same length")

    # handle audio files
    last_audio_duration = 0
    audio_result_list = []
    subtitle_update_list = []
    for subtitle_item, audio_file in zip(subtitle_list, audio_files):
        try:
            # get subtitle item
            index, timestamp_str, text = subtitle_item
            start_time, end_time = timestamp_str.split(' --> ')
            start_ms = utils.time_to_ms(start_time)
            end_ms = utils.time_to_ms(end_time)
            sub_duration = end_ms - start_ms
            if sub_duration <= 0:
                logger.warning(f"invalid timestamp: {timestamp_str}")
                continue

            # load audio file
            tts_audio = AudioSegment.from_file(audio_file)
            audio_duration = len(tts_audio)

            # get real audio duration
            real_audio_duration = sub_duration if sub_duration > audio_duration else audio_duration

            # get data
            real_start_ms = max(start_ms, last_audio_duration)
            data_item = (real_start_ms,audio_file)
            
            # logic
            if start_ms > last_audio_duration:
                last_audio_duration += (start_ms - last_audio_duration + real_audio_duration)
            else:
                last_audio_duration += real_audio_duration
            
            # set data
            audio_result_list.append(data_item)
            subtitle_update_list.append(
                utils.text_to_srt(
                    index,
                    text,
                    float(real_start_ms/1000),
                    float(last_audio_duration/1000)
                )
                )
            if (video_duration * 1000) <= last_audio_duration:
                break

        except Exception as e:
            logger.error(f"merge audio files error: {audio_file} error info: {str(e)}")
            continue
        finally:
            del tts_audio
    
    # update subtitle file
    if len(subtitle_update_list) > 0:
        sub = "\n".join(subtitle_update_list) + "\n"
        with open(merged_subtitle_path, "w", encoding="utf-8") as f:
            f.write(sub)
    
    # create blank audio segment
    final_audio = AudioSegment.silent(duration=last_audio_duration)
    for start_time_ms,audio_file in audio_result_list:
        audio_tts = AudioSegment.from_file(audio_file)
        final_audio = final_audio.overlay(audio_tts, position=start_time_ms)
        del audio_tts

    # save merged audio file
    output_audio_path = os.path.join(out_path, "edit_audio.mp3")
    final_audio.export(output_audio_path, format="mp3")
    logger.info(f"merged audio file saved: {output_audio_path}")

    return output_audio_path

def get_audio_from_video(video_path: str, out_path: str):
    """
    video_path: for example: /path/to/video.mp4
    out_path: for example: /path/to/output/audio.wav
    """
    audio_clip = AudioFileClip(video_path)
    fps = audio_clip.fps
    audio_clip.write_audiofile(
        filename=out_path,
        fps=fps
        )
    audio_clip.close()
