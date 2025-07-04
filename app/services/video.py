from loguru import logger
from moviepy.editor import *
from PIL import ImageFont
import numpy as np
import os,shutil,random,re
from app.models.schema import VideoAspect, SubtitlePosition
from collections import Counter,defaultdict
from app.models.subtitle_position_coord import SubtitlePositionCoord
from app.services import mosaic,paddleocr,subtitle,text_coordinate
from app.utils import str_util
import streamlit as st
import json

def calculate_subtitle_position(position, video_height: int, custom_position: float = 0) -> tuple:
    """
    计算字幕在视频中的具体位置
    
    Args:
        position: 位置配置，top/center/bottom/custom
        video_height: 视频高度
        custom_position: 自定义位置，百分比
    
    Returns:
        tuple: (x, y) 坐标
    """
    # 预设位置
    if position == SubtitlePosition.TOP:
        return ('center', SubtitlePosition.TOP)
    elif position == SubtitlePosition.CENTER:
        return ('center', video_height // 2)
    elif position == SubtitlePosition.BOTTOM:
        return ('center', SubtitlePosition.BOTTOM)
    elif position == SubtitlePosition.CUSTOM:
        return ('center', int(video_height * (custom_position / 100.0)))
    
    # 默认底部
    return ('center', SubtitlePosition.BOTTOM)

def video_subtitle_overall_statistics(video_path:str,
                                    subtitle_merge_distance:int,
                                    sub_rec_area:str,
                                    ignore_min_width:int,
                                    ignore_min_height:int,
                                    ignore_min_word_count:int,
                                    warning_text:str,
                                    title_merge_distance:int,
                                    warning_merge_distance:int) -> dict:
    # get base info
    video_width = st.session_state['video_width']
    video_height = st.session_state['video_height']
    video_fps = st.session_state['video_fps']
    task_path = st.session_state['task_path']
    video_duration = st.session_state['video_duration']

    # frames coordinates
    frame_tmp_path = os.path.join(task_path, "frame_tmp")
    frame_subtitles_position = paddleocr.get_video_frames_coordinates(video_path,frame_tmp_path)

    # filter: sub_rec_area
    frame_subtitles_position = subtitle.filter_frame_subtitles_position_by_area(sub_rec_area,frame_subtitles_position)

    # filter: ignore min width, height, word count,ignore text
    frame_subtitles_position = subtitle.filter_frame_subtitles_position(ignore_min_width, ignore_min_height, ignore_min_word_count, None, frame_subtitles_position)

    # dectect subtitle position
    fixed_regions = text_coordinate.text_coordinate_recognize_video(frame_subtitles_position, video_width, video_height, video_fps,video_duration,warning_text,title_merge_distance,warning_merge_distance,subtitle_merge_distance)

    # recognize coordinates type
    title_bbox = fixed_regions["title"]["bbox"] if fixed_regions["title"] else None
    warning_bbox = fixed_regions["warning"]["bbox"] if fixed_regions["warning"] else None
    subtitle_bbox = fixed_regions["subtitle"]["bbox"] if fixed_regions["subtitle"] else None
    frams = text_coordinate.frames_coordinate_type_recognize(frame_subtitles_position,title_bbox,warning_bbox,subtitle_bbox)

    if not fixed_regions:
        return SubtitlePositionCoord(is_exist=False)
    else:
        return SubtitlePositionCoord(is_exist=True,
                                     fixed_regions=fixed_regions,
                                     time_index={result["t"]:index for index,result in frams.items()},
                                     frames=frams
                                     )

def video_subtitle_mosaic_auto(video_clip,subtitle_position_coord:SubtitlePositionCoord|None,
                                                    all_mosaic:bool,
                                                    title_mosaic:bool,
                                                    warning_mosaic:bool,
                                                    subtitle_mosaic:bool,
                                                    other_mosaic:bool):
    '''
    auto recognize subtitle and mosaic
    subtitle position within the range subtitle_position_coord
    '''
    
    # base check
    if video_clip is None:
        raise Exception("video clip not found")
    if subtitle_position_coord is None:
        logger.info("video subtitle position not recognized,please recognize it first")
        return video_clip
    if not subtitle_position_coord.is_exist:
        logger.info("video subtitle position recognized is empty, no need to mosaic")
        return video_clip

    # get subtitle position
    frame_subtitles_position = subtitle_position_coord.frame_subtitles_position
    fps = video_clip.fps
    # left_top = (subtitle_position_coord.left_top_x,subtitle_position_coord.left_top_y)
    # right_bottom = (subtitle_position_coord.right_bottom_x,subtitle_position_coord.right_bottom_y)

    # load video
    return video_clip.fl(lambda gf, t: make_frame_processor(gf(t), t, frame_subtitles_position,fps))

def make_frame_processor(frame,t:float,frame_subtitles_position:dict[int,list[tuple[tuple[int,int],tuple[int,int],float]]],fps:int):
    frame_copy = frame.copy()
    index = int(np.round(t * fps))
    
    # image_frame_path = "F:\download\\tmp\\frame"
    # cv2.imwrite(image_frame_path + f"/{index}.png", frame_copy)
    # if index == 170:
    #     print(index)

    positions = get_real_subtitle_position(index,3,frame_subtitles_position)

    for top_left, bottom_right,area in positions:
        frame_copy = mosaic.telea_mosaic(frame=frame_copy,
                                            x1=top_left[0],
                                            y1=top_left[1],
                                            x2=bottom_right[0],
                                            y2=bottom_right[1])
    
    #cv2.imwrite(image_frame_path + f"/{index}-new.png", frame_copy)

    return frame_copy

def get_real_subtitle_position(start_index:int,float_num:int,frame_subtitles_position:dict[int,list[tuple[tuple[int,int],tuple[int,int],float]]]):
    positions = frame_subtitles_position.get(start_index, [])
    area = sum(item[2] for item in positions)
    if area > 0:
        return positions

    for num in range(float_num):
        current_index = num + 1

        # up
        up_positions = frame_subtitles_position.get(start_index + current_index,[])
        up_area = sum(item[2] for item in up_positions)
        if up_area > area:
            positions = up_positions
            area = up_area
        # down
        down_positions = frame_subtitles_position.get(start_index - current_index,[])
        down_area = sum(item[2] for item in down_positions)
        if down_area > area:
            positions = down_positions
            area = down_area
    
    return positions

def generate_video_by_images(video_size,image_folder, fps=24, duration=10,last_frame_duration=3):
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    if not image_files:
        raise ValueError("No image files found in the folder.")
    
    base_frames = int(fps * duration)
    selected_images = []
    while len(selected_images) < base_frames:
        random.shuffle(image_files)
        selected_images.extend(image_files)
    selected_images = selected_images[:base_frames]
    
    clips = []
    for i, img_path in enumerate(selected_images):
        img_clip = ImageClip(img_path).set_duration(1/fps)
        clips.append(img_clip)
    
    if last_frame_duration > 0 and selected_images:
        last_image = selected_images[-1]
        last_clip = ImageClip(last_image).set_duration(last_frame_duration)
        clips.append(last_clip)
    
    image_clip = concatenate_videoclips(clips, method="compose")
    image_clip = image_clip.resize(video_size)
    return image_clip

def images_to_video_clip(image_files:list,fps:int):
    return ImageSequenceClip(image_files, fps=fps)

def video_clip_to_video(video_clip,video_path,fps):
    task_path = st.session_state['task_path']
    temp_audio_path = os.path.join(task_path, "temp", "material-audio.aac")
    video_clip.write_videofile(
        video_path,
        codec='libx264',
        audio_codec='aac',
        fps=fps,
        preset='medium',
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

def remove_video_titile_spe_chars(video_name:str):
    return re.sub(r'[\\/*?:"<>“”|]', "", video_name)
