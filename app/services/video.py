from loguru import logger
from moviepy.editor import *
from PIL import ImageFont
import numpy as np
import os,easyocr,shutil,random
from app.models.schema import VideoAspect, SubtitlePosition
from collections import Counter,defaultdict
from app.models.subtitle_position_coord import SubtitlePositionCoord
from app.services import mosaic,paddleocr,subtitle
from app.utils import str_util
import streamlit as st

def wrap_text(text, max_width, font, fontsize=60):
    """
    文本自动换行处理
    Args:
        text: 待处理的文本
        max_width: 最大宽度
        font: 字体文件路径
        fontsize: 字体大小

    Returns:
        tuple: (换行后的文本, 文本高度)
    """
    # 创建字体对象
    font = ImageFont.truetype(font, fontsize)

    def get_text_size(inner_text):
        inner_text = inner_text.strip()
        left, top, right, bottom = font.getbbox(inner_text)
        return right - left, bottom - top

    width, height = get_text_size(text)
    if width <= max_width:
        return text, height

    logger.debug(f"换行文本, 最大宽度: {max_width}, 文本宽度: {width}, 文本: {text}")

    processed = True

    _wrapped_lines_ = []
    words = text.split(" ")
    _txt_ = ""
    for word in words:
        _before = _txt_
        _txt_ += f"{word} "
        _width, _height = get_text_size(_txt_)
        if _width <= max_width:
            continue
        else:
            if _txt_.strip() == word.strip():
                processed = False
                break
            _wrapped_lines_.append(_before)
            _txt_ = f"{word} "
    _wrapped_lines_.append(_txt_)
    if processed:
        _wrapped_lines_ = [line.strip() for line in _wrapped_lines_]
        result = "\n".join(_wrapped_lines_).strip()
        height = len(_wrapped_lines_) * height
        # logger.warning(f"wrapped text: {result}")
        return result, height

    _wrapped_lines_ = []
    chars = list(text)
    _txt_ = ""
    for word in chars:
        _txt_ += word
        _width, _height = get_text_size(_txt_)
        if _width <= max_width:
            continue
        else:
            _wrapped_lines_.append(_txt_)
            _txt_ = ""
    _wrapped_lines_.append(_txt_)
    result = "\n".join(_wrapped_lines_).strip()
    height = len(_wrapped_lines_) * height
    logger.debug(f"换行文本: {result}")
    return result, height

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

def video_subtitle_overall_statistics(video_path:str,min_area:int,distance_threshold:int,sub_rec_area:str,subtitle_auto_mosaic_checkbox_value:bool) -> dict:
    '''
    video_path: video file path

    return: dict[str,int]
        left_top_x:int
        left_top_y:int
        right_bottom_x:int
        right_bottom_y:int
        count:int
        frame_subtitles_position:dict[float,list[tuple[tuple[int,int],tuple[int,int]]]]
    '''
    # frames coordinates
    task_path = st.session_state['task_path']
    frame_tmp_path = os.path.join(task_path, "frame_tmp")
    ignore_text = ["请勿模仿","国外合法饲养请勿","勿模仿"]
    frame_subtitles_position = paddleocr.get_video_frames_coordinates(video_path,frame_tmp_path,subtitle_auto_mosaic_checkbox_value)

    # filter frames coordinates by sub rec area
    frame_subtitles_position = subtitle.filter_frame_subtitles_position_by_area(sub_rec_area,frame_subtitles_position)

    # get frame_time_text_dict
    frame_time_text_dict = {t: [coord[2] for coord in result.get("coordinates") if (coord is not None and not str_util.is_str_contain_list_strs(coord[2],ignore_text))] for t, result in frame_subtitles_position.items()}

    # filter ignore text
    frame_subtitles_position = {result.get("index"): [(coord[0], coord[1]) for coord in result.get("coordinates") if (coord is not None and not str_util.is_str_contain_list_strs(coord[2],ignore_text))] for t, result in frame_subtitles_position.items()}

    # get all coordinates
    all_coords = [coord for result in frame_subtitles_position.values() for coord in result]

    # filter invalid coordinates
    all_coords = filter_coordinates(all_coords,min_area=min_area)

    # merge coordinates
    merged_counts = merge_coordinates_with_count(all_coords,threshold=distance_threshold)

    # get most common region
    if merged_counts:
        max_count = max(merged_counts.values())
        most_common_regions = [region for region, count in merged_counts.items() if count == max_count]
        most_common_region = most_common_regions[0]

        # 根据统计区域过滤每一帧的字幕区域
        for index, position in frame_subtitles_position.items():
            frame_coords = []
            for coord in position:
                frame_coords.append((coord[0], coord[1],(coord[1][0] - coord[0][0])*(coord[1][1] - coord[0][1])))
            if frame_coords and len(frame_coords) >= 0:
                frame_subtitles_position[index] = frame_coords

        return {
            "left_top_x": most_common_region[0][0],
            "left_top_y": most_common_region[0][1],
            "right_bottom_x": most_common_region[1][0],
            "right_bottom_y": most_common_region[1][1],
            "count": max_count,
            "frame_subtitles_position":frame_subtitles_position,
            "frame_time_text_dict":frame_time_text_dict
        }
    else:
        return None

def is_valid_coordinate(top_left, bottom_right, min_area=100):
    '''
    valid coordinate area >= min_area
    '''
    x1, y1 = top_left
    x2, y2 = bottom_right
    if x1 >= x2 or y1 >= y2:
        return False
    area = (x2 - x1) * (y2 - y1)
    return area >= min_area

def filter_coordinates(coords,min_area:int):
    '''
    filter invalid coordinates
    '''
    valid_coords = []
    for top_left, bottom_right in coords:
        if is_valid_coordinate(top_left, bottom_right,min_area=min_area):
            valid_coords.append((top_left, bottom_right))
    return valid_coords

def distance(coord1, coord2):
    '''
    Calculate the distance between the centers of two rectangles.
    Each rectangle is defined by its two diagonal points.
    '''
    # 解析矩形的对角点坐标
    (x1, y1), (x2, y2) = coord1
    (x3, y3), (x4, y4) = coord2

    # 计算两个矩形的中心点坐标
    center1_x = (x1 + x2) / 2
    center1_y = (y1 + y2) / 2
    center2_x = (x3 + x4) / 2
    center2_y = (y3 + y4) / 2

    # 计算两个中心点之间的欧几里得距离
    distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
    return distance
    
def merge_coordinates_with_count(coords, threshold=100):
    '''
    merge coordinates and get region counts
    '''
    # 使用字典记录每个合并后区域的出现次数
    region_counts = defaultdict(int)
    merged_coords = []

    for coord in coords:
        merged = False
        for i, merged_coord in enumerate(merged_coords):
            if distance(coord, merged_coord) < threshold:
                merged_coords[i] = (
                    (int(min(merged_coord[0][0],coord[0][0])), int(min(merged_coord[0][1],coord[0][1]))),
                    (int(max(merged_coord[1][0],coord[1][0])), int(max(merged_coord[1][1],coord[1][1])))
                )
                region_counts[i] += 1
                merged = True
                break
        if not merged:
            region_counts[len(merged_coords)] += 1
            merged_coords.append(coord)
            
    return {tuple(coord): count for coord, count in zip(merged_coords, region_counts.values())}

def is_overlap_over_half(base_rect, other_rect):
    '''
     judge if overlap over half of base_rect
    '''

    # 解析基础矩形和其他矩形的坐标
    (base_left, base_top), (base_right, base_bottom) = base_rect
    (other_left, other_top), (other_right, other_bottom) = other_rect

    # 计算重叠区域的坐标
    overlap_left = max(base_left, other_left)
    overlap_top = max(base_top, other_top)
    overlap_right = min(base_right, other_right)
    overlap_bottom = min(base_bottom, other_bottom)

    # 计算重叠区域的宽度和高度
    overlap_width = overlap_right - overlap_left
    overlap_height = overlap_bottom - overlap_top

    # 判断是否有重叠
    if overlap_width <= 0 or overlap_height <= 0:
        return False

    # 计算重叠面积和其他矩形的面积
    overlap_area = overlap_width * overlap_height
    other_area = (other_right - other_left) * (other_bottom - other_top)

    # 判断重叠面积是否超过其他矩形面积的50%
    return overlap_area > 0.5 * other_area

def video_subtitle_mosaic_auto(video_clip,subtitle_position_coord:SubtitlePositionCoord|None):
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
    left_top = (subtitle_position_coord.left_top_x,subtitle_position_coord.left_top_y)
    right_bottom = (subtitle_position_coord.right_bottom_x,subtitle_position_coord.right_bottom_y)

    # load video
    return video_clip.fl(lambda gf, t: make_frame_processor(gf(t), t, frame_subtitles_position,fps,left_top,right_bottom))

def recognize_subtitle_and_mosaic(frame,base_rect,reader):
    '''
    recognize subtitle and mosaic
    '''

    frame_copy = frame.copy()
    # recognize subtitle
    result = reader.readtext(frame,
                            detail=1,
                            batch_size=10, # 批处理大小
                            )

    # mosaic subtitle
    for item in result:
        text = item[1]
        if text.strip() and item[0] is not None and len(item[0]) == 4:
            top_left = tuple(map(int, item[0][0]))
            bottom_right = tuple(map(int, item[0][2]))
            if is_overlap_over_half(base_rect, (top_left, bottom_right)):
               frame_copy = mosaic.apply_perspective_background_color(frame=frame_copy,
                                                                      x1=top_left[0],
                                                                      y1=top_left[1],
                                                                      x2=bottom_right[0],
                                                                      y2=bottom_right[1],
                                                                      extend_factor = 2)
    
    return frame_copy

def make_frame_processor(frame,t:float,frame_subtitles_position:dict[int,list[tuple[tuple[int,int],tuple[int,int],float]]],fps:int,left_top:tuple[int,int],right_bottom:tuple[int,int]):
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

def video_clip_to_video(video_clip,video_path):
    task_path = st.session_state['task_path']
    temp_audio_path = os.path.join(task_path, "temp", "material-audio.aac")
    video_clip.write_videofile(
        video_path,
        codec='libx264',
        audio_codec='aac',
        fps=video_clip.fps,
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
