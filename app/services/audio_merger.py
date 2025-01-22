import os
import json
import subprocess
import edge_tts
from edge_tts import submaker
from pydub import AudioSegment
from typing import List, Dict
from loguru import logger
from app.utils import utils


def check_ffmpeg():
    """检查FFmpeg是否已安装"""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False


def merge_audio_files(out_path: str, audio_files: list, total_duration: float, subtitle_list: list):
    """
    合并音频文件
    
    Args:
        out_path: 输出目录
        audio_files: TTS生成的音频文件列表
        total_duration: 总时长
        subtitle_list: 脚本列表 ->包含字幕序号、出现时间、和字幕文本的元组列表。
    
    Returns:
        str: 合并后的音频文件路径
    """
    # 检查FFmpeg是否安装
    if not check_ffmpeg():
        logger.error("FFmpeg未安装，无法合并音频文件")
        return None

    # 创建一个空的音频片段
    final_audio = AudioSegment.silent(duration=total_duration * 1000)  # 总时长以毫秒为单位

    # 遍历脚本中的每个片段
    for subtitle_item, audio_file in zip(subtitle_list, audio_files):
        try:
            # 加载TTS音频文件
            tts_audio = AudioSegment.from_file(audio_file)

            # 获取片段信息
            index, timestamp_str, text = subtitle_item

            # 获取片段的开始和结束时间
            start_time, end_time = timestamp_str.split(' --> ')
            start_seconds = utils.time_to_seconds(start_time)
            end_seconds = utils.time_to_seconds(end_time)

            # 根据OST设置处理音频
            final_audio = final_audio.overlay(tts_audio, position=start_seconds * 1000)
        except Exception as e:
            logger.error(f"处理音频文件 {audio_file} 时出错: {str(e)}")
            continue

    # 保存合并后的音频文件
    output_audio_path = os.path.join(out_path, "edit_audio.mp3")
    final_audio.export(output_audio_path, format="mp3")
    logger.info(f"合并后的音频文件已保存: {output_audio_path}")

    return output_audio_path


def time_to_seconds(time_str):
    """
    将时间字符串转换为秒数，支持多种格式：
    1. 'HH:MM:SS,mmm' (时:分:秒,毫秒)
    2. 'MM:SS,mmm' (分:秒,毫秒)
    3. 'SS,mmm' (秒,毫秒)
    """
    try:
        # 处理毫秒部分
        if ',' in time_str:
            time_part, ms_part = time_str.split(',')
            ms = float(ms_part) / 1000
        else:
            time_part = time_str
            ms = 0

        # 分割时间部分
        parts = time_part.split(':')
        
        if len(parts) == 3:  # HH:MM:SS
            h, m, s = map(int, parts)
            seconds = h * 3600 + m * 60 + s
        elif len(parts) == 2:  # MM:SS
            m, s = map(int, parts)
            seconds = m * 60 + s
        else:  # SS
            seconds = int(parts[0])

        return seconds + ms
    except (ValueError, IndexError) as e:
        logger.error(f"Error parsing time {time_str}: {str(e)}")
        return 0.0


def extract_timestamp(filename):
    """
    从文件名中提取开始和结束时间戳
    例如: "audio_00_06,500-00_24,800.mp3" -> (6.5, 24.8)
    """
    try:
        # 从文件名中提取时间部分
        time_part = filename.split('_', 1)[1].split('.')[0]  # 获取 "00_06,500-00_24,800" 部分
        start_time, end_time = time_part.split('-')  # 分割成开始和结束时间
        
        # 将下划线格式转换回冒号格式
        start_time = start_time.replace('_', ':')
        end_time = end_time.replace('_', ':')
        
        # 将时间戳转换为秒
        start_seconds = time_to_seconds(start_time)
        end_seconds = time_to_seconds(end_time)

        return start_seconds, end_seconds
    except Exception as e:
        logger.error(f"Error extracting timestamp from {filename}: {str(e)}")
        return 0.0, 0.0


if __name__ == "__main__":
    # 示例用法
    audio_files =[
        "/Users/apple/Desktop/home/NarratoAI/storage/tasks/test456/audio_00:06-00:24.mp3",
        "/Users/apple/Desktop/home/NarratoAI/storage/tasks/test456/audio_00:32-00:38.mp3",
        "/Users/apple/Desktop/home/NarratoAI/storage/tasks/test456/audio_00:43-00:52.mp3",
        "/Users/apple/Desktop/home/NarratoAI/storage/tasks/test456/audio_00:52-01:09.mp3",
        "/Users/apple/Desktop/home/NarratoAI/storage/tasks/test456/audio_01:13-01:15.mp3",
    ]
    total_duration = 38
    video_script_path = "/Users/apple/Desktop/home/NarratoAI/resource/scripts/test003.json"
    with open(video_script_path, "r", encoding="utf-8") as f:
        video_script = json.load(f)

    output_file = merge_audio_files("test456", audio_files, total_duration, video_script)
    print(output_file)
