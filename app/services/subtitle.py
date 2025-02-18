import json
import os.path
import re
import traceback
from typing import Optional

from faster_whisper import WhisperModel
from timeit import default_timer as timer
from loguru import logger
from moviepy.editor import VideoFileClip
import os
from PIL import ImageFont

from app.config import config
from app.utils import utils

model_size = config.whisper.get("model_size", "faster-whisper-large-v2")
device = config.whisper.get("device", "cpu")
compute_type = config.whisper.get("compute_type", "int8")

def create(audio_file, subtitle_file: str = ""):
    """
    为给定的音频文件创建字幕文件。

    参数:
    - audio_file: 音频文件的路径。
    - subtitle_file: 字幕文件的输出路径（可选）。如果未提供，将根据音频文件的路径生成字幕文件。

    返回:
    无返回值，但会在指定路径生成字幕文件。
    """
    global device, compute_type
    model = None

    # 加载 Whisper 模型check
    model_path = f"{utils.root_dir()}/app/models/faster-whisper-large-v2"
    model_bin_file = f"{model_path}/model.bin"
    if not os.path.isdir(model_path) or not os.path.isfile(model_bin_file):
        logger.error(
            "请先下载 whisper 模型\n\n"
            "********************************************\n"
            "下载地址：https://huggingface.co/guillaumekln/faster-whisper-large-v2\n"
            "存放路径：app/models \n"
            "********************************************\n"
        )
        return None
    else:
        # 尝试使用 CUDA，如果失败则回退到 CPU
        try:
            import torch
            if torch.cuda.is_available():
                try:
                    logger.info(f"尝试使用 CUDA 加载模型: {model_path}")
                    with torch.no_grad():
                        model = WhisperModel(
                            model_size_or_path=model_path,
                            device="cuda",
                            compute_type="float16",
                            local_files_only=True
                        )
                    device = "cuda"
                    compute_type = "float16"
                    logger.info("成功使用 CUDA 加载模型")
                except Exception as e:
                    logger.warning(f"CUDA 加载失败，错误信息: {str(e)}")
                    logger.warning("回退到 CPU 模式")
                    device = "cpu"
                    compute_type = "int8"
                    model = None
                    torch.cuda.empty_cache()
            else:
                logger.info("未检测到 CUDA，使用 CPU 模式")
                device = "cpu"
                compute_type = "int8"
        except ImportError:
            logger.warning("未安装 torch，使用 CPU 模式")
            device = "cpu"
            compute_type = "int8"
            model = None
            torch.cuda.empty_cache()

        if device == "cpu":
            logger.info(f"使用 CPU 加载模型: {model_path}")
            model = WhisperModel(
                model_size_or_path=model_path,
                device=device,
                compute_type=compute_type,
                local_files_only=True
            )

        logger.info(f"模型加载完成，使用设备: {device}, 计算类型: {compute_type}")

    logger.info(f"start, output file: {subtitle_file}")
    if not subtitle_file:
        subtitle_file = f"{audio_file}.srt"

    segments, info = None, None
    try:
        segments, info = model.transcribe(
            audio_file,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            initial_prompt="以下是普通话的句子"
        )
    except Exception as e:
        model = None
        torch.cuda.empty_cache()
        raise Exception(f"Voice contains no text information: {str(e)}")

    logger.info(
        f"检测到的语言: '{info.language}', probability: {info.language_probability:.2f}"
    )

    start = timer()
    subtitles = []

    def recognized(seg_text, seg_start, seg_end):
        seg_text = seg_text.strip()
        if not seg_text:
            return

        msg = "[%.2fs -> %.2fs] %s" % (seg_start, seg_end, seg_text)
        logger.debug(msg)

        subtitles.append(
            {"msg": seg_text, "start_time": seg_start, "end_time": seg_end}
        )

    for segment in segments:
        words_idx = 0
        words_len = len(segment.words)

        seg_start = 0
        seg_end = 0
        seg_text = ""

        if segment.words:
            is_segmented = False
            for word in segment.words:
                if not is_segmented:
                    seg_start = word.start
                    is_segmented = True

                seg_end = word.end
                # 如果包含标点,则断句
                seg_text += word.word

                if utils.str_contains_punctuation(word.word):
                    # remove last char
                    seg_text = seg_text[:-1]
                    if not seg_text:
                        continue

                    recognized(seg_text, seg_start, seg_end)

                    is_segmented = False
                    seg_text = ""

                if words_idx == 0 and segment.start < word.start:
                    seg_start = word.start
                if words_idx == (words_len - 1) and segment.end > word.end:
                    seg_end = word.end
                words_idx += 1

        if not seg_text:
            continue

        recognized(seg_text, seg_start, seg_end)

    end = timer()

    diff = end - start
    logger.info(f"complete, elapsed: {diff:.2f} s")

    idx = 1
    lines = []
    for subtitle in subtitles:
        text = subtitle.get("msg")
        if text:
            lines.append(
                utils.text_to_srt(
                    idx, text, subtitle.get("start_time"), subtitle.get("end_time")
                )
            )
            idx += 1

    sub = "\n".join(lines) + "\n"
    with open(subtitle_file, "w", encoding="utf-8") as f:
        f.write(sub)
    logger.info(f"subtitle file created: {subtitle_file}")

    # clean cache
    del model
    del segments
    del info
    del subtitles
    torch.cuda.empty_cache()

def file_to_subtitles(filename):
    """
    将字幕文件转换为字幕列表。

    参数:
    filename (str): 字幕文件的路径。

    返回:
    list: 包含字幕序号、出现时间、和字幕文本的元组列表。
    """
    if not filename or not os.path.isfile(filename):
        return []

    times_texts = []
    current_times = None
    current_text = ""
    index = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            times = re.findall("([0-9]*:[0-9]*:[0-9]*,[0-9]*)", line)
            if times:
                current_times = line
            elif line.strip() == "" and current_times:
                index += 1
                times_texts.append((index, current_times.strip(), current_text.strip()))
                current_times, current_text = None, ""
            elif current_times:
                current_text += line
    return times_texts

def levenshtein_distance(s1, s2):
    '''
    计算字符串编辑距离
    '''
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def similarity(a, b):
    '''
    计算字符串相似度
    '''
    distance = levenshtein_distance(a.lower(), b.lower())
    max_length = max(len(a), len(b))
    return 1 - (distance / max_length)

def extract_audio_and_create_subtitle(video_file: str, subtitle_file: str = "") -> Optional[str]:
    """
    从视频文件中提取音频并生成字幕文件。

    参数:
    - video_file: MP4视频文件的路径
    - subtitle_file: 输出字幕文件的路径（可选）。如果未提供，将根据视频文件名自动生成。

    返回:
    - str: 生成的字幕文件路径
    - None: 如果处理过程中出现错误
    """
    try:
        # 获取视频文件所在目录
        video_dir = os.path.dirname(video_file)
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        
        # 设置音频文件路径
        audio_file = os.path.join(video_dir, f"{video_name}_audio.wav")
        
        # 如果未指定字幕文件路径，则自动生成
        if not subtitle_file:
            subtitle_file = os.path.join(video_dir, f"{video_name}.srt")
        
        logger.info(f"开始从视频提取音频: {video_file}")
        
        # 加载视频文件
        video = VideoFileClip(video_file)
        
        # 提取音频并保存为WAV格式
        logger.info(f"正在提取音频到: {audio_file}")
        video.audio.write_audiofile(audio_file, codec='pcm_s16le')
        
        # 关闭视频文件
        video.close()
        
        logger.info("音频提取完成，开始生成字幕")
        
        # 使用create函数生成字幕
        create(audio_file, subtitle_file)
        
        # 删除临时音频文件
        if os.path.exists(audio_file):
            os.remove(audio_file)
            logger.info("已清理临时音频文件")
        
        return subtitle_file
        
    except Exception as e:
        logger.error(f"处理视频文件时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def get_text_from_subtitle(sub) -> str:
    '''
    从字幕文件的单个字幕对象中获取字幕文本
    '''

    # 检查字幕文本是否为空
    if not sub or not sub.text or sub.text.strip() == '':
        return None

    # 处理字幕文本：确保是字符串，并处理可能的列表情况
    if isinstance(sub.text, (list, tuple)):
        subtitle_text = ' '.join(str(item) for item in sub.text if item is not None)
    else:
        subtitle_text = str(sub.text)

    return subtitle_text.strip()

def calculate_font_size(video_height):
    '''
    以视频为基准，根据视频高度计算字号
    '''
    # 定义基础参数（以常见的1080p横屏视频为基准）
    BASE_HEIGHT = 1080  # 基准分辨率高度
    BASE_FONT_SIZE = 48  # 在基准分辨率下的字号
    
    # 按当前视频高度与基准高度的比例缩放字号
    font_size = int(BASE_FONT_SIZE * (video_height / BASE_HEIGHT))
    
    # 设置最小和最大字号限制（避免极端情况）
    font_size = max(24, min(font_size, 72))
    return font_size

def auto_wrap_text(text, font_path, font_size, max_width):
    font = ImageFont.truetype(font_path, font_size)
    lines = []
    current_line = []
    
    for word in text.split():
        current_line.append(word)
        line_width = font.getbbox(" ".join(current_line))[2]  # 测量当前行宽度
        if line_width > max_width:
            current_line.pop()
            lines.append(" ".join(current_line))
            current_line = [word]
    
    lines.append(" ".join(current_line))
    return "\n".join(lines)

if __name__ == "__main__":
    task_id = "123456"
    task_dir = utils.task_dir(task_id)
    subtitle_file = f"{task_dir}/subtitle_123456.srt"
    audio_file = f"{task_dir}/audio.wav"
    video_file = "/Users/apple/Desktop/home/NarratoAI/resource/videos/merged_video_1702.mp4"

    extract_audio_and_create_subtitle(video_file, subtitle_file)

    # subtitles = file_to_subtitles(subtitle_file)
    # print(subtitles)

    # # script_file = f"{task_dir}/script.json"
    # # with open(script_file, "r") as f:
    # #     script_content = f.read()
    # # s = json.loads(script_content)
    # # script = s.get("script")
    # #
    # # correct(subtitle_file, script)

    # subtitle_file = f"{task_dir}/subtitle111.srt"
    # create(audio_file, subtitle_file)

    # # # 使用Gemini模型处理音频
    # # gemini_api_key = config.app.get("gemini_api_key")  # 请替换为实际的API密钥
    # # gemini_subtitle_file = create_with_gemini(audio_file, api_key=gemini_api_key)
    # #
    # # if gemini_subtitle_file:
    # #     print(f"Gemini生成的字幕文件: {gemini_subtitle_file}")
