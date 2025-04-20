import json
import os.path
import re
import traceback
from typing import Optional
import streamlit as st
from faster_whisper import WhisperModel
from timeit import default_timer as timer
from loguru import logger
from moviepy.editor import VideoFileClip
import os
from PIL import ImageFont
from app.config import config
from app.utils import utils,str_util
from app.models.subtitle_position_coord import SubtitlePositionCoord

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
    model_path = f"{utils.root_dir()}/app/models/faster-whisper-large-v3"
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
            beam_size=6,#使用束搜索(beam search)算法进行解码，束宽设为5。较大的值可以提高准确性但会降低速度
            best_of=3,#最优解，3个解码器输出，每个解码器输出的最大词数为512，较大的值可以提高准确性但会降低速度
            word_timestamps=True,#为每个单词生成时间戳信息
            vad_filter=True,#启用语音活动检测(VAD)过滤，自动过滤掉静音部分
            temperature=0.4,#控制解码时的随机性，较低的值(如0.3)使输出更确定性和保守
            no_repeat_ngram_size=3,#防止2-gram重复出现，减少重复内容
            vad_parameters=dict(
                min_silence_duration_ms=500,#设置VAD参数，这里指定最小静音持续时间为300毫秒
                speech_pad_ms=300 # 语音前后填充时间（避免截断）
                ),
            language="zh",
            task="transcribe",#执行转录任务(与"translate"翻译任务相对)
            initial_prompt="以下是清晰的标准普通话，文本连贯无重复"
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
    BASE_FONT_SIZE = 45  # 在基准分辨率下的字号
    
    # 按当前视频高度与基准高度的比例缩放字号
    font_size = int(BASE_FONT_SIZE * (video_height / BASE_HEIGHT))
    
    # 设置最小和最大字号限制（避免极端情况）
    font_size = max(24, min(font_size, 72))
    return font_size

def auto_wrap_text(text, font_path, font_size, max_width):
    font = ImageFont.truetype(font_path, font_size)
    lines = []
    current_line = []
    cleaned_text = re.sub(r"\s+", "", text)
    
    for word in cleaned_text:
        current_line.append(word)
        font_postions = font.getbbox("".join(current_line))
        line_width = font_postions[2] -  font_postions[0]  # 测量当前行宽度
        if line_width > max_width:
            current_line.pop()
            lines.append("".join(current_line))
            current_line = [word]
    
    lines.append("".join(current_line))
    return "\n".join(lines)

def analysis_subtitles(subtitles:list[tuple[int,str,str]]):
    def parse_srt_time(time_str):
        """将SRT时间格式转换为秒数"""
        hhmmss, ms = time_str.strip().split(',')
        h, m, s = hhmmss.split(':')
        return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000
    
    subtitle_blocks = []
    for index,times,text in subtitles:
        try:
            time_items = times.split(' --> ')
            start = parse_srt_time(time_items[0])
            end = parse_srt_time(time_items[1])
            subtitle_blocks.append({
                'original_idx': index,
                'start': start,
                'end': end,
                'text': text,
                'duration': end - start
            })
        except Exception as e:
            print(f"警告：解析字幕索引 {index} 时出错：{str(e)}")
            continue

    return subtitle_blocks

def remove_valid_subtitles_by_ocr(subtitle_path:str):
    recognize_position_model = None|SubtitlePositionCoord
    subtitle_position_dict = st.session_state.get('subtitle_position_dict', {})
    recognize_poistion = subtitle_position_dict.get("edit_video.mp4")
    recognize_position_model = SubtitlePositionCoord.model_validate(recognize_poistion)

    if recognize_position_model is None or not recognize_position_model.is_exist:
        print("subtitle position is not exist,not need to remove subtitle")
        return
    frame_subtitles_position = recognize_position_model.frame_subtitles_position

    # 第一步：准确生成OCR时间区间（可能多个不连续区间）格式为[(0，2.36,"你好"，2),(5.12，7.45,"你好呀"，10)]
    def generate_ocr_ranges(positions):
        sorted_times = sorted(positions.keys())
        if not sorted_times:
            return []

        # get text regs
        text_regs = {} # text_regs: {"这个女人"：(开始时间，结束时间，文本出现次数)}
        for time in sorted_times: # time: 0.0, 5.12, 7.33, 10.22
            position_results = positions[time] # position_results :[((155,1630),(919,1716),"这个女人"),((155,1630),(919,1716),"这个女人")]
            for position_text in position_results: # position_text: ((155,1630),(919,1716),"这个女人")
                text = position_text[2] # text: "这个女人"
                time_result = text_regs.get(text,(time,time,0)) # time_result ： (开始时间，结束时间，文本出现次数)

                # update time
                current_start = time_result[0]
                current_end = time
                current_count = time_result[2] + 1

                # update text_regs
                text_regs[text] = (current_start,current_end,current_count)
        
        # 过滤出现次数过少的异常文本
        text_regs = {text:time_result for text,time_result in text_regs.items() if time_result[2] > 3}
        
        # 将text_regs转换为[(开始时间，结束时间，文本，次数)]
        ranges = [(time_result[0],time_result[1],text,time_result[2]) for text,time_result in text_regs.items()]

        return ranges

    ocr_time_ranges = generate_ocr_ranges(frame_subtitles_position)

    # 第二步：处理字幕文件，格式为
    # {
    #             'original_idx': index,
    #             'start': start,秒
    #             'end': end,秒
    #             'text': text,文本
    #             'duration': end - start 秒
    #         }
    subtitles = file_to_subtitles(subtitle_path)
    subtitle_blocks = analysis_subtitles(subtitles)

    # 第三步：过滤字幕
    def is_subtitle_valid(sub, ocr_ranges, time_coverage_threshold=0.7,text_coverage_threshold=0.7):
        """检查字幕是否在OCR时间范围内有足够覆盖"""
        if sub['duration'] <= 0:
            return False
        
        sub_start = sub['start']
        sub_end = sub['end']
        sub_text = sub['text']

        ocr_texts = []
        for ocr_start, ocr_end,ocr_text,ocr_num in ocr_ranges:
            # 如果出现文本
            if sub_text == ocr_text:
                return True
            # 如果未出现则获取时间范围覆盖度达到阀值的ocr识别结果，并对比字符匹配阀值是否达到标准
            overlap_start = max(sub_start, ocr_start)
            overlap_end = min(sub_end, ocr_end)
            if overlap_start < overlap_end:
                if overlap_end - overlap_start >= (sub_end - sub_start) * time_coverage_threshold:
                    ocr_texts.append(ocr_text)
        if not ocr_texts:
            return False
        
        for ocr_text in ocr_texts:
            if str_util.count_common_chars(ocr_text,sub_text) / len(ocr_text) >= text_coverage_threshold:
                return True

        return False

    valid_subtitles = [sub for sub in subtitle_blocks if is_subtitle_valid(sub, ocr_time_ranges)]

    # 第四步：生成新字幕文件
    output_content = []
    for new_idx, sub in enumerate(valid_subtitles, 1):
        output_content.append(utils.text_to_srt(new_idx, sub['text'], sub['start'], sub['end']))

    # 写回文件
    sub_item = "\n".join(output_content) + "\n"
    with open(subtitle_path, "w", encoding="utf-8") as f:
        f.write(sub_item)
    print(f"字幕清理完成：原始字幕 {len(subtitle_blocks)} 条，保留 {len(valid_subtitles)} 条")
    

