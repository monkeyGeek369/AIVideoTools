import os.path
import re
import streamlit as st
from faster_whisper import WhisperModel
from timeit import default_timer as timer
from loguru import logger
import os,json
from PIL import ImageFont
from app.config import config
from app.utils import utils,str_util
from app.models.subtitle_position_coord import SubtitlePositionCoord
from app.services import localhost_llm

device = config.whisper.get("device", "cpu")
compute_type = config.whisper.get("compute_type", "int8")

def create(audio_file, subtitle_file: str = ""):
    global device, compute_type
    model = None

    # 加载 Whisper 模型check
    model_path = f"{utils.root_dir()}/app/models/faster-whisper-large-v3"
    model_bin_file = f"{model_path}/model.bin"
    if not os.path.isdir(model_path) or not os.path.isfile(model_bin_file):
        logger.error(
            "请先下载 whisper 模型\n\n"
            "********************************************\n"
            "下载地址：https://huggingface.co/Systran/faster-whisper-large-v3/tree/main\n"
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
            word_timestamps=True,#为每个单词生成时间戳信息
            # beam_size=10,#使用束搜索(beam search)算法进行解码，束宽设为5。较大的值可以提高准确性但会降低速度
            # best_of=5,#最优解，3个解码器输出，每个解码器输出的最大词数为512，较大的值可以提高准确性但会降低速度
            # vad_filter=True,#启用语音活动检测(VAD)过滤，自动过滤掉静音部分
            # temperature=0.4,#控制解码时的随机性，较低的值(如0.3)使输出更确定性和保守
            # no_repeat_ngram_size=3,#防止2-gram重复出现，减少重复内容
            # vad_parameters=dict(
            #     min_silence_duration_ms=500,#设置VAD参数，这里指定最小静音持续时间为300毫秒
            #     speech_pad_ms=300 # 语音前后填充时间（避免截断）
            #     ),
            # language="zh",
            # task="transcribe",#执行转录任务(与"translate"翻译任务相对)
            initial_prompt="请将以下普通话语音转写为简体中文书面文本，避免重复。"
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
        if segment.words is None:
            continue
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

def get_subtitle_duration(subtitle_duration_str:str) -> int:
    if not subtitle_duration_str:
        return 0
    
    # get time str
    start_time, end_time = subtitle_duration_str.split(' --> ')
    if not start_time or not end_time:
        return 0

    def parse_srt_time(time_str):
        hhmmss, ms = time_str.strip().split(',')
        h, m, s = hhmmss.split(':')
        return int(h)*3600*1000 + int(m)*60*1000 + int(s)*1000 + int(ms)
    
    return parse_srt_time(end_time) - parse_srt_time(start_time)

def remove_valid_subtitles_by_ocr(subtitle_path:str):
    recognize_position_model = None|SubtitlePositionCoord
    subtitle_position_dict = st.session_state.get('subtitle_position_dict', {})
    recognize_poistion = subtitle_position_dict.get("edit_video.mp4")
    recognize_position_model = SubtitlePositionCoord.model_validate(recognize_poistion)

    if recognize_position_model is None or not recognize_position_model.is_exist:
        print("subtitle position is not exist,not need to remove subtitle")
        return
    frame_time_text_dict = recognize_position_model.frame_time_text_dict

    # 第一步：准确生成OCR时间区间（可能多个不连续区间）格式为[(0，2.36,"你好"，2),(5.12，7.45,"你好呀"，10)]
    def generate_ocr_ranges(positions):
        sorted_times = sorted(positions.keys())
        if not sorted_times:
            return []

        # get text regs
        text_regs = {} # text_regs: {"这个女人"：(开始时间，结束时间，文本出现次数)}
        for time in sorted_times: # time: 0.0, 5.12, 7.33, 10.22
            position_results = positions[time] # position_results :["这个女人","这个女人"]
            for position_text in position_results: # position_text: "这个女人"
                text = position_text # text: "这个女人"
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

    ocr_time_ranges = generate_ocr_ranges(frame_time_text_dict)

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
    if  output_content is None or len(output_content) == 0:
        # delete file
        os.remove(subtitle_path)
        print(f"字幕清理完成：原始字幕 {len(subtitle_blocks)} 条，保留 0 条,已删除文件：{subtitle_path}")
    else:
        sub_item = "\n".join(output_content) + "\n"
        with open(subtitle_path, "w", encoding="utf-8") as f:
            f.write(sub_item)
        print(f"字幕清理完成：原始字幕 {len(subtitle_blocks)} 条，保留 {len(valid_subtitles)} 条")
    
def list_to_srt_str(subtitles:list[dict])-> str:
    if subtitles is None or len(subtitles) == 0:
        return None
    result = None
    for sub in subtitles:
        content = f"{str(sub['index'])}\n{sub['timerange']}\n{sub['text']}\n\n"
        if result is None:
            result = content
        else:
            result = result + content
    return result

def str_to_list(file_path: str) -> list[dict]:
    sub_tumples = file_to_subtitles(filename=file_path)
    if sub_tumples is None or len(sub_tumples) == 0:
        return None
    return [{"index":sub[0],"timerange":sub[1],"text":sub[2]} for sub in sub_tumples]

def filter_frame_subtitles_position_by_area(sub_rec_area:str,frame_subtitles_position:dict[float,dict]) -> dict:
    if sub_rec_area == "full_area":
        return frame_subtitles_position
    v_height = int(st.session_state.get("video_height"))

    for t, result in frame_subtitles_position.items():
        if result is None:
            continue
        # origin coordinates
        origin_coordinates = result.get("coordinates")

        # realy coordinates
        realy_coordinates = []
        for coord in origin_coordinates:
            if coord is None:
                continue
            
            # top left
            x1,y1 = coord[0]
            # bottom right
            x2,y2 = coord[1]

            if sub_rec_area == "upper_part_area":
                if y2 <= v_height * 0.5:
                    realy_coordinates.append(coord)
                
            if sub_rec_area == "lower_part_area":
                if y1 >= v_height * 0.5:
                    realy_coordinates.append(coord)

        result["coordinates"] = realy_coordinates

    return frame_subtitles_position

def remove_any_subtitle_duplicates(data):
    seen_text = set()
    seen_timerange = set()
    result = []
    for item in data:
        text = item['text']
        timerange = item['timerange']
        if text not in seen_text and timerange not in seen_timerange:
            seen_text.add(text)
            seen_timerange.add(timerange)
            result.append(item)
    return result

def subtitle_llm_handler(base_url:str,
                        api_key:str,
                        model:str,
                        prompt:str,
                        title:str,
                        subtitle_file_path:str,
                        temperature=0.7,
                        retry_count=3) -> list[dict]:
    # subtitle content
    subtitle_list = str_to_list(subtitle_file_path)
    if subtitle_list is None or len(subtitle_list) == 0:
        logger.error("subtitle content is empty")
        return None
    
    # base param
    retry_contents = ["语义不通顺，无法处理"]

    # content_step_remove
    content_step_remove_path = os.path.join(utils.root_dir(), "app","config","subtitle_remove_prompt.md")
    with open(content_step_remove_path, 'r', encoding='utf-8') as f:
        content_step_remove = f.read()

    # content_step_optimize
    content_step_optimize_path = os.path.join(utils.root_dir(), "app","config","subtitle_optimize_prompt.md")
    with open(content_step_optimize_path, 'r', encoding='utf-8') as f:
        content_step_optimize = f.read()

    # base filter: repeat subtitle
    subtitle_list = remove_any_subtitle_duplicates(subtitle_list)

    # base filter: duration too low
    subtitle_list = [item for item in subtitle_list if get_subtitle_duration(item.get("timerange")) > 600]

    # llm handler: optimize subtitle
    req_content_list = [{"index":sub.get("index"),"text":sub.get("text")} for sub in subtitle_list if sub is not None]
    optimize_content = content_step_optimize.format(sub_title=title,sub_content=json.dumps(req_content_list,ensure_ascii=False))
    optimized_list = localhost_llm.call_llm_get_list(base_url,api_key,model,prompt,optimize_content,retry_contents,temperature,retry_count,True)
    for opt_item in optimized_list:
        if opt_item is None:
            continue
        for sub_item in subtitle_list:
            if sub_item is None:
                continue
            if opt_item.get("index") == sub_item.get("index"):
                sub_item["text"] = opt_item.get("text")
    
    # llm handler: remove error subtitle
    req_remove_content = [{"index":sub.get("index"),"duration":get_subtitle_duration(sub.get("timerange")),"text":sub.get("text")} for sub in subtitle_list if sub is not None]
    remove_content = content_step_remove.format(sub_title=title,sub_content=json.dumps(req_remove_content,ensure_ascii=False))
    error_index_list = localhost_llm.call_llm_get_list(base_url,api_key,model,prompt,remove_content,retry_contents,temperature,retry_count,False)
    if error_index_list is not None and len(error_index_list) > 0:
        subtitle_list = [item for item in subtitle_list if item.get("index") not in error_index_list]

    # base filter
    if len(subtitle_list) == 0:
        raise Exception("no subtitle result")
    
    # format result index
    for i, item in enumerate(subtitle_list, start=1):
        item["index"] = i

    return subtitle_list

def filter_frame_subtitles_position(ignore_min_width:int,
                                    ignore_min_height:int,
                                    ignore_min_word_count:int,
                                    ignore_text:str,
                                    frame_subtitles_position:dict[float,dict]) -> dict:
    for t, result in frame_subtitles_position.items():
        index,coordinates = result.get("index"),result.get("coordinates")

        # filter: ignore min width
        coordinates = [ coord for coord in coordinates if (coord[1][0] - coord[0][0] >= ignore_min_width)]
        # filter: ignore min height
        coordinates = [ coord for coord in coordinates if (coord[1][1] - coord[0][1] >= ignore_min_height)]
        # filter: ignore min word count
        coordinates = [ coord for coord in coordinates if (len(coord[2]) >= ignore_min_word_count)]
        # filter: ignore text
        if ignore_text:
            texts = ignore_text.split("\n") if ignore_text else []
            coordinates = [ coord for coord in coordinates if (not str_util.is_str_contain_list_strs(coord[2],texts))]

        # update result
        frame_subtitles_position[t] = {
            "index": index,
            "coordinates": coordinates
        }

    return frame_subtitles_position

