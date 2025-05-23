import locale
import os
import requests
import threading
from typing import Any
from loguru import logger
import streamlit as st
import json
from uuid import uuid4
import urllib3,shutil
from datetime import datetime, timedelta
from app.models import const
from . import file_utils,streamlit_utils

urllib3.disable_warnings()


def get_response(status: int, data: Any = None, message: str = ""):
    obj = {
        "status": status,
    }
    if data:
        obj["data"] = data
    if message:
        obj["message"] = message
    return obj

def to_json(obj):
    try:
        # 定义一个辅助函数来处理不同类型的对象
        def serialize(o):
            # 如果对象是可序列化类型，直接返回
            if isinstance(o, (int, float, bool, str)) or o is None:
                return o
            # 如果对象是二进制数据，转换为base64编码的字符串
            elif isinstance(o, bytes):
                return "*** binary data ***"
            # 如果象是字典，递归处理每个键值对
            elif isinstance(o, dict):
                return {k: serialize(v) for k, v in o.items()}
            # 如果对象是列表或元组，递归处理每个元素
            elif isinstance(o, (list, tuple)):
                return [serialize(item) for item in o]
            # 如果对象是自定义类型，尝试返回其__dict__属性
            elif hasattr(o, "__dict__"):
                return serialize(o.__dict__)
            # 其他情况返回None（或者可以选择抛出异常）
            else:
                return None

        # 使用serialize函数处理输入对象
        serialized_obj = serialize(obj)

        # 序列化处理后的对象为JSON符串
        return json.dumps(serialized_obj, ensure_ascii=False, indent=4)
    except Exception as e:
        return None

def get_uuid(remove_hyphen: bool = False):
    u = str(uuid4())
    if remove_hyphen:
        u = u.replace("-", "")
    return u

def root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def storage_dir(sub_dir: str = "", create: bool = False):
    d = os.path.join(root_dir(), "storage")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if create and not os.path.exists(d):
        os.makedirs(d)

    return d

def resource_dir(sub_dir: str = ""):
    d = os.path.join(root_dir(), "resource")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    return d

def task_dir(sub_dir: str = ""):
    d = os.path.join(storage_dir(), "tasks")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d

def font_dir(sub_dir: str = ""):
    d = resource_dir("fonts")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d

def end_images_dir(sub_dir: str = ""):
    d = resource_dir("end_images")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d

def public_dir(sub_dir: str = ""):
    d = resource_dir(f"public")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d

def srt_dir(sub_dir: str = ""):
    d = resource_dir(f"srt")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d

def time_convert_seconds_to_hmsm(seconds) -> str:
    hours = int(seconds // 3600)
    seconds = seconds % 3600
    minutes = int(seconds // 60)
    milliseconds = int(seconds * 1000) % 1000
    seconds = int(seconds % 60)
    return "{:02d}:{:02d}:{:02d},{:03d}".format(hours, minutes, seconds, milliseconds)

def text_to_srt(idx: int, msg: str, start_time: float, end_time: float) -> str:
    start_time = time_convert_seconds_to_hmsm(start_time)
    end_time = time_convert_seconds_to_hmsm(end_time)
    srt = """%d
%s --> %s
%s
        """ % (
        idx,
        start_time,
        end_time,
        msg,
    )
    return srt

def str_contains_punctuation(word):
    for p in const.PUNCTUATIONS:
        if p in word:
            return True
    return False

def md5(text):
    import hashlib

    return hashlib.md5(text.encode("utf-8")).hexdigest()

def get_system_locale():
    try:
        loc = locale.getdefaultlocale()
        # zh_CN, zh_TW return zh
        # en_US, en_GB return en
        language_code = loc[0].split("_")[0]
        return language_code
    except Exception as e:
        return "en"

def load_locales(i18n_dir):
    _locales = {}
    for root, dirs, files in os.walk(i18n_dir):
        for file in files:
            if file.endswith(".json"):
                lang = file.split(".")[0]
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    _locales[lang] = json.loads(f.read())
    return _locales

def video_dir(sub_dir: str = ""):
    d = resource_dir(f"videos")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d

def create_dir(dir:str):
    if not os.path.exists(dir):
        os.makedirs(dir)

def time_to_seconds(time_str: str) -> float:
    try:
        # 处理带有'-'的毫秒格式
        if '-' in time_str:
            time_part, ms_part = time_str.split('-')
            ms = float(ms_part) / 1000
        # 处理带有','的毫秒格式
        elif ',' in time_str:
            time_part, ms_part = time_str.split(',')
            ms = float(ms_part) / 1000
        else:
            time_part = time_str
            ms = 0

        # 分割时间部分
        parts = time_part.split(':')

        if len(parts) == 3:  # HH:MM:SS
            h, m, s = map(float, parts)
            seconds = h * 3600 + m * 60 + s
        elif len(parts) == 2:  # MM:SS
            m, s = map(float, parts)
            seconds = m * 60 + s
        else:  # SS
            seconds = float(parts[0])

        return seconds + ms

    except (ValueError, IndexError) as e:
        logger.error(f"时间格式转换错误 {time_str}: {str(e)}")
        return 0.0
    
def time_to_ms(time_str: str) -> int:
    try:
        # get seconds and milliseconds
        if '-' in time_str:
            time_part, ms_part = time_str.split('-')
            ms = int(ms_part)
        elif ',' in time_str:
            time_part, ms_part = time_str.split(',')
            ms = int(ms_part)
        else:
            time_part = time_str
            ms = 0

        # split time part
        parts = time_part.split(':')

        if len(parts) == 3:  # HH:MM:SS
            h, m, s = map(int, parts)
            seconds = h * 3600 + m * 60 + s
        elif len(parts) == 2:  # MM:SS
            m, s = map(int, parts)
            seconds = m * 60 + s
        else:  # SS
            seconds = int(parts[0])

        return seconds * 1000 + ms

    except (ValueError, IndexError) as e:
        logger.error(f"时间格式转换错误 {time_str}: {str(e)}")
        return 0

def seconds_to_time(seconds: float) -> str:
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

def temp_dir(sub_dir: str = ""):
    """
    获取临时文件目录
    Args:
        sub_dir: 子目录名
    Returns:
        str: 临时文件目录路径
    """
    d = os.path.join(storage_dir(), "temp")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d

def init_resources():
    """初始化资源文件"""
    try:
        # 创建字体目录
        font_dir = os.path.join(root_dir(), "resource", "fonts")
        os.makedirs(font_dir, exist_ok=True)

        # 检查字体文件
        font_files = [
            ("SourceHanSansSC-Bold.otf",
             "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Bold.otf"),
            ("simhei.ttf", "C:/Windows/Fonts/simhei.ttf"),  # Windows 黑体
            ("simkai.ttf", "C:/Windows/Fonts/simkai.ttf"),  # Windows 楷体
            ("simsun.ttc", "C:/Windows/Fonts/simsun.ttc"),  # Windows 宋体
        ]

        # 优先使用系统字体
        system_font_found = False
        for font_name, source in font_files:
            if not source.startswith("http") and os.path.exists(source):
                target_path = os.path.join(font_dir, font_name)
                if not os.path.exists(target_path):
                    import shutil
                    shutil.copy2(source, target_path)
                    logger.info(f"已复制系统字体: {font_name}")
                system_font_found = True
                break

        # 如果没有找到系统字体，则下载思源黑体
        if not system_font_found:
            source_han_path = os.path.join(font_dir, "SourceHanSansCN-Regular.otf")
            if not os.path.exists(source_han_path):
                download_font(font_files[0][1], source_han_path)

    except Exception as e:
        logger.error(f"初始化资源文件失败: {e}")

def download_font(url: str, font_path: str):
    """下载字体文件"""
    try:
        logger.info(f"正在下载字体文件: {url}")
        import requests
        response = requests.get(url)
        response.raise_for_status()

        with open(font_path, 'wb') as f:
            f.write(response.content)

        logger.info(f"字体文件下载成功: {font_path}")

    except Exception as e:
        logger.error(f"下载字体文件失败: {e}")
        raise

def cleanup_tasks():
    tasks_path = task_dir()
    file_utils.cleanup_temp_files(tasks_path)

def cleanup_all_closed_tasks():
    active_task_paths = streamlit_utils.get_all_active_session_task_paths()
    all_task_paths = os.listdir(task_dir())
    # get all task paths
    for task_path in all_task_paths:
        task_all_path =  os.path.join(task_dir(), task_path)
        if task_all_path not in active_task_paths:
            shutil.rmtree(task_all_path)

def get_subdir_names(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

