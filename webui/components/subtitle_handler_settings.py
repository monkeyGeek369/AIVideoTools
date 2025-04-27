from streamlit.delta_generator import DeltaGenerator
import streamlit as st
import os
from app.utils import file_utils,utils
from app.services import localhost_llm,subtitle
from app.models.file_info import LocalFileInfo
import pysrt
from moviepy.editor import VideoFileClip
from loguru import logger



def render_subtitle_handler(tr,st_container:DeltaGenerator,container_dict:dict[str,DeltaGenerator]):
    # layout
    subtitle_column,ai_column,processed_column = st_container.columns(3)
    subtitle_container = subtitle_column.container(border=True)
    ai_container = ai_column.container(border=True)
    processed_container = processed_column.container(border=True)

    # get task path
    task_path = st.session_state['task_path']

    # subtitle handler
    material_subtitles_path = os.path.join(task_path, "material_subtitles")
    material_subtitles = file_utils.get_file_list(directory=material_subtitles_path,sort_by="name")
    for material_subtitle in material_subtitles:
        with open(material_subtitle.path, 'r', encoding='utf-8') as f:
            subtitle_content = f.read()
            subtitle_container.write(material_subtitle.name)
            subtitle_container.text_area(
                material_subtitle.name,
                value=subtitle_content,
                height=600,
                label_visibility="collapsed",
                on_change=None,
                key="Subtitle"+":"+material_subtitle.name
            )

    # video handler
    material_videos_path = os.path.join(task_path, "material_videos")
    material_videos = file_utils.get_file_list(directory=material_videos_path,sort_by="name")

    # ai handler
    llm_url = ai_container.text_input(label=tr("base_url"),key="llm_url",value="http://localhost:1234/v1")
    llm_api_key = ai_container.text_input(label=tr("api_key"),key="llm_api_key",value="lm-studio")
    llm_model = ai_container.text_input(label=tr("model"),key="llm_model",value="qwen2.5-14b-instruct")
    llm_prompt = ai_container.text_area(label=tr("prompt"),key="llm_prompt",value="""
# 一、背景信息

你现在是一名中文视频字幕处理专家，需要处理的中文字幕格式参考给定的字幕案例.

### 1.1、字幕案例

```
[
  {
    "index": 1,
    "timerange": "00:00:00,000 --> 00:00:02,920",
    "text": "救了一只没妈的熊宝宝"
  },
  {
    "index": 2,
    "timerange": "00:00:02,920 --> 00:00:03,560",
    "text": "他们给它喂奶并照顾它直到它恢复健康"
  },
  {
    "index": 3,
    "timerange": "00:00:03,600 --> 00:00:05,470",
    "text": "这是一段温暖人心的故事情节"
  }
]
```

### 1.2、字幕案例格式说明

1. 字幕信息为list列表对象
2. 列表中每一个组成对象称为“一条字幕”
3. 每“一条字幕”都由三部分组成,即index“序号”、timerange“字幕时间范围”、text“字幕文本内容”
4. index: 从1开始的连续自然数序列,即第一条字幕的index为1,第二条字幕的index为2...以此类推
5. timerange: 字幕的持续时间字符串
6. text: 字幕的文本信息



# 二、任务要求

### 2.1、要求描述

1. 现在需要你明确自己的角色定位
2. 理解“背景信息”中的字幕格式说明,理解字幕的组成结构
3. 当给到你新的字幕数据后需要按照“2.2、要求细节步骤”进行处理
4. 将处理后的字幕完整输出



### 2.2、要求细节步骤

1. 每一条字幕中的“index”、“timerange”无需任何变动
2. 如果存在“text”内容不符合上下文语意，或者错别字严重的字幕，需要对当前字幕进行删除
3. 如果存在“index”重复或“timerange”重复或“text”重复的字幕,则必须对当前字幕进行删除
4. 针对剩余的字幕中包含的“text”进行改写
5. 改写“text”时对明显不符合上下文语意的错别字进行修正
6. 改写“text”时对其中的名词、主语存在不符合语意的情形进行改写或删除等操作
7. 改写后的“text”要与原来的“text”语意相同但文字要有差异，且字数一定不能超过原字数
8. 改写后的“text”禁止包含标点符号，可以用空格代替
9. 一定要保证第一个字幕中的“index”一定是1，后续的每一个字幕中的“index”一定是前一个的index数值加上1的正整数，不能是None，且不能重复，即index一定是从1开始的连续自然数序列
10. 处理后的字幕数据需要严格遵守1.2中规定的字幕格式，即每一个字幕必须由“index”、“timerange”、“text”三部分组成
11. 输出结果需要保证为字幕list
""")
    llm_temperature = ai_container.text_input(label=tr("temperature"),key="llm_temperature",value="0.7")

    ai_btn_container = ai_container.container(border=True)
    llm_btn_left,llm_btn_mid,llm_btn_right = ai_btn_container.columns(3)

    with ai_container:
        try:
            if llm_btn_left.button(label=tr("llm_test_check")):
                with st.spinner(tr("processing")):
                    status = localhost_llm.check_llm_status(base_url=llm_url,api_key=llm_api_key,model=llm_model)
                    if status:
                        st.success(tr("llm_check_success"))
                    else:
                        raise Exception(tr("llm_check_fail"))
            if llm_btn_mid.button(label=tr("llm_subtitle_process")):
                with st.spinner(tr("processing")):
                    subtitle_ai_handler(llm_url=llm_url,llm_api_key=llm_api_key,llm_model=llm_model,llm_prompt=llm_prompt,llm_temperature=llm_temperature,
                                        st_container=processed_container,material_subtitles=material_subtitles,tr=tr,material_videos=material_videos,
                                        container_dict=container_dict,is_use_llm=True)
                    st.success(tr("llm_handler_success"))
            if llm_btn_right.button(label=tr("use_material_subtitles")):
                subtitle_ai_handler(llm_url=llm_url,llm_api_key=llm_api_key,llm_model=llm_model,llm_prompt=llm_prompt,llm_temperature=llm_temperature,
                                    st_container=processed_container,material_subtitles=material_subtitles,tr=tr,material_videos=material_videos,
                                    container_dict=container_dict,is_use_llm=False)
                st.success(tr("use_original_subtitle_success"))
        except Exception as e:
            st.error(e)

def subtitle_ai_handler(llm_url:str,llm_api_key:str,llm_model:str,llm_prompt:str,llm_temperature:float,
                        st_container:DeltaGenerator,
                        material_subtitles:list[LocalFileInfo],
                        tr,
                        material_videos:list[LocalFileInfo],
                        container_dict:dict[str,DeltaGenerator],
                        is_use_llm:bool=True):
    # check subtitles
    if material_subtitles is None or len(material_subtitles) == 0:
        logger.warning(tr("material_subtitles_empty"))
        return
    if len(material_subtitles) != len(material_videos):
        raise ValueError(tr("material_subtitles_video_count_not_match"))

    # params init
    video_clips = []
    accumulated_duration = 0
    merged_subs = pysrt.SubRipFile()

    try:
        # subtitle handler
        for i, (video_info, subtitle_info) in enumerate(zip(material_videos, material_subtitles)):
            # video process
            video_clip = VideoFileClip(video_info.path)
            video_clips.append(video_clip)

            # subtitle content
            subtitle_content = None
            with open(subtitle_info.path, 'r', encoding='utf-8') as f:
                subtitle_content = f.read()

            # llm process
            llm_result = None
            if is_use_llm:
                llm_result_list = localhost_llm.call_llm_get_list(base_url=llm_url,
                                                            api_key=llm_api_key,
                                                            model=llm_model,
                                                            prompt=llm_prompt,
                                                            content=subtitle_content,
                                                            temperature=llm_temperature,
                                                            retry_count=3)
                if llm_result_list is None or len(llm_result_list) == 0:
                    raise ValueError(tr("llm_result_empty"))
                llm_result = subtitle.list_to_srt_str(llm_result_list)
            else:
                llm_result = subtitle_content

            # subtitle process
            if i == 0:
                current_subs = pysrt.from_string(llm_result)
            else:
                # time adjust
                current_subs = adjust_subtitle_timing(llm_result, accumulated_duration)

            # merge subtitle
            merged_subs.extend(current_subs)

            # add duration
            accumulated_duration += video_clip.duration

        # merge subtitle time interval
        # for i, sub in enumerate(merged_subs):
        #     if i != 0:
        #         sub.start = sub.start + pysrt.SubRipTime(milliseconds=20)

        # merge subtitle path
        task_path = st.session_state['task_path']
        edit_subtitles_path = os.path.join(task_path, "edit_subtitles")
        file_utils.ensure_directory(edit_subtitles_path)
        output_subtitle_path = os.path.join(edit_subtitles_path, "merged.srt")
        merged_subs.save(output_subtitle_path, encoding='utf-8')
        st.session_state['edit_subtitle_path'] = output_subtitle_path

        # show subtitle
        with open(output_subtitle_path, 'r', encoding='utf-8') as f:
            subtitle_content = f.read()
            st_container.text_area(
                "merged_srt",
                value=subtitle_content,
                height=600,
                label_visibility="collapsed",
                on_change=None
            )
            container_dict["edit_subtitle_expander"].text_area(
                "merged.srt",
                value=subtitle_content,
                height=150,
                label_visibility="collapsed",
                on_change=None
            )
    finally:
        # clean up
        for clip in video_clips:
            clip.close()
            del clip
        del merged_subs

def adjust_subtitle_timing(subtitle_content, time_offset):
    """调整字幕时间戳"""
    subs = pysrt.from_string(subtitle_content)

    # 为每个字幕项添加时间偏移
    for sub in subs:
        sub.start.hours += int(time_offset / 3600)
        sub.start.minutes += int((time_offset % 3600) / 60)
        sub.start.seconds += int(time_offset % 60)
        sub.start.milliseconds += int((time_offset * 1000) % 1000)

        sub.end.hours += int(time_offset / 3600)
        sub.end.minutes += int((time_offset % 3600) / 60)
        sub.end.seconds += int(time_offset % 60)
        sub.end.milliseconds += int((time_offset * 1000) % 1000)

    return subs



