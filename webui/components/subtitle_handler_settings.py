from streamlit.delta_generator import DeltaGenerator
import streamlit as st
import os
from app.utils import file_utils,utils
from app.services import localhost_llm,subtitle
from app.models.file_info import LocalFileInfo
import pysrt
from moviepy.editor import VideoFileClip
from loguru import logger
import json



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
    subtitle_prompt = os.path.join(utils.root_dir(), "app","config","subtitle_prompt.md")
    with open(subtitle_prompt, 'r', encoding='utf-8') as f:
        md_content = f.read()
    llm_url = ai_container.text_input(label=tr("base_url"),key="llm_url",value="http://localhost:1234/v1")
    llm_api_key = ai_container.text_input(label=tr("api_key"),key="llm_api_key",value="lm-studio")
    llm_model = ai_container.text_input(label=tr("model"),key="llm_model",value="qwen2.5-14b-instruct")
    llm_prompt = ai_container.text_area(label=tr("prompt"),key="llm_prompt",value=md_content)
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
            subtitle_list = subtitle.str_to_list(subtitle_info.path)

            # llm process
            llm_result = None
            if is_use_llm:
                llm_result_list = localhost_llm.call_llm_get_list(base_url=llm_url,
                                                            api_key=llm_api_key,
                                                            model=llm_model,
                                                            prompt=llm_prompt,
                                                            content=json.dumps(subtitle_list),
                                                            temperature=llm_temperature,
                                                            retry_count=3)
                if llm_result_list is None or len(llm_result_list) == 0:
                    raise ValueError(tr("llm_result_empty"))
                llm_result = subtitle.list_to_srt_str(llm_result_list)
            else:
                llm_result = subtitle.list_to_srt_str(subtitle_content)

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



