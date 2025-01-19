from streamlit.delta_generator import DeltaGenerator
import streamlit as st
import os
from app.utils import file_utils,utils



def render_subtitle_handler(tr,st_container:DeltaGenerator):
    # layout
    subtitle_column,ai_column,processed_column = st_container.columns(3)
    subtitle_container = subtitle_column.container(border=True)
    ai_container = ai_column.container(border=True)
    processed_container = processed_column.container(border=True)

    # get task path
    task_path = st.session_state['task_path']

    # subtitle handler
    material_subtitles_path = os.path.join(task_path, "material_subtitles")
    material_subtitles = file_utils.get_file_list(directory=material_subtitles_path)
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

    # ai handler
    llm_url = ai_container.text_input(label=tr("base_url"),key="llm_url")
    llm_api_key = ai_container.text_input(label=tr("api_key"),key="llm_api_key")
    llm_model = ai_container.text_input(label=tr("model"),key="llm_model")
    llm_prompt = ai_container.text_area(label=tr("prompt"),key="llm_prompt",value="你现在是一名中文视频字幕处理专家，给定中文字幕信息包含字幕index、字幕时间范围、字幕内容，当给到你字幕数据后希望你进行如下处理。-针对每一段字幕一定要重新生成字幕内容-新生成的字幕内容要与原字幕上下文语意相同但文字要有差异-新生成的字幕要满足原视频时间范围-直接输出处理后的中文字幕结果，无需输出其它内容-输出格式要与原格式相同-禁止带标点符号，可以用空格代替")
    llm_temperature = ai_container.text_input(label=tr("temperature"),key="llm_temperature",value="0.7")

    ai_btn_container = ai_container.container(border=True)
    llm_btn_left,llm_btn_mid,llm_btn_right = ai_btn_container.columns(3)
    if llm_btn_left.button(label=tr("llm_test_check")):
        print("llm_url:",llm_url)
    if llm_btn_mid.button(label=tr("llm_subtitle_process")):
        print("llm_api_key:",llm_api_key)
    if llm_btn_right.button(label=tr("use_material_subtitles")):
        print("llm_model:",llm_model)

    # processing handler
    processed_container.subheader("789")


