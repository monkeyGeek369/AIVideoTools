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
    ai_container.subheader("456")

    # processing handler
    processed_container.subheader("789")


