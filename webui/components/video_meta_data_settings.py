from streamlit.delta_generator import DeltaGenerator
import streamlit as st
from app.utils import utils,file_utils
import os

def render_video_meta_data(tr,st_container:DeltaGenerator) -> dict[str,DeltaGenerator]:
    result = {}
    task_path = st.session_state['task_path']

    # material video expander
    material_video_expander = st_container.expander(label=tr("material_video"),expanded=True)
    material_video_expander.write(tr("material_video_tips"))
    result['material_video_expander'] = material_video_expander
    material_videos_path = os.path.join(task_path, "material_videos")
    material_videos = file_utils.get_file_list(directory=material_videos_path)
    for material_video in material_videos:
        material_video_expander.video(material_video.path)

    # material bg music expander 
    #material_bg_music_expander = st_container.expander(label=tr("material_bg_music"),expanded=True)
    #material_bg_music_expander.write(tr("material_bg_music_tips"))
    result['material_bg_music_expander'] = None

    # material voice expander
    material_voice_expander = st_container.expander(label=tr("material_voice"),expanded=True)
    material_voice_expander.write(tr("material_voice_tips"))
    result['material_voice_expander'] = material_voice_expander
    material_voices_path = os.path.join(task_path, "material_voices")
    material_voices = file_utils.get_file_list(directory=material_voices_path)
    for material_voice in material_voices:
        material_voice_expander.audio(material_voice.path)

    # material subtitle expander
    material_subtitle_expander = st_container.expander(label=tr("material_subtitle"),expanded=True)
    material_subtitle_expander.write(tr("material_subtitle_tips"))
    result['material_subtitle_expander'] = material_subtitle_expander
    material_subtitles_path = os.path.join(task_path, "material_subtitles")
    material_subtitles = file_utils.get_file_list(directory=material_subtitles_path)
    for material_subtitle in material_subtitles:
        with open(material_subtitle.path, 'r', encoding='utf-8') as f:
            subtitle_content = f.read()
            material_subtitle_expander.text_area(
                material_subtitle.name,
                value=subtitle_content,
                height=150,
                label_visibility="collapsed",
                key=material_subtitle.name
            )
    
    # subtitle position expander
    subtitle_position_expander = st_container.expander(label=tr("subtitle_position"),expanded=True)
    subtitle_position_expander.write(tr("subtitle_position_tips"))
    result['subtitle_position_expander'] = subtitle_position_expander
    subtitle_position_dict = st.session_state.get('subtitle_position_dict', {})
    for file_name,coord in subtitle_position_dict.items():
        subtitle_position_expander.text_area(
            file_name,
            value=utils.to_json(coord),
            height=150,
            label_visibility="collapsed",
            key=file_name
        )


    return result




