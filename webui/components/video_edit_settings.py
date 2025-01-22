from streamlit.delta_generator import DeltaGenerator
import streamlit as st
import os


def render_video_edit(tr,st_container:DeltaGenerator) -> dict[str,DeltaGenerator]:
    result = {}
    task_path = st.session_state['task_path']

    # material video expander
    edit_video_expander = st_container.expander(label=tr("edit_video"),expanded=True)
    edit_video_expander.write(tr("edit_video_tips"))
    result['edit_video_expander'] = edit_video_expander

    # material bg music expander 
    #edit_bg_music_expander = st_container.expander(label=tr("edit_bg_music"),expanded=True)
    #edit_bg_music_expander.write(tr("edit_bg_music_tips"))
    #result['edit_bg_music_expander'] = edit_bg_music_expander

    # material voice expander
    edit_voice_expander = st_container.expander(label=tr("edit_voice"),expanded=True)
    edit_voice_expander.write(tr("edit_voice_tips"))
    result['edit_voice_expander'] = edit_voice_expander
    edit_voices_path = os.path.join(task_path, "edit_voices","edit_audio.mp3")
    if os.path.exists(edit_voices_path):
        edit_voice_expander.audio(edit_voices_path, format="audio/mp3")

    # material subtitle expander
    edit_subtitle_expander = st_container.expander(label=tr("edit_subtitle"),expanded=True)
    edit_subtitle_expander.write(tr("edit_subtitle_tips"))
    result['edit_subtitle_expander'] = edit_subtitle_expander
    output_subtitle_path = os.path.join(task_path, "edit_subtitles", "merged.srt")
    if os.path.exists(output_subtitle_path):
        with open(output_subtitle_path, 'r', encoding='utf-8') as f:
            subtitle_content = f.read()
            edit_subtitle_expander.text_area(
                "merged.srt",
                value=subtitle_content,
                height=150,
                label_visibility="collapsed",
                on_change=None
            )

    return result
