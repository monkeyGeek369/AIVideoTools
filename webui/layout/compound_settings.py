from streamlit.delta_generator import DeltaGenerator
import streamlit as st
import os


def render_compound_settings(tr,st_container:DeltaGenerator) -> dict[str,DeltaGenerator]:
    result = {}
    task_path = st.session_state['task_path']

    # compound expander
    compound_video_expander = st_container.expander(label=tr("compound_video"),expanded=True)
    compound_video_expander.write(tr("compound_video_tips"))
    result['compound_video_expander'] = compound_video_expander

    compound_video_path = os.path.join(task_path, "compound_videos","compound_video.mp4")
    if os.path.exists(compound_video_path):
        compound_video_expander.video(compound_video_path, format="video/mp4")
        
        # download compound video
        with open(compound_video_path, "rb") as video_file:
            video_bytes = video_file.read()
        compound_video_expander.download_button(label=tr("download_compound_video"),data=video_bytes, file_name="compound_video.mp4",mime="video/mp4")


    return result
