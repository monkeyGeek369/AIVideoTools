from streamlit.delta_generator import DeltaGenerator
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from app.utils import file_utils
from loguru import logger
import os,time


def render_material_handler(tr,st_container:DeltaGenerator):
    # create form
    material_handler_form = st_container.form(key="material_handler_form")

    # create upload file
    uploaded_origin_videos = material_handler_form.file_uploader(
            tr("upload_origin_video"),
            type=["mp4", "webm"],
            accept_multiple_files=True,
            key="origin_videos",
    )

    # create checkbox obj
    video_split_checkbox_value = None
    voice_split_checkbox_value = None
    subtitle_split_checkbox_value = None
    bg_music_split_checkbox_value = None

    # create checkbox
    split_container=material_handler_form.container()
    column1,column2,column3,column4 = split_container.columns(4)
    with column1:
        video_split_checkbox_value = column1.checkbox(label=tr("video_split"),key="video_split")
    with column2:
        voice_split_checkbox_value = column2.checkbox(label=tr("voice_split"),key="voice_split")
    with column3:
        subtitle_split_checkbox_value = column3.checkbox(label=tr("subtitle_split"),key="subtitle_split")
    with column4:
        bg_music_split_checkbox_value = column4.checkbox(label=tr("bg_music_split"),key="bg_music_split")

    # create submit button
    submitted = material_handler_form.form_submit_button(label=tr("material_handler_submit"))
    with split_container:
        with st.spinner(tr("processing")):
            if submitted:
                if not uploaded_origin_videos:
                    logger.error(tr("upload_origin_video_is_empty"))
                    st.error(tr("upload_origin_video_is_empty"))
                    return
                
                try:
                    # save uploaded origin videos
                    save_uploaded_origin_videos(uploaded_origin_videos)
                    
                    # split videos
                    if video_split_checkbox_value:
                        split_videos()
                    if voice_split_checkbox_value:
                        split_voices()
                    if subtitle_split_checkbox_value:
                        split_subtitles()
                    if bg_music_split_checkbox_value:
                        split_bg_musics()
                    st.success(tr("material_handler_submit_success"))
                except Exception as e:
                    logger.error(tr("material_handler_submit_error")+": "+str(e))
                    st.error(tr("material_handler_submit_error")+": "+str(e))

def save_uploaded_origin_videos(videos:list[UploadedFile]):
    # get task path
    task_path = st.session_state['task_path']
    origin_videos = os.path.join(task_path, "origin_videos")

    # cleanup file
    file_utils.cleanup_temp_files(temp_dir=origin_videos)

    # save videos
    for video in videos:
        file_utils.save_uploaded_file(uploaded_file=video,save_dir=origin_videos,allowed_types=['.mp4','.webm'])

def split_videos():
    pass

def split_voices():
    pass

def split_subtitles():
    pass

def split_bg_musics():
    pass


