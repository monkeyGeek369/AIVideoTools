from streamlit.delta_generator import DeltaGenerator
import streamlit as st
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips,vfx
from app.utils import file_utils,utils


def render_video_handler(tr,st_container:DeltaGenerator,container_dict:dict[str,DeltaGenerator]):
    # get cols
    col1,col2 = st_container.columns(2)

    video_mirror = None
    video_stylize = None
    with col1:
        video_mirror = col1.checkbox(label=tr("video_mirror"),key="video_mirror")
    with col2:
        video_stylize = col2.checkbox(label=tr("video_stylize"),key="video_stylize")

    # submit
    submit_button = st_container.button(tr("video_handler_submit"))
    if submit_button:
        with st_container:
            with st.spinner(tr("processing")):
                render_video_edit(tr,st_container,container_dict,video_mirror,video_stylize)


def render_video_edit(tr,st_container:DeltaGenerator,container_dict:dict[str,DeltaGenerator],video_mirror:bool,video_stylize:bool):
    try:
        video_list = []
        task_path = st.session_state['task_path']

        # get all materials videos
        material_videos_path = os.path.join(task_path, "material_videos")
        if not os.path.exists(material_videos_path):
            raise Exception(tr("material_videos_not_exist"))
        for video in os.listdir(material_videos_path):
            if video.endswith(".mp4") or video.endswith(".webm"):
                video_list.append(os.path.join(material_videos_path,video))
        if not video_list or len(video_list) == 0:
            raise Exception(tr("material_videos_not_exist"))

        # merge all videos
        material_videos = []
        for video in video_list:
            material_videos.append(VideoFileClip(video))
        final_clip = concatenate_videoclips(material_videos)

        # mirror
        if video_mirror:
            final_clip = final_clip.fx(vfx.mirror_x)

        # stylize
        if video_stylize:
            pass

        # save
        edit_videos_path = os.path.join(task_path, "edit_videos")
        file_utils.ensure_directory(edit_videos_path)
        final_clip_path = os.path.join(edit_videos_path, "edit_video.mp4")
        final_clip.write_videofile(final_clip_path)

        # show
        container_dict["edit_video_expander"].video(final_clip_path, format="video/mp4")
        
    except Exception as e:
        st_container.error(e)




