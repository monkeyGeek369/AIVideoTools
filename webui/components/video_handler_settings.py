from streamlit.delta_generator import DeltaGenerator
import streamlit as st
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips,vfx
from app.utils import file_utils,utils
from moviepy.video.fx.painting import painting
import numpy as np


def render_video_handler(tr,st_container:DeltaGenerator,container_dict:dict[str,DeltaGenerator]):
    # get cols
    col1,col2,col3 = st_container.columns(3)
    
    filter_options = [
        (tr("none_filter"), ""),
        (tr("warm_sun_filter"), "warm_sun"),
        (tr("setting_sun_filter"), "setting_sun"),
        (tr("sea_blue_filter"), "sea_blue"),
    ]

    video_mirror = None
    video_stylize = None
    video_filter = None
    with col1:
        video_mirror = col1.checkbox(label=tr("video_mirror"),key="video_mirror")
    with col2:
        video_stylize = col2.checkbox(label=tr("video_stylize"),key="video_stylize")
    with col3:
        video_filter_selected = col3.selectbox(
            tr("video_filter"),
            index=0,
            options=range(len(filter_options)),
            format_func=lambda x: filter_options[x][0],
        )
        video_filter = filter_options[video_filter_selected][1]

    # submit
    submit_button = st_container.button(tr("video_handler_submit"))
    if submit_button:
        with st_container:
            with st.spinner(tr("processing")):
                render_video_edit(tr,st_container,container_dict,video_mirror,video_stylize,video_filter)


def render_video_edit(tr,st_container:DeltaGenerator,container_dict:dict[str,DeltaGenerator],video_mirror:bool,video_stylize:bool,video_filter:str):
    try:
        video_list = []
        task_path = st.session_state['task_path']
        video_fps = None

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
        video_fps = material_videos[0].fps
        final_clip = concatenate_videoclips(material_videos)

        # mirror
        if video_mirror:
            final_clip = final_clip.fx(vfx.mirror_x)

        # stylize
        if video_stylize:
            final_clip = final_clip.fx(painting)

        # filter
        if video_filter and len(video_filter) > 0:
            final_clip = video_filter_handler(final_clip,video_filter)

        # save
        edit_videos_path = os.path.join(task_path, "edit_videos")
        file_utils.ensure_directory(edit_videos_path)
        final_clip_path = os.path.join(edit_videos_path, "edit_video.mp4")
        final_clip.write_videofile(
            filename=final_clip_path,
            codec="libx264",
            fps=video_fps,
            bitrate="5000k",
            preset="medium",
            remove_temp=True,
            threads=8
        )

        # show
        container_dict["edit_video_expander"].video(final_clip_path, format="video/mp4")
        st.session_state['edit_video_path'] = final_clip_path
        
    except Exception as e:
        st_container.error(e)

def video_filter_handler(clip:VideoFileClip,video_filter:str):
    if video_filter == "warm_sun":
        clip = clip.fl_image(warm_sun_filter)
    if video_filter == "setting_sun":
        clip = clip.fl_image(setting_sun_filter)
    if video_filter == "sea_blue":
        clip = clip.fl_image(sea_blue_filter)
    return clip

def warm_sun_filter(frame):
    frame_copy = frame.copy()
    frame_copy[:, :, 0] = np.clip(frame_copy[:, :, 0] * 1.15, 0, 255) # 红色
    frame_copy[:, :, 1] = np.clip(frame_copy[:, :, 1] * 1.02, 0, 255) # 绿色
    frame_copy[:, :, 2] = np.clip(frame_copy[:, :, 2] * 0.85, 0, 255) # 蓝色
    return frame_copy

def setting_sun_filter(frame):
    frame_copy = frame.copy()
    frame_copy[:, :, 0] = np.clip(frame_copy[:, :, 0] * 1.2, 0, 255) # 红色
    frame_copy[:, :, 1] = np.clip(frame_copy[:, :, 1] * 1, 0, 255) # 绿色
    frame_copy[:, :, 2] = np.clip(frame_copy[:, :, 2] * 0.9, 0, 255) # 蓝色
    return frame_copy

def sea_blue_filter(frame):
    frame_copy = frame.copy()
    frame_copy[:, :, 0] = np.clip(frame_copy[:, :, 0] * 0.8, 0, 255) # 红色
    frame_copy[:, :, 1] = np.clip(frame_copy[:, :, 1] * 1.03, 0, 255) # 绿色
    frame_copy[:, :, 2] = np.clip(frame_copy[:, :, 2] * 1.21, 0, 255) # 蓝色
    return frame_copy
