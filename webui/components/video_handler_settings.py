from streamlit.delta_generator import DeltaGenerator
import streamlit as st
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips,vfx
from app.utils import file_utils
from moviepy.video.fx.painting import painting
import numpy as np
from app.services import audio,localhost_llm


def render_video_handler(tr,st_container:DeltaGenerator,container_dict:dict[str,DeltaGenerator]):
    # get cols
    col1,col2,col3,col4,col5 = st_container.columns(5)
    
    filter_options = [
        (tr("none_filter"), ""),
        (tr("warm_sun_filter"), "warm_sun"),
        (tr("setting_sun_filter"), "setting_sun"),
        (tr("sea_blue_filter"), "sea_blue"),
    ]

    video_mirror = None
    video_stylize = None
    video_filter = None
    video_audio_vfx = None
    video_title = None
    with col1:
        video_mirror = col1.checkbox(label=tr("video_mirror"),key="video_mirror")
    with col2:
        video_stylize = col2.checkbox(label=tr("video_stylize"),key="video_stylize")
    with col3:
        video_filter_selected = col3.radio(
            label=tr("video_filter"),
            options=[item[0] for item in filter_options],
            index=3,
            key="video_filter_radio"
            )
        video_filter = [item for item in filter_options if item[0] == video_filter_selected][0][1]
    with col4:
        video_audio_vfx = col4.checkbox(label=tr("video_audio_vfx"),key="video_audio_vfx")
    with col5:
        video_title = col5.checkbox(label=tr("video_title_polish"),key="video_title_polish",value=True)

    # submit
    submit_button = st_container.button(tr("video_handler_submit"))
    if submit_button:
        with st_container:
            with st.spinner(tr("processing")):
                render_video_edit(tr,st_container,container_dict,video_mirror,video_stylize,video_filter,video_audio_vfx,video_title)
                st_container.success(tr("video_edit_success"))


def render_video_edit(tr,st_container:DeltaGenerator,container_dict:dict[str,DeltaGenerator],video_mirror:bool,video_stylize:bool,video_filter:str,video_audio_vfx:bool,video_title:bool):
    
    video_list = []
    task_path = st.session_state['task_path']
    video_fps = None
    material_videos = []
    final_clip = None

    try:
        # video title polish
        if video_title:
            video_title_polish()

        # path init
        edit_videos_path = os.path.join(task_path, "edit_videos")
        file_utils.ensure_directory(edit_videos_path)
        final_clip_path = os.path.join(edit_videos_path, "edit_video.mp4")             
        temp_audio_path = os.path.join(task_path, "temp", "edit-audio.aac")
        frame_path = os.path.join(task_path, "frame")
        file_utils.ensure_directory(frame_path)

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

        # audio vfx
        if video_audio_vfx:
            final_clip = audio.audio_visualization_effect(final_clip)

        # save
        final_clip.write_videofile(
            final_clip_path,
            codec='libx264',
            audio_codec='aac',
            fps=video_fps,
            preset='medium',
            threads=os.cpu_count(),
            ffmpeg_params=[
                "-crf", "30",          # 控制视频“质量”，这里的质量主要是指视频的主观视觉质量，即人眼观看视频时的清晰度、细节保留程度以及压缩带来的失真程度
                "-b:v", "2000k", # 设置目标比特率，控制视频每秒数据量，与视频大小有直接关联。
                "-pix_fmt", "yuv420p",#指定像素格式。yuv420p 是一种常见的像素格式，兼容性较好，适用于大多数播放器。
                "-row-mt", "1"#启用行级多线程，允许编码器在单帧内并行处理多行数据，从而提高编码效率。0表示不启用
            ],
            write_logfile=False, #是否写入日志
            remove_temp=True,#是否删除临时文件
            temp_audiofile=temp_audio_path  #指定音频的临时文件路径
        )

        # show
        container_dict["edit_video_expander"].video(final_clip_path, format="video/mp4")
        st.session_state['edit_video_path'] = final_clip_path
        
    except Exception as e:
        st_container.error(e)
    finally:
        del video_list
        del video_fps
        del material_videos
        del final_clip
        

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

def video_title_polish():
    # get llm params
    llm_url = st.session_state['llm_url']
    llm_api_key = st.session_state['llm_api_key']
    llm_model = st.session_state['llm_model']
    llm_temperature = st.session_state['llm_temperature']
    llm_result = localhost_llm.chat_single_content(base_url=llm_url,
                                                api_key=llm_api_key,
                                                model=llm_model,
                                                prompt='''
你现在是一名视频标题润色专家,需要对给到的视频标题进行润色.有如下要求:
1、润色后的标题含义与原标题含义相同但文字表达不同.
2、润色后的标题字数不能超过原视频字数的1.5倍.
3、润色后的标题一定要朗读通顺、无错别字.
4、直接输出标题,不需要额外字符包裹.
5、整体字数绝对不能超过100字.
''',
                                                content=st.session_state['first_video_name'],
                                                temperature=llm_temperature)
    if llm_result:
        st.session_state['video_polish_name'] = llm_result

