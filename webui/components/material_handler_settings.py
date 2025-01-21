from streamlit.delta_generator import DeltaGenerator
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from app.utils import file_utils,utils
from loguru import logger
import os,time
from moviepy.editor import VideoFileClip
from moviepy.editor import AudioFileClip
from app.services import subtitle
from app.models.file_info import LocalFileInfo


def render_material_handler(tr,st_container:DeltaGenerator,container_dict:dict[str,DeltaGenerator]):
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
        pass
        #bg_music_split_checkbox_value = column4.checkbox(label=tr("bg_music_split"),key="bg_music_split",disabled=False)

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
                    split_material_from_origin_videos(video_split_checkbox_value,
                                                      voice_split_checkbox_value,
                                                      subtitle_split_checkbox_value,
                                                      bg_music_split_checkbox_value,
                                                      container_dict)
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

def split_material_from_origin_videos(split_videos:bool,split_voices:bool,split_subtitles:bool,split_bg_musics:bool,container_dict:dict[str,DeltaGenerator]):
    # get task path
    task_path = st.session_state['task_path']

    # all path
    origin_videos = os.path.join(task_path, "origin_videos")
    material_videos_path = os.path.join(task_path, "material_videos")
    material_voices_path = os.path.join(task_path, "material_voices")
    material_subtitles_path = os.path.join(task_path, "material_subtitles")
    material_bg_musics_path = os.path.join(task_path, "material_bg_musics")

    # get all origin videos
    origin_videos = file_utils.get_file_list(directory=origin_videos)
    if not origin_videos:
        raise Exception("origin videos is empty")

    # split origin videos
    for origin_video in origin_videos:
        if split_videos:
            utils.create_dir(material_videos_path)
            video = VideoFileClip(origin_video.path)
            video = video.without_audio()
            video.write_videofile(
                filename=os.path.join(material_videos_path,origin_video.name+".mp4"),
                codec="libx264",
                fps=video.fps,
                bitrate="5000k",
                preset="medium",
                remove_temp=True,
                audio=False,
                threads=8
                )
            video.close()
        if split_voices:
            split_video(material_voices_path,origin_video)
        if split_subtitles:
            utils.create_dir(material_subtitles_path)
            audio_file = os.path.join(material_voices_path,origin_video.name+".wav")
            if not os.path.exists(audio_file):
                split_video(material_voices_path,origin_video)
            subtitle_file = os.path.join(material_subtitles_path,origin_video.name+".srt")
            # create srt
            subtitle.create(audio_file, subtitle_file)
        if split_bg_musics:
            utils.create_dir(material_bg_musics_path)
            audio_file = os.path.join(material_voices_path,origin_video.name+".wav")
            if not os.path.exists(audio_file):
                split_video(material_voices_path,origin_video)
            pass

    # show material
    show_materials(container_dict["material_video_expander"],container_dict["material_bg_music_expander"],container_dict["material_voice_expander"],container_dict["material_subtitle_expander"])

def split_video(material_voices_path:str,origin_video:LocalFileInfo):
    utils.create_dir(material_voices_path)
    audio_clip = AudioFileClip(origin_video.path)
    fps = audio_clip.fps
    audio_clip.write_audiofile(
        filename=os.path.join(material_voices_path,origin_video.name+".wav"),
        fps=fps
        )
    audio_clip.close()

def show_materials(video_container:DeltaGenerator,bg_music_container:DeltaGenerator,voice_container:DeltaGenerator,subtitle_container:DeltaGenerator):
    # get task path
    task_path = st.session_state['task_path']

    # show video
    material_videos_path = os.path.join(task_path, "material_videos")
    material_videos = file_utils.get_file_list(directory=material_videos_path)
    for material_video in material_videos:
        video_container.video(material_video.path)

    # show bg music
    #material_bg_musics_path = os.path.join(task_path, "material_bg_musics")
    #material_bg_musics = file_utils.get_file_list(directory=material_bg_musics_path)
    #for material_bg_music in material_bg_musics:
        #bg_music_container.audio(material_bg_music.path)

    # show voice
    material_voices_path = os.path.join(task_path, "material_voices")
    material_voices = file_utils.get_file_list(directory=material_voices_path)
    for material_voice in material_voices:
        voice_container.audio(material_voice.path)

    # show subtitle
    material_subtitles_path = os.path.join(task_path, "material_subtitles")
    material_subtitles = file_utils.get_file_list(directory=material_subtitles_path)
    for material_subtitle in material_subtitles:
        with open(material_subtitle.path, 'r', encoding='utf-8') as f:
            subtitle_content = f.read()
            subtitle_container.text_area(
                material_subtitle.name,
                value=subtitle_content,
                height=150,
                label_visibility="collapsed",
                key=material_subtitle.name
            )



