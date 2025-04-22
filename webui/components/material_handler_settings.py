from streamlit.delta_generator import DeltaGenerator
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from app.utils import file_utils,utils
from loguru import logger
import os,torch,gc
from moviepy.editor import VideoFileClip
from app.services import subtitle,video,audio
from app.models.file_info import LocalFileInfo
from app.models.subtitle_position_coord import SubtitlePositionCoord


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
    subtitle_position_recognize_checkbox_value = None
    ignore_subtitle_area = None
    min_subtitle_merge_distance = None

    # create checkbox
    split_container=material_handler_form.container()
    column1,column2,column3,column4 = split_container.columns(4)
    with column1:
        video_split_checkbox_value = column1.checkbox(label=tr("video_split"),key="video_split",value=True)
    with column2:
        voice_split_checkbox_value = column2.checkbox(label=tr("voice_split"),key="voice_split",value=True)
    with column3:
        subtitle_split_checkbox_value = column3.checkbox(label=tr("subtitle_split"),key="subtitle_split",value=True)
    with column4:
        subtitle_position_recognize_checkbox_value = column4.checkbox(label=tr("subtitle_position_recognize"),key="subtitle_position_recognize",value=True)
        ignore_subtitle_area = column4.text_input(label=tr("ignore_subtitle_area"),key="ignore_subtitle_area",value=500)
        min_subtitle_merge_distance = column4.text_input(label=tr("min_subtitle_merge_distance"),key="min_subtitle_merge_distance",value=100)

    # create submit button
    submitted = material_handler_form.form_submit_button(label=tr("material_handler_submit"))
    with split_container:
        # print(subtitle_mosaic_checkbox_value)
        # print(subtitle_identification_mode_value.split(" - ")[0])
        # print(mosaic_neighbor_value)
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
                                                      container_dict,
                                                      subtitle_position_recognize_checkbox_value,
                                                      ignore_subtitle_area,
                                                      min_subtitle_merge_distance)
                    st.success(tr("material_handler_submit_success"))
                except Exception as e:
                    logger.error(tr("material_handler_submit_error")+": "+str(e))
                    st.error(tr("material_handler_submit_error")+": "+str(e))
                finally:
                    # clear cache
                    torch.cuda.empty_cache()
                    gc.collect()

def save_uploaded_origin_videos(videos:list[UploadedFile]):
    # get task path
    task_path = st.session_state['task_path']
    origin_videos = os.path.join(task_path, "origin_videos")

    # cleanup file
    file_utils.cleanup_temp_files(temp_dir=origin_videos)

    # save videos
    st.session_state['first_video_name'] = videos[0].name.split(".")[0]
    for video in videos:
        file_utils.save_uploaded_file(uploaded_file=video,save_dir=origin_videos,allowed_types=['.mp4','.webm'])

def split_material_from_origin_videos(split_videos:bool,split_voices:bool,split_subtitles:bool,split_bg_musics:bool,container_dict:dict[str,DeltaGenerator],
                                      subtitle_position_recognize:bool,ignore_subtitle_area:int,min_subtitle_merge_distance:int):
    # get task path
    task_path = st.session_state['task_path']

    # all path
    origin_videos = os.path.join(task_path, "origin_videos")
    material_videos_path = os.path.join(task_path, "material_videos")
    utils.create_dir(material_videos_path)
    material_voices_path = os.path.join(task_path, "material_voices")
    utils.create_dir(material_voices_path)
    material_subtitles_path = os.path.join(task_path, "material_subtitles")
    utils.create_dir(material_subtitles_path)
    material_bg_musics_path = os.path.join(task_path, "material_bg_musics")
    utils.create_dir(material_bg_musics_path)

    # get all origin videos
    origin_videos = file_utils.get_file_list(directory=origin_videos)
    if not origin_videos:
        raise Exception("origin videos is empty")

    # split origin videos
    for origin_video in origin_videos:
        if split_videos:
            video = VideoFileClip(origin_video.path)
            video = video.without_audio()                  
            temp_audio_path = os.path.join(task_path, "temp", "material-audio.aac")
            video.write_videofile(
                os.path.join(material_videos_path,origin_video.name+".mp4"),
                codec='libx264',
                audio_codec='aac',
                fps=video.fps,
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
            st.session_state['video_height'] = video.h
            st.session_state['video_width'] = video.w
            video.close()
        audio_file_path = os.path.join(material_voices_path,origin_video.name+".wav")
        subtitle_file_path = os.path.join(material_subtitles_path,origin_video.name+".srt")
        if split_voices:
            audio.get_audio_from_video(origin_video.path,audio_file_path)
        if split_subtitles:
            if not os.path.exists(audio_file_path):
                audio.get_audio_from_video(origin_video.path,audio_file_path)
            subtitle.create(audio_file_path, subtitle_file_path)
        if split_bg_musics:
            pass
        if subtitle_position_recognize:
            file_name = origin_video.name+".mp4"
            video_path = os.path.join(material_videos_path,file_name)
            recognize_subtitle_position(video_path,int(ignore_subtitle_area),int(min_subtitle_merge_distance))
        if split_subtitles and os.path.exists(subtitle_file_path):
            subtitle.remove_valid_subtitles_by_ocr(subtitle_path=subtitle_file_path)

    # show material
    show_materials(container_dict["material_video_expander"],container_dict["material_bg_music_expander"],
                   container_dict["material_voice_expander"],container_dict["material_subtitle_expander"],
                   container_dict["subtitle_position_expander"])

def split_audio(material_voices_path:str,origin_video:LocalFileInfo):
    utils.create_dir(material_voices_path)
    audio.get_audio_from_video(origin_video.path,os.path.join(material_voices_path,origin_video.name+".wav"))

def show_materials(video_container:DeltaGenerator,bg_music_container:DeltaGenerator,voice_container:DeltaGenerator,
                   subtitle_container:DeltaGenerator,subtitle_position:DeltaGenerator):
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
    
    # show subtitle position
    subtitle_position_dict = st.session_state.get('subtitle_position_dict', {})
    for file_name,coord in subtitle_position_dict.items():
        subtitle_position.text_area(
            file_name,
            value=utils.to_json(coord),
            height=150,
            label_visibility="collapsed",
            key=file_name
        )


def recognize_subtitle_position(video_path:str,ignore_subtitle_area:int,min_subtitle_merge_distance:int):
    # get subtitle position
    position_dict = video.video_subtitle_overall_statistics(video_path,ignore_subtitle_area,min_subtitle_merge_distance)

    coord = None
    if position_dict:
        coord = SubtitlePositionCoord(
            is_exist=True,
            left_top_x=position_dict["left_top_x"],
            left_top_y=position_dict["left_top_y"],
            right_bottom_x=position_dict["right_bottom_x"],
            right_bottom_y=position_dict["right_bottom_y"],
            count=position_dict["count"],
            frame_subtitles_position=position_dict["frame_subtitles_position"]
            )
    else:
        coord = SubtitlePositionCoord(is_exist=False)

    # save subtitle position
    subtitle_position_dict = st.session_state.get('subtitle_position_dict', {})
    subtitle_position_dict["edit_video.mp4"] = coord
    st.session_state['subtitle_position_dict'] = subtitle_position_dict

