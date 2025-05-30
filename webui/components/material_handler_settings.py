from streamlit.delta_generator import DeltaGenerator
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from app.utils import file_utils,utils
from loguru import logger
import os,torch,gc,re
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
    sub_rec_area = None
    subtitle_ocr_filter_checkbox_value = None
    subtitle_auto_mosaic_checkbox_value = None

    # create checkbox
    split_container=material_handler_form.container()
    column1,column2,column3,column4 = split_container.columns(4)
    with column1:
        video_split_checkbox_value = column1.checkbox(label=tr("video_split"),key="video_split",value=True)
    with column2:
        voice_split_checkbox_value = column2.checkbox(label=tr("voice_split"),key="voice_split",value=True)
    with column3:
        subtitle_split_checkbox_value = column3.checkbox(label=tr("subtitle_split"),key="subtitle_split",value=True)
        subtitle_ocr_filter_checkbox_value = column3.checkbox(label=tr("subtitle_ocr_filter"),key="subtitle_ocr_filter",value=True)
    with column4:
        subtitle_position_recognize_checkbox_value = column4.checkbox(label=tr("subtitle_position_recognize"),key="subtitle_position_recognize",value=True)
        sub_rec_params_container,sub_area_container = column4.columns(2)
        with sub_rec_params_container:
            ignore_subtitle_area = sub_rec_params_container.text_input(label=tr("ignore_subtitle_area"),key="ignore_subtitle_area",value=5000)
            min_subtitle_merge_distance = sub_rec_params_container.text_input(label=tr("min_subtitle_merge_distance"),key="min_subtitle_merge_distance",value=100)
        with sub_area_container:
            sub_rec_area_options = [
                (tr("full_area"), "full_area"),
                (tr("upper_part_area"), "upper_part_area"),
                (tr("lower_part_area"), "lower_part_area"),
            ]
            sub_rec_area_selected = sub_area_container.radio(label=tr("subtitle_recognize_area"),options=[item[0] for item in sub_rec_area_options],index=0,key="subtitle_recognize_area")
            subtitle_auto_mosaic_checkbox_value = sub_area_container.checkbox(label=tr("subtitle_auto_mosaic"),key="subtitle_auto_mosaic",value=True)
            sub_rec_area = [item for item in sub_rec_area_options if item[0] == sub_rec_area_selected][0][1]

    # create submit button
    submitted = material_handler_form.form_submit_button(label=tr("material_handler_submit"))
    with split_container:
        #print(sub_rec_area)
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
                                                      min_subtitle_merge_distance,
                                                      sub_rec_area,
                                                      subtitle_ocr_filter_checkbox_value,
                                                      subtitle_auto_mosaic_checkbox_value)
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
    video_name = video.remove_video_titile_spe_chars(videos[0].name.split(".")[0])
    st.session_state['first_video_name'] = video_name
    for video_item in videos:
        file_utils.save_uploaded_file(uploaded_file=video_item,save_dir=origin_videos,allowed_types=['.mp4','.webm'])

def split_material_from_origin_videos(split_videos:bool,split_voices:bool,split_subtitles:bool,split_bg_musics:bool,container_dict:dict[str,DeltaGenerator],
                                      subtitle_position_recognize:bool,ignore_subtitle_area:int,min_subtitle_merge_distance:int,sub_rec_area:str,
                                      subtitle_ocr_filter:bool,
                                      subtitle_auto_mosaic_checkbox_value:bool):
    # get task path
    task_path = st.session_state['task_path']
    recognize_position_model = None
    video_path = None
    temp_video_path = None
    video_name = None

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
    video_duration = 0
    for origin_video in origin_videos:
        video_name = video.remove_video_titile_spe_chars(origin_video.name)
        if split_videos:
            video_clip = VideoFileClip(origin_video.path)
            video_clip = video_clip.without_audio()
            video_path = os.path.join(material_videos_path,video_name+".mp4")
            temp_video_path = os.path.join(material_videos_path,video_name+"_temp.mp4")
            video.video_clip_to_video(video_clip,video_path,video_clip.fps)
            st.session_state['video_height'] = video_clip.h
            st.session_state['video_width'] = video_clip.w
            st.session_state['video_fps'] = video_clip.fps
            video_duration += video_clip.duration
            video_clip.close()
        audio_file_path = os.path.join(material_voices_path,video_name+".wav")
        subtitle_file_path = os.path.join(material_subtitles_path,video_name+".srt")
        if split_voices:
            audio.get_audio_from_video(origin_video.path,audio_file_path)
        if split_subtitles:
            if not os.path.exists(audio_file_path):
                audio.get_audio_from_video(origin_video.path,audio_file_path)
            subtitle.create(audio_file_path, subtitle_file_path)
        if split_bg_musics:
            pass
        if subtitle_position_recognize:
            recognize_position_model = recognize_subtitle_position(video_path,int(ignore_subtitle_area),int(min_subtitle_merge_distance),sub_rec_area)
        if split_subtitles and os.path.exists(subtitle_file_path) and subtitle_ocr_filter:
            subtitle.remove_valid_subtitles_by_ocr(subtitle_path=subtitle_file_path)
        if subtitle_auto_mosaic_checkbox_value:
            with VideoFileClip(video_path) as mosaic_video_clip:
                mosaic_result_clip = video.video_subtitle_mosaic_auto(
                    video_clip=mosaic_video_clip,
                    subtitle_position_coord=recognize_position_model
                )
                
                video.video_clip_to_video(mosaic_result_clip, temp_video_path,mosaic_video_clip.fps)
                mosaic_result_clip.close()
                
                if os.path.exists(temp_video_path) and os.path.getsize(temp_video_path) > 0:
                    os.remove(video_path)
                    os.rename(temp_video_path, video_path)

    # add data
    st.session_state['video_duration'] = video_duration

    # show material
    show_materials(container_dict["material_video_expander"],container_dict["material_bg_music_expander"],
                   container_dict["material_voice_expander"],container_dict["material_subtitle_expander"],
                   container_dict["subtitle_position_expander"])

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


def recognize_subtitle_position(video_path:str,ignore_subtitle_area:int,min_subtitle_merge_distance:int,sub_rec_area:str):
    # get subtitle position
    position_dict = video.video_subtitle_overall_statistics(video_path,ignore_subtitle_area,min_subtitle_merge_distance,sub_rec_area)

    coord = None
    if position_dict:
        coord = SubtitlePositionCoord(
            is_exist=True,
            left_top_x=position_dict["left_top_x"],
            left_top_y=position_dict["left_top_y"],
            right_bottom_x=position_dict["right_bottom_x"],
            right_bottom_y=position_dict["right_bottom_y"],
            count=position_dict["count"],
            frame_subtitles_position=position_dict["frame_subtitles_position"],
            frame_time_text_dict=position_dict["frame_time_text_dict"]
            )
    else:
        coord = SubtitlePositionCoord(is_exist=False)

    # save subtitle position
    subtitle_position_dict = st.session_state.get('subtitle_position_dict', {})
    subtitle_position_dict["edit_video.mp4"] = coord
    st.session_state['subtitle_position_dict'] = subtitle_position_dict

    return coord

