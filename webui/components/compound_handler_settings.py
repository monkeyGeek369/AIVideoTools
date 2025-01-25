from streamlit.delta_generator import DeltaGenerator
import streamlit as st
import os
from moviepy.editor import VideoFileClip,AudioFileClip,CompositeAudioClip
from app.utils import file_utils,utils



def render_compound_handler(tr,st_container:DeltaGenerator,container_dict:dict[str,DeltaGenerator]):
    col1,col2,col3 = st_container.columns(3)

    bg_music_check,voice_check,subtitle_check = None,None,None
    with col1:
        bg_music_check = col1.checkbox(label=tr("compound_bg_music"),key="compound_bg_music",value=True)
    with col2:
        voice_check = col2.checkbox(label=tr("compound_voice"),key="compound_voice",value=True)
    with col3:
        subtitle_check = col3.checkbox(label=tr("compound_subtitle"),key="compound_subtitle",value=True)

    submit_container = st_container.container()
    submit_button = submit_container.button(label=tr("compound_handler_submit"),key="compound_handler_submit")
    if submit_button:
        with submit_container:
            with st.spinner(text=tr("processing")):
                try:
                    compound_video(tr,bg_music_check,voice_check,subtitle_check,container_dict)
                except Exception as e:
                    submit_container.error(e)
    
def compound_video(tr,bg_music_check:bool,voice_check:bool,subtitle_check:bool,container_dict:dict[str,DeltaGenerator]):
    try:
        task_path = st.session_state['task_path']

        # get edit video
        edit_video_path = st.session_state['edit_video_path']
        if edit_video_path is None or os.path.exists(edit_video_path) is False:
            raise Exception(tr("edit_video_not_found"))
        
        video_clip = VideoFileClip(edit_video_path)
        audio_clips = []
        
        # bg music
        if bg_music_check:
            edit_bg_musics_path = st.session_state['edit_bg_musics_path']
            if edit_bg_musics_path is None or os.path.exists(edit_bg_musics_path) is False:
                raise Exception(tr("edit_bg_music_not_found"))
            
            audio_clip = AudioFileClip(edit_bg_musics_path)
            if audio_clip.duration < video_clip.duration:
                audio_clip = audio_clip.set_duration(video_clip.duration).loop(duration=video_clip.duration)
            elif audio_clip.duration > video_clip.duration:
                audio_clip = audio_clip.set_duration(video_clip.duration)
            audio_clips.append(audio_clip)
        
        # voice
        if voice_check:
            edit_voice_path = st.session_state['edit_voice_path']
            if edit_voice_path is None or os.path.exists(edit_voice_path) is False:
                raise Exception(tr("edit_voice_not_found"))
            voice_clip = AudioFileClip(edit_voice_path)
            audio_clips.append(voice_clip)
        
        # subtitle
        if subtitle_check:
            pass

        # save
        mix_audio_clip = CompositeAudioClip(audio_clips)
        video_clip = video_clip.set_audio(mix_audio_clip)
        compound_videos_path = os.path.join(task_path, "compound_videos")
        file_utils.ensure_directory(compound_videos_path)
        final_clip_path = os.path.join(compound_videos_path, "compound_video.mp4")
        video_clip.write_videofile(final_clip_path)

        # show
        container_dict["compound_video_expander"].video(final_clip_path, format="video/mp4")
        st.session_state['compound_video_path'] = final_clip_path
        # download compound video
        with open(final_clip_path, "rb") as video_file:
            video_bytes = video_file.read()
        container_dict["compound_video_expander"].download_button(label=tr("download_compound_video"),data=video_bytes, file_name="compound_video.mp4",mime="video/mp4")

    except Exception as e:
        raise e
    finally:
        if video_clip is not None:
            video_clip.close()
        for audio_clip in audio_clips:
            if audio_clip is not None:
                audio_clip.close()
        if mix_audio_clip is not None:
            mix_audio_clip.close()
