from streamlit.delta_generator import DeltaGenerator
from app.services import voice,subtitle,audio_merger
import streamlit as st
from app.utils import utils,file_utils
import os,random


def render_voice_handler(tr,st_container:DeltaGenerator,container_dict:dict[str,DeltaGenerator]):
    col1,col2,col3,col4,col5 = st_container.columns(5)

    try:
        # speech language
        filter_locals = ["zh-CN", "en-US", "zh-HK", "zh-TW", "vi-VN"]
        speech_language_selected = col1.selectbox(
            tr("speech_language"),
            options=filter_locals,
            index=0,
        )
        st.session_state['speech_language'] = speech_language_selected

        # speech synthesis
        support_locales = ["zh-CN"]
        if speech_language_selected:
            support_locales = [speech_language_selected]
        voices = voice.get_all_azure_voices(filter_locals=support_locales)
        friendly_names = {
            v: v.replace("Female", tr("Female"))
            .replace("Male", tr("Male"))
            .replace("Neural", "")
            for v in voices
        }
        selected_friendly_name = col2.selectbox(
            tr("speech_synthesis"),
            options=list(friendly_names.values()),
            index=random.randint(0, len(friendly_names) - 1),
        )

        # get voice name
        voice_name = list(friendly_names.keys())[
            list(friendly_names.values()).index(selected_friendly_name)
        ]
        st.session_state['voice_name'] = voice_name

        if voice.is_azure_v2_voice(voice_name):
            raise Exception(tr("v2_voice_error"))

        # voice params settings
        voice_volume = col3.text_input(
            tr("speech_volume"),
            value=1.0,
            help=tr("speech_volume_help")
        )

        st.session_state['voice_volume'] = voice_volume

        # speech rate
        voice_rate = col4.text_input(
            tr("speech_rate"),
            value=1.5
        )

        st.session_state['voice_rate'] = voice_rate

        # speech pitch
        voice_pitch = col5.text_input(
            tr("speech_pitch"),
            value=1.0
        )
        st.session_state['voice_pitch'] = voice_pitch

        # submit button
        with st_container:
            with st.spinner(tr("processing")):
                bt1_col,bt2_col,bt3_col,bt4_col,bt5_col = st_container.columns(5)
                submit_button = bt1_col.button(tr("voice_handler_submit"))
                if submit_button:
                    voice_processing(tr,container_dict)
                    st_container.success(tr("voice_create_success"))

                check_button = bt2_col.button(tr("voice_handler_check"))
                if check_button:
                    render_voice_preview(tr, voice_name,st_container)

    except Exception as e:
        st_container.error(e)

def render_voice_preview(tr, voice_name,st_container:DeltaGenerator):
    """渲染语音试听功能"""
    play_content = "感谢关注 AIVideoTools 项目，欢迎使用语音功能。"

    task_path = st.session_state['task_path']
    temp_dir = os.path.join(task_path, "temp")
    file_utils.ensure_directory(temp_dir)
    audio_file = os.path.join(temp_dir, f"tmp-voice-{utils.get_uuid()}.mp3")

    sub_maker = voice.tts(
        text=play_content,
        voice_name=voice_name,
        voice_rate=float(st.session_state.get('voice_rate', 1.0)),
        voice_pitch=float(st.session_state.get('voice_pitch', 1.0)),
        voice_file=audio_file,
        voice_volume=float(st.session_state.get('voice_volume', 1.0)),
    )

    # 如果语音文件生成失败，使用默认内容重试
    if not sub_maker:
        play_content = "This is a example voice. if you hear this, the voice synthesis failed with the original content."
        sub_maker = voice.tts(
            text=play_content,
            voice_name=voice_name,
            voice_rate=float(st.session_state.get('voice_rate', 1.0)),
            voice_pitch=float(st.session_state.get('voice_pitch', 1.0)),
            voice_file=audio_file,
            voice_volume=float(st.session_state.get('voice_volume', 1.0)),
        )

    if sub_maker and os.path.exists(audio_file):
        st_container.audio(audio_file, format="audio/mp3")
        if os.path.exists(audio_file):
            os.remove(audio_file)
    del sub_maker
    del audio_file

def voice_processing(tr,container_dict:dict[str,DeltaGenerator]):
    # get subtitles
    task_path = st.session_state['task_path']
    edit_subtitles_path = os.path.join(task_path, "edit_subtitles")
    file_utils.ensure_directory(edit_subtitles_path)
    merged_subtitle_path = os.path.join(edit_subtitles_path, "merged.srt")
    if not os.path.exists(merged_subtitle_path):
        raise Exception(tr("merged_subtitle_not_found"))
    
    subtitle_texts = subtitle.file_to_subtitles(merged_subtitle_path)

    # get total duration
    last_timestamp = subtitle_texts[-1][1]
    end_time = last_timestamp.split(" --> ")[1]
    total_duration = utils.time_to_seconds(end_time)

    # get audios
    out_path = os.path.join(task_path, "temp")
    file_utils.ensure_directory(out_path)
    audio_files, sub_maker_list = voice.tts_multiple(
        out_path=out_path,
        subtitle_list=subtitle_texts, 
        voice_name=st.session_state.get('voice_name'),
        voice_rate=float(st.session_state.get('voice_rate', 1.0)),
        voice_pitch=float(st.session_state.get('voice_pitch', 1.0)),
        force_regenerate=True,
        volume=float(st.session_state.get('voice_volume', 1.0))
    )
    
    if audio_files is None:
        raise Exception(tr("tts_failed"))

    # merge audios
    final_audio = None
    try:
        out_path = os.path.join(task_path, "edit_voices")
        file_utils.ensure_directory(out_path)
        # save final audio
        final_audio = audio_merger.merge_audio_files(
            out_path=out_path, 
            audio_files=audio_files, 
            total_duration=total_duration+1, 
            subtitle_list=subtitle_texts
        )
    except Exception as e:
        raise Exception(tr("audio_merge_failed") + " : " + str(e))
    finally:
        # remove audio files
        for audio_file in audio_files:
            if os.path.exists(audio_file):
                os.remove(audio_file)

        del audio_files
        del sub_maker_list

    # show final audio
    if final_audio is not None:
        container_dict["edit_voice_expander"].audio(final_audio, format="audio/mp3")
        st.session_state['edit_voice_path'] = final_audio

