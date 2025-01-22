from streamlit.delta_generator import DeltaGenerator
from app.services import voice
import streamlit as st
from app.utils import utils,file_utils
import os


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
            index=1,
        )

        # get voice name
        voice_name = list(friendly_names.keys())[
            list(friendly_names.values()).index(selected_friendly_name)
        ]
        st.session_state['voice_name'] = voice_name

        if voice.is_azure_v2_voice(voice_name):
            raise Exception(tr("v2_voice_error"))

        # voice params settings
        voice_volume = col3.slider(
            tr("speech_volume"),
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
            help=tr("speech_volume_help")
        )
        st.session_state['voice_volume'] = voice_volume


        # speech rate
        voice_rate = col4.selectbox(
            tr("speech_rate"),
            options=[0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0],
            index=2,
        )
        st.session_state['voice_rate'] = voice_rate

        # speech pitch
        voice_pitch = col5.selectbox(
            tr("speech_pitch"),
            options=[0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0],
            index=2,
        )
        st.session_state['voice_pitch'] = voice_pitch

        # submit button
        bt1_col,bt2_col,bt3_col,bt4_col,bt5_col = st_container.columns(5)
        submit_button = bt1_col.button(tr("voice_handler_submit"))
        if submit_button:
            pass

        check_button = bt2_col.button(tr("voice_handler_check"))
        if check_button:
            render_voice_preview(tr, voice_name,st_container)

    except Exception as e:
        st_container.error(e)

def render_voice_preview(tr, voice_name,st_container:DeltaGenerator):
    """渲染语音试听功能"""
    play_content = "感谢关注 AIVideoTools 项目，欢迎使用语音功能。"

    with st_container:
        with st.spinner(tr("processing")):
            task_path = st.session_state['task_path']
            temp_dir = os.path.join(task_path, "temp")
            file_utils.ensure_directory(temp_dir)
            audio_file = os.path.join(temp_dir, f"tmp-voice-{utils.get_uuid()}.mp3")

            sub_maker = voice.tts(
                text=play_content,
                voice_name=voice_name,
                voice_rate=st.session_state.get('voice_rate', 1.0),
                voice_pitch=st.session_state.get('voice_pitch', 1.0),
                voice_file=audio_file,
            )

            # 如果语音文件生成失败，使用默认内容重试
            if not sub_maker:
                play_content = "This is a example voice. if you hear this, the voice synthesis failed with the original content."
                sub_maker = voice.tts(
                    text=play_content,
                    voice_name=voice_name,
                    voice_rate=st.session_state.get('voice_rate', 1.0),
                    voice_pitch=st.session_state.get('voice_pitch', 1.0),
                    voice_file=audio_file,
                )

            if sub_maker and os.path.exists(audio_file):
                st_container.audio(audio_file, format="audio/mp3")
                if os.path.exists(audio_file):
                    os.remove(audio_file)


