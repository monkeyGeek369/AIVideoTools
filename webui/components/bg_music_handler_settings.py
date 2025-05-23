from streamlit.delta_generator import DeltaGenerator
import streamlit as st
import os,shutil,math
from app.utils import utils,file_utils
from pydub import AudioSegment
from app.services import bg_music


def render_bg_music_handler(tr,st_container:DeltaGenerator,container_dict:dict[str,DeltaGenerator]):
    """渲染背景音乐设置"""
    # bgm type
    bgm_options = [
        (tr("no_bg_music"), None),
        (tr("random_bg_music"), "random"),
        (tr("custom_bg_music"), "custom"),
    ]
    selected_index = st_container.selectbox(
        tr("background_music"),
        index=1,
        options=range(len(bgm_options)),
        format_func=lambda x: bgm_options[x][0],
    )
    bgm_type = bgm_options[selected_index][1]

    # custom bgm file if bgm_type is custom
    custom_bgm_file = None
    if bgm_type == "custom":
        custom_bgm_file = st_container.text_input(tr("custom_bgm_file"), value="")
    
    # bgm style
    bgm_style_list = bg_music.get_bgm_type_list()
    bgm_style_selected_index = st_container.selectbox(tr("bgm_style"), index=0, options=range(len(bgm_style_list)), format_func=lambda x: bgm_style_list[x][0])
    bgm_style = bgm_style_list[bgm_style_selected_index][1]

    # bgm volume
    bgm_volume = st_container.text_input(tr("bgm_volume"), value=0.2,help=tr("bgm_volume_help"))

    # get save path
    task_path = st.session_state['task_path']
    edit_bg_musics_path = os.path.join(task_path, "edit_bg_musics")
    file_utils.ensure_directory(edit_bg_musics_path)
    edit_bg_musics_file_path = os.path.join(edit_bg_musics_path, "edit_bg_music.mp3")

    # submit button
    submit_button = st_container.button(tr("bg_music_handler_submit"))
    if submit_button:
        with st_container:
            with st.spinner(tr("processing")):
                bgm_path = bg_music.get_bgm_file(bgm_type=bgm_type, bgm_file=custom_bgm_file, bgm_dir_path=bgm_style)

                if bgm_path is None or bgm_path == "":
                    st_container.error(tr("no_bg_music"))
                    os.remove(edit_bg_musics_file_path)
                else:
                    audio = AudioSegment.from_file(bgm_path)
                    gain_db = 20 * math.log10(float(bgm_volume))
                    adjusted_audio = audio.apply_gain(gain_db)
                    adjusted_audio.export(edit_bg_musics_file_path, format="mp3")
                    st_container.success(tr("bg_music_create_success"))
                    del audio
                    del adjusted_audio
                # show bg music
                if os.path.exists(edit_bg_musics_file_path):
                    container_dict["edit_bg_music_expander"].audio(edit_bg_musics_file_path, format="audio/mp3")
                    st.session_state['edit_bg_musics_path'] = edit_bg_musics_file_path


