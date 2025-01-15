import streamlit as st
import os,sys
from app.config import config
from app.utils import utils
from webui.components import video_meta_data_settings,video_edit_settings,control_panel_settings
from streamlit.delta_generator import DeltaGenerator

def init_log():
    """初始化日志配置"""
    from loguru import logger
    logger.remove()
    _lvl = config.log_level

    def format_record(record):
        # 增加更多需要过滤的警告消息
        ignore_messages = [
            "Examining the path of torch.classes raised",
            "torch.cuda.is_available()",
            "CUDA initialization"
        ]
        
        for msg in ignore_messages:
            if msg in record["message"]:
                return ""
            
        file_path = record["file"].path
        relative_path = os.path.relpath(file_path, config.root_dir)
        record["file"].path = f"./{relative_path}"
        record['message'] = record['message'].replace(config.root_dir, ".")

        _format = '<green>{time:%Y-%m-%d %H:%M:%S}</> | ' + \
                  '<level>{level}</> | ' + \
                  '"{file.path}:{line}":<blue> {function}</> ' + \
                  '- <level>{message}</>' + "\n"
        return _format

    # 优化日志过滤器
    def log_filter(record):
        ignore_messages = [
            "Examining the path of torch.classes raised",
            "torch.cuda.is_available()",
            "CUDA initialization"
        ]
        return not any(msg in record["message"] for msg in ignore_messages)

    logger.add(
        sys.stdout,
        level=_lvl,
        format=format_record,
        colorize=True,
        filter=log_filter
    )

def init_global_state():
    """初始化全局状态"""
    if 'ui_language' not in st.session_state:
        st.session_state['ui_language'] = utils.get_system_locale()

def render_language_settings(st_container):
    """渲染语言设置"""
    system_locale = utils.get_system_locale()
    i18n_dir = os.path.join(os.path.dirname(__file__), "webui", "i18n")
    locales = utils.load_locales(i18n_dir)

    display_languages = []
    selected_index = 0
    for i, code in enumerate(locales.keys()):
        display_languages.append(f"{code} - {locales[code].get('Language')}")
        if code == st.session_state.get('ui_language', system_locale):
            selected_index = i

    selected_language = st_container.selectbox(
        tr("project_version"),
        options=display_languages,
        index=selected_index
    )

    if selected_language:
        code = selected_language.split(" - ")[0].strip()
        st.session_state['ui_language'] = code

def page_layout() -> list[DeltaGenerator]:
    # set wide layout
    st.set_page_config(
        page_title=tr("project_name"),
        page_icon="📽️",
        layout="wide")

    # set default style
    st.markdown(
    """
    <style>
    /* remove default streamlit header*/
    [data-testid="stHeader"] {
        display: none;
    }

    /* set container padding*/
    [data-testid="stMainBlockContainer"] {
        padding-top: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
        padding-bottom: 0rem;
    }

    /* remove vertical block*/
    [data-testid="stVerticalBlock"] {
        display: block;
    }
    </style>
    """,
        unsafe_allow_html=True
    )

    # header container
    header_container = st.container(border=True)
    header_container_title = header_container.container()
    header_container_description = header_container.container()
    header_des_left,header_des_right = header_container_description.columns([0.9,0.1])
    render_language_settings(header_des_right)
    header_container_title.header(tr("project_name") + ":sunflower:📽️")
    header_des_left.write(tr("project_description"))

    # video container
    video_container = st.container(border=True)
    video_meta_data_column,video_edit_column = video_container.columns([0.3,0.7],gap="small")
    video_meta_data_column.subheader(tr("video_meta_data_column_subheader"))
    video_edit_column.subheader(tr("video_edit_column_subheader"))

    # control container
    control_container = st.container(border=True)
    control_container.subheader(tr("control_container_subheader"))

    result = [video_meta_data_column,video_edit_column,control_container]
    return result

def tr(key):
    """翻译函数"""
    i18n_dir = os.path.join(os.path.dirname(__file__), "webui", "i18n")
    locales = utils.load_locales(i18n_dir)
    loc = locales.get(st.session_state['ui_language'], {})
    return loc.get("Translation", {}).get(key, key)

def main():
    # config init
    init_log()
    init_global_state()
    utils.init_resources()

    # page layout
    video_meta_data_column,video_edit_column,control_container = page_layout()

    # render layout
    video_meta_data_settings.render_video_meta_data(tr,video_meta_data_column)
    video_edit_settings.render_video_edit(tr,video_edit_column)
    control_panel_settings.render_control_panel(tr,control_container)
    
if __name__ == "__main__":
    main()
