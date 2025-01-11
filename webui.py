import streamlit as st
import os,sys
from app.config import config
from app.utils import utils
from uuid import uuid4



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
        st.session_state['ui_language'] = config.ui.get("language", utils.get_system_locale())

def page_layout():
    # set wide layout
    st.set_page_config(
        page_title="AI Video Tools",
        page_icon=":camera:",
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
    header_container.header("AI Video Tools")

    # video container
    video_container = st.container(border=True)
    video_meta_data_column,video_result_column = video_container.columns(2)
    video_meta_data_column.video("/Users/monkeygeek/Downloads/test.mp4", start_time=0)
    video_result_column.video("/Users/monkeygeek/Downloads/test.mp4", start_time=0)

    # control container
    control_container = st.container(border=True)
    control_container.write("control_container")

def main():
    # config init
    init_log()
    init_global_state()

    page_layout()
    




if __name__ == "__main__":
    main()



