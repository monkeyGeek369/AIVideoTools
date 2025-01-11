import streamlit as st
import os,sys
from uuid import uuid4

def page_layout():
    import streamlit as st

    # 设置页面容器宽度为 100%
    st.markdown(
        """
        <style>
        .stApp {
            width: 100%;
            margin: 0;
        }
        .stContainer {
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        st.title("这是容器内的标题")
        st.write("这是容器内的内容")

    st.write("这是容器外的内容")

    # col1,col2,col3 = st.columns(3)
    # col1.video("F:\download\king.mp4", start_time=0)
    # col2.video("F:\download\king.mp4", start_time=0)
    # col3.video("F:\download\king.mp4", start_time=0)

    # main_continer_video = st.container(border=True)
    # main_continer_video.write("main_continer_video")
    # main_continer_video.video("F:\download\king.mp4", start_time=0)
    # main_continer_video.video("F:\download\king.mp4", start_time=0)
    # main_continer_video.video("F:\download\king.mp4", start_time=0)


    # main_continer_control = st.container(border=True)
    # main_continer_control.write("main_continer_control")


def main():
    page_layout()
    




if __name__ == "__main__":
    main()



