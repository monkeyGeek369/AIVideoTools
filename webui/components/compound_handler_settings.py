from streamlit.delta_generator import DeltaGenerator
import streamlit as st
import os,time,platform,gc,torch
from moviepy.editor import VideoFileClip,AudioFileClip,CompositeAudioClip,TextClip,CompositeVideoClip
from app.utils import file_utils,utils,cache
import pysrt
from loguru import logger
from app.services import video,subtitle
from app.models.schema import SubtitlePosition
from app.models.subtitle_position_coord import SubtitlePositionCoord



def render_compound_handler(tr,st_container:DeltaGenerator,container_dict:dict[str,DeltaGenerator]):
    col1,col2,col3 = st_container.columns(3)

    bg_music_check,voice_check,subtitle_check = None,None,None
    with col1:
        bg_music_check = col1.checkbox(label=tr("compound_bg_music"),key="compound_bg_music",value=True)
    with col2:
        voice_check = col2.checkbox(label=tr("compound_voice"),key="compound_voice",value=True)
    with col3:
        subtitle_check = col3.checkbox(label=tr("compound_subtitle"),key="compound_subtitle",value=True)
        st.session_state['subtitle_enabled'] = subtitle_check
        subtitle_auto = col3.checkbox(label=tr("compound_subtitle_auto"),key="compound_subtitle_auto",value=True)
        st.session_state['subtitle_auto'] = subtitle_auto

        if subtitle_check:
            render_font_settings(tr)
            render_position_settings(tr)
            render_style_settings(tr)

    submit_container = st_container.container()
    submit_button = submit_container.button(label=tr("compound_handler_submit"),key="compound_handler_submit")
    if submit_button:
        with submit_container:
            with st.spinner(text=tr("processing")):
                try:
                    compound_video(tr,bg_music_check,voice_check,subtitle_check,container_dict)
                except Exception as e:
                    submit_container.error(e)
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()
    
def compound_video(tr,bg_music_check:bool,voice_check:bool,subtitle_check:bool,container_dict:dict[str,DeltaGenerator]):
    try:
        task_path = st.session_state['task_path']
        mix_audio_clip = None
        final_clip = None

        # get edit video
        edit_video_path = st.session_state['edit_video_path']
        if edit_video_path is None or os.path.exists(edit_video_path) is False:
            raise Exception(tr("edit_video_not_found"))
        
        # get video base info
        video_clip = VideoFileClip(edit_video_path)
        video_duration = video_clip.duration
        video_height = video_clip.h
        video_clip.close()
        audio_clips = []
        
        # bg music
        if bg_music_check:
            edit_bg_musics_path = st.session_state['edit_bg_musics_path']
            if edit_bg_musics_path is None or os.path.exists(edit_bg_musics_path) is False:
                raise Exception(tr("edit_bg_music_not_found"))
            
            audio_clip = AudioFileClip(edit_bg_musics_path)
            if audio_clip.duration < video_duration:
                audio_clip = audio_clip.set_duration(video_duration).loop(duration=video_duration)
            elif audio_clip.duration > video_duration:
                audio_clip = audio_clip.set_duration(video_duration)
            audio_clips.append(audio_clip)
        
        # voice
        if voice_check:
            edit_voice_path = st.session_state['edit_voice_path']
            if edit_voice_path is None or os.path.exists(edit_voice_path) is False:
                raise Exception(tr("edit_voice_not_found"))
            voice_clip = AudioFileClip(edit_voice_path)
            audio_clips.append(voice_clip)
        
        # subtitle
        subtitle_clips = []
        if subtitle_check:
            subtitle_clips = get_subtitle_clips(video_height,edit_video_path,task_path)
            if subtitle_clips is None or len(subtitle_clips) == 0:
                raise Exception(tr("subtitle_info_not_found"))

        # get save path
        compound_videos_path = os.path.join(task_path, "compound_videos")
        file_utils.ensure_directory(compound_videos_path)
        final_clip_path = os.path.join(compound_videos_path, "compound_video.mp4")

        # save
        mix_audio_clip = CompositeAudioClip(audio_clips)
        video_clip = VideoFileClip(edit_video_path)
        video_clip = video_clip.set_audio(mix_audio_clip)
        final_clip = video_clip
        if subtitle_clips and len(subtitle_clips) > 0:
            final_clip = CompositeVideoClip([video_clip] + subtitle_clips, size=video_clip.size)        
            
        final_clip.write_videofile(
                final_clip_path,
                codec='libx264',
                audio_codec='aac',
                fps=video_clip.fps
            )

        # show
        container_dict["compound_video_expander"].video(final_clip_path, format="video/mp4")
        st.session_state['compound_video_path'] = final_clip_path
        # download compound video
        with open(final_clip_path, "rb") as video_file:
            video_bytes = video_file.read()
        container_dict["compound_video_expander"].download_button(label=tr("download_compound_video"),
                                                                  data=video_bytes, 
                                                                  file_name="compound_video.mp4",
                                                                  mime="video/mp4",
                                                                  key=f"download_button_"+str(time.time()))
    except Exception as e:
        raise e
    finally:
        if video_clip is not None:
            video_clip.close()
        for audio_clip in audio_clips:
            if audio_clip is not None:
                audio_clip.close()
        for subtitle_clip in subtitle_clips:
            if subtitle_clip is not None:
                subtitle_clip.close()
        if mix_audio_clip is not None:
            mix_audio_clip.close()
        if final_clip is not None:
            final_clip.close()

def render_font_settings(tr):
    """渲染字体设置"""
    # 获取字体列表
    font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "resource", "fonts")
    font_names = cache.get_fonts_cache(font_dir)

    # 字体选择
    font_name = st.selectbox(
        tr("font_name"),
        options=font_names,
        index=0
    )
    st.session_state['font_name'] = font_name

    # 字体颜色
    font_cols = st.columns([0.3, 0.7])
    with font_cols[0]:
        text_fore_color = st.color_picker(
            tr("font_color"),
            "#FFFFFF"
        )
        st.session_state['text_fore_color'] = text_fore_color

    # auto get font size by vidio height
    with font_cols[1]:
        font_size = st.slider(
            tr("font_size"),
            min_value=20,
            max_value=100,
            value=subtitle.calculate_font_size(st.session_state.get('video_height')) if (st.session_state['subtitle_auto'] is not None and st.session_state['subtitle_auto']) else 45
        )
        st.session_state['font_size'] = font_size

def render_position_settings(tr):
    """渲染位置设置"""
    subtitle_positions = [
        (tr("Top"), "top"),
        (tr("Center"), "center"),
        (tr("Bottom"), "bottom"),
        (tr("Custom"), "custom"),
        (tr("Origin_subtitle_position"), "origin_subtitle_position"),
    ]

    selected_index = st.selectbox(
        tr("font_position"),
        index=4,
        options=range(len(subtitle_positions)),
        format_func=lambda x: subtitle_positions[x][0],
    )

    subtitle_position = subtitle_positions[selected_index][1]
    st.session_state['subtitle_position'] = subtitle_position

    # 自定义位置处理
    if subtitle_position == "custom":
        custom_position = st.text_input(
            tr("Custom Position (% from top)"),
            value="70.0"
        )
        try:
            custom_position_value = float(custom_position)
            if custom_position_value < 0 or custom_position_value > 100:
                st.error(tr("Please enter a value between 0 and 100"))
            else:
                st.session_state['custom_position'] = custom_position_value
        except ValueError:
            st.error(tr("Please enter a valid number"))

def render_style_settings(tr):
    """渲染样式设置"""
    stroke_cols = st.columns([0.3, 0.7])

    with stroke_cols[0]:
        stroke_color = st.color_picker(
            tr("stroke_color"),
            value="#000000"
        )
        st.session_state['stroke_color'] = stroke_color

    with stroke_cols[1]:
        stroke_width = st.slider(
            tr("stroke_width"),
            min_value=0.0,
            max_value=10.0,
            value=max(0.5,round(st.session_state['font_size']/50,2)) if (st.session_state['subtitle_auto'] is not None and st.session_state['subtitle_auto']) else float(1),
            step=0.1
        )
        st.session_state['stroke_width'] = stroke_width


def get_subtitle_params():
    """获取字幕参数"""
    return {
        'subtitle_enabled': st.session_state.get('subtitle_enabled', True),
        'font_name': st.session_state.get('font_name', ''),
        'font_size': st.session_state.get('font_size', 60),
        'text_fore_color': st.session_state.get('text_fore_color', '#FFFFFF'),
        'position': st.session_state.get('subtitle_position', 'bottom'),
        'custom_position': st.session_state.get('custom_position', 70.0),
        'stroke_color': st.session_state.get('stroke_color', '#000000'),
        'stroke_width': st.session_state.get('stroke_width', 1.5),
    }

def get_subtitle_clips(video_height,video_path:str,task_path:str) -> list[TextClip]:
    subtitle_path = st.session_state['edit_subtitle_path']
    subtitle_params = get_subtitle_params()
    font_path = utils.font_dir(subtitle_params['font_name'])
    if platform.system() == 'Windows':
        font_path = os.path.relpath(font_path)
        font_path =  "./"+font_path.replace(os.sep, '/')

    # auto subtitle recognized and mosaic
    recognize_position_model = None|SubtitlePositionCoord
    subtitle_position_dict = st.session_state.get('subtitle_position_dict', {})
    recognize_poistion = subtitle_position_dict.get("edit_video.mp4")
    recognize_position_model = SubtitlePositionCoord.model_validate(recognize_poistion)
    if recognize_position_model.is_exist:
        video.video_subtitle_mosaic_auto(video_path=video_path,subtitle_position_coord=recognize_position_model,task_path=task_path)

    # base check
    if not os.path.exists(subtitle_path):
        raise Exception(f"subtitle file not found: {subtitle_path}")
    if not os.path.exists(font_path):
        raise Exception(f"font file not found: {font_path}")

    subtitle_clips = []
    subs = None
    try:
        subs = pysrt.open(subtitle_path)
        logger.info(f"读取到 {len(subs)} 条字幕")

        for index, sub in enumerate(subs):
            start_time = sub.start.ordinal / 1000
            end_time = sub.end.ordinal / 1000

            try:
                # 获取字幕文本
                subtitle_text = subtitle.get_text_from_subtitle(sub)
                if not subtitle_text:
                    logger.info(f"警告：第 {index + 1} 条字幕处理后为空，已跳过")
                    continue

                # 计算自动换行
                subtitle_text = subtitle.auto_wrap_text(subtitle_text,font_path,subtitle_params['font_size'],st.session_state.get('video_width') * 0.8)

                # 计算字幕位置
                position = None
                if subtitle_params['position'] == SubtitlePosition.ORIGIN and recognize_position_model.is_exist:
                    position = ('center',  int(recognize_position_model.left_top_y))
                else:
                    position = video.calculate_subtitle_position(
                        subtitle_params['position'],
                        video_height,
                        subtitle_params['custom_position']
                    )

                # 创建最终的 TextClip
                text_clip = (TextClip(
                    subtitle_text,
                    font=font_path,
                    fontsize=subtitle_params['font_size'],
                    color=subtitle_params['text_fore_color'],
                    stroke_color=subtitle_params['stroke_color'],
                    stroke_width=subtitle_params['stroke_width'],
                    remove_temp=True
                )
                    .set_position(position)
                    .set_duration(end_time - start_time)
                    .set_start(start_time))
                subtitle_clips.append(text_clip)

            except Exception as e:
                logger.error(f"警告：创建第 {index + 1} 条字幕时出错: {str(e)}")

        logger.info(f"成功创建 {len(subtitle_clips)} 条字幕剪辑")
    except Exception as e:
        logger.info(f"警告：处理字幕文件时出错: {str(e)}")
    finally:
        if subs is not None:
            subs = None
            del subs

    return subtitle_clips


