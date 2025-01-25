from streamlit.delta_generator import DeltaGenerator
from .material_handler_settings import render_material_handler
from .subtitle_handler_settings import render_subtitle_handler
from .voice_handler_settings import render_voice_handler
from .bg_music_handler_settings import render_bg_music_handler
from .video_handler_settings import render_video_handler
from .compound_handler_settings import render_compound_handler


def render_control_panel(tr,st_container:DeltaGenerator,container_dict:dict[str,DeltaGenerator]):
    material_handler_tab,subtitle_handler_tab,voice_handler_tab,bg_music_handler_tab,video_handler_tab,compound_handler_tab = st_container.tabs(tabs=[
        tr("material_handler"),
        tr("subtitle_handler"),
        tr("voice_handler"),
        tr("bg_music_handler"),
        tr("video_handler"),
        tr("compound_handler")])
    render_material_handler(tr,material_handler_tab,container_dict)
    render_subtitle_handler(tr,subtitle_handler_tab,container_dict)
    render_voice_handler(tr,voice_handler_tab,container_dict)
    render_bg_music_handler(tr,bg_music_handler_tab,container_dict)
    render_video_handler(tr,video_handler_tab,container_dict)
    render_compound_handler(tr,compound_handler_tab,container_dict)


