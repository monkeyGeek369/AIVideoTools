from streamlit.delta_generator import DeltaGenerator
from .material_handler_settings import render_material_handler
from .subtitle_handler_settings import render_subtitle_handler


def render_control_panel(tr,st_container:DeltaGenerator,container_dict:dict[str,DeltaGenerator]):
    material_handler_tab,subtitle_handler_tab,voice_handler_tab,bg_music_handler_tab,video_handler_tab,material_edit_tab = st_container.tabs(tabs=[
        tr("material_handler"),
        tr("subtitle_handler"),
        tr("voice_handler"),
        tr("bg_music_handler"),
        tr("video_handler"),
        tr("material_edit")])
    render_material_handler(tr,material_handler_tab,container_dict)
    render_subtitle_handler(tr,subtitle_handler_tab)
    render_voice_handler(tr,voice_handler_tab)
    render_bg_music_handler(tr,bg_music_handler_tab)
    render_video_handler(tr,video_handler_tab)
    render_material_edit(tr,material_edit_tab)

def render_voice_handler(tr,st_container:DeltaGenerator):
    pass

def render_bg_music_handler(tr,st_container:DeltaGenerator):
    pass

def render_video_handler(tr,st_container:DeltaGenerator):
    pass
def render_material_edit(tr,st_container:DeltaGenerator):
    pass

