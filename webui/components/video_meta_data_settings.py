from streamlit.delta_generator import DeltaGenerator

def render_video_meta_data(tr,st_container:DeltaGenerator) -> dict[str,DeltaGenerator]:
    result = {}

    # material video expander
    material_video_expander = st_container.expander(label=tr("material_video"),expanded=True)
    material_video_expander.write(tr("material_video_tips"))
    result['material_video_expander'] = material_video_expander

    # material bg music expander 
    material_bg_music_expander = st_container.expander(label=tr("material_bg_music"),expanded=True)
    material_bg_music_expander.write(tr("material_bg_music_tips"))
    result['material_bg_music_expander'] = material_bg_music_expander

    # material voice expander
    material_voice_expander = st_container.expander(label=tr("material_voice"),expanded=True)
    material_voice_expander.write(tr("material_voice_tips"))
    result['material_voice_expander'] = material_voice_expander

    # material subtitle expander
    material_subtitle_expander = st_container.expander(label=tr("material_subtitle"),expanded=True)
    material_subtitle_expander.write(tr("material_subtitle_tips"))
    result['material_subtitle_expander'] = material_subtitle_expander
    return result




