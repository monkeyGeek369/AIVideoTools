from streamlit.delta_generator import DeltaGenerator


def render_video_edit(tr,st_container:DeltaGenerator) -> dict[str,DeltaGenerator]:
    result = {}
    # material video expander
    edit_video_expander = st_container.expander(label=tr("edit_video"),expanded=True)
    edit_video_expander.write(tr("edit_video_tips"))
    result['edit_video_expander'] = edit_video_expander

    # material bg music expander 
    edit_bg_music_expander = st_container.expander(label=tr("edit_bg_music"),expanded=True)
    edit_bg_music_expander.write(tr("edit_bg_music_tips"))
    result['edit_bg_music_expander'] = edit_bg_music_expander

    # material voice expander
    edit_voice_expander = st_container.expander(label=tr("edit_voice"),expanded=True)
    edit_voice_expander.write(tr("edit_voice_tips"))
    result['edit_voice_expander'] = edit_voice_expander

    # material subtitle expander
    edit_subtitle_expander = st_container.expander(label=tr("edit_subtitle"),expanded=True)
    edit_subtitle_expander.write(tr("edit_subtitle_tips"))
    result['edit_subtitle_expander'] = edit_subtitle_expander

    return result
