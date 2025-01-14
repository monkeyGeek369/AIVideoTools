from streamlit.delta_generator import DeltaGenerator

edit_video_expander=None
edit_bg_music_expander=None
edit_voice_expander=None
edit_srt_expander=None


def render_video_edit(tr,st_container:DeltaGenerator):
    global edit_video_expander,edit_bg_music_expander,edit_voice_expander,edit_srt_expander

    # material video expander
    edit_video_expander = st_container.expander(label=tr("edit_video"),expanded=True)
    edit_video_expander.write(tr("edit_video_tips"))

    # material bg music expander 
    edit_bg_music_expander = st_container.expander(label=tr("edit_bg_music"),expanded=True)
    edit_bg_music_expander.write(tr("edit_bg_music_tips"))

    # material voice expander
    edit_voice_expander = st_container.expander(label=tr("edit_voice"),expanded=True)
    edit_voice_expander.write(tr("edit_voice_tips"))

    # material srt expander
    edit_srt_expander = st_container.expander(label=tr("edit_srt"),expanded=True)
    edit_srt_expander.write(tr("edit_srt_tips"))
