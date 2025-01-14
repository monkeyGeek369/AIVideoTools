from streamlit.delta_generator import DeltaGenerator

material_handler_tab=None
voice_handler_tab=None
bg_music_handler_tab=None
subtitle_handler_tab=None
video_handler_tab=None
material_edit_tab=None

def render_control_panel(tr,st_container:DeltaGenerator):
    global material_handler_tab,voice_handler_tab,bg_music_handler_tab,subtitle_handler_tab,video_handler_tab,material_edit_tab
    material_handler_tab,voice_handler_tab,bg_music_handler_tab,subtitle_handler_tab,video_handler_tab,material_edit_tab = st_container.tabs(tabs=[tr("material_handler"),tr("voice_handler"),tr("bg_music_handler"),tr("subtitle_handler"),tr("video_handler"),tr("material_edit")])
    render_material_handler(tr,material_handler_tab)
    render_voice_handler(tr,voice_handler_tab)
    render_bg_music_handler(tr,bg_music_handler_tab)
    render_subtitle_handler(tr,subtitle_handler_tab)
    render_video_handler(tr,video_handler_tab)
    render_material_edit(tr,material_edit_tab)


def render_material_handler(tr,st_container:DeltaGenerator):
    # create form
    material_handler_form = st_container.form(key="material_handler_form")

    # create checkbox obj
    video_split_checkbox_value = None
    voice_split_checkbox_value = None
    subtitle_split_checkbox_value = None
    bg_music_split_checkbox_value = None

    # create checkbox
    split_container=material_handler_form.container()
    column1,column2,column3,column4 = split_container.columns(4)
    with column1:
        video_split_checkbox_value = column1.checkbox(label=tr("video_split"),key="video_split")
    with column2:
        voice_split_checkbox_value = column2.checkbox(label=tr("voice_split"),key="voice_split")
    with column3:
        subtitle_split_checkbox_value = column3.checkbox(label=tr("subtitle_split"),key="subtitle_split")
    with column4:
        bg_music_split_checkbox_value = column4.checkbox(label=tr("bg_music_split"),key="bg_music_split")

    # create submit button
    submitted = material_handler_form.form_submit_button(label=tr("material_handler_submit"))
    if submitted:
        print(video_split_checkbox_value)
        print(voice_split_checkbox_value)
        print(subtitle_split_checkbox_value)
        print(bg_music_split_checkbox_value)

def render_voice_handler(tr,st_container:DeltaGenerator):
    pass

def render_bg_music_handler(tr,st_container:DeltaGenerator):
    pass
def render_subtitle_handler(tr,st_container:DeltaGenerator):
    pass
def render_video_handler(tr,st_container:DeltaGenerator):
    pass
def render_material_edit(tr,st_container:DeltaGenerator):
    pass

