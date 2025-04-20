from pydantic import BaseModel

class SubtitlePositionCoord(BaseModel):
    is_exist:bool
    left_top_x:int
    left_top_y:int
    right_bottom_x:int
    right_bottom_y:int
    count:int
    frame_subtitles_position:dict[float,list[tuple[tuple[int,int],tuple[int,int],str]]]
