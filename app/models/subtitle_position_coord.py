from pydantic import BaseModel

class SubtitlePositionCoord(BaseModel):
    is_exist:bool
    fixed_regions:dict
    time_index:dict[float,int]
    frames:dict[int,dict]
