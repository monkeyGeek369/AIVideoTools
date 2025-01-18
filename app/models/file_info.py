from pydantic import BaseModel

class LocalFileInfo(BaseModel):
    name:str
    suffix:str
    path:str
    size:int
    ctime:float
    mtime:float

