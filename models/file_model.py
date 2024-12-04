from pydantic import BaseModel
from typing import Optional

class FileModel(BaseModel):
    filename: Optional[str] = None
    file_base64: str

class ImageURLModel(BaseModel):
    filename: Optional[str] = None
    file_URL: str

class FileModelTwo(BaseModel):
    filename_1: Optional[str] = None
    file_base64_1: str
    filename_2: Optional[str] = None
    file_base64_2: str

class ImageURLModelTwo(BaseModel):
    filename_1: Optional[str] = None
    file_URL_1: str
    filename_2: Optional[str] = None
    file_URL_2: str