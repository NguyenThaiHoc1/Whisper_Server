from fastapi import UploadFile
from pydantic import BaseModel


class AudioUpload(BaseModel):
    file: UploadFile
