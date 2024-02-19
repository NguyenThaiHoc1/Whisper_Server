import os
from pathlib import Path

# from pydantic import BaseSettings
from pydantic.v1 import BaseSettings

from typing import Union

BASE_DIR = Path(__file__).resolve().parent.parent
WHISPER_SERVICES_MODEL_LIB = BASE_DIR / "services" / "whisper" / "model"


class Settings(BaseSettings):
    # [DEBUG] - Mode
    DEBUG: str = "dev"  # two mode "dev" and "stage".

    # [STORAGE]
    IS_STORAGE: bool = True
    STORAGE_PATH: Union[str, os.PathLike] = None

    # [CORE] - Whisper
    WHISPER_ONNX_PATH: Union[str, os.PathLike] = WHISPER_SERVICES_MODEL_LIB / "whisper-large-v3_beamsearch.onnx"
    WHISPER_PROCESSOR_PATH: Union[str, os.PathLike] = WHISPER_SERVICES_MODEL_LIB / "WhisperProcessor"
    WHISPER_GENERATION_PATH: Union[str, os.PathLike] = WHISPER_SERVICES_MODEL_LIB / "WhisperGenerationConfig"
    WHIPSER_VAD_SEGMENTATION: Union[str, os.PathLike] = WHISPER_SERVICES_MODEL_LIB / "whisperx-vad-segmentation.bin"
    WHIPSER_DEVICE: Union[str, os.PathLike] = 'cpu'

    # [CONFIG-FRAMEWORK] - pydub
    # PYFUB_AUDIOSEGMENT: str = "/opt/homebrew/Cellar/ffmpeg/6.1.1_2/bin/ffmpeg"  # for macbook
    PYFUB_AUDIOSEGMENT: str = "/usr/bin/ffmpeg"  # for linux

    def settings_storage(self):
        if self.DEBUG:
            self.STORAGE_PATH = BASE_DIR / "storage"
        else:  # can you write data to redis ...
            self.STORAGE_PATH = BASE_DIR / "storage"


settings = Settings()
settings.settings_storage()
