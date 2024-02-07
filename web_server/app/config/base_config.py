import os
from pathlib import Path

from pydantic.v1 import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent
WHISPER_SERVICES_MODEL_LIB = BASE_DIR / "services" / "whisper" / "model"


class Settings(BaseSettings):
    # [CORE] - Whisper
    WHISPER_ONNX_PATH = WHISPER_SERVICES_MODEL_LIB / "whisper-large-v3_beamsearch.onnx"
    WHISPER_PROCESSOR_PATH = WHISPER_SERVICES_MODEL_LIB / "WhisperProcessor"
    WHISPER_GENERATION_PATH = WHISPER_SERVICES_MODEL_LIB / "WhisperGenerationConfig"
    WHIPSER_VAD_SEGMENTATION = WHISPER_SERVICES_MODEL_LIB / "whisperx-vad-segmentation.bin"
    WHIPSER_DEVICE = 'cpu'


settings = Settings()
