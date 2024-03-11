import os
from pathlib import Path

# from pydantic import BaseSettings
from pydantic.v1 import BaseSettings

from typing import Union, List

BASE_DIR = Path(__file__).resolve().parent.parent
TRANSLATOR_SERVICES_MODEL_LIB = BASE_DIR / "services" / "translate" / "model"


class Settings(BaseSettings):
    # [DEBUG] - Mode
    DEBUG: str = "dev"  # two mode "dev" and "stage".

    # [MODEL] - Translate model
    TRANSLATOR_ONNX_PATH: Union[str, os.PathLike] = TRANSLATOR_SERVICES_MODEL_LIB / "architecture" / "v1" / "final.onnx"
    TRANSLATOR_DEVICE: Union[str, os.PathLike] = "cpu"


settings = Settings()
