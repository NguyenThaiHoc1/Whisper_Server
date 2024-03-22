
# create model
from app.services.translate.model import MBartModel
from app.services.translate.service import TranslateService

# settings
from app.config.base_config import settings

model_translator = MBartModel(
    model_path=settings.TRANSLATOR_ONNX_PATH,
    tokenizer_path=settings.TRANSLATOR_TOKEN_PATH,
    device=settings.TRANSLATOR_DEVICE
)

service_translate = TranslateService(
    model_translator=model_translator
)
