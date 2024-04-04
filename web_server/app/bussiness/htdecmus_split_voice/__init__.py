from app.config.base_config import settings

from app.bussiness.htdecmus_split_voice.bus import HTDECMuscSplitVoice

from app.bussiness.htdecmus_split_voice.htdecmus_pack import Separator, ProcessInput

# Services
from app.services import service_htdecmus

# init
pro_inp = ProcessInput(num_workers=1)

separator = Separator(device=settings.TRITON_HTDECMUS_SERVICES_DEVICE, overlap=0.25,
                      model_service=service_htdecmus, process_input_class=pro_inp)

bussiness_htdecmus = HTDECMuscSplitVoice(
    device=settings.TRITON_HTDECMUS_SERVICES_DEVICE,
    path_storage=settings.STORAGE_PATH,
    service_model_name=settings.TRITON_HTDECMUS_SERVICES_MODEL_NAME,
    serivce_model_version=settings.TRITON_HTDECMUS_SERVICES_VERSION,
    separator_class=separator,
)
