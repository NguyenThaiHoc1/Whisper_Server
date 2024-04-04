# init - services
from app.services.whisper.service import WhisperService
from app.services.htdecmus.service import HTDecmusServices
from app.services.titanet.service import TitaNetServices
from app.services.verifyspeech.service import VerifySpeechService

# init - adapter

# settings - services
from app.config.base_config import settings

# Init WhisperService
WhisperService.onnx_path = settings.WHISPER_ONNX_PATH
WhisperService.whisper_processor_path = settings.WHISPER_PROCESSOR_PATH
WhisperService.whisper_generation_path = settings.WHISPER_GENERATION_PATH
WhisperService.whisper_vad_segmentation = settings.WHIPSER_VAD_SEGMENTATION
WhisperService.device = settings.WHIPSER_DEVICE
service_whisper = WhisperService()

# Init HTDecmusService
service_htdecmus = HTDecmusServices(
    input_name=settings.TRITON_HTDECMUS_SERVICES_INPUTS_NAME,
    output_name=settings.TRITON_HTDECMUS_SERVICES_OUTPUTS_NAME,
    model_name=settings.TRITON_HTDECMUS_SERVICES_MODEL_NAME,
    model_version=settings.TRITON_HTDECMUS_SERVICES_VERSION,
    triton_protocol=settings.TRITON_SERVICES_PROTOCOL
)

# Init SpeechService
service_titanet = TitaNetServices(
    input_name=settings.TRITON_VERIFIER_SERVICES_INPUTS_NAME,
    output_name=settings.TRITON_VERIFIER_SERVICES_OUTPUTS_NAME,
    model_name=settings.TRITON_VERIFIER_SERVICES_MODEL_NAME,
    model_version=settings.TRITON_VERIFIER_SERVICES_VERSION,
    triton_protocol=settings.TRITON_SERVICES_PROTOCOL
)

# Init VerifyService
service_verifyspeech = VerifySpeechService(
    module_titanet=service_titanet
)

