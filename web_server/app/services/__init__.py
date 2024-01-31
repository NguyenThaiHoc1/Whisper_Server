# init - services
from app.services.whisper.service import WhisperService

# settings - services
from app.config.base_config import settings

# Init WhisperService
WhisperService.onnx_path = settings.WHISPER_ONNX_PATH
WhisperService.whisper_processor_path = settings.WHISPER_PROCESSOR_PATH
WhisperService.whisper_generation_path = settings.WHISPER_GENERATION_PATH
WhisperService.whisper_vad_segmentation = settings.WHIPSER_VAD_SEGMENTATION
WhisperService.device = settings.WHIPSER_DEVICE
service_whisper = WhisperService()


