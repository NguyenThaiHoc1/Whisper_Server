# Services
from app.services import service_whisper

# Utils
from app.bussiness.utils import utils_audio

# logger
from app.logging_utils import logger

# settings
from app.config.base_config import settings

# regularly function
import time
from datetime import datetime


class DetectionLanguageBus:  # regulations for bussiness api speech2text

    def __init__(self, *args, **kwargs):
        pass

    async def __call__(self, *args, **kwargs):
        file = kwargs.get('file')
        assert file is not None, "Please check your file"
        global_s_time = datetime.now()

        # Read extension
        logger.info(f"[func: do_speech2text] starting sfunc: get_extension_file ...")
        start_time = time.time()
        filename, extension_file = utils_audio.get_extension_file(
            file.filename
        )
        end_time = time.time() - start_time
        logger.info(f"[func: do_speech2text]: '{filename}' '{extension_file}' | time: {end_time}s")

        # Decode file
        logger.info(f"[func: do_speech2text] starting sfunc: utils_audio.read_file ...")
        start_time = time.time()
        dict_decode_file = await utils_audio.read_file(
            audio_file_bytes=file,
            path_storage=settings.STORAGE_PATH,
            extension=extension_file,
            filename=filename
        )
        end_time = time.time() - start_time
        logger.info(f"[func: do_speech2text]: Decode file path is successful. | time: {end_time}s")

        # Do service [Whisper]
        logger.info(f"[func: do_speech2text] starting sfunc: service_whisper.do_whisperer ...")
        start_time = time.time()
        response_output = service_whisper.do_detection_language(
            audio_file=dict_decode_file["filepath"]
        )
        end_time = time.time() - start_time
        logger.info(f"[func: do_speech2text]: Do Whisper is successful. | time: {end_time}s")
        global_e_time = datetime.now()

        return {
            "start_time": global_s_time,
            "end_time": global_e_time,
            "filename": filename,
            "file_length": dict_decode_file["duration"],
            "file_size": dict_decode_file["size"],
            "extension": extension_file,
            "model_name": settings.WHISPER_ONNX_PATH.name,
            "output": response_output
        }
