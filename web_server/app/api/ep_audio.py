from typing import Dict
from datetime import datetime
from fastapi import APIRouter, HTTPException, File, UploadFile, Form

# Services
from app.services import service_whisper

# Schema
from app.schema.sch_audio import AudioUpload

# Utils
from app.api.utils import utils_audio

# settings
from app.config.base_config import settings

# logger
from app.logging_utils import logger
import time

router = APIRouter()


@router.post('/speech2text',
             description='do speech audio compile to text',
             status_code=200,
             include_in_schema=True,
             response_model=dict)
async def do_speech2text(file: UploadFile = File(...), task_id: str = Form("123123123")) -> Dict:
    try:
        global_s_time = datetime.now()
        # get extension file
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
        # response_output = service_whisper.do_whisperer(
        #     audio_file=dict_decode_file["filepath"]
        # )
        end_time = time.time() - start_time
        logger.info(f"[func: do_speech2text]: Do Whisper is successful. | time: {end_time}s")
        global_e_time = datetime.now()

        # response
        status_code = 200
        # result = response_output
        result = {
            "start-time": global_s_time,
            "end-time": global_e_time,
            "filename": filename,
            "file-length": dict_decode_file["duration"],
            "file-size": dict_decode_file["size"],
            "extension": extension_file,
            "model-name": settings.WHISPER_ONNX_PATH.name,
            "task-id": task_id
        }
        message_error = None

    except ValueError as e:
        logger.error(
            f"[func: do_speech2text] {400}: {str(e)}"
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.critical(
            f"[func: do_speech2text] {500}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status_code": status_code,
        "result": result,
        "message_error": message_error
    }
