from typing import Dict
from fastapi import APIRouter, HTTPException, File, UploadFile

# Services
from app.services import service_whisper

# Schema
from app.schema.sch_audio import AudioUpload

# Utils
from app.api import utils as utils_audio

router = APIRouter()


@router.post('/speech2text',
             description='do speech audio compile to text',
             status_code=200,
             include_in_schema=True,
             response_model=dict)
async def do_speech2text(file: UploadFile = File(...)) -> Dict:
    try:
        # Decode file
        dict_decode_file = await utils_audio.read_file(
            audio_file_bytes=file
        )
        print(f"[Whisper] do_speech2text: Decode file path is successful.")

        # Do service [Whisper]
        response_output = service_whisper.do_whisperer(
            audio_file=dict_decode_file["filepath"]
        )
        print(f"[Whisper] do_speech2text: Do whisperer is successful.")

        # response
        status_code = 200
        result = response_output
        message_error = None

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status_code": status_code,
        "result": result,
        "message_error": message_error
    }
