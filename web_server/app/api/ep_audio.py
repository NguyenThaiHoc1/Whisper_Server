from typing import Dict

from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Response
from fastapi.responses import FileResponse, JSONResponse

# Bussiness
from app.bussiness import bussiness_speech2text
from app.bussiness import bussiness_htdecmus

# Schema
from app.schema.sch_audio import AudioUpload

# logger
from app.logging_utils import logger
import time

router = APIRouter()


@router.post('/htdecmus',
             description='do htdecmus audio',
             status_code=200)
async def do_htdecmus(file: UploadFile = File(...), task_id: str = Form("1223123123")):
    try:
        # calling bussiness
        outputs_dict = await bussiness_htdecmus(
            file=file
        )

        # add task_id for task
        outputs_dict["task_id"] = task_id

        # response
        status_code = 200
        result = {
            "start_time": outputs_dict["start_time"],
            "end_time": outputs_dict["end_time"],
            "model_name": outputs_dict["model_name"],
        }
        message_error = None

        file_name = outputs_dict["file_name"]
        file_path = outputs_dict["output_path"]
        headers = {
            'message_error': f'{message_error}',
            'result': f'{result}',
            'Content-Disposition': f'attachment; filename="{file_name}"'
        }
        response_data = FileResponse(file_path, headers=headers, media_type='audio/mp3')
        response_data.status_code = status_code

    except ValueError as e:
        logger.error(
            f"[func: do_htdecmus] {400}: {str(e)}"
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            f"[func: do_htdecmus] {500}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=str(e))

    return response_data


@router.post('/speech2text',
             description='do speech audio compile to text',
             status_code=200,
             include_in_schema=True,
             response_model=dict)
async def do_speech2text(file: UploadFile = File(...), task_id: str = Form("123123123")) -> Dict:
    try:
        # calling bussiness
        outputs_dict = await bussiness_speech2text(
            file=file
        )

        # add task_id for task
        outputs_dict["task_id"] = task_id

        # response
        status_code = 200
        result = outputs_dict
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
