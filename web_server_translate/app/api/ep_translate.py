# FastAPI
from fastapi import (
    APIRouter, HTTPException,
    File, UploadFile, Form,
)

# typing
from typing import Dict

# logger
from app.utils import logger

# bus
from app.businesses import bus_translate

router = APIRouter()


@router.post('/text2translate',
             description='do speech audio compile to text',
             status_code=200,
             include_in_schema=True,
             response_model=dict)
async def do_speech2text(text: str = Form(...), src_lang: str = Form(...), tgt_lang: str = Form(...)) -> Dict:
    try:
        # calling bussiness
        out = bus_translate.do_bus(
            texts=text,
            src_lang=src_lang,
            tgt_lang=tgt_lang
        )
        # add task_id for task

        # response
        status_code = 200
        result = out
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
