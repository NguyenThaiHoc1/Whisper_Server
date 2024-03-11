from typing import Dict
from fastapi import APIRouter

router = APIRouter()


@router.get('/hello-world',
            description='testing server hello world',
            status_code=200,
            response_model=dict)
def ping() -> Dict:
    return {
        "message": "hello world"
    }
