from fastapi import APIRouter
from app.api import (
    ep_ping, ep_translate
)


api_router = APIRouter()

api_router.include_router(ep_ping.router, prefix="/ping", tags=["Service: Checking"])
api_router.include_router(ep_translate.router, prefix="/translate", tags=["Service: Handling translate"])
