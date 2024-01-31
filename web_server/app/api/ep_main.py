from fastapi import APIRouter
from app.api import (
    ep_ping, ep_audio
)

api_router = APIRouter()

api_router.include_router(ep_ping.router, prefix="/ping", tags=["Service: Checking"])
api_router.include_router(ep_audio.router, prefix="/audio", tags=["Service: Handling audio"])
