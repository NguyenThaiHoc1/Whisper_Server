from fastapi import APIRouter
from web_server.app.api import (
    ep_ping
)

api_router = APIRouter()

api_router.include_router(ep_ping.router, prefix="/ping", tags=["Service: Checking"])
api_router.include_router(None, prefix="/audio", tags=["Service: Handling audio"])
