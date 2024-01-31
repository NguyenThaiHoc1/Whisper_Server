from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import ep_main


def include_route(app) -> None:
    app.include_router(
        ep_main.api_router, prefix="/test"
    )


def start_application():
    app = FastAPI(title="Hello world", version="1.0")
    include_route(
        app
    )
    return app


app = start_application()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
