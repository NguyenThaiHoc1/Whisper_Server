from app.config.base_config import settings
from app.services.triton.protocol import TritonClientGRPC, TritonClientHTTP

class_triton_client = None

if settings.TRITON_SERVICES_PROTOCOL == "http":
    class_triton_client = TritonClientHTTP(domain=settings.TRITON_SERVICES_URL)

elif settings.TRITON_SERVICES_PROTOCOL == "grpc":
    class_triton_client = TritonClientGRPC(domain=settings.TRITON_SERVICES_URL)

assert class_triton_client is not None, "Triton client class not activate."

triton_client = class_triton_client.triton_client
