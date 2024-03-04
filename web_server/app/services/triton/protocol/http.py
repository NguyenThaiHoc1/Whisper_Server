import tritonclient.http as tritonhttpclient
from app.services.triton.protocol.base import TritonClient


class TritonClientHTTP(TritonClient):
    def __init__(self, domain, port=8000):
        super().__init__(domain=domain)
        self.port = port  # default is 8000
        self.abs_url = f"{self.domain}:{self.port}"
        self._init_triton_client()
        self.show_information()

    def _init_triton_client(self):
        verbose = False
        self.triton_client = tritonhttpclient.InferenceServerClient(url=self.abs_url, verbose=verbose)

    def show_information(self):
        print(f"Triton client")
        print(f"Protocol        :   HTTP")
        print(f"URL             :   {self.abs_url}")
