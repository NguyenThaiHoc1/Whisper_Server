from app.services.triton.inference.infer_htdecmus import TritonHTDecmus
import torch


class HTDecmusServices(object):

    def __init__(self, input_name, output_name, model_name, model_version, triton_protocol):
        self.triton_protocol = triton_protocol

        self.triton_client = TritonHTDecmus(
            input_name=input_name,
            output_name=output_name,
            model_name=model_name,
            model_version=model_version,
            triton_protocol=triton_protocol
        )

    def do_htdecmus(self, torch_tensor):
        # Reprocessing
        dummy_input = torch.randn(1, 2, 343980).float().numpy()
        input_batch = [
            torch_tensor.numpy(),
        ]

        input_shape = [
            torch_tensor.shape
        ]

        # Sending request to triton server
        if self.triton_protocol == "http":
            response_out = self.triton_client.infer_http_active(
                input_batch, input_shape
            )
        elif self.triton_protocol == "grpc":
            response_out = self.triton_client.infer_grpc_active(
                input_batch, input_shape
            )
        else:
            raise ValueError(f'Invalid Triton protocol "{self.triton_protocol}", only accept http and grpc')

        # Postprocessing
        return response_out
