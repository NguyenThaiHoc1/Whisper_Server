import numpy as np

from app.services.triton.inference.infer_verifyspeech import TitaNet
import torch

# import logger


class TitaNetServices(object):

    def __init__(self, input_name, output_name, model_name, model_version, triton_protocol):
        self.triton_protocol = triton_protocol

        self.triton_client = TitaNet(
            input_name=input_name,
            output_name=output_name,
            model_name=model_name,
            model_version=model_version,
            triton_protocol=triton_protocol
        )

    def do_titanet(self, processed_signal, processed_signal_len):
        # Reprocessing
        # dummy_audio_signal = torch.randn(2, 80, 1008).float().numpy()
        # dummy_length = torch.randn(2).float().numpy()

        np_processed_signal = processed_signal.float().numpy()
        np_processed_signal_len = processed_signal_len.float().numpy()[:, np.newaxis]

        input_batch = [
            np_processed_signal,
            np_processed_signal_len
        ]
        input_shape = [
            np_processed_signal.shape,
            np_processed_signal_len.shape
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
        logits = torch.from_numpy(
            response_out[0],
        )

        embs = torch.from_numpy(
            response_out[0],
        )
        return logits, embs
