import tritonclient.http as tritonhttpclient
import tritonclient.grpc as tritongrpcclient
from app.services.triton import triton_client


class TitaNet:
    def __init__(self, input_name, output_name, model_name, model_version, triton_protocol):
        self.input_name = input_name
        self.output_name = output_name
        self.model_name = model_name
        self.model_version = model_version
        self.triton_protocol = triton_protocol

    def infer_http_active(self, input_batch, input_shape):
        # Define input config
        inputs = [
            tritonhttpclient.InferInput(self.input_name[0], input_shape[0], 'FP32'),
            tritonhttpclient.InferInput(self.input_name[1], input_shape[1], 'FP32'),
        ]

        # Attach input
        inputs[0].set_data_from_numpy(input_batch[0], binary_data=True)
        inputs[1].set_data_from_numpy(input_batch[1], binary_data=True)

        # Define output config
        outputs = [
            tritonhttpclient.InferRequestedOutput(self.output_name[0], binary_data=True),
            tritonhttpclient.InferRequestedOutput(self.output_name[1], binary_data=True),
        ]
        response = triton_client.infer(self.model_name, model_version=self.model_version,
                                       inputs=inputs, outputs=outputs)
        param1 = response.as_numpy(self.output_name[0])
        param2 = response.as_numpy(self.output_name[1])
        return param1, param2

    def infer_grpc_active(self, input_batch, input_shape):
        # Define input config
        inputs = [
            tritongrpcclient.InferInput(self.input_name[0], input_shape[0], 'FP32'),
            tritongrpcclient.InferInput(self.input_name[1], input_shape[1], 'FP32'),
        ]

        # Attach input
        inputs[0].set_data_from_numpy(input_batch[0])
        inputs[1].set_data_from_numpy(input_batch[1])

        # Define output config
        outputs = [
            tritongrpcclient.InferRequestedOutput(self.output_name[0]),
            tritongrpcclient.InferRequestedOutput(self.output_name[1]),
        ]

        response = triton_client.infer(self.model_name,
                                       model_version=self.model_version,
                                       inputs=inputs,
                                       outputs=outputs)
        param1 = response.as_numpy(self.output_name[0])
        param2 = response.as_numpy(self.output_name[1])
        return param1, param2
