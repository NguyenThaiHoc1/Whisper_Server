import app.services.whisper.lib as lib_whisper

from transformers import GenerationConfig
from onnxruntime import InferenceSession
from transformers import WhisperProcessor

import time


class WhisperService(object):
    onnx_path = ""
    whisper_processor_path = ""
    whisper_generation_path = ""
    whisper_vad_segmentation = ""
    device = "cpu"
    _exc_provider = []

    def __init__(self, *args, **kwargs):
        if self.device == "cpu":
            self._exc_provider = ["CPUExecutionProvider"]
        elif self.device == "gpu":
            self._exc_provider = ["CPUExecutionProvider", "CUDAExecutionProvider"]
        else:
            raise ValueError("Don't have any type device. Use cpu or cuda")

        assert len(self._exc_provider) > 0, "Please type of device ..."

        generation_config = GenerationConfig.from_pretrained(self.whisper_generation_path)
        repetition_penalty = generation_config.repetition_penalty
        sess = InferenceSession(self.onnx_path, providers=self._exc_provider)
        processor = WhisperProcessor.from_pretrained(self.whisper_processor_path)

        self.model_whisper = lib_whisper.load_model(
            model_fp=self.whisper_vad_segmentation,
            sess=sess,
            processor=processor,
            repetition_penalty=repetition_penalty,
            device=self.device
        )

    def do_whisperer(self, audio_file):
        start_time = time.time()
        print("Processing file audio ...")
        print(f"start-time: {start_time}.")
        audio = lib_whisper.load_audio(
            audio_file
        )
        end_time = time.time() - start_time
        print(f"end-time: {end_time}.")
        print("Processing file audio is done.")

        start_time = time.time()
        print("Processing model ...")
        print(f"start-time: {start_time}.")
        data = self.model_whisper.transcribe(
            audio, batch_size=1
        )
        end_time = time.time() - start_time
        print(f"end-time: {end_time}.")
        print("Processing model is done.")
        return data
