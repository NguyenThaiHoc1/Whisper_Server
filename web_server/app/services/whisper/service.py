import web_server.app.services.whisper.lib as lib_whisper

from transformers import GenerationConfig
from onnxruntime import InferenceSession
from transformers import WhisperProcessor


class WhisperService(object):
    onnx_path = ""
    whisper_processor_path = ""
    whisper_generation_path = ""
    whisper_vad_segmentation = ""

    def __init__(self, *args, **kwargs):
        generation_config = GenerationConfig.from_pretrained(self.whisper_generation_path)
        repetition_penalty = generation_config.repetition_penalty
        sess = InferenceSession(self.onnx_path, providers=["CUDAExecutionProvider"])
        processor = WhisperProcessor.from_pretrained(self.whisper_processor_path)

        self.model_whisper = lib_whisper.load_model(
            model_fp=self.whisper_vad_segmentation,
            sess=sess,
            processor=processor,
            repetition_penalty=repetition_penalty
        )

    def do_whisperer(self, audio_file):
        audio = lib_whisper.load_audio(
            audio_file
        )
        data = self.model_whisper.transcribe(
            audio, batch_size=1
        )
        return data
