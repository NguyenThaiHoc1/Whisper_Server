from transformers import GenerationConfig
from onnxruntime import InferenceSession
from transformers import WhisperProcessor

import app.services.whisper.lib as lib_whisper

onnx_path = '/Users/nguyenthaihoc/Desktop/FUJINET/whisper/web_server/app/services/whisper/model/CPU-whisper-large-v3_beamsearch.onnx'
generation_config = GenerationConfig.from_pretrained("openai/whisper-large-v3")
repetition_penalty = generation_config.repetition_penalty
sess = InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
processor = WhisperProcessor.from_pretrained("./WhisperProcessor")

whisper_vad_segmentation = './whisperx-vad-segmentation.bin'

model_whisper = lib_whisper.load_model(
    model_fp=whisper_vad_segmentation,
    sess=sess,
    processor=processor,
    repetition_penalty=repetition_penalty
)


import onnxruntime

from onnxruntime import RunOptions

hello = RunOptions()
hello.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0;gpu:0")
