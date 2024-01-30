import os
import warnings
from typing import List, Union, Optional, NamedTuple

# import ctranslate2
# import faster_whisper
import numpy as np
import torch
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator

from .audio import N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram
from .vad import load_vad_model, merge_chunks

default_asr_options = {
    "beam_size": 5,
    "best_of": 5,
    "patience": 1,
    "length_penalty": 1,
    "repetition_penalty": 1,
    "no_repeat_ngram_size": 0,
    "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": False,
    "prompt_reset_on_temperature": 0.5,
    "initial_prompt": None,
    "prefix": None,
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": True,
    "max_initial_timestamp": 0.0,
    "word_timestamps": False,
    "prepend_punctuations": "\"'“¿([{-",
    "append_punctuations": "\"'.。,，!！?？:：”)]}、",
    "suppress_numerals": False,
}
default_vad_options = {
    "vad_onset": 0.500,
    "vad_offset": 0.363
}


def find_numeral_symbol_tokens(tokenizer):
    numeral_symbol_tokens = []
    for i in range(tokenizer.eot):
        token = tokenizer.decode([i]).removeprefix(" ")
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(i)
    return numeral_symbol_tokens


def inference_func(features, options, sess, processor, repetition_penalty):
    batch_size = features.shape[0]
    ort_inputs = {
        "input_features": np.array(features, dtype=np.float32),
        "max_length": np.array([448], dtype=np.int32),
        "min_length": np.array([0], dtype=np.int32),
        "num_beams": np.array([5], dtype=np.int32),
        "num_return_sequences": np.array([1], dtype=np.int32),
        "length_penalty": np.array([1], dtype=np.float32),
        "repetition_penalty": np.array([repetition_penalty], dtype=np.float32),
        "decoder_input_ids": np.array([[50258, 50278, 50360, 50364]] * batch_size, dtype=np.int32),
    }

    out = sess.run(None, ort_inputs)[0]
    text = []
    for s in out:
        text.append(processor.batch_decode(s, skip_special_tokens=True))
    return text


class WhisperPipeline(Pipeline):
    def __init__(
            self,
            vad,
            options,
            tokenizer,
            sess,
            processor,
            repetition_penalty,
            **kwargs
    ):
        self.tokenizer = tokenizer
        self.options = options
        self._batch_size = kwargs.pop("batch_size", None)
        self._num_workers = 1
        self._preprocess_params, self._forward_params, self._postprocess_params = \
            self._sanitize_parameters(**kwargs)
        self.call_count = 0
        self.framework = "pt"
        self.device = torch.device('cpu')

        super(Pipeline, self).__init__()
        self.vad_model = vad
        self._vad_params = {
            "vad_onset": 0.500,
            "vad_offset": 0.363
        }

        # additional
        self.sess = sess
        self.processor = processor
        self.repetition_penalty = repetition_penalty

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "tokenizer" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, audio, **kwargs):
        audio = audio['inputs']
        model_n_mels = 128
        features = log_mel_spectrogram(
            audio,
            n_mels=model_n_mels if model_n_mels is not None else 80,
            padding=N_SAMPLES - audio.shape[0],
        )
        return {'inputs': features}

    def _forward(self, model_inputs, **kwargs):
        outputs = inference_func(
            model_inputs['inputs'], self.options,
            sess=self.sess,
            processor=self.processor,
            repetition_penalty=self.repetition_penalty
        )
        return {'text': outputs}

    def postprocess(self, model_outputs, **kwargs):
        return model_outputs

    def get_iterator(
            self, inputs,
            num_workers: int,
            batch_size: int,
            preprocess_params,
            forward_params,
            postprocess_params
    ):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        def stack(items):
            return {'inputs': torch.stack([x['inputs'] for x in items])}

        dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size,
                                                 collate_fn=stack)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

    def transcribe(
            self,
            audio,
            batch_size=4,
            chunk_size=30,

    ):
        def data(audio, segments):
            for seg in segments:
                f1 = int(seg['start'] * SAMPLE_RATE)
                f2 = int(seg['end'] * SAMPLE_RATE)
                yield {'inputs': audio[f1:f2]}

        vad_segments = self.vad_model(
            {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )

        segments = []
        total_segments = len(vad_segments)
        for idx, out in enumerate(self.__call__(
                data(audio, vad_segments), batch_size=batch_size, num_workers=0)):
            text = out['text']
            if batch_size in [0, 1, None]:
                text = text[0]
            segments.append({
                "text": text,
                "start": round(vad_segments[idx]['start'], 3),
                "end": round(vad_segments[idx]['end'], 3)
            })
        return segments


def load_model(model_fp, sess, processor, repetition_penalty):
    vad_model = load_vad_model(torch.device('cuda'),
                               model_fp=model_fp,
                               **default_vad_options)

    tokenizer = processor.tokenizer

    return WhisperPipeline(
        vad=vad_model,
        options=default_asr_options,
        tokenizer=tokenizer,
        sess=sess,
        processor=processor,
        repetition_penalty=repetition_penalty
    )
