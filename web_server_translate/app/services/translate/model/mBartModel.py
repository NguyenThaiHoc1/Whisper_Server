import torch
import numpy as np
from transformers import MBart50TokenizerFast
from onnxruntime import InferenceSession


class MBartModel(object):

    def __init__(self, model_path, tokenizer_path, device, *args, **kwargs):

        # saving parameters
        self.model_path = model_path
        self.device = device

        if self.device == "cpu":
            self._exc_provider = ["CPUExecutionProvider"]
        elif self.device == "cuda":
            self._exc_provider = ["CUDAExecutionProvider"]
        else:
            raise ValueError("Don't have any type device. Use cpu or cuda")

        assert len(self._exc_provider) > 0, "Please type of device ..."

        # Token path: 'facebook/mbart-large-50-many-to-many-mmt'
        self.tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
        self.sess = InferenceSession(self.model_path, providers=self._exc_provider)

        # setup model
        self.num_beams = 5
        self.min_length = 0
        self.max_length = 512
        self.repetition_penalty = 1.0
        self.no_repeat_ngram_size = 0
        self.num_return_sequences = 1

    def do_activate(self, texts, src_lang, tgt_lang):
        self.tokenizer.src_lang = src_lang
        encoder_inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        tgt_lang_ids = self.tokenizer.lang_code_to_id[tgt_lang]
        decoder_input_ids = torch.tensor([2, tgt_lang_ids], dtype=torch.long).tile(encoder_inputs.input_ids.shape[0], 1)

        # setup inputs
        encoder_inputs_ids = encoder_inputs.input_ids.to(torch.int32).to('cpu')
        encoder_attention_mask = encoder_inputs.attention_mask.to(torch.int32).to('cpu')
        decoder_input_ids = decoder_input_ids.to(torch.int32).to('cpu')

        ort_inputs = {
            "input_ids": np.int32(encoder_inputs_ids.cpu().numpy()),
            "max_length": np.array([self.max_length], dtype=np.int32),
            "min_length": np.array([self.min_length], dtype=np.int32),
            "num_beams": np.array([self.num_beams], dtype=np.int32),
            "num_return_sequences": np.array([self.num_return_sequences], dtype=np.int32),
            "length_penalty": np.array([1], dtype=np.float32),
            "repetition_penalty": np.array([self.repetition_penalty], dtype=np.float32),
            "attention_mask": np.int32(encoder_attention_mask.cpu().numpy()),
            "decoder_input_ids": np.int32(decoder_input_ids.cpu().numpy()),
        }

        # run-model
        out = self.sess.run(None, ort_inputs)

        # process ouput
        trans_texts = []
        for out_ids in out[0]:
            trans_text = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            trans_texts.append(trans_text)
        return trans_texts
