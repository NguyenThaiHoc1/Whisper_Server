from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Pipeline

from types import MethodType
import torch

# import logger
from app.logging_utils import logger


class VerifySpeechService:

    def __init__(self, module_titanet):
        self.module_titanet = module_titanet

        self.get_embedding = PretrainedSpeakerEmbedding("nvidia/speakerverification_en_titanet_large")
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                                 use_auth_token="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")

        self._setup(module_titanet)

    def _setup(self, mod_tita):

        def tensorify(lst):
            """
            List must be nested list of tensors (with no varying lengths within a dimension).
            Nested list of nested lengths [D1, D2, ... DN] -> tensor([D1, D2, ..., DN)

            :return: nested list D
            """
            # base case, if the current list is not nested anymore, make it into tensor
            if type(lst[0]) != list:
                if type(lst) == torch.Tensor:
                    return lst
                elif type(lst[0]) == torch.Tensor:
                    tensor_list = torch.stack(lst, dim=0)
                    return tensor_list.squeeze(1)
                else:  # if the elements of lst are floats or something like that
                    return torch.tensor(lst)

            current_dimension_i = len(lst)
            for d_i in range(current_dimension_i):
                tensor = tensorify(lst[d_i])
                lst[d_i] = tensor
            # end of loop lst[d_i] = tensor([D_i, ... D_0])
            tensor_lst = torch.stack(lst, dim=0)
            tensor_lst = tensor_lst.squeeze(1)
            return tensor_lst

        def foward(self, input_signal, input_signal_length):
            processed_signal, processed_signal_len = self.preprocessor(input_signal=input_signal,
                                                                       length=input_signal_length)
            logits = []
            embs = []
            for processed_signal_single, processed_signal_len_single in zip(processed_signal, processed_signal_len):
                processed_signal_single = processed_signal_single.unsqueeze(0)
                processed_signal_len_single = processed_signal_len_single.unsqueeze(0)
                logits_single, embs_single = mod_tita.do_titanet(
                    processed_signal_single,
                    processed_signal_len_single
                )
                logits.append(
                    logits_single
                )
                embs.append(
                    embs_single
                )

            return tensorify(logits), tensorify(embs)

        self.get_embedding.model_.forward = MethodType(
            foward,
            self.get_embedding.model_
        )
        self.pipeline._inferences['_embedding'] = self.get_embedding

    def do_verifyspeech(self, file_path):

        logger.info("[service: do_verifyspeech] Verifyspeech service is running ...")
        # apply pretrained pipeline
        diarization = self.pipeline(file_path)

        # print the result
        result_output = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            result_output.append(
                {
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2),
                    "speaker": speaker
                }
            )
        logger.info("[service: do_verifyspeech] Verifyspeech service is complete.")
        return result_output
