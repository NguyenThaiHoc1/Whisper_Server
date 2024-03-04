import torch
from tqdm import tqdm
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor

from app.bussiness.htdecmus_split_voice.htdecmus_pack.apply import (
    TensorChunk,
    center_trim
)


class ProcessInput(object):

    def __init__(self, num_workers: int = 8):
        self.pool = ThreadPoolExecutor(max_workers=num_workers)

    def _run_model(self, model_service, mix, device, samplerate, segment):
        length = mix.shape[-1]
        valid_length = int(segment * samplerate)
        mix = TensorChunk(mix)
        padded_mix = mix.padded(valid_length).to(device)
        with torch.no_grad():
            out = torch.Tensor(model_service.do_htdecmus(padded_mix))  # call services or model
        return center_trim(out, length)

    def do_progress(self, model_service,
                    mix,
                    overlap,
                    len_model_sources: int = 4,
                    samplerate: int = 44100,
                    device: str = 'cpu',
                    segment=Fraction(39, 5),
                    transition_power: float = 1.,
                    *args, **kwargs):

        model_weights = [1.] * len_model_sources
        totals = [0.] * len_model_sources
        batch, channels, length = mix.shape

        segment_length: int = int(samplerate * segment)
        stride = int((1 - overlap) * segment_length)
        offsets = range(0, length, stride)
        futures = []
        for offset in offsets:
            chunk = TensorChunk(mix, offset, segment_length)
            future = self.pool.submit(self._run_model, model_service, chunk, device, samplerate, segment)
            futures.append((future, offset))
            offset += segment_length

        if True:
            scale = float(format(stride / samplerate, ".2f"))
            futures = tqdm(futures, unit_scale=scale, ncols=120, unit='seconds')

        out = torch.zeros(batch, len_model_sources, channels, length, device=mix.device)
        sum_weight = torch.zeros(length, device=mix.device)
        weight = torch.cat([torch.arange(1, segment_length // 2 + 1, device=device),
                            torch.arange(segment_length - segment_length // 2, 0, -1, device=device)])
        weight = (weight / weight.max()) ** transition_power
        for future, offset in futures:
            chunk_out = future.result()  # get result from pool/features
            chunk_length = chunk_out.shape[-1]
            out[..., offset:offset + segment_length] += (weight[:chunk_length] * chunk_out).to(mix.device)
            sum_weight[offset:offset + segment_length] += weight[:chunk_length].to(mix.device)
        out /= sum_weight

        for k, inst_weight in enumerate(model_weights):
            out[:, k, :, :] *= inst_weight
            totals[k] += inst_weight
        for k in range(out.shape[1]):
            out[:, k, :, :] /= totals[k]
        return out
