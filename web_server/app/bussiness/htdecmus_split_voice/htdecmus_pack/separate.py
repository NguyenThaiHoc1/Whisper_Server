from app.bussiness.htdecmus_split_voice.htdecmus_pack.audio import AudioFile


class Separator:
    def __init__(
            self,
            device="cpu",
            overlap=0.25,
            model_service=None,
            process_input_class=None,
    ):
        self._overlap = overlap
        self._device = device
        self._audio_channels = 2
        self._samplerate = 44100
        self.sources = ['drums', 'bass', 'other', 'vocals']

        # class invlove
        self.model_service = model_service
        self.process_input_class = process_input_class

    def separate_audio_file(self, file):
        assert self.model_service is not None, "Please provide model service"

        wav = AudioFile(file).read(streams=0, samplerate=self._samplerate, channels=self._audio_channels)
        ref = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std() + 1e-8

        # minh nghi minh se phai lam lai khuc nay
        out = self.process_input_class.do_progress(
            model_service=self.model_service,
            mix=wav[None],
            overlap=self._overlap,
            device=self._device,
        )

        out *= ref.std() + 1e-8
        out += ref.mean()
        wav *= ref.std() + 1e-8
        wav += ref.mean()

        return wav, dict(zip(self.sources, out[0]))
