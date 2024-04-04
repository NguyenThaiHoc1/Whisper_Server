# Utils
from app.bussiness.utils import utils_audio

# loggers
from app.logging_utils import logger

# regularly function
import time
from datetime import datetime


class HTDECMuscSplitVoice(object):
    def __init__(self, device, path_storage,
                 service_model_name,
                 serivce_model_version,
                 separator_class,
                 *args, **kwargs):
        assert separator_class is not None, "Please provide separator."
        # func-variables to call
        self.separator = separator_class

        # info-variables to call
        self.path_storage = path_storage
        self.service_model_name = service_model_name
        self.service_model_version = serivce_model_version

    async def __call__(self, *args, **kwargs):
        file = kwargs.get('file')
        assert file is not None, "Please check your file"
        global_s_time = datetime.now()

        # 1: Read extension
        logger.info(f"[func: do_htdecmus] starting sfunc: get_extension_file ...")
        start_time = time.time()
        filename, extension_file = utils_audio.get_extension_file(
            file.filename
        )
        end_time = time.time() - start_time
        logger.info(f"[func: do_htdecmus]: '{filename}' '{extension_file}' | time: {end_time}s")

        # 2: Decode file
        logger.info(f"[func: do_htdecmus] starting sfunc: utils_audio.read_file ...")
        start_time = time.time()
        dict_decode_file = await utils_audio.read_file(
            audio_file_bytes=file,
            path_storage=self.path_storage,
            extension=extension_file,
            filename=filename
        )
        end_time = time.time() - start_time
        logger.info(f"[func: do_htdecmus]: Decode file path is successful. | time: {end_time}s")

        # 3: Create folder
        output_abs_path = utils_audio.create_directory(
            directory_name=str(dict_decode_file["filename_no_ext"]),
            base_path='app/bussiness/htdecmus_split_voice/saved_template'
        )

        logger.info(f"{output_abs_path}")

        # 4: Call Separate audio
        output_file_audio, output_result = self.separator.separate_audio_file(
            file=dict_decode_file["filepath"]
        )

        # 5: Save audio
        for name, source in output_result.items():
            stem = output_abs_path / f"{dict_decode_file['filename_no_ext']}_{name}.wav"
            stem.parent.mkdir(parents=True, exist_ok=True)
            utils_audio.save_audio(source, str(stem), samplerate=self.separator._samplerate)

        filename_with_return = output_abs_path / f"{dict_decode_file['filename_no_ext']}_vocals.wav"
        global_e_time = datetime.now()

        # return FileResponse(str(filename_with_return), media_type="audio/mpeg")
        return {
            "start_time": global_s_time,
            "end_time": global_e_time,
            "model_name": self.service_model_name + " | " + self.service_model_version,
            "file_name": f"{dict_decode_file['filename_no_ext']}_vocals.wav",
            "output_path": filename_with_return
        }
