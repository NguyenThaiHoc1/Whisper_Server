import io
import uuid
from fastapi import UploadFile

from app.api.utils import AudioSegment


def get_extension_file(filename):
    extension_file = filename.split('.')
    return extension_file[0], str.lower(extension_file[1])


async def read_file(audio_file_bytes: UploadFile, **kwargs) -> dict:
    # get debug mode
    path_storage = kwargs.get("path_storage")
    extension_file = kwargs.get("extension")
    raw_filename = kwargs.get("filename")

    # check debug mode
    assert path_storage is not None, "Please check storage path."
    assert extension_file in ["mp3", "wav"], "Please check your file extension. Must be 'mp3' or 'wav'"

    # read data from file
    content = await audio_file_bytes.read()

    # Decode the audio
    if extension_file == "wav":  # (assuming it's in WAV format)
        audio = AudioSegment.from_wav(io.BytesIO(content))
    elif extension_file == "mp3":  # (assuming it's in WAV format)
        audio = AudioSegment.from_file(io.BytesIO(content))
    else:
        raise ValueError("We can't init audio variables of file. Please check ...")

    # Save the decoded content to a file on the server
    decoded_raw_filename = str(raw_filename + "." + extension_file)
    decode_file_name = f"{uuid.uuid4().hex}_decoded_{decoded_raw_filename}"
    decoded_file_path = path_storage / decode_file_name

    audio.export(decoded_file_path, format=extension_file)

    return {
        "filename": decode_file_name,
        "filepath": decoded_file_path,
        "duration": audio.duration_seconds,
        "size": len(content)
    }
