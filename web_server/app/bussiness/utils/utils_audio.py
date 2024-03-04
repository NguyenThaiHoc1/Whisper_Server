import io
import uuid
from pathlib import Path
import torchaudio as ta
from fastapi import UploadFile

from app.bussiness.utils import AudioSegment


def create_directory(directory_name, base_path='.'):
    """
    Creates a directory with the given name under the specified base path.

    Args:
    directory_name (str): The name of the directory to be created.
    base_path (str): The base path where the directory will be created. Default is the current directory.

    Returns:
    Path: The absolute path of the created directory as a Path object.
    """
    # Construct the full path
    directory_path = Path(base_path) / directory_name

    # Check if the directory already exists
    if not directory_path.exists():
        # Create the directory if it doesn't exist
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{directory_name}' created at '{directory_path}'")
    else:
        print(f"Directory '{directory_name}' already exists at '{directory_path}'")

    return directory_path


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
    decoded_raw_filename = str(raw_filename)
    decoded_file_name = f"{uuid.uuid4().hex}_decoded_{decoded_raw_filename}"
    decoded_file_name_with_extension = decoded_file_name + "." + extension_file
    decoded_file_path = path_storage / decoded_file_name_with_extension

    audio.export(decoded_file_path, format=extension_file)

    return {
        "filename_with_ext": decoded_file_name_with_extension,
        "filename_no_ext": decoded_file_name,
        "filepath": decoded_file_path,
        "duration": audio.duration_seconds,
        "size": len(content)
    }


def save_audio(wav, path, samplerate, bitrate=320, bits_per_sample=16, preset=2):
    wav = wav / max(1.01 * wav.abs().max(), 1)
    path = Path(path)
    suffix = path.suffix.lower()
    ta.save(str(path), wav, sample_rate=samplerate, encoding='PCM_S', bits_per_sample=bits_per_sample)
