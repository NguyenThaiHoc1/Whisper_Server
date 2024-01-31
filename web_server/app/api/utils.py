import io
from pydub import AudioSegment
import uuid
from fastapi import UploadFile


async def read_file(audio_file_bytes: UploadFile) -> dict:
    # read data from file
    content = await audio_file_bytes.read()

    # Decode the audio (assuming it's in WAV format)
    audio = AudioSegment.from_wav(io.BytesIO(content))

    # Save the decoded content to a file on the server
    decode_file_name = f"{uuid.uuid4().hex}_decoded_{audio_file_bytes.filename.replace(' ', '_')}"
    decoded_file_path = f"storage/{decode_file_name}"
    audio.export(decoded_file_path, format="wav")

    return {
        "filename": decode_file_name,
        "filepath": decoded_file_path
    }
