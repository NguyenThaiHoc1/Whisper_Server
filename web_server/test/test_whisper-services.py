from app.services.whisper.service import WhisperService

path_audio_file = '/Users/nguyenthaihoc/Desktop/FUJINET/whisper/web_server/test/audio/vi_vocals.wav'

# static  variable
WhisperService.onnx_path = '../app/services/whisper/model/whisper-large-v3_beamsearch.onnx'
WhisperService.whisper_processor_path = '../app/services/whisper/model/WhisperProcessor/'
WhisperService.whisper_generation_path = '../app/services/whisper/model/WhisperGenerationConfig/'
WhisperService.whisper_vad_segmentation = '../app/services/whisper/model/whisperx-vad-segmentation.bin'

# init class
service_whis = WhisperService()

output = service_whis.do_whisperer(
    audio_file=path_audio_file
)

print(output)
