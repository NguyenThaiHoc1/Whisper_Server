# get settings
from app.config.base_config import settings

# package
from pydub import AudioSegment

AudioSegment.converter = settings.PYFUB_AUDIOSEGMENT
