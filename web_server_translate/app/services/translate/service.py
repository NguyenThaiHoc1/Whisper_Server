import time

# logging
from app.utils import logger


class TranslateService(object):

    def __init__(self, model_translator, *args, **kwargs):
        self.model_translator = model_translator

    def do_service(self, texts, src_lang, tgt_lang, *args, **kwargs):
        start_time = time.time()
        logger.info(f"[service: do_service] Processing model ...")
        out = self.model_translator.do_activate(
            texts, src_lang, tgt_lang
        )
        # call service in here
        end_time = time.time() - start_time
        logger.info(f"[service: do_service]: Processing model is done. | time: {end_time}s")
        return out

    def do_service_intermi(self):