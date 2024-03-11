# logger
from app.utils import logger

# other function
import time


class FujiTranslateBusiness(object):

    def __init__(self, services_class, *args, **kwargs):
        self.services_class = services_class

    def do_bus(self, texts, src_lang, tgt_lang, *args, **kwargs):
        if texts is None or texts == "":
            raise ValueError("Please Enter your sentence or short text")

        global_start_time = time.time()

        # Reprocess
        start_time = time.time()
        logger.info(f"[Bus: do_bus_translate] processing ...")
        # please provide script code for process input in here

        end_time = time.time() - start_time
        logger.info(f"[Bus: do_bus_translate] process is done. | time: {end_time}s")

        # Calling service
        start_time = time.time()
        logger.info(f"[Bus: do_bus_translate] in calling progress service ...")
        output_result = self.services_class.do_service(
            texts=texts,
            src_lang=src_lang,
            tgt_lang=tgt_lang
        )
        end_time = time.time() - start_time
        logger.info(f"[Bus: do_bus_translate] process is done. | time: {end_time}s")

        # Postprocess
        start_time = time.time()
        logger.info(f"[Bus: do_bus_translate] post processing ...")
        # please provide script code for post process output in here

        end_time = time.time() - start_time
        logger.info(f"[Bus: do_bus_translate] post processing. | time: {end_time}s")

        global_end_time = time.time() - global_start_time
        return {
            "start_time": global_start_time,
            "end_time": global_end_time,
            "result": output_result
        }
