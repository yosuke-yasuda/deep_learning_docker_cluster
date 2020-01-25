import os
import requests
from logger import Logger

logger = Logger.generate_logger(__name__)

def notify_finish(endpoint=None, *args, **kwargs):
    if endpoint is not None:
        try:
            requests.patch(endpoint)
        except Exception as e:
            logger.error("faild finish notification of frame_key '%s' by error '%s'" % (frame_key, str(e)), exc_info=True)
    else:
        logger.info("ignore finish notification because endpoint is not indicated")