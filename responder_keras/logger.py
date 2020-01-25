from logging import ERROR, INFO
from logging.handlers import WatchedFileHandler
import logging


class Logger:

    @staticmethod
    def generate_logger(clazz_name):
        logger = logging.getLogger(clazz_name)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')

        handler = logging.StreamHandler()
        handler.setLevel(INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        file_handler = WatchedFileHandler('log/app.log')
        file_handler.setLevel(INFO)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(LoggingFilter(ERROR))
        logger.addHandler(file_handler)

        logger.setLevel(INFO)
        logger.propagate = False

        return logger


class LoggingFilter(object):
    def __init__(self, level):
        self.__level = level

    def filter(self, log_record):
        return log_record.levelno <= self.__level