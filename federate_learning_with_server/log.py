import logging
import os
import sys
logger = logging.getLogger('my_logger')

def init_logging(log_level=logging.INFO, log_file='logs/default.log'):
    log_format = '[%(levelname)-5s]%(asctime)s-%(name)-6s-%(filename)-10s-%(lineno)-4s line : %(message)s'
    logger.setLevel(log_level)
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format,date_format)

    with open(log_file,"w"):
        pass
    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)