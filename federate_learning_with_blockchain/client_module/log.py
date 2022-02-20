import logging
import os
import sys
import time
logger = logging.getLogger('=w=')

def init_logging(log_level=logging.INFO, log_file='logs/default.log'):
    log_format = '[%(levelname)-6s]%(asctime)s-%(name)-6s-%(filename)-15s-%(lineno)-4s line : %(message)s'
    date_format = '%Y-%m-%d %H-%M-%S'
    log_dir = '/Users/bytedance/blockchian/federate-learning-computing-market/federate_learning_with_blockchain/client_module/reports/logs'
    file_name  = time.strftime("%Y-%m-%d_%H:%M:%S",time.localtime(time.time())) + '.log'
    log_file = os.path.join(log_dir,file_name)
    logger.setLevel(log_level)
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