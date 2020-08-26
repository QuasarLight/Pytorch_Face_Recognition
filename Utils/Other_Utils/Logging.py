import os
import logging

def init_logger(output_dir):
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create stream handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    # create file handler
    fh = logging.FileHandler(os.path.join(output_dir, 'Training_Log.txt'))
    fh.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y%m%d-%H:%M:%S')

    # add formatter to fh
    fh.setFormatter(formatter)

    # add sh and fh to logger
    # The final log level is the higher one between the default and the one in handler
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger
