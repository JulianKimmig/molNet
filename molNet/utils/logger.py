import logging
import os
import sys
from logging.handlers import RotatingFileHandler
import coloredlogs

from molNet.utils.sys import get_user_folder, _USERFOLDERCHANGELISTENER

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# turn off matplotlib logger
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# turn off RDLogger
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# turn off numba
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

MOLNET_LOGGER = logging.getLogger("molNet")

os.makedirs(os.path.join(get_user_folder(), 'logs'), exist_ok=True)
filehandler = RotatingFileHandler(os.path.join(get_user_folder(), 'logs', 'molNet.log'), maxBytes=2 ** 20,
                                  backupCount=10)
filehandler.setLevel(MOLNET_LOGGER.level)
# create console handler with a higher log level
streamhandler = logging.StreamHandler()
streamhandler.setLevel(MOLNET_LOGGER.level)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
filehandler.setFormatter(formatter)
streamhandler.setFormatter(formatter)
# add the handlers to the logger
MOLNET_LOGGER.addHandler(filehandler)
MOLNET_LOGGER.addHandler(streamhandler)

coloredlogs.install(level=MOLNET_LOGGER.level, logger=MOLNET_LOGGER)


def set_level(level=logging.INFO):
    MOLNET_LOGGER.setLevel(level)
    filehandler.setLevel(level)
    streamhandler.setLevel(level)
    # TODO coloredlogs set_level


def _on_dir_change(d):
    os.makedirs(os.path.join(d, 'logs'), exist_ok=True)
    filehandler.baseFilename = os.path.join(d, 'logs', 'molNet.log')


_USERFOLDERCHANGELISTENER.append(_on_dir_change)
