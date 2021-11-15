import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
MOLNET_LOGGER = logging.getLogger("molNet")

class SMILEError(Exception):
    pass


class MolGenerationError(Exception):
    pass


class ConformerError(Exception):
    pass

from molNet.utils.sys import set_user_folder,get_user_folder