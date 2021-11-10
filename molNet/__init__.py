import logging

MOLNET_LOGGER = logging.getLogger("molNet")


class SMILEError(Exception):
    pass


class MolGenerationError(Exception):
    pass


class ConformerError(Exception):
    pass

from molNet.utils.sys import set_user_folder,get_user_folder