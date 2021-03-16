import os

from json_dict import JsonDict

MOLNET_PATH = os.path.join(os.path.expanduser("~"), ",.molNet")
os.makedirs(MOLNET_PATH, exist_ok=True)

CONFIG = JsonDict(os.path.join("config.json"), autosave=False)
