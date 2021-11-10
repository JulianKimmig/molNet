import os
if "MOLNET_DIR" not in os.environ:
    os.environ["MOLNET_DIR"]=os.path.join(os.path.expanduser("~"),".molNet")

def set_user_folder(path):
    os.environ["MOLNET_DIR"]=os.abspath(path)
    
def get_user_folder():
    return os.environ["MOLNET_DIR"]