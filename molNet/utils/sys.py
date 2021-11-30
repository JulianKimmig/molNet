import os
if "MOLNET_DIR" not in os.environ:
    os.environ["MOLNET_DIR"]=os.path.join(os.path.expanduser("~"),".molNet")

os.makedirs(os.environ["MOLNET_DIR"],exist_ok=True)
_ENV_FILE=os.path.join(os.environ["MOLNET_DIR"],".env")
LOKAL_ENVS={}

_USERFOLDERCHANGELISTENER=[]

def _read_env():
    if not os.path.exists(_ENV_FILE):
        with open(_ENV_FILE,"w+"):
            pass
    with open(_ENV_FILE,"r") as f:
        cont = f.read()
    
    for l in cont.split("\n"):
        try:
            k,v=l.split("=",1)
            LOKAL_ENVS[k]=v
        except Exception:
            pass
    for k,v in LOKAL_ENVS.items():
        os.environ[k]=v

def _write_env():
    cont="\n".join([f"{k}={v}" for k,v in LOKAL_ENVS.items()])+"\n"
    with open(_ENV_FILE,"w+") as f:
        f.write(cont)
        
_read_env()
            
    
def set_user_folder(path,permanent=False):
    os.environ["MOLNET_DIR"]=os.path.abspath(path)
    os.makedirs(os.environ["MOLNET_DIR"],exist_ok=True)


    #update log dir
    for cl in _USERFOLDERCHANGELISTENER:
        cl(get_user_folder())


    if permanent:
        LOKAL_ENVS["MOLNET_DIR"]=os.environ["MOLNET_DIR"]
        _write_env()
        
def get_user_folder():
    return os.environ["MOLNET_DIR"]


