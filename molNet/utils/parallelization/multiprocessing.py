from multiprocessing import cpu_count, Pool

import numpy as np
from tqdm import tqdm

from molNet import MOLNET_LOGGER


def solve_cores(cores="all-1"):
    if cores is None:
        cores = "all-1"
    n_cores = cpu_count()
    if isinstance(cores, int):
        cores = cores
    elif isinstance(cores, str):
        if "all" in cores:
            if "all-" in cores:
                n_cores = n_cores - int(cores.replace("all-", ""))
            elif cores == "all":
                pass
            else:
                raise ValueError("Cannot get core number from '{cores}'")
            cores = n_cores
        else:
            cores = int(cores)
    else:
        raise ValueError(f"unknown core type('{type(cores)}')")
    cores = max(1, min(n_cores, int(cores)))
    return cores


def parallelize(
    func, data, cores=None, progess_bar=True, split_parts=None, progress_bar_kwargs={},target_array=None
):
    # data = np.array(data)
    cores = solve_cores(cores)

    l = len(data)
    if split_parts is None or split_parts < 1:
        split_parts = cores

    # perfect_split=int(np.ceil(l / cores))
    p = min(l, split_parts)
    MOLNET_LOGGER.debug(f"using {cores} cores to work on {p} fragments")

    pl = l // p
    pl1 = pl + 1
    r = l % p
    sr = [[i * pl1, (i + 1) * pl1] for i in range(r)] + [
        [i * pl + r, (i + 1) * pl + r] for i in range(r, p)
    ]
    sub_data = (data[i:k] for i, k in sr)
    # sub_data = np.array_split(data, min(max_split, int(np.ceil(len(data) / cores))))
    class ResAdder():
        def __init__(self,target=None,is_array=False):
            if target is None:
                target=[]
                self.external_target=False
            else:
                self.external_target=True
            
            self.target=target
            self.is_array=is_array
            self.pos=0
            
            if self.is_array and self.external_target:
                self._nan_eq = (np.ones_like(target[0])*np.nan).astype(target.dtype)
            else:
                self._nan_eq=None
        
        def add(self,ri,l):
            if self.is_array and self.external_target:
                for j in range(len(ri)):
                    if ri[j] is None:
                        ri[j]=self._nan_eq
                self.target[self.pos:self.pos+l]=ri
            else:
                self.target.extend(ri)
                
            self.pos+=l
        
        def get_target(self):
            if not self.external_target and self.is_array:
                none_on_none=False
                for j in range(len(self.target)):
                    if ri[j] is None:
                        if self._nan_eq is None:
                            none_on_none=True
                        else:
                            ri[j]=self._nan_eq
                    else:
                        if self._nan_eq is None:
                            self._nan_eq = (np.ones_like(ri[j])*np.nan).astype(ri[j].dtype)
                            if none_on_none:
                                break
                if none_on_none and self._nan_eq is not None:
                    for j in range(len(self.target)):
                        if ri[j] is None:
                            ri[j]=self._nan_eq
                            
                return np.array(self.target)
            return self.target
                
            
    if target_array is None:
        r=ResAdder()
    else:
        r=ResAdder(target_array,is_array=True)
               
    if progess_bar:
        def _iterate(p=None):
            with tqdm(total=len(data), **progress_bar_kwargs) as pbar:
                if p is not None:
                    for ri in p.imap(func, sub_data):
                        l=len(ri)
                        r.add(ri,l)
                        pbar.update(l)
                else:
                    for sd in sub_data:
                        l=len(sd)
                        r.add(func(sd),l)
                        pbar.update(l)
    else:
        def _iterate(p=None):
            if p is not None:
                for ri in p.imap(func, sub_data):
                    l=len(ri)
                    r.add(ri,l)
            else:
                for sd in sub_data:
                    l=len(sd)
                    r.add(func(sd),l)
                
    if cores > 1:
        with Pool(cores) as p:
            _iterate(p)  
    else:
        _iterate(None)   
    
    if r.external_target:
        return r.get_target()
    if r.pos > 0 and r.is_array:
        return np.array(r)
        #return np.concatenate(r,axis=0)
    return r.get_target()
