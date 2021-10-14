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
    func, data, cores=None, progess_bar=True, split_parts=None, progress_bar_kwargs={}
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
    r = []
    if cores > 1:
        with Pool(cores) as p:

            if progess_bar:
                with tqdm(total=len(data), **progress_bar_kwargs) as pbar:
                    for ri in p.imap(func, sub_data):
                        r.extend(ri)
                        pbar.update(len(ri))
            else:
                for ri in p.imap(func, sub_data):
                    r.extend(ri)
    else:
        if progess_bar:
            with tqdm(total=len(data), **progress_bar_kwargs) as pbar:
                for sd in sub_data:
                    r.extend(func(sd))
                    pbar.update(len(sd))
        else:
            for sd in sub_data:
                r.extend(func(sd))

    #for i in r:
    #    print(isinstance(i,np.ndarray))
    if len(r) > 0 and all([isinstance(ri, np.ndarray) for ri in r]):
        return np.array(r)
        #return np.concatenate(r,axis=0)
    return r
