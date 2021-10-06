import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import plotext as plt

if __name__ == "__main__":
    from rdkit import RDLogger
    import sys
    import logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    
    from os.path import abspath,dirname
    sys.path.append(dirname(dirname(abspath(__file__))))
    
    
from molNet.mol.molgraph import mol_graph_from_smiles
from molNet.utils.parallelization.multiprocessing import solve_cores
from molNet import ConformerError
from molNet import MOLNET_LOGGER

def _func(d):
    f = d[0][3]()
    r = np.zeros((len(d), len(f))) * np.nan
    for i, data in enumerate(d):
        mg = mol_graph_from_smiles(data[0], *data[1], **data[2])
        try:
            mg.featurize_mol(f, name="para_feats")
            r[i] = mg.as_arrays()["graph_features"]["para_feats"]
        except ConformerError:
            pass
    return r


def ecdf(data,res=None,smooth=False,unique_only=False):
    if data.ndim>1:
        data = np.squeeze(data)
        if data.ndim>1:
            return [ecdf(data[...,i],res=res,smooth=smooth,unique_only=unique_only) for i in range(data.shape[-1])]
    x=np.sort(data)
    n=len(data)
    y=np.arange(1,n+1)/n
    if smooth:
        x,uindices = np.unique(x,return_index =True)
        y = np.array([a.mean() for a in np.split(y,uindices[1:])])
        
    if res:
        dp=(np.linspace(0,1,res)*(len(x)-1)).astype(int)
        n=res
        x=x[dp]
        y=y[dp]
        
    if unique_only:
        x,uindices = np.unique(x,return_index =True)
        y=y[uindices]
    return x,y

def gen_ecdf(smiles,featurizer_class,ecdres=1000, th_patience=2_000,th=1e-4,es_patience=10_000,split=100,cores="all-1"):
    
    r = np.zeros((len(smiles),len(featurizer_class())))*np.nan
    
    cores = solve_cores(cores)
    MOLNET_LOGGER.info(f"Using {cores} cores")
    progress_bar_kwargs={}
    progess_bar = True
    gen_args=[]
    gen_kwargs={}
    
    
    min_error = np.inf
    
    data=np.array([(s, gen_args, gen_kwargs, featurizer_class) for s in smiles],dtype=object)
    sub_data = np.array_split(data, len(data) / split)
    
    _th_patience=th_patience
    _es_patience=es_patience
    precfd=np.zeros(ecdres*r.shape[1])
    error=[]
    
    rcp=0
    with Pool(cores) as p:
        if progess_bar:
            with tqdm(total=len(data), **progress_bar_kwargs) as pbar:
                for ri in p.imap(_func, sub_data):
                    lri=ri.shape[0]
                    
                    ri=ri[(~np.isnan(ri).any(axis=1))]
                    r[rcp:rcp+ri.shape[0]]=ri
                    rcp+=ri.shape[0]

                    necdf_data=ecdf(r,ecdres,smooth=True)
                    necdf_x=[]
                    necdf_y=[]
                    if isinstance(necdf_data,tuple):
                        necdf_x.append(necdf_data[0])
                        necdf_y.append(necdf_data[1])
                    else:
                        for d in necdf_data:
                            necdf_x.append(d[0])
                            necdf_y.append(d[1])
                    
                    cat_necdf_y = np.concatenate(necdf_y)
                    error.append(
                        np.sqrt(((cat_necdf_y-precfd)**2).mean())
                    )
                    
        
                    if error[-1]<th:
                        _th_patience-=lri
                        if _th_patience<=0:
                            MOLNET_LOGGER.info("stop due to  beeing long under threshold")
                            break
                    else:
                        _th_patience=th_patience
                        
                    if error[-1]<min_error:
                        min_error=error[-1]
                        _es_patience=es_patience
                        
                    else:
                        _es_patience-=lri
                        if _es_patience<=0:
                            MOLNET_LOGGER.info("stop due to not getting any better")
                            break

                    precfd=cat_necdf_y
                    pbar.set_postfix({'diff': error[-1]})
                    pbar.update(lri)
    r=r[(~np.isnan(r).any(axis=1))]
   
    return {"data":r,"diffs":error}
    
if __name__ == "__main__":
    from molNet.utils.smiles.generator import generate_n_random_hetero_carbon_lattice
    from molNet.featurizer.molecule_featurizer import MolWtFeaturizer,NumAtomsFeaturizer,CrippenDescriptorsFeaturizer
    
    smiles = np.array([k for k in generate_n_random_hetero_carbon_lattice(n=100_000, max_c=10)],dtype=object)
    
    ecdres=1000
    featurizer_class=CrippenDescriptorsFeaturizer
    
    ecdf_data = gen_ecdf(smiles,featurizer_class,ecdres=ecdres)
    
    necdf_data=ecdf(ecdf_data["data"],res=ecdres)
    necdf_x=[]
    necdf_y=[]
    if isinstance(necdf_data,tuple):
        necdf_x.append(necdf_data[0])
        necdf_y.append(necdf_data[1])
    else:
        for d in necdf_data:
            necdf_x.append(d[0])
            necdf_y.append(d[1])
    
    for i in range(len(necdf_x)):
        plt.plot(necdf_x[i],necdf_y[i], marker = "small")
    
    
    necdf_data=ecdf(ecdf_data["data"],res=ecdres,smooth=True,unique_only=True)#
    necdf_x=[]
    necdf_y=[]
    if isinstance(necdf_data,tuple):
        necdf_x.append(necdf_data[0])
        necdf_y.append(necdf_data[1])
    else:
        for d in necdf_data:
            necdf_x.append(d[0])
            necdf_y.append(d[1])
    
    for i in range(len(necdf_x)):
        plt.plot(necdf_x[i],necdf_y[i], marker = "small")
        
    plt.title(f"ecdf plot of {featurizer_class()}")
    plt.show()