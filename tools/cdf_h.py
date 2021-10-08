import numpy as np
from molNet.mol.molgraph import mol_graph_from_smiles
from molNet import ConformerError
import os 
import json

def func(d):
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

def generate_code_inject(classname,data_folder="ecdf_data"):
    df_cont=os.listdir(data_folder)
    avail_norms= [f.replace(".data","") for f in df_cont if ".data" in f]
    avail_norms=[f for f in avail_norms if f+".data" in df_cont]
    avail_norms=[f for f in avail_norms if f+".ecdf" in df_cont]

    if classname in avail_norms:
        #print(s["classname"])
        with open(os.path.join(data_folder,classname+".data"),"r") as f:
            ecdf_data=json.load(f)["0"]
        precode=""
        best=None
        for datakey,parakey,best_key in [
            ("linear_norm","linear_norm_parameter", "linear"),
            ("min_max_norm","min_max_norm_parameter","min_max"),
            ("sig_norm","sigmoidal_norm_parameter","sig"),
            ("dual_sig_norm","dual_sigmoidal_norm_parameter", "dual_sig"),
            ("genlog_norm","genlog_norm_parameter","genlog"),
        ]:
            if datakey in ecdf_data:
                norm_data=ecdf_data[datakey]
                precode+=f"    {parakey} = ({', '.join([str(i) for i in norm_data['parameter']])})"+\
                f"  # error of {norm_data['error']:.2E} with sample range ({norm_data['sample_bounds'][0][0]:.2E},{norm_data['sample_bounds'][0][1]:.2E}) resulting in fit range ({norm_data['sample_bounds'][1][0]:.2E},{norm_data['sample_bounds'][1][1]:.2E})\n"
        
            if 'sample_bounds99' not in norm_data or norm_data['sample_bounds'][0][0] == norm_data['sample_bounds'][0][1]:
                best = ("unity",0,norm_data['sample_bounds'])
            else:
                if norm_data['sample_bounds'][1][0]<=0.3 and norm_data['sample_bounds'][1][1]>0.5:
                    if best is None:
                        best = (best_key,norm_data['error'],norm_data['sample_bounds'])
                    else:
                        if norm_data['error']<best[1]:
                            best = (best_key,norm_data['error'],norm_data['sample_bounds'])

        if best is not None:
            precode+=f"    preferred_normalization = '{best[0]}'"
    
        return precode
    return None
