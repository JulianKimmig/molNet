import json

import numpy as np
import pandas as pd
import os

from molNet import get_user_folder, MOLNET_LOGGER



def generate_featurizer_df(featurizer_module):
    all_featurizer = []
    all_featurizer_names = []

    for n,mf in featurizer_module.get_available_featurizer().items():
        inf={}
        try:
            inf["length"] = len(mf)
        except TypeError:
            inf["length"] = -1
        inf["dtype"]=mf.dtype
        inf["instance"]=mf
        inf["class"]=mf.__class__
        inf["module"]=mf.__class__.__module__
        mf.ecdf_info=inf
        all_featurizer.append(mf)
        all_featurizer_names.append(str(n))


    raw_infos=[]
    for featurizer in all_featurizer:
        inf = featurizer.ecdf_info
        raw_infos.append(inf)

    new_infos=pd.DataFrame(raw_infos,
                       index=all_featurizer_names
                      )
    return new_infos

def get_molecule_featurizer_info():
    from molNet.featurizer import molecule_featurizer
    df = generate_featurizer_df(molecule_featurizer)
    return df

def get_atom_featurizer_info():
    from molNet.featurizer import atom_featurizer
    df = generate_featurizer_df(atom_featurizer)
    return df


def get_featurizer_folder():
    f=os.path.join(get_user_folder(),"featurizer")
    os.makedirs(f,exist_ok=True)
    return f


def reset_normalization_data():
    # resets the user featurizer data by the repository data
    MOLNET_LOGGER.debug("resets normalization excel data")
    import shutil
    norm_file=os.path.join(get_featurizer_folder(),"featurizer_norm.xlsx")
    shutil.copyfile(os.path.join(os.path.dirname(__file__),"featurizer_norm.xlsx"), norm_file)


def load_normalization_excel_data():
    # read the user provides normalization data to dataframe (if not provides uses the repository data)
    MOLNET_LOGGER.debug("load normalization excel data")

    norm_file=os.path.join(get_featurizer_folder(),"featurizer_norm.xlsx")
    if not os.path.exists(norm_file):
        reset_normalization_data()
    featurizer_norm = pd.read_excel(norm_file,index_col=[0,1], header=[0, 1,2])
    featurizer_norm.columns = featurizer_norm.columns.to_flat_index()
    return featurizer_norm


def recalculate_normalization_data():
    # recalculates the dictionary normalization data from the
    MOLNET_LOGGER.debug("recalculate normalization data")

    global _NORMALIZATION_DATA

    featurizer_norm = load_normalization_excel_data()
    data={}
    dfc=featurizer_norm.copy()
   # dfc.dropna(axis=0, how='all',inplace=True)
    for i,d in dfc.iterrows():
        featurizer_name, featurizer_position=i
        if featurizer_name not in data:
            data[featurizer_name]=[]

        _data=data[featurizer_name]

        while len(_data)<=featurizer_position:
            _data.append({})

        _sd=_data[featurizer_position]

        for idx in d.index:
            norm_name, data_range, params = idx

            if np.isnan(d[idx]):
                continue

            if norm_name not in _sd:
                _sd[norm_name]={}

            _ssd=_sd[norm_name]


            if data_range not in _ssd:
                _ssd[data_range]={"params":[],"R2":np.nan}

            if params=="R2":
                _ssd[data_range]["R2"]=d[idx]
            else:
                p=_ssd[data_range]["params"]
                while len(p)<=params:
                    p.append(np.nan)
                p[params]=d[idx]

    ndata={}
    for feat_name, d in data.items():
        ndata[feat_name]=[]
        for i,dimdata in enumerate(d):
            sd={}
            for norm_name, norm_data in dimdata.items():
                best_range=None
                for data_range, fit_params in norm_data.items():
                    if best_range is None:
                        best_range = fit_params
                    else:
                        if np.isnan(best_range["R2"]) or fit_params["R2"] < best_range["R2"]:
                            best_range = fit_params

                if np.isnan(best_range["R2"]):
                    continue
                sd[norm_name]=best_range
            ndata[feat_name].append(sd)
    with open(os.path.join(get_featurizer_folder(),"featurizer_norm.json"),"w+") as f:
        json.dump(ndata,f)

    _NORMALIZATION_DATA = {k.lower():v for k,v in ndata.items()}


def load_normalization_dict_data():
    MOLNET_LOGGER.debug("load normalization data")
    norm_file=os.path.join(get_featurizer_folder(),"featurizer_norm.json")

    if not os.path.exists(norm_file) :
        recalculate_normalization_data()
    else:
        xlsx_path = os.path.join(get_featurizer_folder(),"featurizer_norm.xlsx")
        if os.path.exists(xlsx_path) and os.path.getmtime(norm_file)<=os.path.getmtime(xlsx_path):
            recalculate_normalization_data()

    with open(norm_file,"r") as f:
        data = json.load(f)

    data = {k.lower():v for k,v in data.items()}
    return data




_NORMALIZATION_DATA = load_normalization_dict_data()

def get_normalization_data():
    return _NORMALIZATION_DATA

