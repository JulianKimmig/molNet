import pandas as pd

def generate_featurzizer_df(featurizer_module):
    all_featurizer = []
    all_featurizer_names = []

    for n,mf in featurizer_module.get_available_featurizer().items():
        inf={}
        try:
            inf["length"] = len(mf)
        except TypeError:
            inf["length"] = -1
        inf["dtype"]=mf.dtype.__name__
        inf["class"]=mf.__class__
        inf["module"]=mf.__class__.__module__
        mf.ecdf_info=inf
        all_featurizer.append(mf)
        all_featurizer_names.append(f'{inf["module"]}.{n}')


    raw_infos=[]
    for featurizer in all_featurizer:
        inf = featurizer.ecdf_info
        raw_infos.append(inf)

    new_infos=pd.DataFrame(raw_infos,
                       index=all_featurizer_names
                      )
    return new_infos

def get_molecule_featurizer_info():
    df = generate_featurzizer_df(molecule_featurizer)
    return df

def get_atom_featurizer_info():
    df = generate_featurzizer_df(atom_featurizer)
    return df