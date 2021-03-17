import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcNumAromaticRings

from molNet.dataloader.molecule_loader import MoleculeDfLoader
from molNet.mol.molecules import molecule_from_inchi

load_retention_times_df = None
load_retention_times_df_max_limit = 0


def load_retention_times(limit=None):
    global load_retention_times_df, load_retention_times_df_max_limit

    if load_retention_times_df is not None:
        if limit is None:
            return load_retention_times_df.copy()
        elif limit <= load_retention_times_df_max_limit:
            return load_retention_times_df.iloc[:limit].copy()

    print("load df")
    df = pd.read_csv("data/SMRT_dataset.csv", sep=";")
    if limit is None:
        limit = len(df)

    df = df.iloc[:limit].copy()

    dfl = MoleculeDfLoader(
        df,
        inplace=True,
        mol_create_source="inchi",
        mol_create_function=molecule_from_inchi,
    )
    dfl.setup()
    load_retention_times_df = df
    load_retention_times_df_max_limit = max(load_retention_times_df_max_limit, limit)
    return df.copy()


def load_default_df():
    dataset = pd.read_csv("list_chemicals-2020-12-05-21-46-06.tsv", sep="\t")
    dataset = dataset.append(
        {"SMILES": "C", "PREFERRED_NAME": "Methane"}, ignore_index=True
    )

    has_fags = dataset.index[dataset["SMILES"].apply(lambda x: "*" in x)]
    dataset.drop(has_fags, inplace=True)

    dataset["rd_mol"] = dataset["SMILES"].apply(lambda s: Chem.MolFromSmiles(s))

    no_mols = dataset.index[~dataset["rd_mol"].apply(lambda x: isinstance(x, Chem.Mol))]
    dataset.drop(no_mols, inplace=True)

    # dataset["rd_mol"] = dataset['rd_mol'].apply(lambda mol:
    #                                           ReplaceSubstructs(ReplaceSubstructs(Chem.AddHs(mol),
    #                                                             patt1,repl,replaceAll=True)[0],patt2,repl,replaceAll=True)[0])

    dataset["SMILES"] = dataset["rd_mol"].apply(lambda s: Chem.MolToSmiles(s))
    dataset["molar_mass"] = dataset["rd_mol"].apply(
        lambda mol: Chem.Descriptors.MolWt(mol)
    )
    dataset = dataset.rename({"PREFERRED_NAME": "name"}, axis=1)

    _loader = MoleculeGraphFromDfLoader(
        dataset,
        smiles_column="SMILES",
        batch_size=32,
        split=1,
        shuffle=False,
    )
    try:
        _loader.train_dataloader()
    except:
        _loader.setup()
    s = 0
    mol_graphs = []
    for d in _loader.train_dataloader():
        mol_graphs.extend(d)

    dataset["pre_graphs"] = mol_graphs

    dataset["hybridization"] = dataset["pre_graphs"].apply(
        lambda mg: np.array([atom_hybridization_one_hot(a) for a in mg.mol.GetAtoms()])
    )
    # dataset["hybridization_t"] = dataset['hybridization'].apply(lambda h: h.T)
    # for i,s in enumerate(atom_hybridization_one_hot.describe_features()):
    #    dataset[s]=dataset["hybridization"].apply(lambda h:h[:,i])

    return dataset


def find_test_smiles(dataset):
    sdf = dataset.copy()
    sdf = sdf.sort_values("molar_mass", axis=0)

    filter = sdf.rd_mol.apply(lambda mol: CalcNumAromaticRings(mol) > 0)
    if filter.sum() > 0:
        sdf = sdf[filter]

    patern = Chem.MolFromSmarts("[N+]([O-])=O")
    filter = sdf.rd_mol.apply(lambda mol: mol.HasSubstructMatch(patern))
    if filter.sum() > 0:
        sdf = sdf[filter]

    patern = Chem.MolFromSmarts("cOC")
    filter = sdf.rd_mol.apply(lambda mol: mol.HasSubstructMatch(patern))
    if filter.sum() > 0:
        sdf = sdf[filter]

    m = sdf.rd_mol.iloc[0]

    return Chem.MolToSmiles(m)
