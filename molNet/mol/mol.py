from rdkit.Chem import MolFromSmiles, Mol
import inspect
from functools import partial

from molNet import ConformerError
from molNet.utils.parallelization.multiprocessing import parallelize
from typing import Dict, List, Tuple

def _single_call_parallel_featurize_mol(
    mols: List[Mol],
    featurizer
):
    featurizer = featurizer() if inspect.isclass(featurizer) else featurizer
    d=[]
    for mol in mols:
        try:
            d.append(featurizer(mol))
        except (ConformerError, ValueError, ZeroDivisionError):
            d.append(None)
    return d

def _single_call_parallel_featurize_mol_atoms(
        mols: List[Mol],
        featurizer
):
    featurizer = featurizer() if inspect.isclass(featurizer) else featurizer
    d=[]
    for mol in mols:
        for atom in mol.GetAtoms():
            try:
                d.append(featurizer(atom))
            except (ConformerError, ValueError, ZeroDivisionError):
                d.append(None)
    return d
    

def parallel_featurize_mol(
    mols: List[Mol],
    featurizer,
    cores="all-1",
    progess_bar=True,
    target_array=None,
    **kwargs
):
    kwargs["progress_bar_kwargs"]={**dict(unit=" mol"),**kwargs.get("progress_bar_kwargs",{})}
    kwargs["progress_bar_kwargs"]["total"]=len(mols)
    return parallelize(
        partial(_single_call_parallel_featurize_mol,featurizer=featurizer),
        mols,
        cores=cores,
        progess_bar=progess_bar,
        target_array=target_array,
        **kwargs
    )

def parallel_featurize_mol_atoms(
            mols: List[Mol],
            featurizer,
            cores="all-1",
            progess_bar=True,
            target_array=None,
            n_atoms=None,
            **kwargs
    ):
        kwargs["progress_bar_kwargs"]={**dict(unit=" mol"),**kwargs.get("progress_bar_kwargs",{})}
        kwargs["progress_bar_kwargs"]["total"]=n_atoms if n_atoms is not None else sum([m.GetNumAtoms() for m in mols])
        return parallelize(
            partial(_single_call_parallel_featurize_mol_atoms,featurizer=featurizer),
            mols,
            cores=cores,
            progess_bar=progess_bar,
            target_array=target_array,
            **kwargs
    )