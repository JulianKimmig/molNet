from rdkit.Chem import MolFromSmiles
from molNet import MolGenerationError
from rdkit.Chem import Mol


def mol_from_smiles(smiles: str, raise_error: bool = True) -> Mol:
    m = MolFromSmiles(smiles)
    if m is None and raise_error:
        raise MolGenerationError(
            "cannot convert smiles '{}' to molecule".format(smiles)
        )

    return m
