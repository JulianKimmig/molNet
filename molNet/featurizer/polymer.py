from molNet.featurizer.featurizer import (
    FeaturizerList,
    LambdaFeaturizer,
    OneHotFeaturizer,
)
from molNet.featurizer.molecule_featurizer import default_molecule_featurizer


def mol_to_polyunit_featurizer(mol_featurizer):
    return LambdaFeaturizer(
        lamda_call=lambda connectable_group: mol_featurizer(connectable_group.mol),
        length=len(mol_featurizer),
    )


number_of_connections_one_hot = OneHotFeaturizer(
    possible_values=list(range(6)) + [None],
    pre_featurize=lambda connectable_group: len(connectable_group.connection_indices),
)
number_of_connections = LambdaFeaturizer(
    lamda_call=lambda connectable_group: [len(connectable_group.connection_indices)],
    length=1,
)

default_repeating_unit_featurizer = FeaturizerList(
    [
        default_molecule_featurizer,
        number_of_connections_one_hot,
    ]
)
