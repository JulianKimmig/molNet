from ._autogen_molecule_featurizer_list import *
from ._autogen_molecule_featurizer_numeric import *
from ._autogen_molecule_featurizer_numpy_arrays import *
from ._autogen_molecule_featurizer_rdkit_vec import *

from ._autogen_molecule_featurizer_list import (
    _available_featurizer as _available_featurizer_list,
)
from ._autogen_molecule_featurizer_numeric import (
    _available_featurizer as _available_featurizer_numeric,
)
from ._autogen_molecule_featurizer_numpy_arrays import (
    _available_featurizer as _available_featurizer_numpy_arrays,
)
from ._autogen_molecule_featurizer_rdkit_vec import (
    _available_featurizer as _available_featurizer_rdkit_vec,
)

_available_featurizer = {
    **_available_featurizer_list,
    **_available_featurizer_numeric,
    **_available_featurizer_numpy_arrays,
    **_available_featurizer_rdkit_vec,
}
