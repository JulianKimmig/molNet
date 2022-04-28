from typing import Dict

import numpy as np

# import torch
# import torch_geometric
import torch
import torch_geometric
from numpy import ndarray

from molNet.mol.molgraph import MolGraph

def graph_input_from_edgelist(edgelist,node_features,y=None,graph_features=None):
    #assert both connection directions
    edgelist=np.concatenate((edgelist,edgelist[[1,0],:]),axis=1)
    edgelist=np.unique(edgelist,axis=1)

    return torch_geometric.data.Data(
        edge_index=torch.from_numpy(edgelist).long(),
        y=None if y is None else torch.from_numpy(y.astype(float)).float(),
        x=torch.from_numpy(node_features.astype(float)).float(),
        num_nodes=len(node_features),
       # edge_index=torch.from_numpy(edge_index).long(),
        graph_features=None if graph_features is None else torch.from_numpy(graph_features.astype(float)).float()
    )

def molgraph_arrays_to_graph_input(
    size: int,
    eges: ndarray,
    node_features: Dict[str, ndarray],
    graph_features: Dict[str, ndarray],
    keep_string_data=False,
    include_graph_features_titles=False,
    **add_kwargs
):
    # print(graph_features)

    for n, v in node_features.items():
        if v.shape[0] != size:
            raise ValueError(
                "node feature shape mismatch given size ({},{},{})".format(
                    n, v.shape[0], size
                )
            )

    # merge node features
    x_node_features = [
        v.reshape((size, -1))
        for n, v in node_features.items()
        if not n.startswith("_y_")
    ]
    if len(x_node_features) > 0:
        x_node_features = np.concatenate(x_node_features, axis=1)  # Shape[size,n]
    else:
        x_node_features = np.zeros((size, 0))

    y_node_features = [
        v.reshape((size, -1)) for n, v in node_features.items() if n.startswith("_y_")
    ]

    if len(y_node_features) > 0:
        y_node_features = np.concatenate(y_node_features, axis=1)  # Shape[size,m]
    else:
        y_node_features = np.zeros((size, 0))

    x_graph_features = []
    y_graph_features = []
    if include_graph_features_titles:
        add_kwargs["x_graph_features_titles"] = []
        add_kwargs["y_graph_features_titles"] = []
    if keep_string_data:
        add_kwargs["string_data"] = []
        add_kwargs["string_data_titles"] = []
    for n, v in graph_features.items():
        if isinstance(v, ndarray):
            if np.issubdtype(v.dtype, np.number) or np.issubdtype(v.dtype, np.bool_):
                if n.startswith("_y_"):
                    y_graph_features.extend(v.flatten().astype(np.float32))
                    if include_graph_features_titles:
                        add_kwargs["y_graph_features_titles"].append(n)
                else:
                    x_graph_features.extend(v.flatten().astype(np.float32))
                    if include_graph_features_titles:
                        add_kwargs["x_graph_features_titles"].append(n)
        elif isinstance(v, str):
            if keep_string_data:
                add_kwargs["string_data"].append(v)
                add_kwargs["string_data_titles"].append(n)
        else:
            add_kwargs[n] = v

    if len(x_graph_features) > 0:
        x_graph_features = np.expand_dims(
            np.array(x_graph_features, dtype=np.float32), axis=0
        )
    else:
        x_graph_features = np.zeros((1, 0), dtype=np.float32)

    if len(y_graph_features) > 0:
        y_graph_features = np.expand_dims(
            np.array(y_graph_features, dtype=np.float32), axis=0
        )
    else:
        y_graph_features = np.zeros((1, 0), dtype=np.float32)
    # print(x_graph_features, y_graph_features)

    edge_index = np.zeros((2, eges.shape[0] * 2), dtype=int)
    edge_index[:, ::2] = eges.T
    edge_index[0, 1::2] = eges[:, 1]
    edge_index[1, 1::2] = eges[:, 0]

    data = torch_geometric.data.data.Data(
        x=torch.from_numpy(x_node_features.astype(float)).float(),
        y=torch.from_numpy(y_node_features.astype(float)).float(),
        num_nodes=size,
        edge_index=torch.from_numpy(edge_index).long(),
        x_graph_features=torch.from_numpy(x_graph_features).float(),
        y_graph_features=torch.from_numpy(y_graph_features).float(),
        **add_kwargs
    )
    return data


def molgraph_to_graph_input(molgraph: MolGraph, **add_kwargs):
    return molgraph_arrays_to_graph_input(**molgraph.as_arrays(), **add_kwargs)


class GraphInputEqualsException(Exception):
    pass


def assert_graph_input_keys_equal(
    gip1: torch_geometric.data.data.Data, gip2: torch_geometric.data.data.Data
):

    d1 = gip1.to_dict()
    d2 = gip2.to_dict()

    for k in d1.keys():
        if k not in d2:
            raise GraphInputEqualsException("feature keys missmatch('{}')".format(k))
    for k in d2.keys():
        if k not in d1:
            raise GraphInputEqualsException("feature keys missmatch('{}')".format(k))

    return d1, d2


def assert_graph_input_shape_equal(
    gip1: torch_geometric.data.data.Data, gip2: torch_geometric.data.data.Data
):
    d1, d2 = assert_graph_input_keys_equal(gip1, gip2)

    # if node dependent
    s11 = d1["x"].shape[0]
    s21 = d2["x"].shape[0]

    # if edge dependent
    s12 = d1["edge_index"].shape[1]
    s22 = d2["edge_index"].shape[1]
    ss1 = s21 - s11
    ss2 = s22 - s12
    try:
        for k, v in d1.items():
            if hasattr(v, "shape"):
                if not len(v.shape) == len(d2[k].shape):
                    raise GraphInputEqualsException(
                        "feature dimensions missmatch('{}')".format(k)
                    )

                sa1 = np.array(v.shape)
                sa2 = np.array(d2[k].shape)
                z1 = sa1 - sa2
                z2 = z1 + ss1
                z3 = z1 + ss2
                if not np.all((z1 * z2) == 0) and not np.all((z1 * z3) == 0):
                    raise GraphInputEqualsException(
                        "feature shape missmatch('{}')".format(k)
                    )
    except GraphInputEqualsException as e:
        raise e
    except Exception as e:
        raise e.__class__(str(e) + "\n" + str(d1) + "\n" + str(2))
    return d1, d2


def assert_graph_input_data_equal(
    gip1: torch_geometric.data.data.Data, gip2: torch_geometric.data.data.Data
):
    d1, d2 = assert_graph_input_shape_equal(gip1, gip2)

    for k, v in d1.items():
        if isinstance(v, torch.Tensor):
            if not torch.allclose(v, d2[k]):
                raise GraphInputEqualsException(
                    "feature value missmatch('{}')".format(k)
                )


def graph_input_keys_equal(
    gip1: torch_geometric.data.data.Data, gip2: torch_geometric.data.data.Data
):
    try:
        assert_graph_input_keys_equal(gip1, gip2)
    except GraphInputEqualsException:
        return False
    return True


def graph_input_data_equal(
    gip1: torch_geometric.data.data.Data, gip2: torch_geometric.data.data.Data
):
    try:
        assert_graph_input_data_equal(gip1, gip2)
    except GraphInputEqualsException:
        return False
    return True


def graph_input_shape_equal(
    gip1: torch_geometric.data.data.Data, gip2: torch_geometric.data.data.Data
):
    try:
        assert_graph_input_shape_equal(gip1, gip2)
    except GraphInputEqualsException:
        return False
    return True
