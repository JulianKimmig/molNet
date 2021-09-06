from typing import Dict

import numpy as np

# import torch
# import torch_geometric
import torch
import torch_geometric
from numpy import ndarray

from molNet.mol.molgraph import MolGraph


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
        x_node_features = np.concatenate(x_node_features)  # Shape[size,n]
    else:
        x_node_features = np.zeros((size, 0))

    y_node_features = [
        v.reshape((size, -1)) for n, v in node_features.items() if n.startswith("_y_")
    ]

    if len(y_node_features) > 0:
        y_node_features = np.concatenate(y_node_features)  # Shape[size,m]
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
                    y_graph_features.append(v.flatten().astype(np.float32))
                    if include_graph_features_titles:
                        add_kwargs["y_graph_features_titles"].append(n)
                else:
                    x_graph_features.append(v.flatten().astype(np.float32))
                    if include_graph_features_titles:
                        add_kwargs["x_graph_features_titles"].append(n)
        elif isinstance(v, str):
            if keep_string_data:
                add_kwargs["string_data"].append(v)
                add_kwargs["string_data_titles"].append(n)
        else:
            add_kwargs[n] = v

    if len(x_graph_features) > 0:
        x_graph_features = np.array(x_graph_features, dtype=np.float32)
    else:
        x_graph_features = np.zeros((1, 0), dtype=np.float32)

    if len(y_graph_features) > 0:
        y_graph_features = np.array(y_graph_features, dtype=np.float32)
    else:
        y_graph_features = np.zeros((1, 0), dtype=np.float32)
    # print(x_graph_features, y_graph_features)

    edge_index = np.zeros((2, eges.shape[0] * 2), dtype=int)
    edge_index[:, ::2] = eges.T
    edge_index[0, 1::2] = eges[:, 1]
    edge_index[1, 1::2] = eges[:, 0]

    data = torch_geometric.data.data.Data(
        x=torch.from_numpy(x_node_features).float(),
        y=torch.from_numpy(y_node_features).float(),
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


def assert_graph_input_data_equal(
    gip1: torch_geometric.data.data.Data, gip2: torch_geometric.data.data.Data
):
    d1 = gip1.to_dict()
    d2 = gip2.to_dict()

    for _d1, _d2 in ((d1, d2), (d2, d1)):
        for k, v in _d1.items():
            if not np.array_equal(v.shape, _d2[k].shape):
                raise GraphInputEqualsException(
                    "feature shape missmatch('{}')".format(k)
                )
            if not torch.allclose(v, _d2[k]):
                raise GraphInputEqualsException(
                    "feature value missmatch('{}')".format(k)
                )

            print(k, v.shape, _d2[k].shape)


def graph_input_data_equal(
    gip1: torch_geometric.data.data.Data, gip2: torch_geometric.data.data.Data
):
    try:
        assert_graph_input_data_equal(gip1, gip2)
    except GraphInputEqualsException:
        return False
    return True
