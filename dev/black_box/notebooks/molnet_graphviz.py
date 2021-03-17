from graphviz import Digraph


def atom_input_graph(mol_graph, name, reduced=True, reduce_up=2, graph=None):
    if mol_graph is None:
        reduced = True
    else:
        if len(mol_graph) <= reduce_up:
            reduced = False

    if reduced:
        atoms = ["atom{}".format(i) for i in range(reduce_up)] + ["atomn"]
        draw_atoms = [[a, a] for a in atoms[:-1]] + [
            ["reduce", "..."],
            [atoms[-1], atoms[-1]],
        ]
    else:
        atoms = ["atom{}".format(i) for i in range(len(mol_graph))]
        draw_atoms = [[a, a] for a in atoms]

    if graph is None:
        g = Digraph(name, node_attr={"shape": "record"})
        g.attr(rankdir="LR", size="8,8")
        g.attr(compound="true")
        g.attr(splines="polyline")
    else:
        g = graph

    node_name = "{}_node".format(name)
    g.node(
        node_name, table(np.array([[[np, ns]] for np, ns in draw_atoms])), shape="none"
    )
    return g, {
        "ports": atoms,
        "draw_atoms": draw_atoms,
        "reduced": reduced,
        "node_name": node_name,
    }


def poolsum_to_graph(self, name="PoolSum", input_shape=(4, 3), reduced=True):
    g = Digraph(name, node_attr={"shape": "record"})
    g.attr(rankdir="LR", size="8,8")
    g.attr(compound="true")
    g.attr(splines="polyline")

    g.node(
        "{}_op_feats".format(name),
        label=table(
            np.array(
                [
                    [["op_{}".format(x), "opfeat{}".format(x)]]
                    for x in range(input_shape[0])
                ]
            )
        ),
        shape="none",
    )

    for i in range(input_shape[0]):
        g.node("{}_sum{}".format(name, i), label="+", shape="circle")
        g.edge("{}_sum{}".format(name, i), "{}_op_feats:op_{}".format(name, i))
    for i in range(input_shape[1]):
        pg = g
        with g.subgraph(name="cluster_{}_input{}".format(name, i)) as g:
            g.attr(color="#ffffdd", style="filled")
            g.node(
                "{}_ip_feat{}".format(name, i),
                label=table(
                    np.array(
                        [
                            [["ip_{}_{}".format(i, x), "feat{} ip{}".format(x, i)]]
                            for x in range(input_shape[0])
                        ]
                    )
                ),
                shape="none",
            )
        g = pg
        for x in range(input_shape[0]):
            g.edge(
                "{}_ip_feat{}:ip_{}_{}".format(name, i, i, x),
                "{}_sum{}".format(name, x),
                arrowhead="none",
            )

    return g


PoolSum.to_graphviz = poolsum_to_graphviz
