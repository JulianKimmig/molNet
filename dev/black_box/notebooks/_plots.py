import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

import os
import numpy as np
from _defaults import *


def plot_true_pred(model, loader, target_file=None):
    true = []
    pred = []
    try:
        loader.test_dataloader()
    except:
        loader.setup()
    for i, d in enumerate(loader.test_dataloader()):
        pred.extend(model(d.to(model.device)).detach().cpu().numpy().flatten())
        true.extend(d.y.detach().cpu().numpy().flatten())

    true, pred = np.array(true),np.array(pred)
    print(true.shape,pred.shape)
    plt.plot(true, pred, "o")

    if target_file is None:
        plt.show()
    else:
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        plt.savefig(target_file, dpi=DEFAULT_DPI)
    plt.close()


def plot_category_validation(
    model, loader, categories, target_file=None, ignore_empty=True
):
    true = []
    pred_correct = []
    pred_wrong = []
    for i, d in enumerate(loader.test_dataloader()):
        pred_m = model(d.to(model.device)).detach().cpu().numpy()
        p = pred_m.argmax(1)
        t = d.y.detach().cpu().numpy().argmax(1)
        pred_correct.extend(p[p == t])
        pred_wrong.extend(p[p != t])
        true.extend(t)

    # plt.hist(true)
    # plt.hist(pred)
    categories = np.array(categories)

    labels_true, counts_true = np.unique(true, return_counts=True)
    labels_pred_correct, counts_pred_correct = np.unique(
        pred_correct, return_counts=True
    )
    labels_pred_wrong, counts_pred_wrong = np.unique(pred_wrong, return_counts=True)

    if ignore_empty:
        all_labels = np.array(
            list(set(labels_true) | set(labels_pred_correct) | set(labels_pred_wrong))
        )
        label_list = all_labels.tolist()
        labels_true = np.array([label_list.index(l) for l in labels_true])
        labels_pred_correct = np.array(
            [label_list.index(l) for l in labels_pred_correct]
        )
        labels_pred_wrong = np.array([label_list.index(l) for l in labels_pred_wrong])

    plt.bar(labels_true - 0.2, counts_true, align="center", width=0.2, label="true")
    plt.bar(
        labels_pred_correct,
        counts_pred_correct,
        align="center",
        width=0.2,
        label="correct predicted",
    )
    plt.bar(
        labels_pred_wrong + 0.2,
        counts_pred_wrong,
        align="center",
        width=0.2,
        label="wrong predicted",
    )

    # n, bins, patches = plt.hist([true,pred], len(categories), density=False)
    # print(bins, len(categories))
    x = np.arange(len(categories))
    if ignore_empty:
        x = np.arange(len(all_labels))
        categories = categories[all_labels]
    # print(x,categories)
    plt.xticks(ticks=x, labels=categories, rotation=90, horizontalalignment="left")
    plt.legend()
    plt.tight_layout()
    if target_file is None:
        plt.show()
    else:
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        plt.savefig(target_file, dpi=DEFAULT_DPI)
    plt.close()


class MidpointNormalize(mpl.colors.Normalize):
    """Normalise the colorbar."""

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def plot_fcnn(
    layer_sizes,
    weights=None,
    biases=False,
    show_bar=False,
    input_labels=None,
    weight_position=None,
    round_weights=2,
    edge_width=1,
    save=None,
    show=None,
    hide_loose=False,
    cmap=plt.cm.coolwarm,
    nodes_cmap=plt.cm.coolwarm,
    input_array=None,
    layer_norm=False,
):
    g = nx.Graph()
    pos = {}
    w = 0
    if weights is not None:
        for i in range(len(weights)):
            assert weights[i].shape[0] == layer_sizes[i + 1], (
                weights[i].shape[0],
                layer_sizes[i + 1],
            )
            assert weights[i].shape[1] == layer_sizes[i], (
                weights[i].shape[1],
                layer_sizes[i],
            )

            if layer_norm:
                weights[i] = weights[i] / np.abs(weights[i]).max()
            weights[i] = np.round(weights[i], round_weights)

    for l, n in enumerate(layer_sizes):
        for i in range(n):
            node = "{}_{}".format(l, i)
            node_d = {"layer": l, "layer_pos": i, "show": True}

            if input_labels is not None:
                if len(input_labels) > len(g):
                    node_d["label"] = input_labels[len(g)]
            g.add_node(node, **node_d)
            # pos[node]=(l*50,-(i-n/2)*10)
            if l > 0:
                for j in range(layer_sizes[l - 1]):
                    pnode = "{}_{}".format(l - 1, j)
                    ed = {"show": True}
                    if weights is not None:
                        # display(weights[l-1])
                        # display(l,n,j,i)
                        ed["w"] = weights[l - 1][i][j]
                        if hide_loose and ed["w"] == 0:
                            ed["show"] = False
                        w += 1
                    g.add_edge(pnode, node, **ed)

    if hide_loose:
        for node, nd in g.nodes(data=True):
            if g.edges(node) == 0 or all(
                [not g.get_edge_data(*e)["show"] for e in g.edges(node)]
            ):
                g.nodes[node]["show"] = False

    nodes_kwargs = {}
    if input_array is not None and weights is not None:
        node_values = [input_array[: layer_sizes[0]]]
        for i, w in enumerate(weights):
            node_values.append(np.dot(w, node_values[i]))

        if layer_norm:
            for i in range(len(node_values)):
                node_values[i] = node_values[i] / np.abs(node_values[i]).max()

        for node, nd in g.nodes(data=True):
            g.nodes[node]["value"] = node_values[nd["layer"]][nd["layer_pos"]]

        vmin = min([w.min() for w in node_values])
        vmax = max([w.max() for w in node_values])
        sm_nodes = plt.cm.ScalarMappable(
            cmap=nodes_cmap, norm=MidpointNormalize(vmin, vmax, 0.0)
        )

        nodes_kwargs["node_color"] = [
            sm_nodes.to_rgba(nd["value"]) for n, nd in g.nodes(data=True) if nd["show"]
        ]

    # reposition
    layer_pos = {ln: 0 for ln in range(l + 1)}
    showing_layer_size = [0] * len(layer_sizes)
    for node, nd in g.nodes(data=True):
        if nd["show"]:
            showing_layer_size[nd["layer"]] += 1

    for node, nd in g.nodes(data=True):
        if nd["show"]:
            l = nd["layer"]
            pos[node] = (l * 50, -((layer_pos[l] - showing_layer_size[l] / 2) * 10))
            layer_pos[l] += 1
        # else:
        #    pos[node]=(0,0)

    while showing_layer_size[-1] == 0:
        showing_layer_size.pop(-1)

    fs = (2 * (len(showing_layer_size) + 1), 1 + max(showing_layer_size) / 3)

    if weights is not None:
        vmin = min([w.min() for w in weights])
        vmax = max([w.max() for w in weights])
        sm_edges = plt.cm.ScalarMappable(
            cmap=cmap, norm=MidpointNormalize(vmin, vmax, 0.0)
        )

    fig = plt.figure(figsize=fs)
    nodes = nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=[n for n, nd in g.nodes(data=True) if nd["show"]],
        **nodes_kwargs,
    )

    max_end_width = 0
    if input_labels:
        for i, (node, data) in enumerate(g.nodes(data=True)):
            if data["show"] and "label" in data and data["label"]:
                if data["layer"] == 0:
                    x, y = pos[node]
                    x = x - 6
                    plt.text(
                        x,
                        y,
                        s=data["label"],
                        bbox=dict(facecolor="white", alpha=0.5),
                        horizontalalignment="right",
                        verticalalignment="center_baseline",
                    )
                elif data["layer"] == len(layer_sizes) - 1:
                    x, y = pos[node]
                    x = x + 6
                    t = plt.text(
                        x,
                        y,
                        s=data["label"],
                        bbox=dict(facecolor="white", alpha=0.5),
                        horizontalalignment="left",
                        verticalalignment="center_baseline",
                    )
                    r = fig.canvas.get_renderer()
                    bb = t.get_window_extent(renderer=r)
                    max_end_width = max(max_end_width, bb.width)
                else:
                    x, y = pos[node]
                    plt.text(
                        x,
                        y,
                        s=data["label"],
                        bbox=dict(facecolor="white", alpha=0.5),
                        horizontalalignment="center",
                        verticalalignment="center_baseline",
                    )

    ed = {
        "width": edge_width,
        "edgelist": [(n1, n2) for n1, n2, v in g.edges(data=True) if v["show"]],
    }
    if weights is not None:
        ed = {
            **ed,
            **dict(
                edge_cmap=cmap,
                edge_color=[
                    sm_edges.to_rgba(v["w"])
                    for n1, n2, v in g.edges(data=True)
                    if v["show"]
                ],
            ),
        }

        if weight_position is not None:

            def draw_networkx_edge_labels(edge_labels, label_pos):
                nodes = nx.draw_networkx_edge_labels(
                    g,
                    pos,
                    edge_labels=edge_labels,
                    rotate=False,
                    label_pos=label_pos,
                    # norm=sm_edges,
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
                )

            if isinstance(weight_position, (float, int)):
                edge_labels = {
                    (n1, n2): v["w"] for n1, n2, v in g.edges(data=True) if v["show"]
                }
                label_pos = 1 - weight_position
                draw_networkx_edge_labels(edge_labels, label_pos)
            else:
                assert len(weight_position) == len(layer_sizes) - 1
                for i, wp in enumerate(weight_position):
                    if wp is None:
                        continue
                    if isinstance(wp, (float, int)):
                        edge_labels = {}
                        for n1, n2, v in g.edges(data=True):
                            if n1.startswith("{}_".format(i)) and v["show"]:
                                edge_labels[(n1, n2)] = v["w"]
                        nodes = draw_networkx_edge_labels(
                            edge_labels=edge_labels,
                            label_pos=1 - wp,
                        )

        if show_bar:
            bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

            plt.colorbar(sm_edges, pad=0.1 + max_end_width / (bbox.width * fig.dpi))
            if "node_color" in nodes_kwargs:
                plt.colorbar(sm_nodes, pad=0.1 + max_end_width / (bbox.width * fig.dpi))

    edges = nx.draw_networkx_edges(g, pos, **ed)

    # plt.tight_layout
    plt.axis("off")
    if save is not None:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save, dpi=DEFAULT_DPI)
    if show is None:
        if save is not None:
            show = False
        else:
            show = True
    cut = 1.15
    xmax = max(xx for xx, yy in pos.values()) + 10
    ymax = max(yy for xx, yy in pos.values()) + 10
    xmin = min(xx for xx, yy in pos.values()) - 10
    ymin = min(yy for xx, yy in pos.values()) - 10
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    # fig.tight_layout()
    if show:
        plt.show()
    plt.close()


# copied from rdkit
import rdkit.Chem.Draw as Draw
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap


def customGetSimilarityMapFromWeights(
    mol,
    weights,
    colorMap=None,
    scale=-1,
    size=(250, 250),
    sigma=None,
    coordScale=1.5,
    step=0.01,
    colors="k",
    contourLines=10,
    alpha=0.5,
    vmin=None,
    vmax=None,
    **kwargs,
):
    if mol.GetNumAtoms() < 2:
        raise ValueError("too few atoms")

    fig = Draw.MolToMPL(mol, coordScale=coordScale, size=size, **kwargs)
    if sigma is None:
        if mol.GetNumBonds() > 0:
            bond = mol.GetBondWithIdx(0)
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            sigma = 0.3 * np.sqrt(
                sum(
                    [
                        (mol._atomPs[idx1][i] - mol._atomPs[idx2][i]) ** 2
                        for i in range(2)
                    ]
                )
            )
        else:
            sigma = 0.3 * np.sqrt(
                sum([(mol._atomPs[0][i] - mol._atomPs[1][i]) ** 2 for i in range(2)])
            )
        sigma = round(sigma, 2)
    x, y, z = Draw.calcAtomGaussians(mol, sigma, weights=weights, step=step)
    z = z / 100
    # scaling
    if scale <= 0.0:
        maxScale = max(np.fabs(np.min(z)), np.fabs(np.max(z)))
    else:
        maxScale = scale
    # coloring
    if colorMap is None:
        if cm is None:
            raise RuntimeError("matplotlib failed to import")
        PiYG_cmap = cm.get_cmap("PiYG", 2)
        colorMap = LinearSegmentedColormap.from_list(
            "PiWG", [PiYG_cmap(0), (1.0, 1.0, 1.0), PiYG_cmap(1)], N=255
        )

    if vmin is None:
        vmin = -maxScale
    if vmax is None:
        vmax = maxScale
    sm_nodes = plt.cm.ScalarMappable(
        cmap=colorMap, norm=MidpointNormalize(vmin, vmax, 0.0)
    )

    z += 1e-6

    a = fig.axes[0].imshow(
        z,
        cmap=colorMap,
        interpolation="bilinear",
        origin="lower",
        extent=(0, 1, 0, 1),
        norm=MidpointNormalize(vmin, vmax, 0.0),
    )

    ax = fig.axes[0]
    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.1,
            ax.get_position().height,
        ]
    )
    fig.colorbar(a, cax=cax)
    # contour lines
    # only draw them when at least one weight is not zero
    if len([w for w in weights if w != 0.0]):
        contourset = fig.axes[0].contour(
            x, y, z, contourLines, colors=colors, alpha=alpha, **kwargs
        )
        for j, c in enumerate(contourset.collections):
            if contourset.levels[j] == 0.0:
                c.set_linewidth(0.0)
            elif contourset.levels[j] < 0:
                c.set_dashes([(0, (3.0, 3.0))])
    fig.axes[0].set_axis_off()
    return fig


def plot_features_to_mol(features, mol, title=None, path=None, prefix="", plot=True):
    vmin, vmax = min(0, features.min()), max(1e-6, features.max())
    files = []
    for d in range(features.shape[1]):
        if path:
            filepath = os.path.join(path, "{}{}.png".format(prefix, d))
            if os.path.exists(filepath) and not REDRAW:
                files.append(filepath)
                continue
        f = customGetSimilarityMapFromWeights(
            mol, features[:, d], colorMap="jet", vmin=vmin, vmax=vmax
        )
        if title:
            f.axes[0].set_title(title[d], fontsize=20)
        if path:
            files.append(filepath)
            plt.savefig(files[-1], bbox_inches="tight", dpi=DEFAULT_DPI)

        if plot:
            plt.show()

        plt.close()
    return files


def gallery(images, height="auto", captions=None):
    if isinstance(height, (int, float)):
        height = str(height) + "px"
    if not captions:
        captions = [None] * len(images)
    if len(captions) < len(images):
        captions = captions + [None] * (len(captions) - len(images))

    figures = []
    for i, image in enumerate(images):
        src = image
        caption = f"<figcaption>{captions[i]}</figcaption>"
        figures.append(
            f"""
            <figure style="margin: 5px !important;">
              <img src="{src}" style="height: {height}">
              {caption}
            </figure>
        """
        )
    return HTML(
        data=f"""
        <div style="display: flex; flex-flow: row wrap; text-align: center;">
        {''.join(figures)}
        </div>
    """
    )


def show_false_atom_predictions(loader, model, ignore_subgroups=[]):
    try:
        loader.test_dataloader()
    except:
        loader.setup()

    subgroups = ignore_subgroups.copy()

    sgd = []
    for s in subgroups:
        ind_map = {}
        qmol = Chem.MolFromSmarts(s)
        for atom in qmol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num:
                ind_map[map_num - 1] = atom.GetIdx()
        map_list = np.array([ind_map[x] for x in sorted(ind_map)])
        sgd.append((qmol, map_list))

    for _loader in [
        loader.test_dataloader(),
        loader.val_dataloader(),
        loader.train_dataloader(),
    ]:
        for d in _loader:
            pred = model(d)
            bad_pred = pred.argmax(1) != d.y.argmax(1)
            for batch in d.batch[bad_pred].unique():
                indices = d.batch == batch

                graph = d.mol_graph[batch]
                l_true = short_hybrid[d.y[indices].detach().numpy().argmax(1)].astype(
                    np.object
                )
                l_pred = short_hybrid[pred[indices].detach().numpy().argmax(1)].astype(
                    np.object
                )

                wrong_l = l_true != l_pred

                node_color = np.array(["#1f78b4"] * len(graph))
                node_color[wrong_l] = "red"
                l = l_true.copy()
                l[wrong_l] = l_pred[wrong_l] + "(" + l_true[wrong_l] + ")"

                mol = graph.molecule.mol
                found = False
                for sg in sgd:
                    if found:
                        break
                    # display(sg[0])
                    for match in mol.GetSubstructMatches(sg[0]):
                        match = np.array(match)
                        ##print(sg[1])
                        mas = match[sg[1]]
                        if any(np.where(wrong_l)[0] == mas):
                            found = True
                            break

                if not found:
                    display(graph.molecule)
                    f = graph.get_fig(labels=l.tolist(), node_color=node_color)
                    plt.show()
                    plt.close()
                    display(Chem.MolToSmiles(graph.molecule.mol))
