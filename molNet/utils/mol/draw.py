from typing import Dict, Tuple, Any

from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import Mol


def mol_to_svg(
    mol: Mol,
    size: Tuple[int, int] = (200, 200),
    svg_data: Dict[str, Any] = None,
    drawOptions: Dict[str, Any] = None,
) -> str:
    if svg_data is None:
        svg_data = {}
    if drawOptions is None:
        drawOptions = {}

    if "clearBackground" not in drawOptions:
        drawOptions["clearBackground"] = False

    d = rdMolDraw2D.MolDraw2DSVG(*size)
    opts = d.drawOptions()
    for k, v in drawOptions.items():
        setattr(opts, k, v)

    d.DrawMolecule(mol, **svg_data)
    d.FinishDrawing()
    return d.GetDrawingText()
