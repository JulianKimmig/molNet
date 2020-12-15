from rdkit.Chem.Draw import rdMolDraw2D


def mol_to_svg(mol, size=(200, 200), svg_data=None):
    if svg_data is None:
        svg_data = {}

    d = rdMolDraw2D.MolDraw2DSVG(*size)

    d.DrawMolecule(mol,**svg_data)
    d.FinishDrawing()
    return d.GetDrawingText()