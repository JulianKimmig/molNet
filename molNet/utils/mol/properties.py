from rdkit.Chem import AllChem

def assert_confomers(mol,iterations=10):
    if iterations<=0:
        return mol
    
    if  mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, useRandomCoords=False, maxAttempts=iterations)
    else:
        return mol
    
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=iterations)
        
    
    if  mol.GetNumConformers() == 0:
        ps = AllChem.ETKDGv2()
        ps.maxIterations=iterations
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol,ps)
    
    if  mol.GetNumConformers() == 0:
        ps = AllChem.ETKDGv3()
        ps.maxIterations=iterations
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol,ps)
    
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=iterations,ignoreSmoothingFailures=True)
    
    if mol.GetNumConformers() > 0:
        AllChem.MMFFOptimizeMolecule(mol)
    return mol


def has_confomers(mol,create_if_not=True,*args,**kwargs):
    if mol is None:
        return False
    try:
        if create_if_not:
            assert_confomers(mol,*args,**kwargs)
        return mol.GetNumConformers()> 0
            
    except:
        return False
