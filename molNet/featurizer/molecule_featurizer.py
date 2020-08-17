from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds, CalcExactMolWt, CalcNumLipinskiHBD, CalcNumRings, \
    CalcNumLipinskiHBA, CalcNumHBA, CalcNumHBD, CalcNumAromaticRings, CalcNumSaturatedRings, CalcNumHeterocycles, \
    CalcNumAromaticHeterocycles, CalcNumAromaticCarbocycles, CalcNumSaturatedHeterocycles, CalcNumSaturatedCarbocycles, \
    CalcNumAliphaticRings, CalcNumAliphaticHeterocycles, CalcNumAliphaticCarbocycles, CalcNumHeteroatoms, \
    CalcNumAmideBonds, CalcFractionCSP3, CalcLabuteASA, CalcTPSA, CalcChi0v, CalcChi1v, CalcChi2v, CalcChi3v, CalcChi4v, \
    CalcChi0n, CalcChi1n, CalcChi2n, CalcChi3n, CalcChi4n, CalcHallKierAlpha, CalcKappa1, CalcKappa2, CalcKappa3, \
    CalcNumSpiroAtoms, CalcNumBridgeheadAtoms, CalcNumAtomStereoCenters, CalcNumUnspecifiedAtomStereoCenters, CalcPBF, \
    CalcNPR1, CalcNPR2, CalcPMI1, CalcPMI2, CalcPMI3, CalcRadiusOfGyration, CalcInertialShapeFactor, CalcEccentricity, \
    CalcAsphericity, CalcSpherocityIndex, CalcCrippenDescriptors, GetUSR, GetUSRCAT, SlogP_VSA_, SMR_VSA_, PEOE_VSA_, \
    CalcWHIM, CalcGETAWAY, CalcRDF, CalcMORSE, CalcAUTOCORR3D, CalcAUTOCORR2D

from .featurizer import FeaturizerList, LambdaFeaturizer

__all__ = ['default_molecule_featurizer', 'extend_molnet_featurzier', 'molecule_asphericity', 'molecule_autocorr2d',
           'molecule_autocorr3d', 'molecule_chi0n', 'molecule_chi0v', 'molecule_chi1n', 'molecule_chi1v',
           'molecule_chi2n', 'molecule_chi2v', 'molecule_chi3n', 'molecule_chi3v', 'molecule_chi4n', 'molecule_chi4v',
           'molecule_crippen_descriptors', 'molecule_eccentricity', 'molecule_exact_mol_wt', 'molecule_fraction_csp3',
           'molecule_getaway', 'molecule_hall_kier_alpha', 'molecule_inertial_shape_factor', 'molecule_kappa1',
           'molecule_kappa2', 'molecule_kappa3', 'molecule_labute_asa', 'molecule_logp', 'molecule_molecular_ri',
           'molecule_morse', 'molecule_npr1', 'molecule_npr2', 'molecule_num_aliphatic_carbocycles',
           'molecule_num_aliphatic_heterocycles', 'molecule_num_aliphatic_rings', 'molecule_num_amide_bonds',
           'molecule_num_aromatic_carbocycles', 'molecule_num_aromatic_heterocycles', 'molecule_num_aromatic_rings',
           'molecule_num_atom_stereo_centers', 'molecule_num_bridgehead_atoms', 'molecule_num_hba', 'molecule_num_hbd',
           'molecule_num_heteroatoms', 'molecule_num_heterocycles', 'molecule_num_lipinski_hba',
           'molecule_num_lipinski_hbd', 'molecule_num_rings', 'molecule_num_rotatable_bonds',
           'molecule_num_saturated_carbocycles', 'molecule_num_saturated_heterocycles', 'molecule_num_saturated_rings',
           'molecule_num_spiro_atoms', 'molecule_num_unspecified_atom_stereo_centers', 'molecule_pbf',
           'molecule_peoe_vsa', 'molecule_pmi1', 'molecule_pmi2', 'molecule_pmi3', 'molecule_radius_of_gyration',
           'molecule_rdf', 'molecule_slogp_vsa', 'molecule_smr_vsa', 'molecule_spherocity_index', 'molecule_tpsa',
           'molecule_usr', 'molecule_usrcat', 'molecule_whim','molecule_num_atoms','molecule_num_heavy_atoms',
           'molecule_num_hs'
           ]

molecule_num_atoms = LambdaFeaturizer(
    lamda_call=lambda mol: [mol.GetNumAtoms(onlyExplicit=False)],
    length=1
)

molecule_num_heavy_atoms = LambdaFeaturizer(
    lamda_call=lambda mol: [mol.GetNumHeavyAtoms()],
    length=1
)

molecule_num_hs = LambdaFeaturizer(
    lamda_call=lambda mol: [mol.GetNumAtoms(onlyExplicit=False)-mol.GetNumHeavyAtoms()],
    length=1
)


molecule_num_rotatable_bonds = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumRotatableBonds(mol)],
    length=1
)
molecule_exact_mol_wt = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcExactMolWt(mol)],
    length=1
)
molecule_num_lipinski_hbd = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumLipinskiHBD(mol)],
    length=1
)
molecule_num_lipinski_hba = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumLipinskiHBA(mol)],
    length=1
)
molecule_num_hbd = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumHBD(mol)],
    length=1
)
molecule_num_hba = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumHBA(mol)],
    length=1
)
molecule_num_rings = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumRings(mol)],
    length=1
)
molecule_num_aromatic_rings = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumAromaticRings(mol)],
    length=1
)
molecule_num_saturated_rings = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumSaturatedRings(mol)],
    length=1
)
molecule_num_heterocycles = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumHeterocycles(mol)],
    length=1
)
molecule_num_aromatic_heterocycles = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumAromaticHeterocycles(mol)],
    length=1
)
molecule_num_aromatic_carbocycles = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumAromaticCarbocycles(mol)],
    length=1
)
molecule_num_saturated_heterocycles = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumSaturatedHeterocycles(mol)],
    length=1
)
molecule_num_saturated_carbocycles = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumSaturatedCarbocycles(mol)],
    length=1
)
molecule_num_aliphatic_rings = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumAliphaticRings(mol)],
    length=1
)
molecule_num_aliphatic_heterocycles = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumAliphaticHeterocycles(mol)],
    length=1
)
molecule_num_aliphatic_carbocycles = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumAliphaticCarbocycles(mol)],
    length=1
)
molecule_num_heteroatoms = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumHeteroatoms(mol)],
    length=1
)
molecule_num_amide_bonds = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumAmideBonds(mol)],
    length=1
)

molecule_fraction_csp3 = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcFractionCSP3(mol)],
    length=1
)
molecule_labute_asa = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcLabuteASA(mol)],
    length=1
)
molecule_tpsa = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcTPSA(mol)],
    length=1
)
molecule_chi0v = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcChi0v(mol)],
    length=1
)
molecule_chi1v = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcChi1v(mol)],
    length=1
)
molecule_chi2v = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcChi2v(mol)],
    length=1
)
molecule_chi3v = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcChi3v(mol)],
    length=1
)
molecule_chi4v = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcChi4v(mol)],
    length=1
)
molecule_chi0n = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcChi0n(mol)],
    length=1
)
molecule_chi1n = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcChi1n(mol)],
    length=1
)
molecule_chi2n = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcChi2n(mol)],
    length=1
)
molecule_chi3n = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcChi3n(mol)],
    length=1
)
molecule_chi4n = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcChi4n(mol)],
    length=1
)
molecule_hall_kier_alpha = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcHallKierAlpha(mol)],
    length=1
)
molecule_kappa1 = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcKappa1(mol)],
    length=1
)
molecule_kappa2 = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcKappa2(mol)],
    length=1
)
molecule_kappa3 = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcKappa3(mol)],
    length=1
)
molecule_num_spiro_atoms = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumSpiroAtoms(mol)],
    length=1
)
molecule_num_bridgehead_atoms = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumBridgeheadAtoms(mol)],
    length=1
)
molecule_num_atom_stereo_centers = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumAtomStereoCenters(mol)],
    length=1
)
molecule_num_unspecified_atom_stereo_centers = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNumUnspecifiedAtomStereoCenters(mol)],
    length=1
)
molecule_pbf = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcPBF(_assert_confomers(mol))],
    length=1
)
molecule_npr1 = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNPR1(_assert_confomers(mol))],
    length=1
)
molecule_npr2 = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcNPR2(_assert_confomers(mol))],
    length=1
)
molecule_pmi1 = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcPMI1(_assert_confomers(mol))],
    length=1
)
molecule_pmi2 = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcPMI2(_assert_confomers(mol))],
    length=1
)
molecule_pmi3 = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcPMI3(_assert_confomers(mol))],
    length=1
)
molecule_radius_of_gyration = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcRadiusOfGyration(_assert_confomers(mol))],
    length=1
)
molecule_inertial_shape_factor = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcInertialShapeFactor(_assert_confomers(mol))],
    length=1
)
molecule_eccentricity = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcEccentricity(_assert_confomers(mol))],
    length=1
)

def _assert_confomers(mol):
    if mol.GetNumConformers()==0:
        AllChem.EmbedMolecule(mol,useRandomCoords=True,maxAttempts=5000)
    return mol
molecule_asphericity = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcAsphericity(_assert_confomers(mol))],
    length=1
)
molecule_spherocity_index = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcSpherocityIndex(_assert_confomers(mol))],
    length=1
)
molecule_logp = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcCrippenDescriptors(mol)[0]],
    length=1
)

molecule_molecular_ri = LambdaFeaturizer(
    lamda_call=lambda mol: [CalcCrippenDescriptors(mol)[1]],
    length=1
)

molecule_crippen_descriptors = LambdaFeaturizer(
    lamda_call=lambda mol: list(CalcCrippenDescriptors(mol)),
    length=2
)
molecule_usr = LambdaFeaturizer(
    lamda_call=lambda mol: list(GetUSR(_assert_confomers(mol))),
    length=12
)
molecule_usrcat = LambdaFeaturizer(
    lamda_call=lambda mol: list(GetUSRCAT(_assert_confomers(mol))),
    length=60
)
molecule_slogp_vsa = LambdaFeaturizer(
    lamda_call=lambda mol: list(SlogP_VSA_(mol)),
    length=12
)
molecule_smr_vsa = LambdaFeaturizer(
    lamda_call=lambda mol: list(SMR_VSA_(mol)),
    length=10
)
molecule_peoe_vsa = LambdaFeaturizer(
    lamda_call=lambda mol: list(PEOE_VSA_(mol)),
    length=14
)
molecule_whim = LambdaFeaturizer(
    lamda_call=lambda mol: list(CalcWHIM(_assert_confomers(mol))),
    length=114
)
molecule_getaway = LambdaFeaturizer(
    lamda_call=lambda mol: list(CalcGETAWAY(_assert_confomers(mol))),
    length=273
)
molecule_rdf = LambdaFeaturizer(
    lamda_call=lambda mol: list(CalcRDF(_assert_confomers(mol))),
    length=210
)
molecule_morse = LambdaFeaturizer(
    lamda_call=lambda mol: list(CalcMORSE(_assert_confomers(mol))),
    length=224
)
molecule_autocorr3d = LambdaFeaturizer(
    lamda_call=lambda mol: list(CalcAUTOCORR3D(_assert_confomers(mol))),
    length=80
)
molecule_autocorr2d = LambdaFeaturizer(
    lamda_call=lambda mol: list(CalcAUTOCORR2D(mol)),
    length=192
)

extend_molnet_featurzier = LambdaFeaturizer(
    lamda_call=lambda mol: mol.molnet_features if hasattr(mol, "molnet_features") else [],
    length=None
)

default_molecule_featurizer = FeaturizerList([
    molecule_num_heavy_atoms,
    molecule_crippen_descriptors,
    molecule_num_atoms,
    molecule_num_aliphatic_carbocycles,
    molecule_num_aliphatic_heterocycles,
    molecule_num_aliphatic_rings,
    molecule_num_aromatic_carbocycles,
    molecule_num_aromatic_heterocycles,
    molecule_num_aromatic_rings,
    #molecule_num_atom_stereo_centers,
    molecule_num_bridgehead_atoms,
    molecule_num_hba,
    molecule_num_hbd,
    molecule_num_heteroatoms,
#    molecule_num_heterocycles,
#    molecule_num_lipinski_hba,
#    molecule_num_lipinski_hbd,
    molecule_num_rings,
    molecule_num_rotatable_bonds,
#    molecule_num_saturated_carbocycles,
#    molecule_num_saturated_heterocycles,
    molecule_num_saturated_rings,
    molecule_num_spiro_atoms,
#    molecule_num_unspecified_atom_stereo_centers
])
