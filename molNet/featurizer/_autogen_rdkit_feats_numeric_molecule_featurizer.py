import numpy as np
from molNet.featurizer._molecule_featurizer import SingleValueMoleculeFeaturizer
from rdkit.Chem.AllChem import (
    CalcNPR2,
    CalcRadiusOfGyration,
    CalcLabuteASA,
    CalcPMI2,
    CalcChi2n,
    CalcNumAromaticHeterocycles,
    CalcNumHBD,
    CalcNumLipinskiHBA,
    CalcAsphericity,
    CalcNumAliphaticHeterocycles,
    CalcNumAromaticRings,
    CalcNumHBA,
    CalcNumSpiroAtoms,
    CalcNumBridgeheadAtoms,
    CalcChi3v,
    CalcKappa3,
    CalcNumAliphaticCarbocycles,
    ComputeMolVolume,
    CalcChi1n,
    CalcNumAtoms,
    CalcChi0n,
    CalcNumHeterocycles,
    CalcNumSaturatedCarbocycles,
    CalcPMI1,
    CalcChi3n,
    CalcNumAromaticCarbocycles,
    CalcKappa1,
    CalcKappa2,
    CalcNumSaturatedHeterocycles,
    CalcNPR1,
    CalcNumSaturatedRings,
    CalcHallKierAlpha,
    CalcNumRings,
    CalcNumRotatableBonds,
    CalcNumHeteroatoms,
    CalcChi4v,
    CalcNumHeavyAtoms,
    CalcExactMolWt,
    EmbedMolecule,
    CalcNumLipinskiHBD,
    CalcSpherocityIndex,
    CalcChi2v,
    CalcChi0v,
    CalcNumAmideBonds,
    Compute2DCoords,
    CalcEccentricity,
    CalcChi1v,
    CalcNumAliphaticRings,
    CalcPhi,
    CalcChi4n,
    CalcPMI3,
    CalcPBF,
    CalcTPSA,
    CalcInertialShapeFactor,
    CalcFractionCSP3,
)
from rdkit.Chem import GetFormalCharge, GetSSSR
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem.Descriptors3D import (
    NPR1,
    PMI1,
    PMI2,
    SpherocityIndex,
    InertialShapeFactor,
    RadiusOfGyration,
    NPR2,
    Eccentricity,
    Asphericity,
    PMI3,
)
from rdkit.Chem.Descriptors import (
    VSA_EState1,
    VSA_EState10,
    MaxAbsPartialCharge,
    Chi1v,
    MinPartialCharge,
    VSA_EState4,
    VSA_EState2,
    Kappa3,
    EState_VSA6,
    BCUT2D_MRHI,
    BCUT2D_LOGPLOW,
    ExactMolWt,
    Kappa2,
    BCUT2D_LOGPHI,
    BCUT2D_MWHI,
    Chi4n,
    MaxAbsEStateIndex,
    BCUT2D_CHGLO,
    Chi3v,
    Chi0v,
    NumRadicalElectrons,
    NumValenceElectrons,
    Chi0n,
    Chi1,
    MinEStateIndex,
    VSA_EState6,
    FpDensityMorgan1,
    HeavyAtomMolWt,
    Kappa1,
    EState_VSA3,
    BCUT2D_MWLOW,
    VSA_EState9,
    BCUT2D_MRLOW,
    BCUT2D_CHGHI,
    EState_VSA7,
    Chi1n,
    VSA_EState3,
    EState_VSA10,
    Chi2n,
    EState_VSA5,
    MolWt,
    EState_VSA8,
    EState_VSA2,
    Ipc,
    VSA_EState8,
    EState_VSA4,
    FpDensityMorgan3,
    BalabanJ,
    MaxPartialCharge,
    EState_VSA9,
    Chi3n,
    Chi2v,
    HallKierAlpha,
    FpDensityMorgan2,
    Chi4v,
    MinAbsPartialCharge,
    VSA_EState5,
    Chi0,
    VSA_EState7,
    EState_VSA11,
    BertzCT,
    EState_VSA1,
)
from rdkit.Chem.EState import (
    MinAbsEStateIndex,
    MaxAbsEStateIndex,
    MinEStateIndex,
    MaxEStateIndex,
)
from rdkit.Chem.EnumerateStereoisomers import GetStereoisomerCount, EmbedMolecule
from rdkit.Chem.Fragments import (
    fr_methoxy,
    fr_C_O,
    fr_isothiocyan,
    fr_Al_OH_noTert,
    fr_hdrzine,
    fr_alkyl_carbamate,
    fr_thiazole,
    fr_phos_ester,
    fr_thiocyan,
    fr_HOCCN,
    fr_Ar_NH,
    fr_C_O_noCOO,
    fr_C_S,
    fr_unbrch_alkane,
    fr_tetrazole,
    fr_oxime,
    fr_ether,
    fr_prisulfonamd,
    fr_aldehyde,
    fr_amide,
    fr_azide,
    fr_piperzine,
    fr_azo,
    fr_ester,
    fr_nitrile,
    fr_quatN,
    fr_allylic_oxid,
    fr_morpholine,
    fr_lactam,
    fr_imidazole,
    fr_guanido,
    fr_diazo,
    fr_NH2,
    fr_lactone,
    fr_urea,
    fr_barbitur,
    fr_dihydropyridine,
    fr_aryl_methyl,
    fr_Al_COO,
    fr_isocyan,
    fr_oxazole,
    fr_para_hydroxylation,
    fr_Nhpyrrole,
    fr_ketone_Topliss,
    fr_benzodiazepine,
    fr_pyridine,
    fr_epoxide,
    fr_nitro_arom_nonortho,
    fr_Ar_COO,
    fr_nitro_arom,
    fr_benzene,
    fr_imide,
    fr_Ndealkylation1,
    fr_N_O,
    fr_NH0,
    fr_priamide,
    fr_sulfide,
    fr_phos_acid,
    fr_thiophene,
    fr_sulfonamd,
    fr_Ar_N,
    fr_Imine,
    fr_COO,
    fr_furan,
    fr_NH1,
    fr_piperdine,
    fr_Al_OH,
    fr_Ar_OH,
    fr_sulfone,
    fr_ArN,
    fr_SH,
    fr_phenol,
    fr_halogen,
    fr_alkyl_halide,
    fr_Ndealkylation2,
    fr_hdrzone,
    fr_amidine,
    fr_bicyclic,
    fr_ketone,
    fr_term_acetylene,
    fr_nitroso,
    fr_COO2,
    fr_phenol_noOrthoHbond,
    fr_aniline,
    fr_nitro,
)
from rdkit.Chem.Lipinski import (
    NumSaturatedRings,
    NumAromaticCarbocycles,
    NumSaturatedCarbocycles,
    FractionCSP3,
    NumAliphaticHeterocycles,
    NumAromaticHeterocycles,
    NOCount,
    NumRotatableBonds,
    NumSaturatedHeterocycles,
    NumAliphaticCarbocycles,
    RingCount,
    HeavyAtomCount,
    NumHDonors,
    NumAromaticRings,
    NumHeteroatoms,
    NHOHCount,
    NumAliphaticRings,
    NumHAcceptors,
)
from rdkit.Chem.MolSurf import (
    PEOE_VSA8,
    pyLabuteASA,
    SlogP_VSA8,
    PEOE_VSA2,
    SMR_VSA3,
    SMR_VSA10,
    SMR_VSA5,
    SMR_VSA4,
    PEOE_VSA3,
    SMR_VSA1,
    SMR_VSA7,
    SlogP_VSA11,
    SlogP_VSA1,
    SlogP_VSA7,
    PEOE_VSA12,
    SlogP_VSA2,
    SlogP_VSA9,
    PEOE_VSA7,
    SlogP_VSA12,
    PEOE_VSA5,
    PEOE_VSA10,
    PEOE_VSA14,
    SMR_VSA6,
    SlogP_VSA4,
    PEOE_VSA11,
    SlogP_VSA5,
    SMR_VSA2,
    TPSA,
    PEOE_VSA6,
    SMR_VSA9,
    LabuteASA,
    SlogP_VSA3,
    PEOE_VSA4,
    SMR_VSA8,
    SlogP_VSA10,
    PEOE_VSA1,
    SlogP_VSA6,
    PEOE_VSA9,
    PEOE_VSA13,
)
from rdkit.Chem.QED import default, weights_mean, qed, weights_none, weights_max
from rdkit.Chem.rdmolops import GetFormalCharge, GetSSSR


class Molecule_AllChem_Asphericity_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcAsphericity
    dtype = np.float32
    featurize = staticmethod(CalcAsphericity)


class Molecule_AllChem_Chi0n_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcChi0n
    dtype = np.float32
    featurize = staticmethod(CalcChi0n)


class Molecule_AllChem_Chi0v_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcChi0v
    dtype = np.float32
    featurize = staticmethod(CalcChi0v)


class Molecule_AllChem_Chi1n_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcChi1n
    dtype = np.float32
    featurize = staticmethod(CalcChi1n)


class Molecule_AllChem_Chi1v_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcChi1v
    dtype = np.float32
    featurize = staticmethod(CalcChi1v)


class Molecule_AllChem_Chi2n_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcChi2n
    dtype = np.float32
    featurize = staticmethod(CalcChi2n)


class Molecule_AllChem_Chi2v_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcChi2v
    dtype = np.float32
    featurize = staticmethod(CalcChi2v)


class Molecule_AllChem_Chi3n_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcChi3n
    dtype = np.float32
    featurize = staticmethod(CalcChi3n)


class Molecule_AllChem_Chi3v_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcChi3v
    dtype = np.float32
    featurize = staticmethod(CalcChi3v)


class Molecule_AllChem_Chi4n_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcChi4n
    dtype = np.float32
    featurize = staticmethod(CalcChi4n)


class Molecule_AllChem_Chi4v_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcChi4v
    dtype = np.float32
    featurize = staticmethod(CalcChi4v)


class Molecule_AllChem_Compute2DCoords_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.Compute2DCoords
    dtype = np.int32
    featurize = staticmethod(Compute2DCoords)


class Molecule_AllChem_ComputeMolVolume_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.ComputeMolVolume
    dtype = np.float32
    featurize = staticmethod(ComputeMolVolume)


class Molecule_AllChem_Eccentricity_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcEccentricity
    dtype = np.float32
    featurize = staticmethod(CalcEccentricity)


class Molecule_AllChem_EmbedMolecule_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.EmbedMolecule
    dtype = np.int32
    featurize = staticmethod(EmbedMolecule)


class Molecule_AllChem_ExactMolWt_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcExactMolWt
    dtype = np.float32
    featurize = staticmethod(CalcExactMolWt)


class Molecule_AllChem_FractionCSP3_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcFractionCSP3
    dtype = np.float32
    featurize = staticmethod(CalcFractionCSP3)


class Molecule_AllChem_HallKierAlpha_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcHallKierAlpha
    dtype = np.float32
    featurize = staticmethod(CalcHallKierAlpha)


class Molecule_AllChem_InertialShapeFactor_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcInertialShapeFactor
    dtype = np.float32
    featurize = staticmethod(CalcInertialShapeFactor)


class Molecule_AllChem_Kappa1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcKappa1
    dtype = np.float32
    featurize = staticmethod(CalcKappa1)


class Molecule_AllChem_Kappa2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcKappa2
    dtype = np.float32
    featurize = staticmethod(CalcKappa2)


class Molecule_AllChem_Kappa3_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcKappa3
    dtype = np.float32
    featurize = staticmethod(CalcKappa3)


class Molecule_AllChem_LabuteASA_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcLabuteASA
    dtype = np.float32
    featurize = staticmethod(CalcLabuteASA)


class Molecule_AllChem_NPR1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNPR1
    dtype = np.float32
    featurize = staticmethod(CalcNPR1)


class Molecule_AllChem_NPR2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNPR2
    dtype = np.float32
    featurize = staticmethod(CalcNPR2)


class Molecule_AllChem_NumAliphaticCarbocycles_Featurizer(
    SingleValueMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumAliphaticCarbocycles
    dtype = np.int32
    featurize = staticmethod(CalcNumAliphaticCarbocycles)


class Molecule_AllChem_NumAliphaticHeterocycles_Featurizer(
    SingleValueMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumAliphaticHeterocycles
    dtype = np.int32
    featurize = staticmethod(CalcNumAliphaticHeterocycles)


class Molecule_AllChem_NumAliphaticRings_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumAliphaticRings
    dtype = np.int32
    featurize = staticmethod(CalcNumAliphaticRings)


class Molecule_AllChem_NumAmideBonds_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumAmideBonds
    dtype = np.int32
    featurize = staticmethod(CalcNumAmideBonds)


class Molecule_AllChem_NumAromaticCarbocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumAromaticCarbocycles
    dtype = np.int32
    featurize = staticmethod(CalcNumAromaticCarbocycles)


class Molecule_AllChem_NumAromaticHeterocycles_Featurizer(
    SingleValueMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumAromaticHeterocycles
    dtype = np.int32
    featurize = staticmethod(CalcNumAromaticHeterocycles)


class Molecule_AllChem_NumAromaticRings_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumAromaticRings
    dtype = np.int32
    featurize = staticmethod(CalcNumAromaticRings)


class Molecule_AllChem_NumAtoms_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumAtoms
    dtype = np.int32
    featurize = staticmethod(CalcNumAtoms)


class Molecule_AllChem_NumBridgeheadAtoms_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumBridgeheadAtoms
    dtype = np.int32
    featurize = staticmethod(CalcNumBridgeheadAtoms)


class Molecule_AllChem_NumHBA_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumHBA
    dtype = np.int32
    featurize = staticmethod(CalcNumHBA)


class Molecule_AllChem_NumHBD_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumHBD
    dtype = np.int32
    featurize = staticmethod(CalcNumHBD)


class Molecule_AllChem_NumHeavyAtoms_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumHeavyAtoms
    dtype = np.int32
    featurize = staticmethod(CalcNumHeavyAtoms)


class Molecule_AllChem_NumHeteroatoms_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumHeteroatoms
    dtype = np.int32
    featurize = staticmethod(CalcNumHeteroatoms)


class Molecule_AllChem_NumHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumHeterocycles
    dtype = np.int32
    featurize = staticmethod(CalcNumHeterocycles)


class Molecule_AllChem_NumLipinskiHBA_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumLipinskiHBA
    dtype = np.int32
    featurize = staticmethod(CalcNumLipinskiHBA)


class Molecule_AllChem_NumLipinskiHBD_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumLipinskiHBD
    dtype = np.int32
    featurize = staticmethod(CalcNumLipinskiHBD)


class Molecule_AllChem_NumRings_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumRings
    dtype = np.int32
    featurize = staticmethod(CalcNumRings)


class Molecule_AllChem_NumRotatableBonds_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumRotatableBonds
    dtype = np.int32
    featurize = staticmethod(CalcNumRotatableBonds)


class Molecule_AllChem_NumSaturatedCarbocycles_Featurizer(
    SingleValueMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumSaturatedCarbocycles
    dtype = np.int32
    featurize = staticmethod(CalcNumSaturatedCarbocycles)


class Molecule_AllChem_NumSaturatedHeterocycles_Featurizer(
    SingleValueMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumSaturatedHeterocycles
    dtype = np.int32
    featurize = staticmethod(CalcNumSaturatedHeterocycles)


class Molecule_AllChem_NumSaturatedRings_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumSaturatedRings
    dtype = np.int32
    featurize = staticmethod(CalcNumSaturatedRings)


class Molecule_AllChem_NumSpiroAtoms_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumSpiroAtoms
    dtype = np.int32
    featurize = staticmethod(CalcNumSpiroAtoms)


class Molecule_AllChem_PBF_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcPBF
    dtype = np.float32
    featurize = staticmethod(CalcPBF)


class Molecule_AllChem_PMI1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcPMI1
    dtype = np.float32
    featurize = staticmethod(CalcPMI1)


class Molecule_AllChem_PMI2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcPMI2
    dtype = np.float32
    featurize = staticmethod(CalcPMI2)


class Molecule_AllChem_PMI3_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcPMI3
    dtype = np.float32
    featurize = staticmethod(CalcPMI3)


class Molecule_AllChem_Phi_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcPhi
    dtype = np.float32
    featurize = staticmethod(CalcPhi)


class Molecule_AllChem_RadiusOfGyration_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcRadiusOfGyration
    dtype = np.float32
    featurize = staticmethod(CalcRadiusOfGyration)


class Molecule_AllChem_SpherocityIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcSpherocityIndex
    dtype = np.float32
    featurize = staticmethod(CalcSpherocityIndex)


class Molecule_AllChem_TPSA_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcTPSA
    dtype = np.float32
    featurize = staticmethod(CalcTPSA)


class Molecule_Chem_FormalCharge_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.GetFormalCharge
    dtype = np.int32
    featurize = staticmethod(GetFormalCharge)


class Molecule_Chem_SSSR_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.GetSSSR
    dtype = np.int32
    featurize = staticmethod(GetSSSR)


class Molecule_Crippen_MolLogP_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Crippen.MolLogP
    dtype = np.float32
    featurize = staticmethod(MolLogP)


class Molecule_Crippen_MolMR_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Crippen.MolMR
    dtype = np.float32
    featurize = staticmethod(MolMR)


class Molecule_Descriptors3D_Asphericity_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors3D.Asphericity
    dtype = np.float32
    featurize = staticmethod(Asphericity)


class Molecule_Descriptors3D_Eccentricity_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors3D.Eccentricity
    dtype = np.float32
    featurize = staticmethod(Eccentricity)


class Molecule_Descriptors3D_InertialShapeFactor_Featurizer(
    SingleValueMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.Descriptors3D.InertialShapeFactor
    dtype = np.float32
    featurize = staticmethod(InertialShapeFactor)


class Molecule_Descriptors3D_NPR1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors3D.NPR1
    dtype = np.float32
    featurize = staticmethod(NPR1)


class Molecule_Descriptors3D_NPR2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors3D.NPR2
    dtype = np.float32
    featurize = staticmethod(NPR2)


class Molecule_Descriptors3D_PMI1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors3D.PMI1
    dtype = np.float32
    featurize = staticmethod(PMI1)


class Molecule_Descriptors3D_PMI2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors3D.PMI2
    dtype = np.float32
    featurize = staticmethod(PMI2)


class Molecule_Descriptors3D_PMI3_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors3D.PMI3
    dtype = np.float32
    featurize = staticmethod(PMI3)


class Molecule_Descriptors3D_RadiusOfGyration_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors3D.RadiusOfGyration
    dtype = np.float32
    featurize = staticmethod(RadiusOfGyration)


class Molecule_Descriptors3D_SpherocityIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors3D.SpherocityIndex
    dtype = np.float32
    featurize = staticmethod(SpherocityIndex)


class Molecule_Descriptors_BCUT2D_CHGHI_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.BCUT2D_CHGHI
    dtype = np.float32
    featurize = staticmethod(BCUT2D_CHGHI)


class Molecule_Descriptors_BCUT2D_CHGLO_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.BCUT2D_CHGLO
    dtype = np.float32
    featurize = staticmethod(BCUT2D_CHGLO)


class Molecule_Descriptors_BCUT2D_LOGPHI_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.BCUT2D_LOGPHI
    dtype = np.float32
    featurize = staticmethod(BCUT2D_LOGPHI)


class Molecule_Descriptors_BCUT2D_LOGPLOW_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.BCUT2D_LOGPLOW
    dtype = np.float32
    featurize = staticmethod(BCUT2D_LOGPLOW)


class Molecule_Descriptors_BCUT2D_MRHI_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.BCUT2D_MRHI
    dtype = np.float32
    featurize = staticmethod(BCUT2D_MRHI)


class Molecule_Descriptors_BCUT2D_MRLOW_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.BCUT2D_MRLOW
    dtype = np.float32
    featurize = staticmethod(BCUT2D_MRLOW)


class Molecule_Descriptors_BCUT2D_MWHI_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.BCUT2D_MWHI
    dtype = np.float32
    featurize = staticmethod(BCUT2D_MWHI)


class Molecule_Descriptors_BCUT2D_MWLOW_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.BCUT2D_MWLOW
    dtype = np.float32
    featurize = staticmethod(BCUT2D_MWLOW)


class Molecule_Descriptors_BalabanJ_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.BalabanJ
    dtype = np.float32
    featurize = staticmethod(BalabanJ)


class Molecule_Descriptors_BertzCT_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.BertzCT
    dtype = np.float32
    featurize = staticmethod(BertzCT)


class Molecule_Descriptors_Chi0_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Chi0
    dtype = np.float32
    featurize = staticmethod(Chi0)


class Molecule_Descriptors_Chi0n_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Chi0n
    dtype = np.float32
    featurize = staticmethod(Chi0n)


class Molecule_Descriptors_Chi0v_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Chi0v
    dtype = np.float32
    featurize = staticmethod(Chi0v)


class Molecule_Descriptors_Chi1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Chi1
    dtype = np.float32
    featurize = staticmethod(Chi1)


class Molecule_Descriptors_Chi1n_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Chi1n
    dtype = np.float32
    featurize = staticmethod(Chi1n)


class Molecule_Descriptors_Chi1v_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Chi1v
    dtype = np.float32
    featurize = staticmethod(Chi1v)


class Molecule_Descriptors_Chi2n_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Chi2n
    dtype = np.float32
    featurize = staticmethod(Chi2n)


class Molecule_Descriptors_Chi2v_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Chi2v
    dtype = np.float32
    featurize = staticmethod(Chi2v)


class Molecule_Descriptors_Chi3n_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Chi3n
    dtype = np.float32
    featurize = staticmethod(Chi3n)


class Molecule_Descriptors_Chi3v_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Chi3v
    dtype = np.float32
    featurize = staticmethod(Chi3v)


class Molecule_Descriptors_Chi4n_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Chi4n
    dtype = np.float32
    featurize = staticmethod(Chi4n)


class Molecule_Descriptors_Chi4v_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Chi4v
    dtype = np.float32
    featurize = staticmethod(Chi4v)


class Molecule_Descriptors_EState_VSA10_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA10
    dtype = np.float32
    featurize = staticmethod(EState_VSA10)


class Molecule_Descriptors_EState_VSA11_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA11
    dtype = np.float32
    featurize = staticmethod(EState_VSA11)


class Molecule_Descriptors_EState_VSA1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA1
    dtype = np.float32
    featurize = staticmethod(EState_VSA1)


class Molecule_Descriptors_EState_VSA2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA2
    dtype = np.float32
    featurize = staticmethod(EState_VSA2)


class Molecule_Descriptors_EState_VSA3_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA3
    dtype = np.float32
    featurize = staticmethod(EState_VSA3)


class Molecule_Descriptors_EState_VSA4_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA4
    dtype = np.float32
    featurize = staticmethod(EState_VSA4)


class Molecule_Descriptors_EState_VSA5_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA5
    dtype = np.float32
    featurize = staticmethod(EState_VSA5)


class Molecule_Descriptors_EState_VSA6_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA6
    dtype = np.float32
    featurize = staticmethod(EState_VSA6)


class Molecule_Descriptors_EState_VSA7_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA7
    dtype = np.float32
    featurize = staticmethod(EState_VSA7)


class Molecule_Descriptors_EState_VSA8_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA8
    dtype = np.float32
    featurize = staticmethod(EState_VSA8)


class Molecule_Descriptors_EState_VSA9_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA9
    dtype = np.float32
    featurize = staticmethod(EState_VSA9)


class Molecule_Descriptors_ExactMolWt_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.ExactMolWt
    dtype = np.float32
    featurize = staticmethod(ExactMolWt)


class Molecule_Descriptors_FpDensityMorgan1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.FpDensityMorgan1
    dtype = np.float32
    featurize = staticmethod(FpDensityMorgan1)


class Molecule_Descriptors_FpDensityMorgan2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.FpDensityMorgan2
    dtype = np.float32
    featurize = staticmethod(FpDensityMorgan2)


class Molecule_Descriptors_FpDensityMorgan3_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.FpDensityMorgan3
    dtype = np.float32
    featurize = staticmethod(FpDensityMorgan3)


class Molecule_Descriptors_HallKierAlpha_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.HallKierAlpha
    dtype = np.float32
    featurize = staticmethod(HallKierAlpha)


class Molecule_Descriptors_HeavyAtomMolWt_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.HeavyAtomMolWt
    dtype = np.float32
    featurize = staticmethod(HeavyAtomMolWt)


class Molecule_Descriptors_Ipc_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Ipc
    dtype = np.float32
    featurize = staticmethod(Ipc)


class Molecule_Descriptors_Kappa1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Kappa1
    dtype = np.float32
    featurize = staticmethod(Kappa1)


class Molecule_Descriptors_Kappa2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Kappa2
    dtype = np.float32
    featurize = staticmethod(Kappa2)


class Molecule_Descriptors_Kappa3_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Kappa3
    dtype = np.float32
    featurize = staticmethod(Kappa3)


class Molecule_Descriptors_MaxAbsEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.MaxAbsEStateIndex
    dtype = np.float32
    featurize = staticmethod(MaxAbsEStateIndex)


class Molecule_Descriptors_MaxAbsPartialCharge_Featurizer(
    SingleValueMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.Descriptors.MaxAbsPartialCharge
    dtype = np.float32
    featurize = staticmethod(MaxAbsPartialCharge)


class Molecule_Descriptors_MaxPartialCharge_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.MaxPartialCharge
    dtype = np.float32
    featurize = staticmethod(MaxPartialCharge)


class Molecule_Descriptors_MinAbsPartialCharge_Featurizer(
    SingleValueMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.Descriptors.MinAbsPartialCharge
    dtype = np.float32
    featurize = staticmethod(MinAbsPartialCharge)


class Molecule_Descriptors_MinEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.MinEStateIndex
    dtype = np.float32
    featurize = staticmethod(MinEStateIndex)


class Molecule_Descriptors_MinPartialCharge_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.MinPartialCharge
    dtype = np.float32
    featurize = staticmethod(MinPartialCharge)


class Molecule_Descriptors_MolWt_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.MolWt
    dtype = np.float32
    featurize = staticmethod(MolWt)


class Molecule_Descriptors_NumRadicalElectrons_Featurizer(
    SingleValueMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.Descriptors.NumRadicalElectrons
    dtype = np.int32
    featurize = staticmethod(NumRadicalElectrons)


class Molecule_Descriptors_NumValenceElectrons_Featurizer(
    SingleValueMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.Descriptors.NumValenceElectrons
    dtype = np.int32
    featurize = staticmethod(NumValenceElectrons)


class Molecule_Descriptors_VSA_EState10_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.VSA_EState10
    dtype = np.float32
    featurize = staticmethod(VSA_EState10)


class Molecule_Descriptors_VSA_EState1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.VSA_EState1
    dtype = np.float32
    featurize = staticmethod(VSA_EState1)


class Molecule_Descriptors_VSA_EState2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.VSA_EState2
    dtype = np.float32
    featurize = staticmethod(VSA_EState2)


class Molecule_Descriptors_VSA_EState3_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.VSA_EState3
    dtype = np.float32
    featurize = staticmethod(VSA_EState3)


class Molecule_Descriptors_VSA_EState4_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.VSA_EState4
    dtype = np.float32
    featurize = staticmethod(VSA_EState4)


class Molecule_Descriptors_VSA_EState5_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.VSA_EState5
    dtype = np.float32
    featurize = staticmethod(VSA_EState5)


class Molecule_Descriptors_VSA_EState6_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.VSA_EState6
    dtype = np.float32
    featurize = staticmethod(VSA_EState6)


class Molecule_Descriptors_VSA_EState7_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.VSA_EState7
    dtype = np.float32
    featurize = staticmethod(VSA_EState7)


class Molecule_Descriptors_VSA_EState8_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.VSA_EState8
    dtype = np.float32
    featurize = staticmethod(VSA_EState8)


class Molecule_Descriptors_VSA_EState9_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.VSA_EState9
    dtype = np.float32
    featurize = staticmethod(VSA_EState9)


class Molecule_EState_MaxAbsEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.EState.MaxAbsEStateIndex
    dtype = np.float32
    featurize = staticmethod(MaxAbsEStateIndex)


class Molecule_EState_MaxEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.EState.MaxEStateIndex
    dtype = np.float32
    featurize = staticmethod(MaxEStateIndex)


class Molecule_EState_MinAbsEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.EState.MinAbsEStateIndex
    dtype = np.float32
    featurize = staticmethod(MinAbsEStateIndex)


class Molecule_EState_MinEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.EState.MinEStateIndex
    dtype = np.float32
    featurize = staticmethod(MinEStateIndex)


class Molecule_EnumerateStereoisomers_EmbedMolecule_Featurizer(
    SingleValueMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.EnumerateStereoisomers.EmbedMolecule
    dtype = np.int32
    featurize = staticmethod(EmbedMolecule)


class Molecule_EnumerateStereoisomers_StereoisomerCount_Featurizer(
    SingleValueMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.EnumerateStereoisomers.GetStereoisomerCount
    dtype = np.int32
    featurize = staticmethod(GetStereoisomerCount)


class Molecule_Fragments_fr_Al_COO_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Al_COO
    dtype = np.int32
    featurize = staticmethod(fr_Al_COO)


class Molecule_Fragments_fr_Al_OH_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Al_OH
    dtype = np.int32
    featurize = staticmethod(fr_Al_OH)


class Molecule_Fragments_fr_Al_OH_noTert_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Al_OH_noTert
    dtype = np.int32
    featurize = staticmethod(fr_Al_OH_noTert)


class Molecule_Fragments_fr_ArN_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_ArN
    dtype = np.int32
    featurize = staticmethod(fr_ArN)


class Molecule_Fragments_fr_Ar_COO_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Ar_COO
    dtype = np.int32
    featurize = staticmethod(fr_Ar_COO)


class Molecule_Fragments_fr_Ar_NH_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Ar_NH
    dtype = np.int32
    featurize = staticmethod(fr_Ar_NH)


class Molecule_Fragments_fr_Ar_N_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Ar_N
    dtype = np.int32
    featurize = staticmethod(fr_Ar_N)


class Molecule_Fragments_fr_Ar_OH_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Ar_OH
    dtype = np.int32
    featurize = staticmethod(fr_Ar_OH)


class Molecule_Fragments_fr_COO2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_COO2
    dtype = np.int32
    featurize = staticmethod(fr_COO2)


class Molecule_Fragments_fr_COO_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_COO
    dtype = np.int32
    featurize = staticmethod(fr_COO)


class Molecule_Fragments_fr_C_O_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_C_O
    dtype = np.int32
    featurize = staticmethod(fr_C_O)


class Molecule_Fragments_fr_C_O_noCOO_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_C_O_noCOO
    dtype = np.int32
    featurize = staticmethod(fr_C_O_noCOO)


class Molecule_Fragments_fr_C_S_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_C_S
    dtype = np.int32
    featurize = staticmethod(fr_C_S)


class Molecule_Fragments_fr_HOCCN_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_HOCCN
    dtype = np.int32
    featurize = staticmethod(fr_HOCCN)


class Molecule_Fragments_fr_Imine_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Imine
    dtype = np.int32
    featurize = staticmethod(fr_Imine)


class Molecule_Fragments_fr_NH0_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_NH0
    dtype = np.int32
    featurize = staticmethod(fr_NH0)


class Molecule_Fragments_fr_NH1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_NH1
    dtype = np.int32
    featurize = staticmethod(fr_NH1)


class Molecule_Fragments_fr_NH2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_NH2
    dtype = np.int32
    featurize = staticmethod(fr_NH2)


class Molecule_Fragments_fr_N_O_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_N_O
    dtype = np.int32
    featurize = staticmethod(fr_N_O)


class Molecule_Fragments_fr_Ndealkylation1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Ndealkylation1
    dtype = np.int32
    featurize = staticmethod(fr_Ndealkylation1)


class Molecule_Fragments_fr_Ndealkylation2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Ndealkylation2
    dtype = np.int32
    featurize = staticmethod(fr_Ndealkylation2)


class Molecule_Fragments_fr_Nhpyrrole_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Nhpyrrole
    dtype = np.int32
    featurize = staticmethod(fr_Nhpyrrole)


class Molecule_Fragments_fr_SH_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_SH
    dtype = np.int32
    featurize = staticmethod(fr_SH)


class Molecule_Fragments_fr_aldehyde_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_aldehyde
    dtype = np.int32
    featurize = staticmethod(fr_aldehyde)


class Molecule_Fragments_fr_alkyl_carbamate_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_alkyl_carbamate
    dtype = np.int32
    featurize = staticmethod(fr_alkyl_carbamate)


class Molecule_Fragments_fr_alkyl_halide_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_alkyl_halide
    dtype = np.int32
    featurize = staticmethod(fr_alkyl_halide)


class Molecule_Fragments_fr_allylic_oxid_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_allylic_oxid
    dtype = np.int32
    featurize = staticmethod(fr_allylic_oxid)


class Molecule_Fragments_fr_amide_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_amide
    dtype = np.int32
    featurize = staticmethod(fr_amide)


class Molecule_Fragments_fr_amidine_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_amidine
    dtype = np.int32
    featurize = staticmethod(fr_amidine)


class Molecule_Fragments_fr_aniline_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_aniline
    dtype = np.int32
    featurize = staticmethod(fr_aniline)


class Molecule_Fragments_fr_aryl_methyl_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_aryl_methyl
    dtype = np.int32
    featurize = staticmethod(fr_aryl_methyl)


class Molecule_Fragments_fr_azide_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_azide
    dtype = np.int32
    featurize = staticmethod(fr_azide)


class Molecule_Fragments_fr_azo_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_azo
    dtype = np.int32
    featurize = staticmethod(fr_azo)


class Molecule_Fragments_fr_barbitur_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_barbitur
    dtype = np.int32
    featurize = staticmethod(fr_barbitur)


class Molecule_Fragments_fr_benzene_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_benzene
    dtype = np.int32
    featurize = staticmethod(fr_benzene)


class Molecule_Fragments_fr_benzodiazepine_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_benzodiazepine
    dtype = np.int32
    featurize = staticmethod(fr_benzodiazepine)


class Molecule_Fragments_fr_bicyclic_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_bicyclic
    dtype = np.int32
    featurize = staticmethod(fr_bicyclic)


class Molecule_Fragments_fr_diazo_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_diazo
    dtype = np.int32
    featurize = staticmethod(fr_diazo)


class Molecule_Fragments_fr_dihydropyridine_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_dihydropyridine
    dtype = np.int32
    featurize = staticmethod(fr_dihydropyridine)


class Molecule_Fragments_fr_epoxide_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_epoxide
    dtype = np.int32
    featurize = staticmethod(fr_epoxide)


class Molecule_Fragments_fr_ester_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_ester
    dtype = np.int32
    featurize = staticmethod(fr_ester)


class Molecule_Fragments_fr_ether_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_ether
    dtype = np.int32
    featurize = staticmethod(fr_ether)


class Molecule_Fragments_fr_furan_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_furan
    dtype = np.int32
    featurize = staticmethod(fr_furan)


class Molecule_Fragments_fr_guanido_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_guanido
    dtype = np.int32
    featurize = staticmethod(fr_guanido)


class Molecule_Fragments_fr_halogen_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_halogen
    dtype = np.int32
    featurize = staticmethod(fr_halogen)


class Molecule_Fragments_fr_hdrzine_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_hdrzine
    dtype = np.int32
    featurize = staticmethod(fr_hdrzine)


class Molecule_Fragments_fr_hdrzone_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_hdrzone
    dtype = np.int32
    featurize = staticmethod(fr_hdrzone)


class Molecule_Fragments_fr_imidazole_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_imidazole
    dtype = np.int32
    featurize = staticmethod(fr_imidazole)


class Molecule_Fragments_fr_imide_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_imide
    dtype = np.int32
    featurize = staticmethod(fr_imide)


class Molecule_Fragments_fr_isocyan_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_isocyan
    dtype = np.int32
    featurize = staticmethod(fr_isocyan)


class Molecule_Fragments_fr_isothiocyan_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_isothiocyan
    dtype = np.int32
    featurize = staticmethod(fr_isothiocyan)


class Molecule_Fragments_fr_ketone_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_ketone
    dtype = np.int32
    featurize = staticmethod(fr_ketone)


class Molecule_Fragments_fr_ketone_Topliss_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_ketone_Topliss
    dtype = np.int32
    featurize = staticmethod(fr_ketone_Topliss)


class Molecule_Fragments_fr_lactam_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_lactam
    dtype = np.int32
    featurize = staticmethod(fr_lactam)


class Molecule_Fragments_fr_lactone_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_lactone
    dtype = np.int32
    featurize = staticmethod(fr_lactone)


class Molecule_Fragments_fr_methoxy_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_methoxy
    dtype = np.int32
    featurize = staticmethod(fr_methoxy)


class Molecule_Fragments_fr_morpholine_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_morpholine
    dtype = np.int32
    featurize = staticmethod(fr_morpholine)


class Molecule_Fragments_fr_nitrile_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_nitrile
    dtype = np.int32
    featurize = staticmethod(fr_nitrile)


class Molecule_Fragments_fr_nitro_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_nitro
    dtype = np.int32
    featurize = staticmethod(fr_nitro)


class Molecule_Fragments_fr_nitro_arom_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_nitro_arom
    dtype = np.int32
    featurize = staticmethod(fr_nitro_arom)


class Molecule_Fragments_fr_nitro_arom_nonortho_Featurizer(
    SingleValueMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.Fragments.fr_nitro_arom_nonortho
    dtype = np.int32
    featurize = staticmethod(fr_nitro_arom_nonortho)


class Molecule_Fragments_fr_nitroso_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_nitroso
    dtype = np.int32
    featurize = staticmethod(fr_nitroso)


class Molecule_Fragments_fr_oxazole_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_oxazole
    dtype = np.int32
    featurize = staticmethod(fr_oxazole)


class Molecule_Fragments_fr_oxime_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_oxime
    dtype = np.int32
    featurize = staticmethod(fr_oxime)


class Molecule_Fragments_fr_para_hydroxylation_Featurizer(
    SingleValueMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.Fragments.fr_para_hydroxylation
    dtype = np.int32
    featurize = staticmethod(fr_para_hydroxylation)


class Molecule_Fragments_fr_phenol_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_phenol
    dtype = np.int32
    featurize = staticmethod(fr_phenol)


class Molecule_Fragments_fr_phenol_noOrthoHbond_Featurizer(
    SingleValueMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.Fragments.fr_phenol_noOrthoHbond
    dtype = np.int32
    featurize = staticmethod(fr_phenol_noOrthoHbond)


class Molecule_Fragments_fr_phos_acid_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_phos_acid
    dtype = np.int32
    featurize = staticmethod(fr_phos_acid)


class Molecule_Fragments_fr_phos_ester_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_phos_ester
    dtype = np.int32
    featurize = staticmethod(fr_phos_ester)


class Molecule_Fragments_fr_piperdine_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_piperdine
    dtype = np.int32
    featurize = staticmethod(fr_piperdine)


class Molecule_Fragments_fr_piperzine_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_piperzine
    dtype = np.int32
    featurize = staticmethod(fr_piperzine)


class Molecule_Fragments_fr_priamide_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_priamide
    dtype = np.int32
    featurize = staticmethod(fr_priamide)


class Molecule_Fragments_fr_prisulfonamd_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_prisulfonamd
    dtype = np.int32
    featurize = staticmethod(fr_prisulfonamd)


class Molecule_Fragments_fr_pyridine_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_pyridine
    dtype = np.int32
    featurize = staticmethod(fr_pyridine)


class Molecule_Fragments_fr_quatN_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_quatN
    dtype = np.int32
    featurize = staticmethod(fr_quatN)


class Molecule_Fragments_fr_sulfide_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_sulfide
    dtype = np.int32
    featurize = staticmethod(fr_sulfide)


class Molecule_Fragments_fr_sulfonamd_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_sulfonamd
    dtype = np.int32
    featurize = staticmethod(fr_sulfonamd)


class Molecule_Fragments_fr_sulfone_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_sulfone
    dtype = np.int32
    featurize = staticmethod(fr_sulfone)


class Molecule_Fragments_fr_term_acetylene_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_term_acetylene
    dtype = np.int32
    featurize = staticmethod(fr_term_acetylene)


class Molecule_Fragments_fr_tetrazole_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_tetrazole
    dtype = np.int32
    featurize = staticmethod(fr_tetrazole)


class Molecule_Fragments_fr_thiazole_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_thiazole
    dtype = np.int32
    featurize = staticmethod(fr_thiazole)


class Molecule_Fragments_fr_thiocyan_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_thiocyan
    dtype = np.int32
    featurize = staticmethod(fr_thiocyan)


class Molecule_Fragments_fr_thiophene_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_thiophene
    dtype = np.int32
    featurize = staticmethod(fr_thiophene)


class Molecule_Fragments_fr_unbrch_alkane_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_unbrch_alkane
    dtype = np.int32
    featurize = staticmethod(fr_unbrch_alkane)


class Molecule_Fragments_fr_urea_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_urea
    dtype = np.int32
    featurize = staticmethod(fr_urea)


class Molecule_Lipinski_FractionCSP3_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.FractionCSP3
    dtype = np.float32
    featurize = staticmethod(FractionCSP3)


class Molecule_Lipinski_HeavyAtomCount_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.HeavyAtomCount
    dtype = np.int32
    featurize = staticmethod(HeavyAtomCount)


class Molecule_Lipinski_NHOHCount_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NHOHCount
    dtype = np.int32
    featurize = staticmethod(NHOHCount)


class Molecule_Lipinski_NOCount_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NOCount
    dtype = np.int32
    featurize = staticmethod(NOCount)


class Molecule_Lipinski_NumAliphaticCarbocycles_Featurizer(
    SingleValueMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.Lipinski.NumAliphaticCarbocycles
    dtype = np.int32
    featurize = staticmethod(NumAliphaticCarbocycles)


class Molecule_Lipinski_NumAliphaticHeterocycles_Featurizer(
    SingleValueMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.Lipinski.NumAliphaticHeterocycles
    dtype = np.int32
    featurize = staticmethod(NumAliphaticHeterocycles)


class Molecule_Lipinski_NumAliphaticRings_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NumAliphaticRings
    dtype = np.int32
    featurize = staticmethod(NumAliphaticRings)


class Molecule_Lipinski_NumAromaticCarbocycles_Featurizer(
    SingleValueMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.Lipinski.NumAromaticCarbocycles
    dtype = np.int32
    featurize = staticmethod(NumAromaticCarbocycles)


class Molecule_Lipinski_NumAromaticHeterocycles_Featurizer(
    SingleValueMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.Lipinski.NumAromaticHeterocycles
    dtype = np.int32
    featurize = staticmethod(NumAromaticHeterocycles)


class Molecule_Lipinski_NumAromaticRings_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NumAromaticRings
    dtype = np.int32
    featurize = staticmethod(NumAromaticRings)


class Molecule_Lipinski_NumHAcceptors_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NumHAcceptors
    dtype = np.int32
    featurize = staticmethod(NumHAcceptors)


class Molecule_Lipinski_NumHDonors_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NumHDonors
    dtype = np.int32
    featurize = staticmethod(NumHDonors)


class Molecule_Lipinski_NumHeteroatoms_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NumHeteroatoms
    dtype = np.int32
    featurize = staticmethod(NumHeteroatoms)


class Molecule_Lipinski_NumRotatableBonds_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NumRotatableBonds
    dtype = np.int32
    featurize = staticmethod(NumRotatableBonds)


class Molecule_Lipinski_NumSaturatedCarbocycles_Featurizer(
    SingleValueMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.Lipinski.NumSaturatedCarbocycles
    dtype = np.int32
    featurize = staticmethod(NumSaturatedCarbocycles)


class Molecule_Lipinski_NumSaturatedHeterocycles_Featurizer(
    SingleValueMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.Lipinski.NumSaturatedHeterocycles
    dtype = np.int32
    featurize = staticmethod(NumSaturatedHeterocycles)


class Molecule_Lipinski_NumSaturatedRings_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NumSaturatedRings
    dtype = np.int32
    featurize = staticmethod(NumSaturatedRings)


class Molecule_Lipinski_RingCount_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.RingCount
    dtype = np.int32
    featurize = staticmethod(RingCount)


class Molecule_MolSurf_LabuteASA_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.LabuteASA
    dtype = np.float32
    featurize = staticmethod(LabuteASA)


class Molecule_MolSurf_PEOE_VSA10_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA10
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA10)


class Molecule_MolSurf_PEOE_VSA11_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA11
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA11)


class Molecule_MolSurf_PEOE_VSA12_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA12
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA12)


class Molecule_MolSurf_PEOE_VSA13_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA13
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA13)


class Molecule_MolSurf_PEOE_VSA14_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA14
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA14)


class Molecule_MolSurf_PEOE_VSA1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA1
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA1)


class Molecule_MolSurf_PEOE_VSA2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA2
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA2)


class Molecule_MolSurf_PEOE_VSA3_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA3
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA3)


class Molecule_MolSurf_PEOE_VSA4_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA4
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA4)


class Molecule_MolSurf_PEOE_VSA5_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA5
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA5)


class Molecule_MolSurf_PEOE_VSA6_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA6
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA6)


class Molecule_MolSurf_PEOE_VSA7_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA7
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA7)


class Molecule_MolSurf_PEOE_VSA8_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA8
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA8)


class Molecule_MolSurf_PEOE_VSA9_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA9
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA9)


class Molecule_MolSurf_SMR_VSA10_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SMR_VSA10
    dtype = np.float32
    featurize = staticmethod(SMR_VSA10)


class Molecule_MolSurf_SMR_VSA1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SMR_VSA1
    dtype = np.float32
    featurize = staticmethod(SMR_VSA1)


class Molecule_MolSurf_SMR_VSA2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SMR_VSA2
    dtype = np.float32
    featurize = staticmethod(SMR_VSA2)


class Molecule_MolSurf_SMR_VSA3_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SMR_VSA3
    dtype = np.float32
    featurize = staticmethod(SMR_VSA3)


class Molecule_MolSurf_SMR_VSA4_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SMR_VSA4
    dtype = np.float32
    featurize = staticmethod(SMR_VSA4)


class Molecule_MolSurf_SMR_VSA5_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SMR_VSA5
    dtype = np.float32
    featurize = staticmethod(SMR_VSA5)


class Molecule_MolSurf_SMR_VSA6_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SMR_VSA6
    dtype = np.float32
    featurize = staticmethod(SMR_VSA6)


class Molecule_MolSurf_SMR_VSA7_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SMR_VSA7
    dtype = np.float32
    featurize = staticmethod(SMR_VSA7)


class Molecule_MolSurf_SMR_VSA8_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SMR_VSA8
    dtype = np.float32
    featurize = staticmethod(SMR_VSA8)


class Molecule_MolSurf_SMR_VSA9_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SMR_VSA9
    dtype = np.float32
    featurize = staticmethod(SMR_VSA9)


class Molecule_MolSurf_SlogP_VSA10_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA10
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA10)


class Molecule_MolSurf_SlogP_VSA11_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA11
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA11)


class Molecule_MolSurf_SlogP_VSA12_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA12
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA12)


class Molecule_MolSurf_SlogP_VSA1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA1
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA1)


class Molecule_MolSurf_SlogP_VSA2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA2
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA2)


class Molecule_MolSurf_SlogP_VSA3_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA3
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA3)


class Molecule_MolSurf_SlogP_VSA4_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA4
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA4)


class Molecule_MolSurf_SlogP_VSA5_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA5
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA5)


class Molecule_MolSurf_SlogP_VSA6_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA6
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA6)


class Molecule_MolSurf_SlogP_VSA7_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA7
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA7)


class Molecule_MolSurf_SlogP_VSA8_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA8
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA8)


class Molecule_MolSurf_SlogP_VSA9_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA9
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA9)


class Molecule_MolSurf_TPSA_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.TPSA
    dtype = np.float32
    featurize = staticmethod(TPSA)


class Molecule_MolSurf_pyLabuteASA_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.pyLabuteASA
    dtype = np.float32
    featurize = staticmethod(pyLabuteASA)


class Molecule_NumAtoms_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=GetNumAtoms
    dtype = np.int32

    def featurize(self, mol):
        return mol.GetNumAtoms()


class Molecule_NumBonds_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=GetNumBonds
    dtype = np.int32

    def featurize(self, mol):
        return mol.GetNumBonds()


class Molecule_NumHeavyAtoms_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=GetNumHeavyAtoms
    dtype = np.int32

    def featurize(self, mol):
        return mol.GetNumHeavyAtoms()


class Molecule_QED_default_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.QED.default
    dtype = np.float32
    featurize = staticmethod(default)


class Molecule_QED_qed_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.QED.qed
    dtype = np.float32
    featurize = staticmethod(qed)


class Molecule_QED_weights_max_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.QED.weights_max
    dtype = np.float32
    featurize = staticmethod(weights_max)


class Molecule_QED_weights_mean_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.QED.weights_mean
    dtype = np.float32
    featurize = staticmethod(weights_mean)


class Molecule_QED_weights_none_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.QED.weights_none
    dtype = np.float32
    featurize = staticmethod(weights_none)


class Molecule_rdmolops_FormalCharge_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.rdmolops.GetFormalCharge
    dtype = np.int32
    featurize = staticmethod(GetFormalCharge)


class Molecule_rdmolops_SSSR_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.rdmolops.GetSSSR
    dtype = np.int32
    featurize = staticmethod(GetSSSR)


molecule_AllChem_Asphericity_featurizer = Molecule_AllChem_Asphericity_Featurizer()
molecule_AllChem_Chi0n_featurizer = Molecule_AllChem_Chi0n_Featurizer()
molecule_AllChem_Chi0v_featurizer = Molecule_AllChem_Chi0v_Featurizer()
molecule_AllChem_Chi1n_featurizer = Molecule_AllChem_Chi1n_Featurizer()
molecule_AllChem_Chi1v_featurizer = Molecule_AllChem_Chi1v_Featurizer()
molecule_AllChem_Chi2n_featurizer = Molecule_AllChem_Chi2n_Featurizer()
molecule_AllChem_Chi2v_featurizer = Molecule_AllChem_Chi2v_Featurizer()
molecule_AllChem_Chi3n_featurizer = Molecule_AllChem_Chi3n_Featurizer()
molecule_AllChem_Chi3v_featurizer = Molecule_AllChem_Chi3v_Featurizer()
molecule_AllChem_Chi4n_featurizer = Molecule_AllChem_Chi4n_Featurizer()
molecule_AllChem_Chi4v_featurizer = Molecule_AllChem_Chi4v_Featurizer()
molecule_AllChem_Compute2DCoords_featurizer = (
    Molecule_AllChem_Compute2DCoords_Featurizer()
)
molecule_AllChem_ComputeMolVolume_featurizer = (
    Molecule_AllChem_ComputeMolVolume_Featurizer()
)
molecule_AllChem_Eccentricity_featurizer = Molecule_AllChem_Eccentricity_Featurizer()
molecule_AllChem_EmbedMolecule_featurizer = Molecule_AllChem_EmbedMolecule_Featurizer()
molecule_AllChem_ExactMolWt_featurizer = Molecule_AllChem_ExactMolWt_Featurizer()
molecule_AllChem_FractionCSP3_featurizer = Molecule_AllChem_FractionCSP3_Featurizer()
molecule_AllChem_HallKierAlpha_featurizer = Molecule_AllChem_HallKierAlpha_Featurizer()
molecule_AllChem_InertialShapeFactor_featurizer = (
    Molecule_AllChem_InertialShapeFactor_Featurizer()
)
molecule_AllChem_Kappa1_featurizer = Molecule_AllChem_Kappa1_Featurizer()
molecule_AllChem_Kappa2_featurizer = Molecule_AllChem_Kappa2_Featurizer()
molecule_AllChem_Kappa3_featurizer = Molecule_AllChem_Kappa3_Featurizer()
molecule_AllChem_LabuteASA_featurizer = Molecule_AllChem_LabuteASA_Featurizer()
molecule_AllChem_NPR1_featurizer = Molecule_AllChem_NPR1_Featurizer()
molecule_AllChem_NPR2_featurizer = Molecule_AllChem_NPR2_Featurizer()
molecule_AllChem_NumAliphaticCarbocycles_featurizer = (
    Molecule_AllChem_NumAliphaticCarbocycles_Featurizer()
)
molecule_AllChem_NumAliphaticHeterocycles_featurizer = (
    Molecule_AllChem_NumAliphaticHeterocycles_Featurizer()
)
molecule_AllChem_NumAliphaticRings_featurizer = (
    Molecule_AllChem_NumAliphaticRings_Featurizer()
)
molecule_AllChem_NumAmideBonds_featurizer = Molecule_AllChem_NumAmideBonds_Featurizer()
molecule_AllChem_NumAromaticCarbocycles_featurizer = (
    Molecule_AllChem_NumAromaticCarbocycles_Featurizer()
)
molecule_AllChem_NumAromaticHeterocycles_featurizer = (
    Molecule_AllChem_NumAromaticHeterocycles_Featurizer()
)
molecule_AllChem_NumAromaticRings_featurizer = (
    Molecule_AllChem_NumAromaticRings_Featurizer()
)
molecule_AllChem_NumAtoms_featurizer = Molecule_AllChem_NumAtoms_Featurizer()
molecule_AllChem_NumBridgeheadAtoms_featurizer = (
    Molecule_AllChem_NumBridgeheadAtoms_Featurizer()
)
molecule_AllChem_NumHBA_featurizer = Molecule_AllChem_NumHBA_Featurizer()
molecule_AllChem_NumHBD_featurizer = Molecule_AllChem_NumHBD_Featurizer()
molecule_AllChem_NumHeavyAtoms_featurizer = Molecule_AllChem_NumHeavyAtoms_Featurizer()
molecule_AllChem_NumHeteroatoms_featurizer = (
    Molecule_AllChem_NumHeteroatoms_Featurizer()
)
molecule_AllChem_NumHeterocycles_featurizer = (
    Molecule_AllChem_NumHeterocycles_Featurizer()
)
molecule_AllChem_NumLipinskiHBA_featurizer = (
    Molecule_AllChem_NumLipinskiHBA_Featurizer()
)
molecule_AllChem_NumLipinskiHBD_featurizer = (
    Molecule_AllChem_NumLipinskiHBD_Featurizer()
)
molecule_AllChem_NumRings_featurizer = Molecule_AllChem_NumRings_Featurizer()
molecule_AllChem_NumRotatableBonds_featurizer = (
    Molecule_AllChem_NumRotatableBonds_Featurizer()
)
molecule_AllChem_NumSaturatedCarbocycles_featurizer = (
    Molecule_AllChem_NumSaturatedCarbocycles_Featurizer()
)
molecule_AllChem_NumSaturatedHeterocycles_featurizer = (
    Molecule_AllChem_NumSaturatedHeterocycles_Featurizer()
)
molecule_AllChem_NumSaturatedRings_featurizer = (
    Molecule_AllChem_NumSaturatedRings_Featurizer()
)
molecule_AllChem_NumSpiroAtoms_featurizer = Molecule_AllChem_NumSpiroAtoms_Featurizer()
molecule_AllChem_PBF_featurizer = Molecule_AllChem_PBF_Featurizer()
molecule_AllChem_PMI1_featurizer = Molecule_AllChem_PMI1_Featurizer()
molecule_AllChem_PMI2_featurizer = Molecule_AllChem_PMI2_Featurizer()
molecule_AllChem_PMI3_featurizer = Molecule_AllChem_PMI3_Featurizer()
molecule_AllChem_Phi_featurizer = Molecule_AllChem_Phi_Featurizer()
molecule_AllChem_RadiusOfGyration_featurizer = (
    Molecule_AllChem_RadiusOfGyration_Featurizer()
)
molecule_AllChem_SpherocityIndex_featurizer = (
    Molecule_AllChem_SpherocityIndex_Featurizer()
)
molecule_AllChem_TPSA_featurizer = Molecule_AllChem_TPSA_Featurizer()
molecule_Chem_FormalCharge_featurizer = Molecule_Chem_FormalCharge_Featurizer()
molecule_Chem_SSSR_featurizer = Molecule_Chem_SSSR_Featurizer()
molecule_Crippen_MolLogP_featurizer = Molecule_Crippen_MolLogP_Featurizer()
molecule_Crippen_MolMR_featurizer = Molecule_Crippen_MolMR_Featurizer()
molecule_Descriptors3D_Asphericity_featurizer = (
    Molecule_Descriptors3D_Asphericity_Featurizer()
)
molecule_Descriptors3D_Eccentricity_featurizer = (
    Molecule_Descriptors3D_Eccentricity_Featurizer()
)
molecule_Descriptors3D_InertialShapeFactor_featurizer = (
    Molecule_Descriptors3D_InertialShapeFactor_Featurizer()
)
molecule_Descriptors3D_NPR1_featurizer = Molecule_Descriptors3D_NPR1_Featurizer()
molecule_Descriptors3D_NPR2_featurizer = Molecule_Descriptors3D_NPR2_Featurizer()
molecule_Descriptors3D_PMI1_featurizer = Molecule_Descriptors3D_PMI1_Featurizer()
molecule_Descriptors3D_PMI2_featurizer = Molecule_Descriptors3D_PMI2_Featurizer()
molecule_Descriptors3D_PMI3_featurizer = Molecule_Descriptors3D_PMI3_Featurizer()
molecule_Descriptors3D_RadiusOfGyration_featurizer = (
    Molecule_Descriptors3D_RadiusOfGyration_Featurizer()
)
molecule_Descriptors3D_SpherocityIndex_featurizer = (
    Molecule_Descriptors3D_SpherocityIndex_Featurizer()
)
molecule_Descriptors_BCUT2D_CHGHI_featurizer = (
    Molecule_Descriptors_BCUT2D_CHGHI_Featurizer()
)
molecule_Descriptors_BCUT2D_CHGLO_featurizer = (
    Molecule_Descriptors_BCUT2D_CHGLO_Featurizer()
)
molecule_Descriptors_BCUT2D_LOGPHI_featurizer = (
    Molecule_Descriptors_BCUT2D_LOGPHI_Featurizer()
)
molecule_Descriptors_BCUT2D_LOGPLOW_featurizer = (
    Molecule_Descriptors_BCUT2D_LOGPLOW_Featurizer()
)
molecule_Descriptors_BCUT2D_MRHI_featurizer = (
    Molecule_Descriptors_BCUT2D_MRHI_Featurizer()
)
molecule_Descriptors_BCUT2D_MRLOW_featurizer = (
    Molecule_Descriptors_BCUT2D_MRLOW_Featurizer()
)
molecule_Descriptors_BCUT2D_MWHI_featurizer = (
    Molecule_Descriptors_BCUT2D_MWHI_Featurizer()
)
molecule_Descriptors_BCUT2D_MWLOW_featurizer = (
    Molecule_Descriptors_BCUT2D_MWLOW_Featurizer()
)
molecule_Descriptors_BalabanJ_featurizer = Molecule_Descriptors_BalabanJ_Featurizer()
molecule_Descriptors_BertzCT_featurizer = Molecule_Descriptors_BertzCT_Featurizer()
molecule_Descriptors_Chi0_featurizer = Molecule_Descriptors_Chi0_Featurizer()
molecule_Descriptors_Chi0n_featurizer = Molecule_Descriptors_Chi0n_Featurizer()
molecule_Descriptors_Chi0v_featurizer = Molecule_Descriptors_Chi0v_Featurizer()
molecule_Descriptors_Chi1_featurizer = Molecule_Descriptors_Chi1_Featurizer()
molecule_Descriptors_Chi1n_featurizer = Molecule_Descriptors_Chi1n_Featurizer()
molecule_Descriptors_Chi1v_featurizer = Molecule_Descriptors_Chi1v_Featurizer()
molecule_Descriptors_Chi2n_featurizer = Molecule_Descriptors_Chi2n_Featurizer()
molecule_Descriptors_Chi2v_featurizer = Molecule_Descriptors_Chi2v_Featurizer()
molecule_Descriptors_Chi3n_featurizer = Molecule_Descriptors_Chi3n_Featurizer()
molecule_Descriptors_Chi3v_featurizer = Molecule_Descriptors_Chi3v_Featurizer()
molecule_Descriptors_Chi4n_featurizer = Molecule_Descriptors_Chi4n_Featurizer()
molecule_Descriptors_Chi4v_featurizer = Molecule_Descriptors_Chi4v_Featurizer()
molecule_Descriptors_EState_VSA10_featurizer = (
    Molecule_Descriptors_EState_VSA10_Featurizer()
)
molecule_Descriptors_EState_VSA11_featurizer = (
    Molecule_Descriptors_EState_VSA11_Featurizer()
)
molecule_Descriptors_EState_VSA1_featurizer = (
    Molecule_Descriptors_EState_VSA1_Featurizer()
)
molecule_Descriptors_EState_VSA2_featurizer = (
    Molecule_Descriptors_EState_VSA2_Featurizer()
)
molecule_Descriptors_EState_VSA3_featurizer = (
    Molecule_Descriptors_EState_VSA3_Featurizer()
)
molecule_Descriptors_EState_VSA4_featurizer = (
    Molecule_Descriptors_EState_VSA4_Featurizer()
)
molecule_Descriptors_EState_VSA5_featurizer = (
    Molecule_Descriptors_EState_VSA5_Featurizer()
)
molecule_Descriptors_EState_VSA6_featurizer = (
    Molecule_Descriptors_EState_VSA6_Featurizer()
)
molecule_Descriptors_EState_VSA7_featurizer = (
    Molecule_Descriptors_EState_VSA7_Featurizer()
)
molecule_Descriptors_EState_VSA8_featurizer = (
    Molecule_Descriptors_EState_VSA8_Featurizer()
)
molecule_Descriptors_EState_VSA9_featurizer = (
    Molecule_Descriptors_EState_VSA9_Featurizer()
)
molecule_Descriptors_ExactMolWt_featurizer = (
    Molecule_Descriptors_ExactMolWt_Featurizer()
)
molecule_Descriptors_FpDensityMorgan1_featurizer = (
    Molecule_Descriptors_FpDensityMorgan1_Featurizer()
)
molecule_Descriptors_FpDensityMorgan2_featurizer = (
    Molecule_Descriptors_FpDensityMorgan2_Featurizer()
)
molecule_Descriptors_FpDensityMorgan3_featurizer = (
    Molecule_Descriptors_FpDensityMorgan3_Featurizer()
)
molecule_Descriptors_HallKierAlpha_featurizer = (
    Molecule_Descriptors_HallKierAlpha_Featurizer()
)
molecule_Descriptors_HeavyAtomMolWt_featurizer = (
    Molecule_Descriptors_HeavyAtomMolWt_Featurizer()
)
molecule_Descriptors_Ipc_featurizer = Molecule_Descriptors_Ipc_Featurizer()
molecule_Descriptors_Kappa1_featurizer = Molecule_Descriptors_Kappa1_Featurizer()
molecule_Descriptors_Kappa2_featurizer = Molecule_Descriptors_Kappa2_Featurizer()
molecule_Descriptors_Kappa3_featurizer = Molecule_Descriptors_Kappa3_Featurizer()
molecule_Descriptors_MaxAbsEStateIndex_featurizer = (
    Molecule_Descriptors_MaxAbsEStateIndex_Featurizer()
)
molecule_Descriptors_MaxAbsPartialCharge_featurizer = (
    Molecule_Descriptors_MaxAbsPartialCharge_Featurizer()
)
molecule_Descriptors_MaxPartialCharge_featurizer = (
    Molecule_Descriptors_MaxPartialCharge_Featurizer()
)
molecule_Descriptors_MinAbsPartialCharge_featurizer = (
    Molecule_Descriptors_MinAbsPartialCharge_Featurizer()
)
molecule_Descriptors_MinEStateIndex_featurizer = (
    Molecule_Descriptors_MinEStateIndex_Featurizer()
)
molecule_Descriptors_MinPartialCharge_featurizer = (
    Molecule_Descriptors_MinPartialCharge_Featurizer()
)
molecule_Descriptors_MolWt_featurizer = Molecule_Descriptors_MolWt_Featurizer()
molecule_Descriptors_NumRadicalElectrons_featurizer = (
    Molecule_Descriptors_NumRadicalElectrons_Featurizer()
)
molecule_Descriptors_NumValenceElectrons_featurizer = (
    Molecule_Descriptors_NumValenceElectrons_Featurizer()
)
molecule_Descriptors_VSA_EState10_featurizer = (
    Molecule_Descriptors_VSA_EState10_Featurizer()
)
molecule_Descriptors_VSA_EState1_featurizer = (
    Molecule_Descriptors_VSA_EState1_Featurizer()
)
molecule_Descriptors_VSA_EState2_featurizer = (
    Molecule_Descriptors_VSA_EState2_Featurizer()
)
molecule_Descriptors_VSA_EState3_featurizer = (
    Molecule_Descriptors_VSA_EState3_Featurizer()
)
molecule_Descriptors_VSA_EState4_featurizer = (
    Molecule_Descriptors_VSA_EState4_Featurizer()
)
molecule_Descriptors_VSA_EState5_featurizer = (
    Molecule_Descriptors_VSA_EState5_Featurizer()
)
molecule_Descriptors_VSA_EState6_featurizer = (
    Molecule_Descriptors_VSA_EState6_Featurizer()
)
molecule_Descriptors_VSA_EState7_featurizer = (
    Molecule_Descriptors_VSA_EState7_Featurizer()
)
molecule_Descriptors_VSA_EState8_featurizer = (
    Molecule_Descriptors_VSA_EState8_Featurizer()
)
molecule_Descriptors_VSA_EState9_featurizer = (
    Molecule_Descriptors_VSA_EState9_Featurizer()
)
molecule_EState_MaxAbsEStateIndex_featurizer = (
    Molecule_EState_MaxAbsEStateIndex_Featurizer()
)
molecule_EState_MaxEStateIndex_featurizer = Molecule_EState_MaxEStateIndex_Featurizer()
molecule_EState_MinAbsEStateIndex_featurizer = (
    Molecule_EState_MinAbsEStateIndex_Featurizer()
)
molecule_EState_MinEStateIndex_featurizer = Molecule_EState_MinEStateIndex_Featurizer()
molecule_EnumerateStereoisomers_EmbedMolecule_featurizer = (
    Molecule_EnumerateStereoisomers_EmbedMolecule_Featurizer()
)
molecule_EnumerateStereoisomers_StereoisomerCount_featurizer = (
    Molecule_EnumerateStereoisomers_StereoisomerCount_Featurizer()
)
molecule_Fragments_fr_Al_COO_featurizer = Molecule_Fragments_fr_Al_COO_Featurizer()
molecule_Fragments_fr_Al_OH_featurizer = Molecule_Fragments_fr_Al_OH_Featurizer()
molecule_Fragments_fr_Al_OH_noTert_featurizer = (
    Molecule_Fragments_fr_Al_OH_noTert_Featurizer()
)
molecule_Fragments_fr_ArN_featurizer = Molecule_Fragments_fr_ArN_Featurizer()
molecule_Fragments_fr_Ar_COO_featurizer = Molecule_Fragments_fr_Ar_COO_Featurizer()
molecule_Fragments_fr_Ar_NH_featurizer = Molecule_Fragments_fr_Ar_NH_Featurizer()
molecule_Fragments_fr_Ar_N_featurizer = Molecule_Fragments_fr_Ar_N_Featurizer()
molecule_Fragments_fr_Ar_OH_featurizer = Molecule_Fragments_fr_Ar_OH_Featurizer()
molecule_Fragments_fr_COO2_featurizer = Molecule_Fragments_fr_COO2_Featurizer()
molecule_Fragments_fr_COO_featurizer = Molecule_Fragments_fr_COO_Featurizer()
molecule_Fragments_fr_C_O_featurizer = Molecule_Fragments_fr_C_O_Featurizer()
molecule_Fragments_fr_C_O_noCOO_featurizer = (
    Molecule_Fragments_fr_C_O_noCOO_Featurizer()
)
molecule_Fragments_fr_C_S_featurizer = Molecule_Fragments_fr_C_S_Featurizer()
molecule_Fragments_fr_HOCCN_featurizer = Molecule_Fragments_fr_HOCCN_Featurizer()
molecule_Fragments_fr_Imine_featurizer = Molecule_Fragments_fr_Imine_Featurizer()
molecule_Fragments_fr_NH0_featurizer = Molecule_Fragments_fr_NH0_Featurizer()
molecule_Fragments_fr_NH1_featurizer = Molecule_Fragments_fr_NH1_Featurizer()
molecule_Fragments_fr_NH2_featurizer = Molecule_Fragments_fr_NH2_Featurizer()
molecule_Fragments_fr_N_O_featurizer = Molecule_Fragments_fr_N_O_Featurizer()
molecule_Fragments_fr_Ndealkylation1_featurizer = (
    Molecule_Fragments_fr_Ndealkylation1_Featurizer()
)
molecule_Fragments_fr_Ndealkylation2_featurizer = (
    Molecule_Fragments_fr_Ndealkylation2_Featurizer()
)
molecule_Fragments_fr_Nhpyrrole_featurizer = (
    Molecule_Fragments_fr_Nhpyrrole_Featurizer()
)
molecule_Fragments_fr_SH_featurizer = Molecule_Fragments_fr_SH_Featurizer()
molecule_Fragments_fr_aldehyde_featurizer = Molecule_Fragments_fr_aldehyde_Featurizer()
molecule_Fragments_fr_alkyl_carbamate_featurizer = (
    Molecule_Fragments_fr_alkyl_carbamate_Featurizer()
)
molecule_Fragments_fr_alkyl_halide_featurizer = (
    Molecule_Fragments_fr_alkyl_halide_Featurizer()
)
molecule_Fragments_fr_allylic_oxid_featurizer = (
    Molecule_Fragments_fr_allylic_oxid_Featurizer()
)
molecule_Fragments_fr_amide_featurizer = Molecule_Fragments_fr_amide_Featurizer()
molecule_Fragments_fr_amidine_featurizer = Molecule_Fragments_fr_amidine_Featurizer()
molecule_Fragments_fr_aniline_featurizer = Molecule_Fragments_fr_aniline_Featurizer()
molecule_Fragments_fr_aryl_methyl_featurizer = (
    Molecule_Fragments_fr_aryl_methyl_Featurizer()
)
molecule_Fragments_fr_azide_featurizer = Molecule_Fragments_fr_azide_Featurizer()
molecule_Fragments_fr_azo_featurizer = Molecule_Fragments_fr_azo_Featurizer()
molecule_Fragments_fr_barbitur_featurizer = Molecule_Fragments_fr_barbitur_Featurizer()
molecule_Fragments_fr_benzene_featurizer = Molecule_Fragments_fr_benzene_Featurizer()
molecule_Fragments_fr_benzodiazepine_featurizer = (
    Molecule_Fragments_fr_benzodiazepine_Featurizer()
)
molecule_Fragments_fr_bicyclic_featurizer = Molecule_Fragments_fr_bicyclic_Featurizer()
molecule_Fragments_fr_diazo_featurizer = Molecule_Fragments_fr_diazo_Featurizer()
molecule_Fragments_fr_dihydropyridine_featurizer = (
    Molecule_Fragments_fr_dihydropyridine_Featurizer()
)
molecule_Fragments_fr_epoxide_featurizer = Molecule_Fragments_fr_epoxide_Featurizer()
molecule_Fragments_fr_ester_featurizer = Molecule_Fragments_fr_ester_Featurizer()
molecule_Fragments_fr_ether_featurizer = Molecule_Fragments_fr_ether_Featurizer()
molecule_Fragments_fr_furan_featurizer = Molecule_Fragments_fr_furan_Featurizer()
molecule_Fragments_fr_guanido_featurizer = Molecule_Fragments_fr_guanido_Featurizer()
molecule_Fragments_fr_halogen_featurizer = Molecule_Fragments_fr_halogen_Featurizer()
molecule_Fragments_fr_hdrzine_featurizer = Molecule_Fragments_fr_hdrzine_Featurizer()
molecule_Fragments_fr_hdrzone_featurizer = Molecule_Fragments_fr_hdrzone_Featurizer()
molecule_Fragments_fr_imidazole_featurizer = (
    Molecule_Fragments_fr_imidazole_Featurizer()
)
molecule_Fragments_fr_imide_featurizer = Molecule_Fragments_fr_imide_Featurizer()
molecule_Fragments_fr_isocyan_featurizer = Molecule_Fragments_fr_isocyan_Featurizer()
molecule_Fragments_fr_isothiocyan_featurizer = (
    Molecule_Fragments_fr_isothiocyan_Featurizer()
)
molecule_Fragments_fr_ketone_featurizer = Molecule_Fragments_fr_ketone_Featurizer()
molecule_Fragments_fr_ketone_Topliss_featurizer = (
    Molecule_Fragments_fr_ketone_Topliss_Featurizer()
)
molecule_Fragments_fr_lactam_featurizer = Molecule_Fragments_fr_lactam_Featurizer()
molecule_Fragments_fr_lactone_featurizer = Molecule_Fragments_fr_lactone_Featurizer()
molecule_Fragments_fr_methoxy_featurizer = Molecule_Fragments_fr_methoxy_Featurizer()
molecule_Fragments_fr_morpholine_featurizer = (
    Molecule_Fragments_fr_morpholine_Featurizer()
)
molecule_Fragments_fr_nitrile_featurizer = Molecule_Fragments_fr_nitrile_Featurizer()
molecule_Fragments_fr_nitro_featurizer = Molecule_Fragments_fr_nitro_Featurizer()
molecule_Fragments_fr_nitro_arom_featurizer = (
    Molecule_Fragments_fr_nitro_arom_Featurizer()
)
molecule_Fragments_fr_nitro_arom_nonortho_featurizer = (
    Molecule_Fragments_fr_nitro_arom_nonortho_Featurizer()
)
molecule_Fragments_fr_nitroso_featurizer = Molecule_Fragments_fr_nitroso_Featurizer()
molecule_Fragments_fr_oxazole_featurizer = Molecule_Fragments_fr_oxazole_Featurizer()
molecule_Fragments_fr_oxime_featurizer = Molecule_Fragments_fr_oxime_Featurizer()
molecule_Fragments_fr_para_hydroxylation_featurizer = (
    Molecule_Fragments_fr_para_hydroxylation_Featurizer()
)
molecule_Fragments_fr_phenol_featurizer = Molecule_Fragments_fr_phenol_Featurizer()
molecule_Fragments_fr_phenol_noOrthoHbond_featurizer = (
    Molecule_Fragments_fr_phenol_noOrthoHbond_Featurizer()
)
molecule_Fragments_fr_phos_acid_featurizer = (
    Molecule_Fragments_fr_phos_acid_Featurizer()
)
molecule_Fragments_fr_phos_ester_featurizer = (
    Molecule_Fragments_fr_phos_ester_Featurizer()
)
molecule_Fragments_fr_piperdine_featurizer = (
    Molecule_Fragments_fr_piperdine_Featurizer()
)
molecule_Fragments_fr_piperzine_featurizer = (
    Molecule_Fragments_fr_piperzine_Featurizer()
)
molecule_Fragments_fr_priamide_featurizer = Molecule_Fragments_fr_priamide_Featurizer()
molecule_Fragments_fr_prisulfonamd_featurizer = (
    Molecule_Fragments_fr_prisulfonamd_Featurizer()
)
molecule_Fragments_fr_pyridine_featurizer = Molecule_Fragments_fr_pyridine_Featurizer()
molecule_Fragments_fr_quatN_featurizer = Molecule_Fragments_fr_quatN_Featurizer()
molecule_Fragments_fr_sulfide_featurizer = Molecule_Fragments_fr_sulfide_Featurizer()
molecule_Fragments_fr_sulfonamd_featurizer = (
    Molecule_Fragments_fr_sulfonamd_Featurizer()
)
molecule_Fragments_fr_sulfone_featurizer = Molecule_Fragments_fr_sulfone_Featurizer()
molecule_Fragments_fr_term_acetylene_featurizer = (
    Molecule_Fragments_fr_term_acetylene_Featurizer()
)
molecule_Fragments_fr_tetrazole_featurizer = (
    Molecule_Fragments_fr_tetrazole_Featurizer()
)
molecule_Fragments_fr_thiazole_featurizer = Molecule_Fragments_fr_thiazole_Featurizer()
molecule_Fragments_fr_thiocyan_featurizer = Molecule_Fragments_fr_thiocyan_Featurizer()
molecule_Fragments_fr_thiophene_featurizer = (
    Molecule_Fragments_fr_thiophene_Featurizer()
)
molecule_Fragments_fr_unbrch_alkane_featurizer = (
    Molecule_Fragments_fr_unbrch_alkane_Featurizer()
)
molecule_Fragments_fr_urea_featurizer = Molecule_Fragments_fr_urea_Featurizer()
molecule_Lipinski_FractionCSP3_featurizer = Molecule_Lipinski_FractionCSP3_Featurizer()
molecule_Lipinski_HeavyAtomCount_featurizer = (
    Molecule_Lipinski_HeavyAtomCount_Featurizer()
)
molecule_Lipinski_NHOHCount_featurizer = Molecule_Lipinski_NHOHCount_Featurizer()
molecule_Lipinski_NOCount_featurizer = Molecule_Lipinski_NOCount_Featurizer()
molecule_Lipinski_NumAliphaticCarbocycles_featurizer = (
    Molecule_Lipinski_NumAliphaticCarbocycles_Featurizer()
)
molecule_Lipinski_NumAliphaticHeterocycles_featurizer = (
    Molecule_Lipinski_NumAliphaticHeterocycles_Featurizer()
)
molecule_Lipinski_NumAliphaticRings_featurizer = (
    Molecule_Lipinski_NumAliphaticRings_Featurizer()
)
molecule_Lipinski_NumAromaticCarbocycles_featurizer = (
    Molecule_Lipinski_NumAromaticCarbocycles_Featurizer()
)
molecule_Lipinski_NumAromaticHeterocycles_featurizer = (
    Molecule_Lipinski_NumAromaticHeterocycles_Featurizer()
)
molecule_Lipinski_NumAromaticRings_featurizer = (
    Molecule_Lipinski_NumAromaticRings_Featurizer()
)
molecule_Lipinski_NumHAcceptors_featurizer = (
    Molecule_Lipinski_NumHAcceptors_Featurizer()
)
molecule_Lipinski_NumHDonors_featurizer = Molecule_Lipinski_NumHDonors_Featurizer()
molecule_Lipinski_NumHeteroatoms_featurizer = (
    Molecule_Lipinski_NumHeteroatoms_Featurizer()
)
molecule_Lipinski_NumRotatableBonds_featurizer = (
    Molecule_Lipinski_NumRotatableBonds_Featurizer()
)
molecule_Lipinski_NumSaturatedCarbocycles_featurizer = (
    Molecule_Lipinski_NumSaturatedCarbocycles_Featurizer()
)
molecule_Lipinski_NumSaturatedHeterocycles_featurizer = (
    Molecule_Lipinski_NumSaturatedHeterocycles_Featurizer()
)
molecule_Lipinski_NumSaturatedRings_featurizer = (
    Molecule_Lipinski_NumSaturatedRings_Featurizer()
)
molecule_Lipinski_RingCount_featurizer = Molecule_Lipinski_RingCount_Featurizer()
molecule_MolSurf_LabuteASA_featurizer = Molecule_MolSurf_LabuteASA_Featurizer()
molecule_MolSurf_PEOE_VSA10_featurizer = Molecule_MolSurf_PEOE_VSA10_Featurizer()
molecule_MolSurf_PEOE_VSA11_featurizer = Molecule_MolSurf_PEOE_VSA11_Featurizer()
molecule_MolSurf_PEOE_VSA12_featurizer = Molecule_MolSurf_PEOE_VSA12_Featurizer()
molecule_MolSurf_PEOE_VSA13_featurizer = Molecule_MolSurf_PEOE_VSA13_Featurizer()
molecule_MolSurf_PEOE_VSA14_featurizer = Molecule_MolSurf_PEOE_VSA14_Featurizer()
molecule_MolSurf_PEOE_VSA1_featurizer = Molecule_MolSurf_PEOE_VSA1_Featurizer()
molecule_MolSurf_PEOE_VSA2_featurizer = Molecule_MolSurf_PEOE_VSA2_Featurizer()
molecule_MolSurf_PEOE_VSA3_featurizer = Molecule_MolSurf_PEOE_VSA3_Featurizer()
molecule_MolSurf_PEOE_VSA4_featurizer = Molecule_MolSurf_PEOE_VSA4_Featurizer()
molecule_MolSurf_PEOE_VSA5_featurizer = Molecule_MolSurf_PEOE_VSA5_Featurizer()
molecule_MolSurf_PEOE_VSA6_featurizer = Molecule_MolSurf_PEOE_VSA6_Featurizer()
molecule_MolSurf_PEOE_VSA7_featurizer = Molecule_MolSurf_PEOE_VSA7_Featurizer()
molecule_MolSurf_PEOE_VSA8_featurizer = Molecule_MolSurf_PEOE_VSA8_Featurizer()
molecule_MolSurf_PEOE_VSA9_featurizer = Molecule_MolSurf_PEOE_VSA9_Featurizer()
molecule_MolSurf_SMR_VSA10_featurizer = Molecule_MolSurf_SMR_VSA10_Featurizer()
molecule_MolSurf_SMR_VSA1_featurizer = Molecule_MolSurf_SMR_VSA1_Featurizer()
molecule_MolSurf_SMR_VSA2_featurizer = Molecule_MolSurf_SMR_VSA2_Featurizer()
molecule_MolSurf_SMR_VSA3_featurizer = Molecule_MolSurf_SMR_VSA3_Featurizer()
molecule_MolSurf_SMR_VSA4_featurizer = Molecule_MolSurf_SMR_VSA4_Featurizer()
molecule_MolSurf_SMR_VSA5_featurizer = Molecule_MolSurf_SMR_VSA5_Featurizer()
molecule_MolSurf_SMR_VSA6_featurizer = Molecule_MolSurf_SMR_VSA6_Featurizer()
molecule_MolSurf_SMR_VSA7_featurizer = Molecule_MolSurf_SMR_VSA7_Featurizer()
molecule_MolSurf_SMR_VSA8_featurizer = Molecule_MolSurf_SMR_VSA8_Featurizer()
molecule_MolSurf_SMR_VSA9_featurizer = Molecule_MolSurf_SMR_VSA9_Featurizer()
molecule_MolSurf_SlogP_VSA10_featurizer = Molecule_MolSurf_SlogP_VSA10_Featurizer()
molecule_MolSurf_SlogP_VSA11_featurizer = Molecule_MolSurf_SlogP_VSA11_Featurizer()
molecule_MolSurf_SlogP_VSA12_featurizer = Molecule_MolSurf_SlogP_VSA12_Featurizer()
molecule_MolSurf_SlogP_VSA1_featurizer = Molecule_MolSurf_SlogP_VSA1_Featurizer()
molecule_MolSurf_SlogP_VSA2_featurizer = Molecule_MolSurf_SlogP_VSA2_Featurizer()
molecule_MolSurf_SlogP_VSA3_featurizer = Molecule_MolSurf_SlogP_VSA3_Featurizer()
molecule_MolSurf_SlogP_VSA4_featurizer = Molecule_MolSurf_SlogP_VSA4_Featurizer()
molecule_MolSurf_SlogP_VSA5_featurizer = Molecule_MolSurf_SlogP_VSA5_Featurizer()
molecule_MolSurf_SlogP_VSA6_featurizer = Molecule_MolSurf_SlogP_VSA6_Featurizer()
molecule_MolSurf_SlogP_VSA7_featurizer = Molecule_MolSurf_SlogP_VSA7_Featurizer()
molecule_MolSurf_SlogP_VSA8_featurizer = Molecule_MolSurf_SlogP_VSA8_Featurizer()
molecule_MolSurf_SlogP_VSA9_featurizer = Molecule_MolSurf_SlogP_VSA9_Featurizer()
molecule_MolSurf_TPSA_featurizer = Molecule_MolSurf_TPSA_Featurizer()
molecule_MolSurf_pyLabuteASA_featurizer = Molecule_MolSurf_pyLabuteASA_Featurizer()
molecule_NumAtoms_featurizer = Molecule_NumAtoms_Featurizer()
molecule_NumBonds_featurizer = Molecule_NumBonds_Featurizer()
molecule_NumHeavyAtoms_featurizer = Molecule_NumHeavyAtoms_Featurizer()
molecule_QED_default_featurizer = Molecule_QED_default_Featurizer()
molecule_QED_qed_featurizer = Molecule_QED_qed_Featurizer()
molecule_QED_weights_max_featurizer = Molecule_QED_weights_max_Featurizer()
molecule_QED_weights_mean_featurizer = Molecule_QED_weights_mean_Featurizer()
molecule_QED_weights_none_featurizer = Molecule_QED_weights_none_Featurizer()
molecule_rdmolops_FormalCharge_featurizer = Molecule_rdmolops_FormalCharge_Featurizer()
molecule_rdmolops_SSSR_featurizer = Molecule_rdmolops_SSSR_Featurizer()
_available_featurizer = {
    "molecule_AllChem_Asphericity_featurizer": molecule_AllChem_Asphericity_featurizer,
    "molecule_AllChem_Chi0n_featurizer": molecule_AllChem_Chi0n_featurizer,
    "molecule_AllChem_Chi0v_featurizer": molecule_AllChem_Chi0v_featurizer,
    "molecule_AllChem_Chi1n_featurizer": molecule_AllChem_Chi1n_featurizer,
    "molecule_AllChem_Chi1v_featurizer": molecule_AllChem_Chi1v_featurizer,
    "molecule_AllChem_Chi2n_featurizer": molecule_AllChem_Chi2n_featurizer,
    "molecule_AllChem_Chi2v_featurizer": molecule_AllChem_Chi2v_featurizer,
    "molecule_AllChem_Chi3n_featurizer": molecule_AllChem_Chi3n_featurizer,
    "molecule_AllChem_Chi3v_featurizer": molecule_AllChem_Chi3v_featurizer,
    "molecule_AllChem_Chi4n_featurizer": molecule_AllChem_Chi4n_featurizer,
    "molecule_AllChem_Chi4v_featurizer": molecule_AllChem_Chi4v_featurizer,
    "molecule_AllChem_Compute2DCoords_featurizer": molecule_AllChem_Compute2DCoords_featurizer,
    "molecule_AllChem_ComputeMolVolume_featurizer": molecule_AllChem_ComputeMolVolume_featurizer,
    "molecule_AllChem_Eccentricity_featurizer": molecule_AllChem_Eccentricity_featurizer,
    "molecule_AllChem_EmbedMolecule_featurizer": molecule_AllChem_EmbedMolecule_featurizer,
    "molecule_AllChem_ExactMolWt_featurizer": molecule_AllChem_ExactMolWt_featurizer,
    "molecule_AllChem_FractionCSP3_featurizer": molecule_AllChem_FractionCSP3_featurizer,
    "molecule_AllChem_HallKierAlpha_featurizer": molecule_AllChem_HallKierAlpha_featurizer,
    "molecule_AllChem_InertialShapeFactor_featurizer": molecule_AllChem_InertialShapeFactor_featurizer,
    "molecule_AllChem_Kappa1_featurizer": molecule_AllChem_Kappa1_featurizer,
    "molecule_AllChem_Kappa2_featurizer": molecule_AllChem_Kappa2_featurizer,
    "molecule_AllChem_Kappa3_featurizer": molecule_AllChem_Kappa3_featurizer,
    "molecule_AllChem_LabuteASA_featurizer": molecule_AllChem_LabuteASA_featurizer,
    "molecule_AllChem_NPR1_featurizer": molecule_AllChem_NPR1_featurizer,
    "molecule_AllChem_NPR2_featurizer": molecule_AllChem_NPR2_featurizer,
    "molecule_AllChem_NumAliphaticCarbocycles_featurizer": molecule_AllChem_NumAliphaticCarbocycles_featurizer,
    "molecule_AllChem_NumAliphaticHeterocycles_featurizer": molecule_AllChem_NumAliphaticHeterocycles_featurizer,
    "molecule_AllChem_NumAliphaticRings_featurizer": molecule_AllChem_NumAliphaticRings_featurizer,
    "molecule_AllChem_NumAmideBonds_featurizer": molecule_AllChem_NumAmideBonds_featurizer,
    "molecule_AllChem_NumAromaticCarbocycles_featurizer": molecule_AllChem_NumAromaticCarbocycles_featurizer,
    "molecule_AllChem_NumAromaticHeterocycles_featurizer": molecule_AllChem_NumAromaticHeterocycles_featurizer,
    "molecule_AllChem_NumAromaticRings_featurizer": molecule_AllChem_NumAromaticRings_featurizer,
    "molecule_AllChem_NumAtoms_featurizer": molecule_AllChem_NumAtoms_featurizer,
    "molecule_AllChem_NumBridgeheadAtoms_featurizer": molecule_AllChem_NumBridgeheadAtoms_featurizer,
    "molecule_AllChem_NumHBA_featurizer": molecule_AllChem_NumHBA_featurizer,
    "molecule_AllChem_NumHBD_featurizer": molecule_AllChem_NumHBD_featurizer,
    "molecule_AllChem_NumHeavyAtoms_featurizer": molecule_AllChem_NumHeavyAtoms_featurizer,
    "molecule_AllChem_NumHeteroatoms_featurizer": molecule_AllChem_NumHeteroatoms_featurizer,
    "molecule_AllChem_NumHeterocycles_featurizer": molecule_AllChem_NumHeterocycles_featurizer,
    "molecule_AllChem_NumLipinskiHBA_featurizer": molecule_AllChem_NumLipinskiHBA_featurizer,
    "molecule_AllChem_NumLipinskiHBD_featurizer": molecule_AllChem_NumLipinskiHBD_featurizer,
    "molecule_AllChem_NumRings_featurizer": molecule_AllChem_NumRings_featurizer,
    "molecule_AllChem_NumRotatableBonds_featurizer": molecule_AllChem_NumRotatableBonds_featurizer,
    "molecule_AllChem_NumSaturatedCarbocycles_featurizer": molecule_AllChem_NumSaturatedCarbocycles_featurizer,
    "molecule_AllChem_NumSaturatedHeterocycles_featurizer": molecule_AllChem_NumSaturatedHeterocycles_featurizer,
    "molecule_AllChem_NumSaturatedRings_featurizer": molecule_AllChem_NumSaturatedRings_featurizer,
    "molecule_AllChem_NumSpiroAtoms_featurizer": molecule_AllChem_NumSpiroAtoms_featurizer,
    "molecule_AllChem_PBF_featurizer": molecule_AllChem_PBF_featurizer,
    "molecule_AllChem_PMI1_featurizer": molecule_AllChem_PMI1_featurizer,
    "molecule_AllChem_PMI2_featurizer": molecule_AllChem_PMI2_featurizer,
    "molecule_AllChem_PMI3_featurizer": molecule_AllChem_PMI3_featurizer,
    "molecule_AllChem_Phi_featurizer": molecule_AllChem_Phi_featurizer,
    "molecule_AllChem_RadiusOfGyration_featurizer": molecule_AllChem_RadiusOfGyration_featurizer,
    "molecule_AllChem_SpherocityIndex_featurizer": molecule_AllChem_SpherocityIndex_featurizer,
    "molecule_AllChem_TPSA_featurizer": molecule_AllChem_TPSA_featurizer,
    "molecule_Chem_FormalCharge_featurizer": molecule_Chem_FormalCharge_featurizer,
    "molecule_Chem_SSSR_featurizer": molecule_Chem_SSSR_featurizer,
    "molecule_Crippen_MolLogP_featurizer": molecule_Crippen_MolLogP_featurizer,
    "molecule_Crippen_MolMR_featurizer": molecule_Crippen_MolMR_featurizer,
    "molecule_Descriptors3D_Asphericity_featurizer": molecule_Descriptors3D_Asphericity_featurizer,
    "molecule_Descriptors3D_Eccentricity_featurizer": molecule_Descriptors3D_Eccentricity_featurizer,
    "molecule_Descriptors3D_InertialShapeFactor_featurizer": molecule_Descriptors3D_InertialShapeFactor_featurizer,
    "molecule_Descriptors3D_NPR1_featurizer": molecule_Descriptors3D_NPR1_featurizer,
    "molecule_Descriptors3D_NPR2_featurizer": molecule_Descriptors3D_NPR2_featurizer,
    "molecule_Descriptors3D_PMI1_featurizer": molecule_Descriptors3D_PMI1_featurizer,
    "molecule_Descriptors3D_PMI2_featurizer": molecule_Descriptors3D_PMI2_featurizer,
    "molecule_Descriptors3D_PMI3_featurizer": molecule_Descriptors3D_PMI3_featurizer,
    "molecule_Descriptors3D_RadiusOfGyration_featurizer": molecule_Descriptors3D_RadiusOfGyration_featurizer,
    "molecule_Descriptors3D_SpherocityIndex_featurizer": molecule_Descriptors3D_SpherocityIndex_featurizer,
    "molecule_Descriptors_BCUT2D_CHGHI_featurizer": molecule_Descriptors_BCUT2D_CHGHI_featurizer,
    "molecule_Descriptors_BCUT2D_CHGLO_featurizer": molecule_Descriptors_BCUT2D_CHGLO_featurizer,
    "molecule_Descriptors_BCUT2D_LOGPHI_featurizer": molecule_Descriptors_BCUT2D_LOGPHI_featurizer,
    "molecule_Descriptors_BCUT2D_LOGPLOW_featurizer": molecule_Descriptors_BCUT2D_LOGPLOW_featurizer,
    "molecule_Descriptors_BCUT2D_MRHI_featurizer": molecule_Descriptors_BCUT2D_MRHI_featurizer,
    "molecule_Descriptors_BCUT2D_MRLOW_featurizer": molecule_Descriptors_BCUT2D_MRLOW_featurizer,
    "molecule_Descriptors_BCUT2D_MWHI_featurizer": molecule_Descriptors_BCUT2D_MWHI_featurizer,
    "molecule_Descriptors_BCUT2D_MWLOW_featurizer": molecule_Descriptors_BCUT2D_MWLOW_featurizer,
    "molecule_Descriptors_BalabanJ_featurizer": molecule_Descriptors_BalabanJ_featurizer,
    "molecule_Descriptors_BertzCT_featurizer": molecule_Descriptors_BertzCT_featurizer,
    "molecule_Descriptors_Chi0_featurizer": molecule_Descriptors_Chi0_featurizer,
    "molecule_Descriptors_Chi0n_featurizer": molecule_Descriptors_Chi0n_featurizer,
    "molecule_Descriptors_Chi0v_featurizer": molecule_Descriptors_Chi0v_featurizer,
    "molecule_Descriptors_Chi1_featurizer": molecule_Descriptors_Chi1_featurizer,
    "molecule_Descriptors_Chi1n_featurizer": molecule_Descriptors_Chi1n_featurizer,
    "molecule_Descriptors_Chi1v_featurizer": molecule_Descriptors_Chi1v_featurizer,
    "molecule_Descriptors_Chi2n_featurizer": molecule_Descriptors_Chi2n_featurizer,
    "molecule_Descriptors_Chi2v_featurizer": molecule_Descriptors_Chi2v_featurizer,
    "molecule_Descriptors_Chi3n_featurizer": molecule_Descriptors_Chi3n_featurizer,
    "molecule_Descriptors_Chi3v_featurizer": molecule_Descriptors_Chi3v_featurizer,
    "molecule_Descriptors_Chi4n_featurizer": molecule_Descriptors_Chi4n_featurizer,
    "molecule_Descriptors_Chi4v_featurizer": molecule_Descriptors_Chi4v_featurizer,
    "molecule_Descriptors_EState_VSA10_featurizer": molecule_Descriptors_EState_VSA10_featurizer,
    "molecule_Descriptors_EState_VSA11_featurizer": molecule_Descriptors_EState_VSA11_featurizer,
    "molecule_Descriptors_EState_VSA1_featurizer": molecule_Descriptors_EState_VSA1_featurizer,
    "molecule_Descriptors_EState_VSA2_featurizer": molecule_Descriptors_EState_VSA2_featurizer,
    "molecule_Descriptors_EState_VSA3_featurizer": molecule_Descriptors_EState_VSA3_featurizer,
    "molecule_Descriptors_EState_VSA4_featurizer": molecule_Descriptors_EState_VSA4_featurizer,
    "molecule_Descriptors_EState_VSA5_featurizer": molecule_Descriptors_EState_VSA5_featurizer,
    "molecule_Descriptors_EState_VSA6_featurizer": molecule_Descriptors_EState_VSA6_featurizer,
    "molecule_Descriptors_EState_VSA7_featurizer": molecule_Descriptors_EState_VSA7_featurizer,
    "molecule_Descriptors_EState_VSA8_featurizer": molecule_Descriptors_EState_VSA8_featurizer,
    "molecule_Descriptors_EState_VSA9_featurizer": molecule_Descriptors_EState_VSA9_featurizer,
    "molecule_Descriptors_ExactMolWt_featurizer": molecule_Descriptors_ExactMolWt_featurizer,
    "molecule_Descriptors_FpDensityMorgan1_featurizer": molecule_Descriptors_FpDensityMorgan1_featurizer,
    "molecule_Descriptors_FpDensityMorgan2_featurizer": molecule_Descriptors_FpDensityMorgan2_featurizer,
    "molecule_Descriptors_FpDensityMorgan3_featurizer": molecule_Descriptors_FpDensityMorgan3_featurizer,
    "molecule_Descriptors_HallKierAlpha_featurizer": molecule_Descriptors_HallKierAlpha_featurizer,
    "molecule_Descriptors_HeavyAtomMolWt_featurizer": molecule_Descriptors_HeavyAtomMolWt_featurizer,
    "molecule_Descriptors_Ipc_featurizer": molecule_Descriptors_Ipc_featurizer,
    "molecule_Descriptors_Kappa1_featurizer": molecule_Descriptors_Kappa1_featurizer,
    "molecule_Descriptors_Kappa2_featurizer": molecule_Descriptors_Kappa2_featurizer,
    "molecule_Descriptors_Kappa3_featurizer": molecule_Descriptors_Kappa3_featurizer,
    "molecule_Descriptors_MaxAbsEStateIndex_featurizer": molecule_Descriptors_MaxAbsEStateIndex_featurizer,
    "molecule_Descriptors_MaxAbsPartialCharge_featurizer": molecule_Descriptors_MaxAbsPartialCharge_featurizer,
    "molecule_Descriptors_MaxPartialCharge_featurizer": molecule_Descriptors_MaxPartialCharge_featurizer,
    "molecule_Descriptors_MinAbsPartialCharge_featurizer": molecule_Descriptors_MinAbsPartialCharge_featurizer,
    "molecule_Descriptors_MinEStateIndex_featurizer": molecule_Descriptors_MinEStateIndex_featurizer,
    "molecule_Descriptors_MinPartialCharge_featurizer": molecule_Descriptors_MinPartialCharge_featurizer,
    "molecule_Descriptors_MolWt_featurizer": molecule_Descriptors_MolWt_featurizer,
    "molecule_Descriptors_NumRadicalElectrons_featurizer": molecule_Descriptors_NumRadicalElectrons_featurizer,
    "molecule_Descriptors_NumValenceElectrons_featurizer": molecule_Descriptors_NumValenceElectrons_featurizer,
    "molecule_Descriptors_VSA_EState10_featurizer": molecule_Descriptors_VSA_EState10_featurizer,
    "molecule_Descriptors_VSA_EState1_featurizer": molecule_Descriptors_VSA_EState1_featurizer,
    "molecule_Descriptors_VSA_EState2_featurizer": molecule_Descriptors_VSA_EState2_featurizer,
    "molecule_Descriptors_VSA_EState3_featurizer": molecule_Descriptors_VSA_EState3_featurizer,
    "molecule_Descriptors_VSA_EState4_featurizer": molecule_Descriptors_VSA_EState4_featurizer,
    "molecule_Descriptors_VSA_EState5_featurizer": molecule_Descriptors_VSA_EState5_featurizer,
    "molecule_Descriptors_VSA_EState6_featurizer": molecule_Descriptors_VSA_EState6_featurizer,
    "molecule_Descriptors_VSA_EState7_featurizer": molecule_Descriptors_VSA_EState7_featurizer,
    "molecule_Descriptors_VSA_EState8_featurizer": molecule_Descriptors_VSA_EState8_featurizer,
    "molecule_Descriptors_VSA_EState9_featurizer": molecule_Descriptors_VSA_EState9_featurizer,
    "molecule_EState_MaxAbsEStateIndex_featurizer": molecule_EState_MaxAbsEStateIndex_featurizer,
    "molecule_EState_MaxEStateIndex_featurizer": molecule_EState_MaxEStateIndex_featurizer,
    "molecule_EState_MinAbsEStateIndex_featurizer": molecule_EState_MinAbsEStateIndex_featurizer,
    "molecule_EState_MinEStateIndex_featurizer": molecule_EState_MinEStateIndex_featurizer,
    "molecule_EnumerateStereoisomers_EmbedMolecule_featurizer": molecule_EnumerateStereoisomers_EmbedMolecule_featurizer,
    "molecule_EnumerateStereoisomers_StereoisomerCount_featurizer": molecule_EnumerateStereoisomers_StereoisomerCount_featurizer,
    "molecule_Fragments_fr_Al_COO_featurizer": molecule_Fragments_fr_Al_COO_featurizer,
    "molecule_Fragments_fr_Al_OH_featurizer": molecule_Fragments_fr_Al_OH_featurizer,
    "molecule_Fragments_fr_Al_OH_noTert_featurizer": molecule_Fragments_fr_Al_OH_noTert_featurizer,
    "molecule_Fragments_fr_ArN_featurizer": molecule_Fragments_fr_ArN_featurizer,
    "molecule_Fragments_fr_Ar_COO_featurizer": molecule_Fragments_fr_Ar_COO_featurizer,
    "molecule_Fragments_fr_Ar_NH_featurizer": molecule_Fragments_fr_Ar_NH_featurizer,
    "molecule_Fragments_fr_Ar_N_featurizer": molecule_Fragments_fr_Ar_N_featurizer,
    "molecule_Fragments_fr_Ar_OH_featurizer": molecule_Fragments_fr_Ar_OH_featurizer,
    "molecule_Fragments_fr_COO2_featurizer": molecule_Fragments_fr_COO2_featurizer,
    "molecule_Fragments_fr_COO_featurizer": molecule_Fragments_fr_COO_featurizer,
    "molecule_Fragments_fr_C_O_featurizer": molecule_Fragments_fr_C_O_featurizer,
    "molecule_Fragments_fr_C_O_noCOO_featurizer": molecule_Fragments_fr_C_O_noCOO_featurizer,
    "molecule_Fragments_fr_C_S_featurizer": molecule_Fragments_fr_C_S_featurizer,
    "molecule_Fragments_fr_HOCCN_featurizer": molecule_Fragments_fr_HOCCN_featurizer,
    "molecule_Fragments_fr_Imine_featurizer": molecule_Fragments_fr_Imine_featurizer,
    "molecule_Fragments_fr_NH0_featurizer": molecule_Fragments_fr_NH0_featurizer,
    "molecule_Fragments_fr_NH1_featurizer": molecule_Fragments_fr_NH1_featurizer,
    "molecule_Fragments_fr_NH2_featurizer": molecule_Fragments_fr_NH2_featurizer,
    "molecule_Fragments_fr_N_O_featurizer": molecule_Fragments_fr_N_O_featurizer,
    "molecule_Fragments_fr_Ndealkylation1_featurizer": molecule_Fragments_fr_Ndealkylation1_featurizer,
    "molecule_Fragments_fr_Ndealkylation2_featurizer": molecule_Fragments_fr_Ndealkylation2_featurizer,
    "molecule_Fragments_fr_Nhpyrrole_featurizer": molecule_Fragments_fr_Nhpyrrole_featurizer,
    "molecule_Fragments_fr_SH_featurizer": molecule_Fragments_fr_SH_featurizer,
    "molecule_Fragments_fr_aldehyde_featurizer": molecule_Fragments_fr_aldehyde_featurizer,
    "molecule_Fragments_fr_alkyl_carbamate_featurizer": molecule_Fragments_fr_alkyl_carbamate_featurizer,
    "molecule_Fragments_fr_alkyl_halide_featurizer": molecule_Fragments_fr_alkyl_halide_featurizer,
    "molecule_Fragments_fr_allylic_oxid_featurizer": molecule_Fragments_fr_allylic_oxid_featurizer,
    "molecule_Fragments_fr_amide_featurizer": molecule_Fragments_fr_amide_featurizer,
    "molecule_Fragments_fr_amidine_featurizer": molecule_Fragments_fr_amidine_featurizer,
    "molecule_Fragments_fr_aniline_featurizer": molecule_Fragments_fr_aniline_featurizer,
    "molecule_Fragments_fr_aryl_methyl_featurizer": molecule_Fragments_fr_aryl_methyl_featurizer,
    "molecule_Fragments_fr_azide_featurizer": molecule_Fragments_fr_azide_featurizer,
    "molecule_Fragments_fr_azo_featurizer": molecule_Fragments_fr_azo_featurizer,
    "molecule_Fragments_fr_barbitur_featurizer": molecule_Fragments_fr_barbitur_featurizer,
    "molecule_Fragments_fr_benzene_featurizer": molecule_Fragments_fr_benzene_featurizer,
    "molecule_Fragments_fr_benzodiazepine_featurizer": molecule_Fragments_fr_benzodiazepine_featurizer,
    "molecule_Fragments_fr_bicyclic_featurizer": molecule_Fragments_fr_bicyclic_featurizer,
    "molecule_Fragments_fr_diazo_featurizer": molecule_Fragments_fr_diazo_featurizer,
    "molecule_Fragments_fr_dihydropyridine_featurizer": molecule_Fragments_fr_dihydropyridine_featurizer,
    "molecule_Fragments_fr_epoxide_featurizer": molecule_Fragments_fr_epoxide_featurizer,
    "molecule_Fragments_fr_ester_featurizer": molecule_Fragments_fr_ester_featurizer,
    "molecule_Fragments_fr_ether_featurizer": molecule_Fragments_fr_ether_featurizer,
    "molecule_Fragments_fr_furan_featurizer": molecule_Fragments_fr_furan_featurizer,
    "molecule_Fragments_fr_guanido_featurizer": molecule_Fragments_fr_guanido_featurizer,
    "molecule_Fragments_fr_halogen_featurizer": molecule_Fragments_fr_halogen_featurizer,
    "molecule_Fragments_fr_hdrzine_featurizer": molecule_Fragments_fr_hdrzine_featurizer,
    "molecule_Fragments_fr_hdrzone_featurizer": molecule_Fragments_fr_hdrzone_featurizer,
    "molecule_Fragments_fr_imidazole_featurizer": molecule_Fragments_fr_imidazole_featurizer,
    "molecule_Fragments_fr_imide_featurizer": molecule_Fragments_fr_imide_featurizer,
    "molecule_Fragments_fr_isocyan_featurizer": molecule_Fragments_fr_isocyan_featurizer,
    "molecule_Fragments_fr_isothiocyan_featurizer": molecule_Fragments_fr_isothiocyan_featurizer,
    "molecule_Fragments_fr_ketone_featurizer": molecule_Fragments_fr_ketone_featurizer,
    "molecule_Fragments_fr_ketone_Topliss_featurizer": molecule_Fragments_fr_ketone_Topliss_featurizer,
    "molecule_Fragments_fr_lactam_featurizer": molecule_Fragments_fr_lactam_featurizer,
    "molecule_Fragments_fr_lactone_featurizer": molecule_Fragments_fr_lactone_featurizer,
    "molecule_Fragments_fr_methoxy_featurizer": molecule_Fragments_fr_methoxy_featurizer,
    "molecule_Fragments_fr_morpholine_featurizer": molecule_Fragments_fr_morpholine_featurizer,
    "molecule_Fragments_fr_nitrile_featurizer": molecule_Fragments_fr_nitrile_featurizer,
    "molecule_Fragments_fr_nitro_featurizer": molecule_Fragments_fr_nitro_featurizer,
    "molecule_Fragments_fr_nitro_arom_featurizer": molecule_Fragments_fr_nitro_arom_featurizer,
    "molecule_Fragments_fr_nitro_arom_nonortho_featurizer": molecule_Fragments_fr_nitro_arom_nonortho_featurizer,
    "molecule_Fragments_fr_nitroso_featurizer": molecule_Fragments_fr_nitroso_featurizer,
    "molecule_Fragments_fr_oxazole_featurizer": molecule_Fragments_fr_oxazole_featurizer,
    "molecule_Fragments_fr_oxime_featurizer": molecule_Fragments_fr_oxime_featurizer,
    "molecule_Fragments_fr_para_hydroxylation_featurizer": molecule_Fragments_fr_para_hydroxylation_featurizer,
    "molecule_Fragments_fr_phenol_featurizer": molecule_Fragments_fr_phenol_featurizer,
    "molecule_Fragments_fr_phenol_noOrthoHbond_featurizer": molecule_Fragments_fr_phenol_noOrthoHbond_featurizer,
    "molecule_Fragments_fr_phos_acid_featurizer": molecule_Fragments_fr_phos_acid_featurizer,
    "molecule_Fragments_fr_phos_ester_featurizer": molecule_Fragments_fr_phos_ester_featurizer,
    "molecule_Fragments_fr_piperdine_featurizer": molecule_Fragments_fr_piperdine_featurizer,
    "molecule_Fragments_fr_piperzine_featurizer": molecule_Fragments_fr_piperzine_featurizer,
    "molecule_Fragments_fr_priamide_featurizer": molecule_Fragments_fr_priamide_featurizer,
    "molecule_Fragments_fr_prisulfonamd_featurizer": molecule_Fragments_fr_prisulfonamd_featurizer,
    "molecule_Fragments_fr_pyridine_featurizer": molecule_Fragments_fr_pyridine_featurizer,
    "molecule_Fragments_fr_quatN_featurizer": molecule_Fragments_fr_quatN_featurizer,
    "molecule_Fragments_fr_sulfide_featurizer": molecule_Fragments_fr_sulfide_featurizer,
    "molecule_Fragments_fr_sulfonamd_featurizer": molecule_Fragments_fr_sulfonamd_featurizer,
    "molecule_Fragments_fr_sulfone_featurizer": molecule_Fragments_fr_sulfone_featurizer,
    "molecule_Fragments_fr_term_acetylene_featurizer": molecule_Fragments_fr_term_acetylene_featurizer,
    "molecule_Fragments_fr_tetrazole_featurizer": molecule_Fragments_fr_tetrazole_featurizer,
    "molecule_Fragments_fr_thiazole_featurizer": molecule_Fragments_fr_thiazole_featurizer,
    "molecule_Fragments_fr_thiocyan_featurizer": molecule_Fragments_fr_thiocyan_featurizer,
    "molecule_Fragments_fr_thiophene_featurizer": molecule_Fragments_fr_thiophene_featurizer,
    "molecule_Fragments_fr_unbrch_alkane_featurizer": molecule_Fragments_fr_unbrch_alkane_featurizer,
    "molecule_Fragments_fr_urea_featurizer": molecule_Fragments_fr_urea_featurizer,
    "molecule_Lipinski_FractionCSP3_featurizer": molecule_Lipinski_FractionCSP3_featurizer,
    "molecule_Lipinski_HeavyAtomCount_featurizer": molecule_Lipinski_HeavyAtomCount_featurizer,
    "molecule_Lipinski_NHOHCount_featurizer": molecule_Lipinski_NHOHCount_featurizer,
    "molecule_Lipinski_NOCount_featurizer": molecule_Lipinski_NOCount_featurizer,
    "molecule_Lipinski_NumAliphaticCarbocycles_featurizer": molecule_Lipinski_NumAliphaticCarbocycles_featurizer,
    "molecule_Lipinski_NumAliphaticHeterocycles_featurizer": molecule_Lipinski_NumAliphaticHeterocycles_featurizer,
    "molecule_Lipinski_NumAliphaticRings_featurizer": molecule_Lipinski_NumAliphaticRings_featurizer,
    "molecule_Lipinski_NumAromaticCarbocycles_featurizer": molecule_Lipinski_NumAromaticCarbocycles_featurizer,
    "molecule_Lipinski_NumAromaticHeterocycles_featurizer": molecule_Lipinski_NumAromaticHeterocycles_featurizer,
    "molecule_Lipinski_NumAromaticRings_featurizer": molecule_Lipinski_NumAromaticRings_featurizer,
    "molecule_Lipinski_NumHAcceptors_featurizer": molecule_Lipinski_NumHAcceptors_featurizer,
    "molecule_Lipinski_NumHDonors_featurizer": molecule_Lipinski_NumHDonors_featurizer,
    "molecule_Lipinski_NumHeteroatoms_featurizer": molecule_Lipinski_NumHeteroatoms_featurizer,
    "molecule_Lipinski_NumRotatableBonds_featurizer": molecule_Lipinski_NumRotatableBonds_featurizer,
    "molecule_Lipinski_NumSaturatedCarbocycles_featurizer": molecule_Lipinski_NumSaturatedCarbocycles_featurizer,
    "molecule_Lipinski_NumSaturatedHeterocycles_featurizer": molecule_Lipinski_NumSaturatedHeterocycles_featurizer,
    "molecule_Lipinski_NumSaturatedRings_featurizer": molecule_Lipinski_NumSaturatedRings_featurizer,
    "molecule_Lipinski_RingCount_featurizer": molecule_Lipinski_RingCount_featurizer,
    "molecule_MolSurf_LabuteASA_featurizer": molecule_MolSurf_LabuteASA_featurizer,
    "molecule_MolSurf_PEOE_VSA10_featurizer": molecule_MolSurf_PEOE_VSA10_featurizer,
    "molecule_MolSurf_PEOE_VSA11_featurizer": molecule_MolSurf_PEOE_VSA11_featurizer,
    "molecule_MolSurf_PEOE_VSA12_featurizer": molecule_MolSurf_PEOE_VSA12_featurizer,
    "molecule_MolSurf_PEOE_VSA13_featurizer": molecule_MolSurf_PEOE_VSA13_featurizer,
    "molecule_MolSurf_PEOE_VSA14_featurizer": molecule_MolSurf_PEOE_VSA14_featurizer,
    "molecule_MolSurf_PEOE_VSA1_featurizer": molecule_MolSurf_PEOE_VSA1_featurizer,
    "molecule_MolSurf_PEOE_VSA2_featurizer": molecule_MolSurf_PEOE_VSA2_featurizer,
    "molecule_MolSurf_PEOE_VSA3_featurizer": molecule_MolSurf_PEOE_VSA3_featurizer,
    "molecule_MolSurf_PEOE_VSA4_featurizer": molecule_MolSurf_PEOE_VSA4_featurizer,
    "molecule_MolSurf_PEOE_VSA5_featurizer": molecule_MolSurf_PEOE_VSA5_featurizer,
    "molecule_MolSurf_PEOE_VSA6_featurizer": molecule_MolSurf_PEOE_VSA6_featurizer,
    "molecule_MolSurf_PEOE_VSA7_featurizer": molecule_MolSurf_PEOE_VSA7_featurizer,
    "molecule_MolSurf_PEOE_VSA8_featurizer": molecule_MolSurf_PEOE_VSA8_featurizer,
    "molecule_MolSurf_PEOE_VSA9_featurizer": molecule_MolSurf_PEOE_VSA9_featurizer,
    "molecule_MolSurf_SMR_VSA10_featurizer": molecule_MolSurf_SMR_VSA10_featurizer,
    "molecule_MolSurf_SMR_VSA1_featurizer": molecule_MolSurf_SMR_VSA1_featurizer,
    "molecule_MolSurf_SMR_VSA2_featurizer": molecule_MolSurf_SMR_VSA2_featurizer,
    "molecule_MolSurf_SMR_VSA3_featurizer": molecule_MolSurf_SMR_VSA3_featurizer,
    "molecule_MolSurf_SMR_VSA4_featurizer": molecule_MolSurf_SMR_VSA4_featurizer,
    "molecule_MolSurf_SMR_VSA5_featurizer": molecule_MolSurf_SMR_VSA5_featurizer,
    "molecule_MolSurf_SMR_VSA6_featurizer": molecule_MolSurf_SMR_VSA6_featurizer,
    "molecule_MolSurf_SMR_VSA7_featurizer": molecule_MolSurf_SMR_VSA7_featurizer,
    "molecule_MolSurf_SMR_VSA8_featurizer": molecule_MolSurf_SMR_VSA8_featurizer,
    "molecule_MolSurf_SMR_VSA9_featurizer": molecule_MolSurf_SMR_VSA9_featurizer,
    "molecule_MolSurf_SlogP_VSA10_featurizer": molecule_MolSurf_SlogP_VSA10_featurizer,
    "molecule_MolSurf_SlogP_VSA11_featurizer": molecule_MolSurf_SlogP_VSA11_featurizer,
    "molecule_MolSurf_SlogP_VSA12_featurizer": molecule_MolSurf_SlogP_VSA12_featurizer,
    "molecule_MolSurf_SlogP_VSA1_featurizer": molecule_MolSurf_SlogP_VSA1_featurizer,
    "molecule_MolSurf_SlogP_VSA2_featurizer": molecule_MolSurf_SlogP_VSA2_featurizer,
    "molecule_MolSurf_SlogP_VSA3_featurizer": molecule_MolSurf_SlogP_VSA3_featurizer,
    "molecule_MolSurf_SlogP_VSA4_featurizer": molecule_MolSurf_SlogP_VSA4_featurizer,
    "molecule_MolSurf_SlogP_VSA5_featurizer": molecule_MolSurf_SlogP_VSA5_featurizer,
    "molecule_MolSurf_SlogP_VSA6_featurizer": molecule_MolSurf_SlogP_VSA6_featurizer,
    "molecule_MolSurf_SlogP_VSA7_featurizer": molecule_MolSurf_SlogP_VSA7_featurizer,
    "molecule_MolSurf_SlogP_VSA8_featurizer": molecule_MolSurf_SlogP_VSA8_featurizer,
    "molecule_MolSurf_SlogP_VSA9_featurizer": molecule_MolSurf_SlogP_VSA9_featurizer,
    "molecule_MolSurf_TPSA_featurizer": molecule_MolSurf_TPSA_featurizer,
    "molecule_MolSurf_pyLabuteASA_featurizer": molecule_MolSurf_pyLabuteASA_featurizer,
    "molecule_NumAtoms_featurizer": molecule_NumAtoms_featurizer,
    "molecule_NumBonds_featurizer": molecule_NumBonds_featurizer,
    "molecule_NumHeavyAtoms_featurizer": molecule_NumHeavyAtoms_featurizer,
    "molecule_QED_default_featurizer": molecule_QED_default_featurizer,
    "molecule_QED_qed_featurizer": molecule_QED_qed_featurizer,
    "molecule_QED_weights_max_featurizer": molecule_QED_weights_max_featurizer,
    "molecule_QED_weights_mean_featurizer": molecule_QED_weights_mean_featurizer,
    "molecule_QED_weights_none_featurizer": molecule_QED_weights_none_featurizer,
    "molecule_rdmolops_FormalCharge_featurizer": molecule_rdmolops_FormalCharge_featurizer,
    "molecule_rdmolops_SSSR_featurizer": molecule_rdmolops_SSSR_featurizer,
}
__all__ = [
    "Molecule_AllChem_Asphericity_Featurizer",
    "molecule_AllChem_Asphericity_featurizer",
    "Molecule_AllChem_Chi0n_Featurizer",
    "molecule_AllChem_Chi0n_featurizer",
    "Molecule_AllChem_Chi0v_Featurizer",
    "molecule_AllChem_Chi0v_featurizer",
    "Molecule_AllChem_Chi1n_Featurizer",
    "molecule_AllChem_Chi1n_featurizer",
    "Molecule_AllChem_Chi1v_Featurizer",
    "molecule_AllChem_Chi1v_featurizer",
    "Molecule_AllChem_Chi2n_Featurizer",
    "molecule_AllChem_Chi2n_featurizer",
    "Molecule_AllChem_Chi2v_Featurizer",
    "molecule_AllChem_Chi2v_featurizer",
    "Molecule_AllChem_Chi3n_Featurizer",
    "molecule_AllChem_Chi3n_featurizer",
    "Molecule_AllChem_Chi3v_Featurizer",
    "molecule_AllChem_Chi3v_featurizer",
    "Molecule_AllChem_Chi4n_Featurizer",
    "molecule_AllChem_Chi4n_featurizer",
    "Molecule_AllChem_Chi4v_Featurizer",
    "molecule_AllChem_Chi4v_featurizer",
    "Molecule_AllChem_Compute2DCoords_Featurizer",
    "molecule_AllChem_Compute2DCoords_featurizer",
    "Molecule_AllChem_ComputeMolVolume_Featurizer",
    "molecule_AllChem_ComputeMolVolume_featurizer",
    "Molecule_AllChem_Eccentricity_Featurizer",
    "molecule_AllChem_Eccentricity_featurizer",
    "Molecule_AllChem_EmbedMolecule_Featurizer",
    "molecule_AllChem_EmbedMolecule_featurizer",
    "Molecule_AllChem_ExactMolWt_Featurizer",
    "molecule_AllChem_ExactMolWt_featurizer",
    "Molecule_AllChem_FractionCSP3_Featurizer",
    "molecule_AllChem_FractionCSP3_featurizer",
    "Molecule_AllChem_HallKierAlpha_Featurizer",
    "molecule_AllChem_HallKierAlpha_featurizer",
    "Molecule_AllChem_InertialShapeFactor_Featurizer",
    "molecule_AllChem_InertialShapeFactor_featurizer",
    "Molecule_AllChem_Kappa1_Featurizer",
    "molecule_AllChem_Kappa1_featurizer",
    "Molecule_AllChem_Kappa2_Featurizer",
    "molecule_AllChem_Kappa2_featurizer",
    "Molecule_AllChem_Kappa3_Featurizer",
    "molecule_AllChem_Kappa3_featurizer",
    "Molecule_AllChem_LabuteASA_Featurizer",
    "molecule_AllChem_LabuteASA_featurizer",
    "Molecule_AllChem_NPR1_Featurizer",
    "molecule_AllChem_NPR1_featurizer",
    "Molecule_AllChem_NPR2_Featurizer",
    "molecule_AllChem_NPR2_featurizer",
    "Molecule_AllChem_NumAliphaticCarbocycles_Featurizer",
    "molecule_AllChem_NumAliphaticCarbocycles_featurizer",
    "Molecule_AllChem_NumAliphaticHeterocycles_Featurizer",
    "molecule_AllChem_NumAliphaticHeterocycles_featurizer",
    "Molecule_AllChem_NumAliphaticRings_Featurizer",
    "molecule_AllChem_NumAliphaticRings_featurizer",
    "Molecule_AllChem_NumAmideBonds_Featurizer",
    "molecule_AllChem_NumAmideBonds_featurizer",
    "Molecule_AllChem_NumAromaticCarbocycles_Featurizer",
    "molecule_AllChem_NumAromaticCarbocycles_featurizer",
    "Molecule_AllChem_NumAromaticHeterocycles_Featurizer",
    "molecule_AllChem_NumAromaticHeterocycles_featurizer",
    "Molecule_AllChem_NumAromaticRings_Featurizer",
    "molecule_AllChem_NumAromaticRings_featurizer",
    "Molecule_AllChem_NumAtoms_Featurizer",
    "molecule_AllChem_NumAtoms_featurizer",
    "Molecule_AllChem_NumBridgeheadAtoms_Featurizer",
    "molecule_AllChem_NumBridgeheadAtoms_featurizer",
    "Molecule_AllChem_NumHBA_Featurizer",
    "molecule_AllChem_NumHBA_featurizer",
    "Molecule_AllChem_NumHBD_Featurizer",
    "molecule_AllChem_NumHBD_featurizer",
    "Molecule_AllChem_NumHeavyAtoms_Featurizer",
    "molecule_AllChem_NumHeavyAtoms_featurizer",
    "Molecule_AllChem_NumHeteroatoms_Featurizer",
    "molecule_AllChem_NumHeteroatoms_featurizer",
    "Molecule_AllChem_NumHeterocycles_Featurizer",
    "molecule_AllChem_NumHeterocycles_featurizer",
    "Molecule_AllChem_NumLipinskiHBA_Featurizer",
    "molecule_AllChem_NumLipinskiHBA_featurizer",
    "Molecule_AllChem_NumLipinskiHBD_Featurizer",
    "molecule_AllChem_NumLipinskiHBD_featurizer",
    "Molecule_AllChem_NumRings_Featurizer",
    "molecule_AllChem_NumRings_featurizer",
    "Molecule_AllChem_NumRotatableBonds_Featurizer",
    "molecule_AllChem_NumRotatableBonds_featurizer",
    "Molecule_AllChem_NumSaturatedCarbocycles_Featurizer",
    "molecule_AllChem_NumSaturatedCarbocycles_featurizer",
    "Molecule_AllChem_NumSaturatedHeterocycles_Featurizer",
    "molecule_AllChem_NumSaturatedHeterocycles_featurizer",
    "Molecule_AllChem_NumSaturatedRings_Featurizer",
    "molecule_AllChem_NumSaturatedRings_featurizer",
    "Molecule_AllChem_NumSpiroAtoms_Featurizer",
    "molecule_AllChem_NumSpiroAtoms_featurizer",
    "Molecule_AllChem_PBF_Featurizer",
    "molecule_AllChem_PBF_featurizer",
    "Molecule_AllChem_PMI1_Featurizer",
    "molecule_AllChem_PMI1_featurizer",
    "Molecule_AllChem_PMI2_Featurizer",
    "molecule_AllChem_PMI2_featurizer",
    "Molecule_AllChem_PMI3_Featurizer",
    "molecule_AllChem_PMI3_featurizer",
    "Molecule_AllChem_Phi_Featurizer",
    "molecule_AllChem_Phi_featurizer",
    "Molecule_AllChem_RadiusOfGyration_Featurizer",
    "molecule_AllChem_RadiusOfGyration_featurizer",
    "Molecule_AllChem_SpherocityIndex_Featurizer",
    "molecule_AllChem_SpherocityIndex_featurizer",
    "Molecule_AllChem_TPSA_Featurizer",
    "molecule_AllChem_TPSA_featurizer",
    "Molecule_Chem_FormalCharge_Featurizer",
    "molecule_Chem_FormalCharge_featurizer",
    "Molecule_Chem_SSSR_Featurizer",
    "molecule_Chem_SSSR_featurizer",
    "Molecule_Crippen_MolLogP_Featurizer",
    "molecule_Crippen_MolLogP_featurizer",
    "Molecule_Crippen_MolMR_Featurizer",
    "molecule_Crippen_MolMR_featurizer",
    "Molecule_Descriptors3D_Asphericity_Featurizer",
    "molecule_Descriptors3D_Asphericity_featurizer",
    "Molecule_Descriptors3D_Eccentricity_Featurizer",
    "molecule_Descriptors3D_Eccentricity_featurizer",
    "Molecule_Descriptors3D_InertialShapeFactor_Featurizer",
    "molecule_Descriptors3D_InertialShapeFactor_featurizer",
    "Molecule_Descriptors3D_NPR1_Featurizer",
    "molecule_Descriptors3D_NPR1_featurizer",
    "Molecule_Descriptors3D_NPR2_Featurizer",
    "molecule_Descriptors3D_NPR2_featurizer",
    "Molecule_Descriptors3D_PMI1_Featurizer",
    "molecule_Descriptors3D_PMI1_featurizer",
    "Molecule_Descriptors3D_PMI2_Featurizer",
    "molecule_Descriptors3D_PMI2_featurizer",
    "Molecule_Descriptors3D_PMI3_Featurizer",
    "molecule_Descriptors3D_PMI3_featurizer",
    "Molecule_Descriptors3D_RadiusOfGyration_Featurizer",
    "molecule_Descriptors3D_RadiusOfGyration_featurizer",
    "Molecule_Descriptors3D_SpherocityIndex_Featurizer",
    "molecule_Descriptors3D_SpherocityIndex_featurizer",
    "Molecule_Descriptors_BCUT2D_CHGHI_Featurizer",
    "molecule_Descriptors_BCUT2D_CHGHI_featurizer",
    "Molecule_Descriptors_BCUT2D_CHGLO_Featurizer",
    "molecule_Descriptors_BCUT2D_CHGLO_featurizer",
    "Molecule_Descriptors_BCUT2D_LOGPHI_Featurizer",
    "molecule_Descriptors_BCUT2D_LOGPHI_featurizer",
    "Molecule_Descriptors_BCUT2D_LOGPLOW_Featurizer",
    "molecule_Descriptors_BCUT2D_LOGPLOW_featurizer",
    "Molecule_Descriptors_BCUT2D_MRHI_Featurizer",
    "molecule_Descriptors_BCUT2D_MRHI_featurizer",
    "Molecule_Descriptors_BCUT2D_MRLOW_Featurizer",
    "molecule_Descriptors_BCUT2D_MRLOW_featurizer",
    "Molecule_Descriptors_BCUT2D_MWHI_Featurizer",
    "molecule_Descriptors_BCUT2D_MWHI_featurizer",
    "Molecule_Descriptors_BCUT2D_MWLOW_Featurizer",
    "molecule_Descriptors_BCUT2D_MWLOW_featurizer",
    "Molecule_Descriptors_BalabanJ_Featurizer",
    "molecule_Descriptors_BalabanJ_featurizer",
    "Molecule_Descriptors_BertzCT_Featurizer",
    "molecule_Descriptors_BertzCT_featurizer",
    "Molecule_Descriptors_Chi0_Featurizer",
    "molecule_Descriptors_Chi0_featurizer",
    "Molecule_Descriptors_Chi0n_Featurizer",
    "molecule_Descriptors_Chi0n_featurizer",
    "Molecule_Descriptors_Chi0v_Featurizer",
    "molecule_Descriptors_Chi0v_featurizer",
    "Molecule_Descriptors_Chi1_Featurizer",
    "molecule_Descriptors_Chi1_featurizer",
    "Molecule_Descriptors_Chi1n_Featurizer",
    "molecule_Descriptors_Chi1n_featurizer",
    "Molecule_Descriptors_Chi1v_Featurizer",
    "molecule_Descriptors_Chi1v_featurizer",
    "Molecule_Descriptors_Chi2n_Featurizer",
    "molecule_Descriptors_Chi2n_featurizer",
    "Molecule_Descriptors_Chi2v_Featurizer",
    "molecule_Descriptors_Chi2v_featurizer",
    "Molecule_Descriptors_Chi3n_Featurizer",
    "molecule_Descriptors_Chi3n_featurizer",
    "Molecule_Descriptors_Chi3v_Featurizer",
    "molecule_Descriptors_Chi3v_featurizer",
    "Molecule_Descriptors_Chi4n_Featurizer",
    "molecule_Descriptors_Chi4n_featurizer",
    "Molecule_Descriptors_Chi4v_Featurizer",
    "molecule_Descriptors_Chi4v_featurizer",
    "Molecule_Descriptors_EState_VSA10_Featurizer",
    "molecule_Descriptors_EState_VSA10_featurizer",
    "Molecule_Descriptors_EState_VSA11_Featurizer",
    "molecule_Descriptors_EState_VSA11_featurizer",
    "Molecule_Descriptors_EState_VSA1_Featurizer",
    "molecule_Descriptors_EState_VSA1_featurizer",
    "Molecule_Descriptors_EState_VSA2_Featurizer",
    "molecule_Descriptors_EState_VSA2_featurizer",
    "Molecule_Descriptors_EState_VSA3_Featurizer",
    "molecule_Descriptors_EState_VSA3_featurizer",
    "Molecule_Descriptors_EState_VSA4_Featurizer",
    "molecule_Descriptors_EState_VSA4_featurizer",
    "Molecule_Descriptors_EState_VSA5_Featurizer",
    "molecule_Descriptors_EState_VSA5_featurizer",
    "Molecule_Descriptors_EState_VSA6_Featurizer",
    "molecule_Descriptors_EState_VSA6_featurizer",
    "Molecule_Descriptors_EState_VSA7_Featurizer",
    "molecule_Descriptors_EState_VSA7_featurizer",
    "Molecule_Descriptors_EState_VSA8_Featurizer",
    "molecule_Descriptors_EState_VSA8_featurizer",
    "Molecule_Descriptors_EState_VSA9_Featurizer",
    "molecule_Descriptors_EState_VSA9_featurizer",
    "Molecule_Descriptors_ExactMolWt_Featurizer",
    "molecule_Descriptors_ExactMolWt_featurizer",
    "Molecule_Descriptors_FpDensityMorgan1_Featurizer",
    "molecule_Descriptors_FpDensityMorgan1_featurizer",
    "Molecule_Descriptors_FpDensityMorgan2_Featurizer",
    "molecule_Descriptors_FpDensityMorgan2_featurizer",
    "Molecule_Descriptors_FpDensityMorgan3_Featurizer",
    "molecule_Descriptors_FpDensityMorgan3_featurizer",
    "Molecule_Descriptors_HallKierAlpha_Featurizer",
    "molecule_Descriptors_HallKierAlpha_featurizer",
    "Molecule_Descriptors_HeavyAtomMolWt_Featurizer",
    "molecule_Descriptors_HeavyAtomMolWt_featurizer",
    "Molecule_Descriptors_Ipc_Featurizer",
    "molecule_Descriptors_Ipc_featurizer",
    "Molecule_Descriptors_Kappa1_Featurizer",
    "molecule_Descriptors_Kappa1_featurizer",
    "Molecule_Descriptors_Kappa2_Featurizer",
    "molecule_Descriptors_Kappa2_featurizer",
    "Molecule_Descriptors_Kappa3_Featurizer",
    "molecule_Descriptors_Kappa3_featurizer",
    "Molecule_Descriptors_MaxAbsEStateIndex_Featurizer",
    "molecule_Descriptors_MaxAbsEStateIndex_featurizer",
    "Molecule_Descriptors_MaxAbsPartialCharge_Featurizer",
    "molecule_Descriptors_MaxAbsPartialCharge_featurizer",
    "Molecule_Descriptors_MaxPartialCharge_Featurizer",
    "molecule_Descriptors_MaxPartialCharge_featurizer",
    "Molecule_Descriptors_MinAbsPartialCharge_Featurizer",
    "molecule_Descriptors_MinAbsPartialCharge_featurizer",
    "Molecule_Descriptors_MinEStateIndex_Featurizer",
    "molecule_Descriptors_MinEStateIndex_featurizer",
    "Molecule_Descriptors_MinPartialCharge_Featurizer",
    "molecule_Descriptors_MinPartialCharge_featurizer",
    "Molecule_Descriptors_MolWt_Featurizer",
    "molecule_Descriptors_MolWt_featurizer",
    "Molecule_Descriptors_NumRadicalElectrons_Featurizer",
    "molecule_Descriptors_NumRadicalElectrons_featurizer",
    "Molecule_Descriptors_NumValenceElectrons_Featurizer",
    "molecule_Descriptors_NumValenceElectrons_featurizer",
    "Molecule_Descriptors_VSA_EState10_Featurizer",
    "molecule_Descriptors_VSA_EState10_featurizer",
    "Molecule_Descriptors_VSA_EState1_Featurizer",
    "molecule_Descriptors_VSA_EState1_featurizer",
    "Molecule_Descriptors_VSA_EState2_Featurizer",
    "molecule_Descriptors_VSA_EState2_featurizer",
    "Molecule_Descriptors_VSA_EState3_Featurizer",
    "molecule_Descriptors_VSA_EState3_featurizer",
    "Molecule_Descriptors_VSA_EState4_Featurizer",
    "molecule_Descriptors_VSA_EState4_featurizer",
    "Molecule_Descriptors_VSA_EState5_Featurizer",
    "molecule_Descriptors_VSA_EState5_featurizer",
    "Molecule_Descriptors_VSA_EState6_Featurizer",
    "molecule_Descriptors_VSA_EState6_featurizer",
    "Molecule_Descriptors_VSA_EState7_Featurizer",
    "molecule_Descriptors_VSA_EState7_featurizer",
    "Molecule_Descriptors_VSA_EState8_Featurizer",
    "molecule_Descriptors_VSA_EState8_featurizer",
    "Molecule_Descriptors_VSA_EState9_Featurizer",
    "molecule_Descriptors_VSA_EState9_featurizer",
    "Molecule_EState_MaxAbsEStateIndex_Featurizer",
    "molecule_EState_MaxAbsEStateIndex_featurizer",
    "Molecule_EState_MaxEStateIndex_Featurizer",
    "molecule_EState_MaxEStateIndex_featurizer",
    "Molecule_EState_MinAbsEStateIndex_Featurizer",
    "molecule_EState_MinAbsEStateIndex_featurizer",
    "Molecule_EState_MinEStateIndex_Featurizer",
    "molecule_EState_MinEStateIndex_featurizer",
    "Molecule_EnumerateStereoisomers_EmbedMolecule_Featurizer",
    "molecule_EnumerateStereoisomers_EmbedMolecule_featurizer",
    "Molecule_EnumerateStereoisomers_StereoisomerCount_Featurizer",
    "molecule_EnumerateStereoisomers_StereoisomerCount_featurizer",
    "Molecule_Fragments_fr_Al_COO_Featurizer",
    "molecule_Fragments_fr_Al_COO_featurizer",
    "Molecule_Fragments_fr_Al_OH_Featurizer",
    "molecule_Fragments_fr_Al_OH_featurizer",
    "Molecule_Fragments_fr_Al_OH_noTert_Featurizer",
    "molecule_Fragments_fr_Al_OH_noTert_featurizer",
    "Molecule_Fragments_fr_ArN_Featurizer",
    "molecule_Fragments_fr_ArN_featurizer",
    "Molecule_Fragments_fr_Ar_COO_Featurizer",
    "molecule_Fragments_fr_Ar_COO_featurizer",
    "Molecule_Fragments_fr_Ar_NH_Featurizer",
    "molecule_Fragments_fr_Ar_NH_featurizer",
    "Molecule_Fragments_fr_Ar_N_Featurizer",
    "molecule_Fragments_fr_Ar_N_featurizer",
    "Molecule_Fragments_fr_Ar_OH_Featurizer",
    "molecule_Fragments_fr_Ar_OH_featurizer",
    "Molecule_Fragments_fr_COO2_Featurizer",
    "molecule_Fragments_fr_COO2_featurizer",
    "Molecule_Fragments_fr_COO_Featurizer",
    "molecule_Fragments_fr_COO_featurizer",
    "Molecule_Fragments_fr_C_O_Featurizer",
    "molecule_Fragments_fr_C_O_featurizer",
    "Molecule_Fragments_fr_C_O_noCOO_Featurizer",
    "molecule_Fragments_fr_C_O_noCOO_featurizer",
    "Molecule_Fragments_fr_C_S_Featurizer",
    "molecule_Fragments_fr_C_S_featurizer",
    "Molecule_Fragments_fr_HOCCN_Featurizer",
    "molecule_Fragments_fr_HOCCN_featurizer",
    "Molecule_Fragments_fr_Imine_Featurizer",
    "molecule_Fragments_fr_Imine_featurizer",
    "Molecule_Fragments_fr_NH0_Featurizer",
    "molecule_Fragments_fr_NH0_featurizer",
    "Molecule_Fragments_fr_NH1_Featurizer",
    "molecule_Fragments_fr_NH1_featurizer",
    "Molecule_Fragments_fr_NH2_Featurizer",
    "molecule_Fragments_fr_NH2_featurizer",
    "Molecule_Fragments_fr_N_O_Featurizer",
    "molecule_Fragments_fr_N_O_featurizer",
    "Molecule_Fragments_fr_Ndealkylation1_Featurizer",
    "molecule_Fragments_fr_Ndealkylation1_featurizer",
    "Molecule_Fragments_fr_Ndealkylation2_Featurizer",
    "molecule_Fragments_fr_Ndealkylation2_featurizer",
    "Molecule_Fragments_fr_Nhpyrrole_Featurizer",
    "molecule_Fragments_fr_Nhpyrrole_featurizer",
    "Molecule_Fragments_fr_SH_Featurizer",
    "molecule_Fragments_fr_SH_featurizer",
    "Molecule_Fragments_fr_aldehyde_Featurizer",
    "molecule_Fragments_fr_aldehyde_featurizer",
    "Molecule_Fragments_fr_alkyl_carbamate_Featurizer",
    "molecule_Fragments_fr_alkyl_carbamate_featurizer",
    "Molecule_Fragments_fr_alkyl_halide_Featurizer",
    "molecule_Fragments_fr_alkyl_halide_featurizer",
    "Molecule_Fragments_fr_allylic_oxid_Featurizer",
    "molecule_Fragments_fr_allylic_oxid_featurizer",
    "Molecule_Fragments_fr_amide_Featurizer",
    "molecule_Fragments_fr_amide_featurizer",
    "Molecule_Fragments_fr_amidine_Featurizer",
    "molecule_Fragments_fr_amidine_featurizer",
    "Molecule_Fragments_fr_aniline_Featurizer",
    "molecule_Fragments_fr_aniline_featurizer",
    "Molecule_Fragments_fr_aryl_methyl_Featurizer",
    "molecule_Fragments_fr_aryl_methyl_featurizer",
    "Molecule_Fragments_fr_azide_Featurizer",
    "molecule_Fragments_fr_azide_featurizer",
    "Molecule_Fragments_fr_azo_Featurizer",
    "molecule_Fragments_fr_azo_featurizer",
    "Molecule_Fragments_fr_barbitur_Featurizer",
    "molecule_Fragments_fr_barbitur_featurizer",
    "Molecule_Fragments_fr_benzene_Featurizer",
    "molecule_Fragments_fr_benzene_featurizer",
    "Molecule_Fragments_fr_benzodiazepine_Featurizer",
    "molecule_Fragments_fr_benzodiazepine_featurizer",
    "Molecule_Fragments_fr_bicyclic_Featurizer",
    "molecule_Fragments_fr_bicyclic_featurizer",
    "Molecule_Fragments_fr_diazo_Featurizer",
    "molecule_Fragments_fr_diazo_featurizer",
    "Molecule_Fragments_fr_dihydropyridine_Featurizer",
    "molecule_Fragments_fr_dihydropyridine_featurizer",
    "Molecule_Fragments_fr_epoxide_Featurizer",
    "molecule_Fragments_fr_epoxide_featurizer",
    "Molecule_Fragments_fr_ester_Featurizer",
    "molecule_Fragments_fr_ester_featurizer",
    "Molecule_Fragments_fr_ether_Featurizer",
    "molecule_Fragments_fr_ether_featurizer",
    "Molecule_Fragments_fr_furan_Featurizer",
    "molecule_Fragments_fr_furan_featurizer",
    "Molecule_Fragments_fr_guanido_Featurizer",
    "molecule_Fragments_fr_guanido_featurizer",
    "Molecule_Fragments_fr_halogen_Featurizer",
    "molecule_Fragments_fr_halogen_featurizer",
    "Molecule_Fragments_fr_hdrzine_Featurizer",
    "molecule_Fragments_fr_hdrzine_featurizer",
    "Molecule_Fragments_fr_hdrzone_Featurizer",
    "molecule_Fragments_fr_hdrzone_featurizer",
    "Molecule_Fragments_fr_imidazole_Featurizer",
    "molecule_Fragments_fr_imidazole_featurizer",
    "Molecule_Fragments_fr_imide_Featurizer",
    "molecule_Fragments_fr_imide_featurizer",
    "Molecule_Fragments_fr_isocyan_Featurizer",
    "molecule_Fragments_fr_isocyan_featurizer",
    "Molecule_Fragments_fr_isothiocyan_Featurizer",
    "molecule_Fragments_fr_isothiocyan_featurizer",
    "Molecule_Fragments_fr_ketone_Featurizer",
    "molecule_Fragments_fr_ketone_featurizer",
    "Molecule_Fragments_fr_ketone_Topliss_Featurizer",
    "molecule_Fragments_fr_ketone_Topliss_featurizer",
    "Molecule_Fragments_fr_lactam_Featurizer",
    "molecule_Fragments_fr_lactam_featurizer",
    "Molecule_Fragments_fr_lactone_Featurizer",
    "molecule_Fragments_fr_lactone_featurizer",
    "Molecule_Fragments_fr_methoxy_Featurizer",
    "molecule_Fragments_fr_methoxy_featurizer",
    "Molecule_Fragments_fr_morpholine_Featurizer",
    "molecule_Fragments_fr_morpholine_featurizer",
    "Molecule_Fragments_fr_nitrile_Featurizer",
    "molecule_Fragments_fr_nitrile_featurizer",
    "Molecule_Fragments_fr_nitro_Featurizer",
    "molecule_Fragments_fr_nitro_featurizer",
    "Molecule_Fragments_fr_nitro_arom_Featurizer",
    "molecule_Fragments_fr_nitro_arom_featurizer",
    "Molecule_Fragments_fr_nitro_arom_nonortho_Featurizer",
    "molecule_Fragments_fr_nitro_arom_nonortho_featurizer",
    "Molecule_Fragments_fr_nitroso_Featurizer",
    "molecule_Fragments_fr_nitroso_featurizer",
    "Molecule_Fragments_fr_oxazole_Featurizer",
    "molecule_Fragments_fr_oxazole_featurizer",
    "Molecule_Fragments_fr_oxime_Featurizer",
    "molecule_Fragments_fr_oxime_featurizer",
    "Molecule_Fragments_fr_para_hydroxylation_Featurizer",
    "molecule_Fragments_fr_para_hydroxylation_featurizer",
    "Molecule_Fragments_fr_phenol_Featurizer",
    "molecule_Fragments_fr_phenol_featurizer",
    "Molecule_Fragments_fr_phenol_noOrthoHbond_Featurizer",
    "molecule_Fragments_fr_phenol_noOrthoHbond_featurizer",
    "Molecule_Fragments_fr_phos_acid_Featurizer",
    "molecule_Fragments_fr_phos_acid_featurizer",
    "Molecule_Fragments_fr_phos_ester_Featurizer",
    "molecule_Fragments_fr_phos_ester_featurizer",
    "Molecule_Fragments_fr_piperdine_Featurizer",
    "molecule_Fragments_fr_piperdine_featurizer",
    "Molecule_Fragments_fr_piperzine_Featurizer",
    "molecule_Fragments_fr_piperzine_featurizer",
    "Molecule_Fragments_fr_priamide_Featurizer",
    "molecule_Fragments_fr_priamide_featurizer",
    "Molecule_Fragments_fr_prisulfonamd_Featurizer",
    "molecule_Fragments_fr_prisulfonamd_featurizer",
    "Molecule_Fragments_fr_pyridine_Featurizer",
    "molecule_Fragments_fr_pyridine_featurizer",
    "Molecule_Fragments_fr_quatN_Featurizer",
    "molecule_Fragments_fr_quatN_featurizer",
    "Molecule_Fragments_fr_sulfide_Featurizer",
    "molecule_Fragments_fr_sulfide_featurizer",
    "Molecule_Fragments_fr_sulfonamd_Featurizer",
    "molecule_Fragments_fr_sulfonamd_featurizer",
    "Molecule_Fragments_fr_sulfone_Featurizer",
    "molecule_Fragments_fr_sulfone_featurizer",
    "Molecule_Fragments_fr_term_acetylene_Featurizer",
    "molecule_Fragments_fr_term_acetylene_featurizer",
    "Molecule_Fragments_fr_tetrazole_Featurizer",
    "molecule_Fragments_fr_tetrazole_featurizer",
    "Molecule_Fragments_fr_thiazole_Featurizer",
    "molecule_Fragments_fr_thiazole_featurizer",
    "Molecule_Fragments_fr_thiocyan_Featurizer",
    "molecule_Fragments_fr_thiocyan_featurizer",
    "Molecule_Fragments_fr_thiophene_Featurizer",
    "molecule_Fragments_fr_thiophene_featurizer",
    "Molecule_Fragments_fr_unbrch_alkane_Featurizer",
    "molecule_Fragments_fr_unbrch_alkane_featurizer",
    "Molecule_Fragments_fr_urea_Featurizer",
    "molecule_Fragments_fr_urea_featurizer",
    "Molecule_Lipinski_FractionCSP3_Featurizer",
    "molecule_Lipinski_FractionCSP3_featurizer",
    "Molecule_Lipinski_HeavyAtomCount_Featurizer",
    "molecule_Lipinski_HeavyAtomCount_featurizer",
    "Molecule_Lipinski_NHOHCount_Featurizer",
    "molecule_Lipinski_NHOHCount_featurizer",
    "Molecule_Lipinski_NOCount_Featurizer",
    "molecule_Lipinski_NOCount_featurizer",
    "Molecule_Lipinski_NumAliphaticCarbocycles_Featurizer",
    "molecule_Lipinski_NumAliphaticCarbocycles_featurizer",
    "Molecule_Lipinski_NumAliphaticHeterocycles_Featurizer",
    "molecule_Lipinski_NumAliphaticHeterocycles_featurizer",
    "Molecule_Lipinski_NumAliphaticRings_Featurizer",
    "molecule_Lipinski_NumAliphaticRings_featurizer",
    "Molecule_Lipinski_NumAromaticCarbocycles_Featurizer",
    "molecule_Lipinski_NumAromaticCarbocycles_featurizer",
    "Molecule_Lipinski_NumAromaticHeterocycles_Featurizer",
    "molecule_Lipinski_NumAromaticHeterocycles_featurizer",
    "Molecule_Lipinski_NumAromaticRings_Featurizer",
    "molecule_Lipinski_NumAromaticRings_featurizer",
    "Molecule_Lipinski_NumHAcceptors_Featurizer",
    "molecule_Lipinski_NumHAcceptors_featurizer",
    "Molecule_Lipinski_NumHDonors_Featurizer",
    "molecule_Lipinski_NumHDonors_featurizer",
    "Molecule_Lipinski_NumHeteroatoms_Featurizer",
    "molecule_Lipinski_NumHeteroatoms_featurizer",
    "Molecule_Lipinski_NumRotatableBonds_Featurizer",
    "molecule_Lipinski_NumRotatableBonds_featurizer",
    "Molecule_Lipinski_NumSaturatedCarbocycles_Featurizer",
    "molecule_Lipinski_NumSaturatedCarbocycles_featurizer",
    "Molecule_Lipinski_NumSaturatedHeterocycles_Featurizer",
    "molecule_Lipinski_NumSaturatedHeterocycles_featurizer",
    "Molecule_Lipinski_NumSaturatedRings_Featurizer",
    "molecule_Lipinski_NumSaturatedRings_featurizer",
    "Molecule_Lipinski_RingCount_Featurizer",
    "molecule_Lipinski_RingCount_featurizer",
    "Molecule_MolSurf_LabuteASA_Featurizer",
    "molecule_MolSurf_LabuteASA_featurizer",
    "Molecule_MolSurf_PEOE_VSA10_Featurizer",
    "molecule_MolSurf_PEOE_VSA10_featurizer",
    "Molecule_MolSurf_PEOE_VSA11_Featurizer",
    "molecule_MolSurf_PEOE_VSA11_featurizer",
    "Molecule_MolSurf_PEOE_VSA12_Featurizer",
    "molecule_MolSurf_PEOE_VSA12_featurizer",
    "Molecule_MolSurf_PEOE_VSA13_Featurizer",
    "molecule_MolSurf_PEOE_VSA13_featurizer",
    "Molecule_MolSurf_PEOE_VSA14_Featurizer",
    "molecule_MolSurf_PEOE_VSA14_featurizer",
    "Molecule_MolSurf_PEOE_VSA1_Featurizer",
    "molecule_MolSurf_PEOE_VSA1_featurizer",
    "Molecule_MolSurf_PEOE_VSA2_Featurizer",
    "molecule_MolSurf_PEOE_VSA2_featurizer",
    "Molecule_MolSurf_PEOE_VSA3_Featurizer",
    "molecule_MolSurf_PEOE_VSA3_featurizer",
    "Molecule_MolSurf_PEOE_VSA4_Featurizer",
    "molecule_MolSurf_PEOE_VSA4_featurizer",
    "Molecule_MolSurf_PEOE_VSA5_Featurizer",
    "molecule_MolSurf_PEOE_VSA5_featurizer",
    "Molecule_MolSurf_PEOE_VSA6_Featurizer",
    "molecule_MolSurf_PEOE_VSA6_featurizer",
    "Molecule_MolSurf_PEOE_VSA7_Featurizer",
    "molecule_MolSurf_PEOE_VSA7_featurizer",
    "Molecule_MolSurf_PEOE_VSA8_Featurizer",
    "molecule_MolSurf_PEOE_VSA8_featurizer",
    "Molecule_MolSurf_PEOE_VSA9_Featurizer",
    "molecule_MolSurf_PEOE_VSA9_featurizer",
    "Molecule_MolSurf_SMR_VSA10_Featurizer",
    "molecule_MolSurf_SMR_VSA10_featurizer",
    "Molecule_MolSurf_SMR_VSA1_Featurizer",
    "molecule_MolSurf_SMR_VSA1_featurizer",
    "Molecule_MolSurf_SMR_VSA2_Featurizer",
    "molecule_MolSurf_SMR_VSA2_featurizer",
    "Molecule_MolSurf_SMR_VSA3_Featurizer",
    "molecule_MolSurf_SMR_VSA3_featurizer",
    "Molecule_MolSurf_SMR_VSA4_Featurizer",
    "molecule_MolSurf_SMR_VSA4_featurizer",
    "Molecule_MolSurf_SMR_VSA5_Featurizer",
    "molecule_MolSurf_SMR_VSA5_featurizer",
    "Molecule_MolSurf_SMR_VSA6_Featurizer",
    "molecule_MolSurf_SMR_VSA6_featurizer",
    "Molecule_MolSurf_SMR_VSA7_Featurizer",
    "molecule_MolSurf_SMR_VSA7_featurizer",
    "Molecule_MolSurf_SMR_VSA8_Featurizer",
    "molecule_MolSurf_SMR_VSA8_featurizer",
    "Molecule_MolSurf_SMR_VSA9_Featurizer",
    "molecule_MolSurf_SMR_VSA9_featurizer",
    "Molecule_MolSurf_SlogP_VSA10_Featurizer",
    "molecule_MolSurf_SlogP_VSA10_featurizer",
    "Molecule_MolSurf_SlogP_VSA11_Featurizer",
    "molecule_MolSurf_SlogP_VSA11_featurizer",
    "Molecule_MolSurf_SlogP_VSA12_Featurizer",
    "molecule_MolSurf_SlogP_VSA12_featurizer",
    "Molecule_MolSurf_SlogP_VSA1_Featurizer",
    "molecule_MolSurf_SlogP_VSA1_featurizer",
    "Molecule_MolSurf_SlogP_VSA2_Featurizer",
    "molecule_MolSurf_SlogP_VSA2_featurizer",
    "Molecule_MolSurf_SlogP_VSA3_Featurizer",
    "molecule_MolSurf_SlogP_VSA3_featurizer",
    "Molecule_MolSurf_SlogP_VSA4_Featurizer",
    "molecule_MolSurf_SlogP_VSA4_featurizer",
    "Molecule_MolSurf_SlogP_VSA5_Featurizer",
    "molecule_MolSurf_SlogP_VSA5_featurizer",
    "Molecule_MolSurf_SlogP_VSA6_Featurizer",
    "molecule_MolSurf_SlogP_VSA6_featurizer",
    "Molecule_MolSurf_SlogP_VSA7_Featurizer",
    "molecule_MolSurf_SlogP_VSA7_featurizer",
    "Molecule_MolSurf_SlogP_VSA8_Featurizer",
    "molecule_MolSurf_SlogP_VSA8_featurizer",
    "Molecule_MolSurf_SlogP_VSA9_Featurizer",
    "molecule_MolSurf_SlogP_VSA9_featurizer",
    "Molecule_MolSurf_TPSA_Featurizer",
    "molecule_MolSurf_TPSA_featurizer",
    "Molecule_MolSurf_pyLabuteASA_Featurizer",
    "molecule_MolSurf_pyLabuteASA_featurizer",
    "Molecule_NumAtoms_Featurizer",
    "molecule_NumAtoms_featurizer",
    "Molecule_NumBonds_Featurizer",
    "molecule_NumBonds_featurizer",
    "Molecule_NumHeavyAtoms_Featurizer",
    "molecule_NumHeavyAtoms_featurizer",
    "Molecule_QED_default_Featurizer",
    "molecule_QED_default_featurizer",
    "Molecule_QED_qed_Featurizer",
    "molecule_QED_qed_featurizer",
    "Molecule_QED_weights_max_Featurizer",
    "molecule_QED_weights_max_featurizer",
    "Molecule_QED_weights_mean_Featurizer",
    "molecule_QED_weights_mean_featurizer",
    "Molecule_QED_weights_none_Featurizer",
    "molecule_QED_weights_none_featurizer",
    "Molecule_rdmolops_FormalCharge_Featurizer",
    "molecule_rdmolops_FormalCharge_featurizer",
    "Molecule_rdmolops_SSSR_Featurizer",
    "molecule_rdmolops_SSSR_featurizer",
]


def get_available_featurizer():
    return _available_featurizer


def main():
    from rdkit import Chem
    from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization

    testdata = prepare_mol_for_featurization(Chem.MolFromSmiles("c1ccccc1"))
    for n, f in get_available_featurizer().items():
        print(n, end=" ")
        print(f(testdata))
    print(len(get_available_featurizer()))


if __name__ == "__main__":
    main()
